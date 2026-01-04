"""
Data handling module for time series forecasting.

This module provides classes and functions for downloading, processing,
and managing financial time series data with proper leakage prevention.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yfinance as yf
from omegaconf import DictConfig

from ..utils import ensure_dir, validate_time_series_data


class DataLoader:
    """Data loader for financial time series data."""
    
    def __init__(self, config: DictConfig):
        """Initialize data loader.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("data")
        ensure_dir(self.data_dir)
    
    def download_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        save: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Download stock data from Yahoo Finance.
        
        Args:
            symbols: List of stock symbols.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            save: Whether to save data to disk.
            
        Returns:
            Dictionary mapping symbols to DataFrames.
        """
        data = {}
        
        for symbol in symbols:
            self.logger.info(f"Downloading data for {symbol}")
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                
                if df.empty:
                    self.logger.warning(f"No data found for {symbol}")
                    continue
                
                # Clean column names
                df.columns = [col.lower() for col in df.columns]
                df.index.name = "date"
                
                # Validate data
                if not validate_time_series_data(df):
                    self.logger.warning(f"Invalid data for {symbol}")
                    continue
                
                data[symbol] = df
                
                if save:
                    file_path = self.data_dir / f"{symbol.lower()}_data.csv"
                    df.to_csv(file_path)
                    self.logger.info(f"Saved data for {symbol} to {file_path}")
                
            except Exception as e:
                self.logger.error(f"Error downloading {symbol}: {e}")
                continue
        
        return data
    
    def load_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data from disk.
        
        Args:
            symbol: Stock symbol.
            
        Returns:
            DataFrame or None if not found.
        """
        file_path = self.data_dir / f"{symbol.lower()}_data.csv"
        
        if not file_path.exists():
            self.logger.warning(f"Data file not found: {file_path}")
            return None
        
        try:
            df = pd.read_csv(file_path, index_col="date", parse_dates=True)
            return df
        except Exception as e:
            self.logger.error(f"Error loading data for {symbol}: {e}")
            return None
    
    def create_synthetic_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_price: float = 100.0,
        volatility: float = 0.2,
        drift: float = 0.05,
    ) -> pd.DataFrame:
        """Create synthetic stock data for testing.
        
        Args:
            symbol: Stock symbol.
            start_date: Start date.
            end_date: End date.
            initial_price: Initial price.
            volatility: Annual volatility.
            drift: Annual drift.
            
        Returns:
            Synthetic DataFrame.
        """
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        dates = dates[dates.weekday < 5]  # Only weekdays
        
        n_days = len(dates)
        dt = 1 / 252  # Daily time step
        
        # Generate random returns using GBM
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(
            drift * dt, volatility * np.sqrt(dt), n_days
        )
        
        # Calculate prices
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Create OHLCV data
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Simple OHLCV generation
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i - 1] if i > 0 else price
            volume = np.random.randint(1000000, 10000000)
            
            data.append({
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "volume": volume,
            })
        
        df = pd.DataFrame(data, index=dates)
        df.index.name = "date"
        
        # Save synthetic data
        file_path = self.data_dir / f"{symbol.lower()}_data.csv"
        df.to_csv(file_path)
        
        return df


class TimeSeriesSplitter:
    """Time series data splitter with leakage prevention."""
    
    def __init__(self, config: DictConfig):
        """Initialize splitter.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def split_data(
        self,
        data: pd.DataFrame,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets.
        
        Args:
            data: Input DataFrame.
            train_split: Training set proportion.
            val_split: Validation set proportion.
            test_split: Test set proportion.
            
        Returns:
            Tuple of (train, val, test) DataFrames.
        """
        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            raise ValueError("Splits must sum to 1.0")
        
        n_samples = len(data)
        train_end = int(n_samples * train_split)
        val_end = int(n_samples * (train_split + val_split))
        
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]
        
        self.logger.info(
            f"Split data: train={len(train_data)}, "
            f"val={len(val_data)}, test={len(test_data)}"
        )
        
        return train_data, val_data, test_data
    
    def walk_forward_split(
        self,
        data: pd.DataFrame,
        n_windows: int = 5,
        min_train_size: int = 252,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create walk-forward validation splits.
        
        Args:
            data: Input DataFrame.
            n_windows: Number of validation windows.
            min_train_size: Minimum training set size.
            
        Returns:
            List of (train, test) tuples.
        """
        splits = []
        n_samples = len(data)
        
        # Calculate step size
        step_size = (n_samples - min_train_size) // n_windows
        
        for i in range(n_windows):
            train_end = min_train_size + i * step_size
            test_end = min(train_end + step_size, n_samples)
            
            if test_end <= train_end:
                break
            
            train_data = data.iloc[:train_end]
            test_data = data.iloc[train_end:test_end]
            
            splits.append((train_data, test_data))
        
        self.logger.info(f"Created {len(splits)} walk-forward splits")
        return splits


class FeatureEngineer:
    """Feature engineering for time series data."""
    
    def __init__(self, config: DictConfig):
        """Initialize feature engineer.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data.
        
        Args:
            data: Input DataFrame with OHLCV data.
            
        Returns:
            DataFrame with additional technical indicators.
        """
        df = data.copy()
        
        # Simple Moving Averages
        for window in [5, 10, 20, 50]:
            df[f"sma_{window}"] = df["close"].rolling(window=window).mean()
        
        # Exponential Moving Averages
        for span in [12, 26]:
            df[f"ema_{span}"] = df["close"].ewm(span=span).mean()
        
        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        bb_std = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
        df["bb_width"] = df["bb_upper"] - df["bb_lower"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / df["bb_width"]
        
        # Volume indicators
        df["volume_sma"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]
        
        # Price-based features
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        df["volatility"] = df["returns"].rolling(window=20).std()
        
        # High-Low features
        df["hl_ratio"] = df["high"] / df["low"]
        df["oc_ratio"] = df["open"] / df["close"]
        
        self.logger.info(f"Added {len(df.columns) - len(data.columns)} technical indicators")
        return df
    
    def add_lag_features(
        self,
        data: pd.DataFrame,
        target_col: str,
        lags: List[int],
    ) -> pd.DataFrame:
        """Add lagged features.
        
        Args:
            data: Input DataFrame.
            target_col: Target column name.
            lags: List of lag periods.
            
        Returns:
            DataFrame with lagged features.
        """
        df = data.copy()
        
        for lag in lags:
            df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
        
        self.logger.info(f"Added {len(lags)} lag features")
        return df
    
    def add_rolling_features(
        self,
        data: pd.DataFrame,
        target_col: str,
        windows: List[int],
        functions: List[str] = ["mean", "std", "min", "max"],
    ) -> pd.DataFrame:
        """Add rolling window features.
        
        Args:
            data: Input DataFrame.
            target_col: Target column name.
            windows: List of window sizes.
            functions: List of aggregation functions.
            
        Returns:
            DataFrame with rolling features.
        """
        df = data.copy()
        
        for window in windows:
            for func in functions:
                df[f"{target_col}_rolling_{func}_{window}"] = getattr(
                    df[target_col].rolling(window=window), func
                )()
        
        self.logger.info(f"Added {len(windows) * len(functions)} rolling features")
        return df
