"""
Core utilities for the Time Series Forecasting project.

This module provides essential utilities for data handling, configuration,
logging, and reproducibility across the project.
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        level: Logging level.
        
    Returns:
        Configured logger.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration object.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return OmegaConf.load(config_path)


def get_device() -> torch.device:
    """Get the best available device (CUDA, MPS, or CPU).
    
    Returns:
        PyTorch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path.
        
    Returns:
        Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pickle(obj: Any, path: Union[str, Path]) -> None:
    """Save object to pickle file.
    
    Args:
        obj: Object to save.
        path: File path.
    """
    import pickle
    
    path = Path(path)
    ensure_dir(path.parent)
    
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Union[str, Path]) -> Any:
    """Load object from pickle file.
    
    Args:
        path: File path.
        
    Returns:
        Loaded object.
    """
    import pickle
    
    with open(path, "rb") as f:
        return pickle.load(f)


def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate returns for given periods.
    
    Args:
        prices: Price series.
        periods: Number of periods for return calculation.
        
    Returns:
        Returns series.
    """
    return prices.pct_change(periods=periods)


def calculate_volatility(returns: pd.Series, window: int = 252) -> pd.Series:
    """Calculate rolling volatility.
    
    Args:
        returns: Returns series.
        window: Rolling window size.
        
    Returns:
        Volatility series.
    """
    return returns.rolling(window=window).std() * np.sqrt(252)


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio.
    
    Args:
        returns: Returns series.
        risk_free_rate: Risk-free rate.
        
    Returns:
        Sharpe ratio.
    """
    excess_returns = returns.mean() - risk_free_rate / 252
    return excess_returns / returns.std() * np.sqrt(252)


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown.
    
    Args:
        returns: Returns series.
        
    Returns:
        Maximum drawdown.
    """
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()


def validate_time_series_data(data: pd.DataFrame) -> bool:
    """Validate time series data for common issues.
    
    Args:
        data: DataFrame to validate.
        
    Returns:
        True if valid, False otherwise.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        return False
    
    if data.isnull().any().any():
        return False
    
    if len(data) < 2:
        return False
    
    return True
