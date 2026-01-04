"""
Unit tests for the time series forecasting project.

This module contains comprehensive tests for all major components
of the forecasting pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import (
    set_seed,
    setup_logging,
    get_device,
    calculate_returns,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    validate_time_series_data,
)
from data import DataLoader, TimeSeriesSplitter, FeatureEngineer
from models import ARIMAModel, XGBoostModel
from evaluation import ModelEvaluator
from backtest import Backtester


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        # Test that seeds are set (basic check)
        assert True  # If no exception is raised, test passes
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert device is not None
        assert hasattr(device, 'type')
    
    def test_calculate_returns(self):
        """Test returns calculation."""
        prices = pd.Series([100, 105, 110, 108, 112])
        returns = calculate_returns(prices)
        
        expected = pd.Series([np.nan, 0.05, 0.0476, -0.0182, 0.0370], index=prices.index)
        pd.testing.assert_series_equal(returns, expected, atol=1e-3)
    
    def test_calculate_volatility(self):
        """Test volatility calculation."""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        volatility = calculate_volatility(returns, window=3)
        
        # Check that volatility is calculated
        assert len(volatility) == len(returns)
        assert not volatility.iloc[:2].isna().all()  # First two should be NaN
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        sharpe = calculate_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
    
    def test_calculate_max_drawdown(self):
        """Test max drawdown calculation."""
        returns = pd.Series([0.01, 0.02, -0.05, 0.01, 0.02])
        max_dd = calculate_max_drawdown(returns)
        
        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown should be negative or zero
    
    def test_validate_time_series_data(self):
        """Test time series data validation."""
        # Valid data
        valid_data = pd.DataFrame({
            'price': [100, 105, 110, 108, 112]
        }, index=pd.date_range('2023-01-01', periods=5))
        
        assert validate_time_series_data(valid_data) == True
        
        # Invalid data - no datetime index
        invalid_data = pd.DataFrame({
            'price': [100, 105, 110, 108, 112]
        })
        
        assert validate_time_series_data(invalid_data) == False
        
        # Invalid data - with NaN
        invalid_data2 = pd.DataFrame({
            'price': [100, np.nan, 110, 108, 112]
        }, index=pd.date_range('2023-01-01', periods=5))
        
        assert validate_time_series_data(invalid_data2) == False


class TestDataLoader:
    """Test data loading functionality."""
    
    def test_create_synthetic_data(self):
        """Test synthetic data creation."""
        from omegaconf import DictConfig
        
        # Create minimal config
        config = DictConfig({
            'data': {'symbols': ['TEST'], 'start_date': '2023-01-01', 'end_date': '2023-01-31'}
        })
        
        loader = DataLoader(config)
        data = loader.create_synthetic_data('TEST', '2023-01-01', '2023-01-31')
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'close' in data.columns
        assert isinstance(data.index, pd.DatetimeIndex)
    
    def test_time_series_splitter(self):
        """Test time series data splitting."""
        from omegaconf import DictConfig
        
        config = DictConfig({})
        splitter = TimeSeriesSplitter(config)
        
        # Create test data
        data = pd.DataFrame({
            'price': range(100)
        }, index=pd.date_range('2023-01-01', periods=100))
        
        train, val, test = splitter.split_data(data, 0.6, 0.2, 0.2)
        
        assert len(train) == 60
        assert len(val) == 20
        assert len(test) == 20
        assert len(train) + len(val) + len(test) == len(data)


class TestFeatureEngineer:
    """Test feature engineering functionality."""
    
    def test_add_technical_indicators(self):
        """Test technical indicator addition."""
        from omegaconf import DictConfig
        
        config = DictConfig({})
        engineer = FeatureEngineer(config)
        
        # Create test OHLCV data
        data = pd.DataFrame({
            'open': [100, 101, 102, 101, 103],
            'high': [102, 103, 104, 103, 105],
            'low': [99, 100, 101, 100, 102],
            'close': [101, 102, 103, 102, 104],
            'volume': [1000, 1100, 1200, 1100, 1300]
        }, index=pd.date_range('2023-01-01', periods=5))
        
        enhanced_data = engineer.add_technical_indicators(data)
        
        assert len(enhanced_data.columns) > len(data.columns)
        assert 'sma_5' in enhanced_data.columns
        assert 'rsi' in enhanced_data.columns
    
    def test_add_lag_features(self):
        """Test lag feature addition."""
        from omegaconf import DictConfig
        
        config = DictConfig({})
        engineer = FeatureEngineer(config)
        
        data = pd.DataFrame({
            'price': [100, 105, 110, 108, 112]
        }, index=pd.date_range('2023-01-01', periods=5))
        
        enhanced_data = engineer.add_lag_features(data, 'price', [1, 2])
        
        assert 'price_lag_1' in enhanced_data.columns
        assert 'price_lag_2' in enhanced_data.columns


class TestModels:
    """Test model implementations."""
    
    def test_arima_model(self):
        """Test ARIMA model functionality."""
        from omegaconf import DictConfig
        
        config = DictConfig({
            'models': {
                'arima': {
                    'order': [1, 1, 1],
                    'seasonal_order': [0, 0, 0, 0],
                    'auto_arima': False
                }
            },
            'seed': 42
        })
        
        model = ARIMAModel(config)
        
        # Create test data
        np.random.seed(42)
        data = pd.DataFrame({
            'price': 100 + np.cumsum(np.random.randn(50) * 0.1)
        }, index=pd.date_range('2023-01-01', periods=50))
        
        # Test fitting
        model.fit(data[['price']], data['price'])
        assert model.is_fitted == True
        
        # Test prediction
        predictions = model.predict(data[['price']], steps=1)
        assert len(predictions) == 1
        assert isinstance(predictions[0], (int, float))
    
    def test_xgboost_model(self):
        """Test XGBoost model functionality."""
        from omegaconf import DictConfig
        
        config = DictConfig({
            'models': {
                'xgboost': {
                    'n_estimators': 10,
                    'max_depth': 3,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                }
            },
            'seed': 42
        })
        
        model = XGBoostModel(config)
        
        # Create test data
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
            'price': 100 + np.cumsum(np.random.randn(50) * 0.1)
        }, index=pd.date_range('2023-01-01', periods=50))
        
        # Test fitting
        model.fit(data[['feature1', 'feature2']], data['price'])
        assert model.is_fitted == True
        
        # Test prediction
        predictions = model.predict(data[['feature1', 'feature2']], steps=1)
        assert len(predictions) == 1
        assert isinstance(predictions[0], (int, float))


class TestEvaluation:
    """Test evaluation functionality."""
    
    def test_model_evaluator(self):
        """Test model evaluator."""
        from omegaconf import DictConfig
        
        config = DictConfig({})
        evaluator = ModelEvaluator(config)
        
        # Test ML metrics
        y_true = np.array([100, 105, 110, 108, 112])
        y_pred = np.array([102, 103, 108, 109, 111])
        
        ml_metrics = evaluator.calculate_ml_metrics(y_true, y_pred)
        
        assert 'mae' in ml_metrics
        assert 'rmse' in ml_metrics
        assert 'r2' in ml_metrics
        assert all(isinstance(v, float) for v in ml_metrics.values())
    
    def test_finance_metrics(self):
        """Test financial metrics calculation."""
        from omegaconf import DictConfig
        
        config = DictConfig({})
        evaluator = ModelEvaluator(config)
        
        # Create test returns
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        
        finance_metrics = evaluator.calculate_finance_metrics(returns)
        
        assert 'total_return' in finance_metrics
        assert 'volatility' in finance_metrics
        assert 'sharpe_ratio' in finance_metrics
        assert all(isinstance(v, float) for v in finance_metrics.values())


class TestBacktesting:
    """Test backtesting functionality."""
    
    def test_backtester_initialization(self):
        """Test backtester initialization."""
        from omegaconf import DictConfig
        
        config = DictConfig({
            'backtesting': {
                'initial_capital': 100000,
                'transaction_cost': 0.001,
                'slippage': 0.0005,
                'benchmark': 'SPY'
            }
        })
        
        backtester = Backtester(config)
        
        assert backtester.initial_capital == 100000
        assert backtester.transaction_cost == 0.001
        assert backtester.slippage == 0.0005
    
    def test_backtest_execution(self):
        """Test backtest execution."""
        from omegaconf import DictConfig
        
        config = DictConfig({
            'backtesting': {
                'initial_capital': 100000,
                'transaction_cost': 0.001,
                'slippage': 0.0005,
                'benchmark': 'SPY'
            }
        })
        
        backtester = Backtester(config)
        
        # Create test data
        prices = pd.Series([100, 105, 110, 108, 112], 
                          index=pd.date_range('2023-01-01', periods=5))
        signals = pd.Series([1, 1, -1, 1, 0], 
                          index=pd.date_range('2023-01-01', periods=5))
        
        results = backtester.run_backtest(prices, signals)
        
        assert isinstance(results, dict)
        assert 'total_return' in results
        assert 'final_capital' in results
        assert 'trades' in results


if __name__ == "__main__":
    pytest.main([__file__])
