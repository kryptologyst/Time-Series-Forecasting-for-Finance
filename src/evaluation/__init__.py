"""
Evaluation framework for time series forecasting models.

This module provides comprehensive evaluation metrics for both
machine learning performance and financial performance.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..utils import calculate_max_drawdown, calculate_sharpe_ratio


class ModelEvaluator:
    """Comprehensive model evaluator for time series forecasting."""
    
    def __init__(self, config: DictConfig):
        """Initialize evaluator.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_ml_metrics(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
    ) -> Dict[str, float]:
        """Calculate machine learning metrics.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            
        Returns:
            Dictionary of ML metrics.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return {}
        
        metrics = {}
        
        # Basic regression metrics
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["mse"] = mean_squared_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["r2"] = r2_score(y_true, y_pred)
        
        # Percentage errors
        mask_nonzero = y_true != 0
        if np.any(mask_nonzero):
            mape = np.mean(np.abs((y_true[mask_nonzero] - y_pred[mask_nonzero]) / y_true[mask_nonzero])) * 100
            metrics["mape"] = mape
        
        # Symmetric Mean Absolute Percentage Error
        smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
        metrics["smape"] = smape
        
        # Mean Absolute Scaled Error (MASE)
        if len(y_true) > 1:
            naive_mae = np.mean(np.abs(np.diff(y_true)))
            if naive_mae != 0:
                mase = metrics["mae"] / naive_mae
                metrics["mase"] = mase
        
        return metrics
    
    def calculate_finance_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02,
    ) -> Dict[str, float]:
        """Calculate financial performance metrics.
        
        Args:
            returns: Strategy returns.
            benchmark_returns: Benchmark returns for comparison.
            risk_free_rate: Risk-free rate (annual).
            
        Returns:
            Dictionary of financial metrics.
        """
        metrics = {}
        
        if len(returns) == 0:
            return metrics
        
        # Basic return metrics
        metrics["total_return"] = (1 + returns).prod() - 1
        metrics["annualized_return"] = (1 + returns).prod() ** (252 / len(returns)) - 1
        metrics["volatility"] = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        metrics["sharpe_ratio"] = calculate_sharpe_ratio(returns, risk_free_rate)
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std() * np.sqrt(252)
            excess_return = metrics["annualized_return"] - risk_free_rate
            metrics["sortino_ratio"] = excess_return / downside_std if downside_std != 0 else 0
        else:
            metrics["sortino_ratio"] = float('inf') if metrics["annualized_return"] > risk_free_rate else 0
        
        # Calmar ratio
        max_dd = calculate_max_drawdown(returns)
        metrics["max_drawdown"] = max_dd
        metrics["calmar_ratio"] = metrics["annualized_return"] / abs(max_dd) if max_dd != 0 else 0
        
        # Hit rate
        positive_returns = returns > 0
        metrics["hit_rate"] = positive_returns.mean() if len(returns) > 0 else 0
        
        # Skewness and Kurtosis
        metrics["skewness"] = returns.skew()
        metrics["kurtosis"] = returns.kurtosis()
        
        # Benchmark comparison
        if benchmark_returns is not None:
            # Align returns
            aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
            
            if len(aligned_returns) > 0:
                # Beta calculation
                covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                benchmark_variance = np.var(aligned_benchmark)
                metrics["beta"] = covariance / benchmark_variance if benchmark_variance != 0 else 0
                
                # Alpha calculation
                benchmark_return = aligned_benchmark.mean() * 252
                metrics["alpha"] = metrics["annualized_return"] - risk_free_rate - metrics["beta"] * (benchmark_return - risk_free_rate)
                
                # Information ratio
                excess_returns = aligned_returns - aligned_benchmark
                metrics["information_ratio"] = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 else 0
        
        return metrics
    
    def calculate_trading_metrics(
        self,
        prices: pd.Series,
        predictions: np.ndarray,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
    ) -> Dict[str, float]:
        """Calculate trading strategy metrics.
        
        Args:
            prices: Actual prices.
            predictions: Predicted prices.
            transaction_cost: Transaction cost per trade.
            slippage: Slippage per trade.
            
        Returns:
            Dictionary of trading metrics.
        """
        metrics = {}
        
        if len(prices) != len(predictions):
            self.logger.warning("Price and prediction lengths don't match")
            return metrics
        
        # Generate trading signals (simple momentum strategy)
        signals = np.where(predictions > prices.shift(1), 1, -1)
        signals[0] = 0  # No signal for first period
        
        # Calculate returns
        returns = prices.pct_change()
        
        # Apply transaction costs
        total_cost = transaction_cost + slippage
        strategy_returns = signals * returns - np.abs(np.diff(signals, prepend=0)) * total_cost
        
        # Calculate metrics
        metrics.update(self.calculate_finance_metrics(pd.Series(strategy_returns)))
        
        # Additional trading metrics
        metrics["total_trades"] = np.sum(np.abs(np.diff(signals, prepend=0)))
        metrics["avg_trade_return"] = strategy_returns.mean()
        metrics["win_rate"] = (strategy_returns > 0).mean()
        
        return metrics
    
    def evaluate_model(
        self,
        model_name: str,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        prices: Optional[pd.Series] = None,
        benchmark_prices: Optional[pd.Series] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Comprehensive model evaluation.
        
        Args:
            model_name: Name of the model.
            y_true: True values.
            y_pred: Predicted values.
            prices: Price series for trading metrics.
            benchmark_prices: Benchmark prices.
            
        Returns:
            Dictionary containing all evaluation metrics.
        """
        results = {"model": model_name}
        
        # ML metrics
        results["ml_metrics"] = self.calculate_ml_metrics(y_true, y_pred)
        
        # Financial metrics
        if prices is not None:
            returns = prices.pct_change().dropna()
            benchmark_returns = None
            
            if benchmark_prices is not None:
                benchmark_returns = benchmark_prices.pct_change().dropna()
            
            results["finance_metrics"] = self.calculate_finance_metrics(
                returns, benchmark_returns
            )
            
            # Trading metrics
            results["trading_metrics"] = self.calculate_trading_metrics(
                prices, y_pred
            )
        
        return results
    
    def create_evaluation_report(
        self,
        results: List[Dict[str, Dict[str, float]]],
    ) -> pd.DataFrame:
        """Create a comprehensive evaluation report.
        
        Args:
            results: List of model evaluation results.
            
        Returns:
            DataFrame with evaluation metrics.
        """
        report_data = []
        
        for result in results:
            model_name = result["model"]
            row = {"Model": model_name}
            
            # Add ML metrics
            if "ml_metrics" in result:
                for metric, value in result["ml_metrics"].items():
                    row[f"ML_{metric.upper()}"] = value
            
            # Add finance metrics
            if "finance_metrics" in result:
                for metric, value in result["finance_metrics"].items():
                    row[f"FIN_{metric.upper()}"] = value
            
            # Add trading metrics
            if "trading_metrics" in result:
                for metric, value in result["trading_metrics"].items():
                    row[f"TRADE_{metric.upper()}"] = value
            
            report_data.append(row)
        
        return pd.DataFrame(report_data)
    
    def plot_evaluation_results(
        self,
        results: List[Dict[str, Dict[str, float]]],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot evaluation results.
        
        Args:
            results: List of model evaluation results.
            save_path: Optional path to save plots.
        """
        import matplotlib.pyplot as plt
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Model Evaluation Results", fontsize=16)
        
        # Extract data for plotting
        models = [r["model"] for r in results]
        
        # ML Metrics
        ml_metrics = ["MAE", "RMSE", "MAPE", "R2"]
        ml_data = {metric: [] for metric in ml_metrics}
        
        for result in results:
            if "ml_metrics" in result:
                for metric in ml_metrics:
                    ml_data[metric].append(result["ml_metrics"].get(metric.lower(), 0))
            else:
                for metric in ml_metrics:
                    ml_data[metric].append(0)
        
        # Plot ML metrics
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(ml_metrics):
            axes[0, 0].bar(x + i * width, ml_data[metric], width, label=metric)
        
        axes[0, 0].set_title("ML Metrics")
        axes[0, 0].set_xlabel("Models")
        axes[0, 0].set_ylabel("Score")
        axes[0, 0].set_xticks(x + width * 1.5)
        axes[0, 0].set_xticklabels(models, rotation=45)
        axes[0, 0].legend()
        
        # Finance Metrics
        fin_metrics = ["SHARPE_RATIO", "SORTINO_RATIO", "CALMAR_RATIO", "MAX_DRAWDOWN"]
        fin_data = {metric: [] for metric in fin_metrics}
        
        for result in results:
            if "finance_metrics" in result:
                for metric in fin_metrics:
                    fin_data[metric].append(result["finance_metrics"].get(metric.lower(), 0))
            else:
                for metric in fin_metrics:
                    fin_data[metric].append(0)
        
        # Plot finance metrics
        for i, metric in enumerate(fin_metrics):
            axes[0, 1].bar(x + i * width, fin_data[metric], width, label=metric)
        
        axes[0, 1].set_title("Finance Metrics")
        axes[0, 1].set_xlabel("Models")
        axes[0, 1].set_ylabel("Score")
        axes[0, 1].set_xticks(x + width * 1.5)
        axes[0, 1].set_xticklabels(models, rotation=45)
        axes[0, 1].legend()
        
        # Trading Metrics
        trade_metrics = ["HIT_RATE", "WIN_RATE", "TOTAL_TRADES"]
        trade_data = {metric: [] for metric in trade_metrics}
        
        for result in results:
            if "trading_metrics" in result:
                for metric in trade_metrics:
                    trade_data[metric].append(result["trading_metrics"].get(metric.lower(), 0))
            else:
                for metric in trade_metrics:
                    trade_data[metric].append(0)
        
        # Plot trading metrics
        for i, metric in enumerate(trade_metrics):
            axes[1, 0].bar(x + i * width, trade_data[metric], width, label=metric)
        
        axes[1, 0].set_title("Trading Metrics")
        axes[1, 0].set_xlabel("Models")
        axes[1, 0].set_ylabel("Score")
        axes[1, 0].set_xticks(x + width * 1.5)
        axes[1, 0].set_xticklabels(models, rotation=45)
        axes[1, 0].legend()
        
        # Performance comparison
        if "finance_metrics" in results[0]:
            returns = [r["finance_metrics"].get("annualized_return", 0) for r in results]
            volatilities = [r["finance_metrics"].get("volatility", 0) for r in results]
            
            axes[1, 1].scatter(volatilities, returns, s=100, alpha=0.7)
            axes[1, 1].set_xlabel("Volatility")
            axes[1, 1].set_ylabel("Annualized Return")
            axes[1, 1].set_title("Risk-Return Profile")
            
            # Add model labels
            for i, model in enumerate(models):
                axes[1, 1].annotate(model, (volatilities[i], returns[i]), 
                                   xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Evaluation plots saved to {save_path}")
        
        plt.show()
