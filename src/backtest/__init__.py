"""
Backtesting framework for time series forecasting strategies.

This module provides realistic backtesting capabilities with
transaction costs, slippage, and proper position sizing.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from ..utils import calculate_max_drawdown, calculate_sharpe_ratio


class Backtester:
    """Realistic backtesting engine for trading strategies."""
    
    def __init__(self, config: DictConfig):
        """Initialize backtester.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Backtesting parameters
        self.initial_capital = config.backtesting.initial_capital
        self.transaction_cost = config.backtesting.transaction_cost
        self.slippage = config.backtesting.slippage
        self.benchmark_symbol = config.backtesting.benchmark
        
        # Portfolio state
        self.reset()
    
    def reset(self) -> None:
        """Reset backtester state."""
        self.capital = self.initial_capital
        self.position = 0.0  # Number of shares
        self.cash = self.initial_capital
        self.trades = []
        self.portfolio_values = []
        self.returns = []
    
    def _calculate_position_size(
        self,
        price: float,
        signal: float,
        method: str = "fixed",
        risk_per_trade: float = 0.02,
    ) -> float:
        """Calculate position size based on signal and risk management.
        
        Args:
            price: Current price.
            signal: Trading signal (-1, 0, 1).
            method: Position sizing method.
            risk_per_trade: Risk per trade as fraction of capital.
            
        Returns:
            Position size in shares.
        """
        if signal == 0:
            return 0
        
        if method == "fixed":
            # Fixed dollar amount
            dollar_amount = self.capital * risk_per_trade
            return (dollar_amount / price) * signal
        
        elif method == "kelly":
            # Kelly criterion (simplified)
            # This is a placeholder - in practice, you'd need win rate and avg win/loss
            kelly_fraction = 0.1  # Simplified
            dollar_amount = self.capital * kelly_fraction
            return (dollar_amount / price) * signal
        
        elif method == "volatility_target":
            # Volatility targeting
            target_vol = 0.15  # 15% annual volatility target
            # This would require historical volatility calculation
            dollar_amount = self.capital * risk_per_trade
            return (dollar_amount / price) * signal
        
        else:
            # Default to fixed
            dollar_amount = self.capital * risk_per_trade
            return (dollar_amount / price) * signal
    
    def _execute_trade(
        self,
        price: float,
        signal: float,
        timestamp: pd.Timestamp,
        position_sizing_method: str = "fixed",
    ) -> Dict:
        """Execute a trade with realistic costs.
        
        Args:
            price: Execution price.
            signal: Trading signal.
            timestamp: Trade timestamp.
            position_sizing_method: Position sizing method.
            
        Returns:
            Trade details dictionary.
        """
        # Calculate desired position
        desired_position = self._calculate_position_size(
            price, signal, position_sizing_method
        )
        
        # Calculate position change
        position_change = desired_position - self.position
        
        if abs(position_change) < 1e-6:  # No significant change
            return None
        
        # Apply slippage
        execution_price = price * (1 + self.slippage * np.sign(position_change))
        
        # Calculate trade value
        trade_value = abs(position_change) * execution_price
        
        # Calculate costs
        transaction_cost = trade_value * self.transaction_cost
        
        # Check if we have enough cash
        required_cash = trade_value + transaction_cost
        
        if position_change > 0 and required_cash > self.cash:
            # Can't afford the trade
            self.logger.warning(f"Insufficient cash for trade at {timestamp}")
            return None
        
        # Execute trade
        self.position += position_change
        self.cash -= required_cash if position_change > 0 else -(trade_value - transaction_cost)
        
        # Record trade
        trade = {
            "timestamp": timestamp,
            "signal": signal,
            "price": price,
            "execution_price": execution_price,
            "position_change": position_change,
            "position": self.position,
            "trade_value": trade_value,
            "transaction_cost": transaction_cost,
            "cash": self.cash,
        }
        
        self.trades.append(trade)
        
        return trade
    
    def run_backtest(
        self,
        prices: pd.Series,
        signals: pd.Series,
        benchmark_prices: Optional[pd.Series] = None,
        position_sizing_method: str = "fixed",
    ) -> Dict[str, Union[float, pd.DataFrame, List[Dict]]]:
        """Run backtest on given prices and signals.
        
        Args:
            prices: Price series.
            signals: Trading signals (-1, 0, 1).
            benchmark_prices: Benchmark prices for comparison.
            position_sizing_method: Position sizing method.
            
        Returns:
            Backtest results dictionary.
        """
        self.reset()
        
        # Align data
        data = pd.DataFrame({
            "price": prices,
            "signal": signals,
        }).dropna()
        
        if len(data) == 0:
            self.logger.error("No valid data for backtesting")
            return {}
        
        # Run backtest
        for timestamp, row in data.iterrows():
            price = row["price"]
            signal = row["signal"]
            
            # Execute trade
            trade = self._execute_trade(
                price, signal, timestamp, position_sizing_method
            )
            
            # Calculate portfolio value
            portfolio_value = self.cash + self.position * price
            self.portfolio_values.append({
                "timestamp": timestamp,
                "portfolio_value": portfolio_value,
                "cash": self.cash,
                "position": self.position,
                "price": price,
            })
            
            # Calculate returns
            if len(self.portfolio_values) > 1:
                prev_value = self.portfolio_values[-2]["portfolio_value"]
                ret = (portfolio_value - prev_value) / prev_value
                self.returns.append(ret)
        
        # Calculate results
        results = self._calculate_backtest_results(
            data, benchmark_prices
        )
        
        return results
    
    def _calculate_backtest_results(
        self,
        data: pd.DataFrame,
        benchmark_prices: Optional[pd.Series] = None,
    ) -> Dict[str, Union[float, pd.DataFrame, List[Dict]]]:
        """Calculate comprehensive backtest results.
        
        Args:
            data: Backtest data.
            benchmark_prices: Benchmark prices.
            
        Returns:
            Results dictionary.
        """
        if not self.portfolio_values:
            return {}
        
        # Convert to DataFrames
        portfolio_df = pd.DataFrame(self.portfolio_values)
        portfolio_df.set_index("timestamp", inplace=True)
        
        returns_series = pd.Series(self.returns, index=portfolio_df.index[1:])
        
        # Basic metrics
        total_return = (portfolio_df["portfolio_value"].iloc[-1] / self.initial_capital) - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_df)) - 1
        volatility = returns_series.std() * np.sqrt(252)
        
        # Risk metrics
        sharpe_ratio = calculate_sharpe_ratio(returns_series)
        max_drawdown = calculate_max_drawdown(returns_series)
        
        # Trading metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade["position_change"] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Transaction costs
        total_costs = sum(trade["transaction_cost"] for trade in self.trades)
        cost_ratio = total_costs / self.initial_capital
        
        # Benchmark comparison
        benchmark_metrics = {}
        if benchmark_prices is not None:
            benchmark_returns = benchmark_prices.pct_change().dropna()
            benchmark_return = (benchmark_prices.iloc[-1] / benchmark_prices.iloc[0]) - 1
            benchmark_annualized = (1 + benchmark_return) ** (252 / len(benchmark_prices)) - 1
            
            # Align returns
            aligned_returns, aligned_benchmark = returns_series.align(benchmark_returns, join='inner')
            
            if len(aligned_returns) > 0:
                # Beta and Alpha
                covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                benchmark_variance = np.var(aligned_benchmark)
                beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
                
                alpha = annualized_return - 0.02 - beta * (benchmark_annualized - 0.02)
                
                benchmark_metrics = {
                    "benchmark_return": benchmark_return,
                    "benchmark_annualized": benchmark_annualized,
                    "beta": beta,
                    "alpha": alpha,
                    "excess_return": annualized_return - benchmark_annualized,
                }
        
        results = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_costs": total_costs,
            "cost_ratio": cost_ratio,
            "final_capital": portfolio_df["portfolio_value"].iloc[-1],
            "portfolio_values": portfolio_df,
            "returns": returns_series,
            "trades": self.trades,
            **benchmark_metrics,
        }
        
        return results
    
    def plot_backtest_results(
        self,
        results: Dict[str, Union[float, pd.DataFrame, List[Dict]]],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot backtest results.
        
        Args:
            results: Backtest results.
            save_path: Optional path to save plots.
        """
        import matplotlib.pyplot as plt
        
        if "portfolio_values" not in results:
            self.logger.error("No portfolio values to plot")
            return
        
        portfolio_df = results["portfolio_values"]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Backtest Results", fontsize=16)
        
        # Portfolio value over time
        axes[0, 0].plot(portfolio_df.index, portfolio_df["portfolio_value"])
        axes[0, 0].set_title("Portfolio Value")
        axes[0, 0].set_ylabel("Value ($)")
        axes[0, 0].grid(True)
        
        # Drawdown
        if "returns" in results:
            returns_series = results["returns"]
            cumulative = (1 + returns_series).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            
            axes[0, 1].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
            axes[0, 1].set_title("Drawdown")
            axes[0, 1].set_ylabel("Drawdown")
            axes[0, 1].grid(True)
        
        # Position over time
        axes[1, 0].plot(portfolio_df.index, portfolio_df["position"])
        axes[1, 0].set_title("Position")
        axes[1, 0].set_ylabel("Shares")
        axes[1, 0].grid(True)
        
        # Returns distribution
        if "returns" in results:
            returns_series = results["returns"]
            axes[1, 1].hist(returns_series, bins=50, alpha=0.7)
            axes[1, 1].set_title("Returns Distribution")
            axes[1, 1].set_xlabel("Daily Returns")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Backtest plots saved to {save_path}")
        
        plt.show()
    
    def generate_trade_report(self, results: Dict) -> pd.DataFrame:
        """Generate detailed trade report.
        
        Args:
            results: Backtest results.
            
        Returns:
            DataFrame with trade details.
        """
        if "trades" not in results or not results["trades"]:
            return pd.DataFrame()
        
        trades_df = pd.DataFrame(results["trades"])
        trades_df.set_index("timestamp", inplace=True)
        
        return trades_df
