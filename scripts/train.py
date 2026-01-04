"""
Main training script for time series forecasting models.

This script demonstrates the complete pipeline from data loading
to model training, evaluation, and backtesting.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import DataLoader, FeatureEngineer, TimeSeriesSplitter
from models import ARIMAModel, LSTMForecaster, XGBoostModel, ModelEnsemble
from evaluation import ModelEvaluator
from backtest import Backtester
from utils import load_config, setup_logging, set_seed, ensure_dir


class ForecastingPipeline:
    """Complete forecasting pipeline."""
    
    def __init__(self, config_path: str):
        """Initialize pipeline.
        
        Args:
            config_path: Path to configuration file.
        """
        self.config = load_config(config_path)
        self.logger = setup_logging(self.config.logging.level)
        
        # Set random seed for reproducibility
        set_seed(self.config.seed)
        
        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.splitter = TimeSeriesSplitter(self.config)
        self.evaluator = ModelEvaluator(self.config)
        self.backtester = Backtester(self.config)
        
        # Initialize models
        self.models = {
            "ARIMA": ARIMAModel(self.config),
            "LSTM": LSTMForecaster(self.config),
            "XGBoost": XGBoostModel(self.config),
        }
        
        self.logger.info("Forecasting pipeline initialized")
    
    def load_and_prepare_data(self, symbol: str) -> pd.DataFrame:
        """Load and prepare data for a given symbol.
        
        Args:
            symbol: Stock symbol.
            
        Returns:
            Prepared DataFrame.
        """
        self.logger.info(f"Loading data for {symbol}")
        
        # Try to load existing data first
        data = self.data_loader.load_data(symbol)
        
        if data is None:
            # Download new data
            self.logger.info(f"Downloading data for {symbol}")
            downloaded_data = self.data_loader.download_data(
                [symbol],
                self.config.data.start_date,
                self.config.data.end_date,
                save=True,
            )
            
            if symbol not in downloaded_data:
                # Create synthetic data for demonstration
                self.logger.info(f"Creating synthetic data for {symbol}")
                data = self.data_loader.create_synthetic_data(
                    symbol,
                    self.config.data.start_date,
                    self.config.data.end_date,
                )
            else:
                data = downloaded_data[symbol]
        
        # Add technical indicators
        data = self.feature_engineer.add_technical_indicators(data)
        
        # Add lag features
        target_col = self.config.data.target.lower()
        lags = [1, 2, 3, 5, 10, 20]
        data = self.feature_engineer.add_lag_features(data, target_col, lags)
        
        # Add rolling features
        windows = [5, 10, 20]
        data = self.feature_engineer.add_rolling_features(
            data, target_col, windows, ["mean", "std", "min", "max"]
        )
        
        # Remove NaN values
        data = data.dropna()
        
        self.logger.info(f"Prepared data shape: {data.shape}")
        return data
    
    def train_models(self, train_data: pd.DataFrame, target_col: str) -> Dict[str, any]:
        """Train all models on training data.
        
        Args:
            train_data: Training data.
            target_col: Target column name.
            
        Returns:
            Dictionary of trained models.
        """
        trained_models = {}
        
        # Prepare features and target
        feature_cols = [col for col in train_data.columns if col != target_col]
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        
        for name, model in self.models.items():
            self.logger.info(f"Training {name} model")
            
            try:
                model.fit(X_train, y_train)
                trained_models[name] = model
                self.logger.info(f"{name} model trained successfully")
                
            except Exception as e:
                self.logger.error(f"Error training {name} model: {e}")
                continue
        
        return trained_models
    
    def evaluate_models(
        self,
        models: Dict[str, any],
        test_data: pd.DataFrame,
        target_col: str,
    ) -> List[Dict]:
        """Evaluate all trained models.
        
        Args:
            models: Dictionary of trained models.
            test_data: Test data.
            target_col: Target column name.
            
        Returns:
            List of evaluation results.
        """
        results = []
        
        # Prepare test data
        feature_cols = [col for col in test_data.columns if col != target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        
        for name, model in models.items():
            self.logger.info(f"Evaluating {name} model")
            
            try:
                # Make predictions
                predictions = model.predict(X_test, steps=1)
                
                # Evaluate model
                result = self.evaluator.evaluate_model(
                    name,
                    y_test,
                    predictions,
                    test_data[self.config.data.target.lower()],
                )
                
                results.append(result)
                self.logger.info(f"{name} evaluation completed")
                
            except Exception as e:
                self.logger.error(f"Error evaluating {name} model: {e}")
                continue
        
        return results
    
    def run_backtests(
        self,
        models: Dict[str, any],
        test_data: pd.DataFrame,
        target_col: str,
    ) -> Dict[str, Dict]:
        """Run backtests for all models.
        
        Args:
            models: Dictionary of trained models.
            test_data: Test data.
            target_col: Target column name.
            
        Returns:
            Dictionary of backtest results.
        """
        backtest_results = {}
        
        # Prepare test data
        feature_cols = [col for col in test_data.columns if col != target_col]
        X_test = test_data[feature_cols]
        prices = test_data[self.config.data.target.lower()]
        
        for name, model in models.items():
            self.logger.info(f"Running backtest for {name} model")
            
            try:
                # Generate predictions
                predictions = model.predict(X_test, steps=1)
                
                # Generate trading signals (simple momentum strategy)
                signals = np.where(predictions > prices.shift(1), 1, -1)
                signals[0] = 0  # No signal for first period
                
                # Run backtest
                results = self.backtester.run_backtest(
                    prices,
                    pd.Series(signals, index=prices.index),
                )
                
                backtest_results[name] = results
                self.logger.info(f"{name} backtest completed")
                
            except Exception as e:
                self.logger.error(f"Error running backtest for {name}: {e}")
                continue
        
        return backtest_results
    
    def run_pipeline(self, symbol: str = "AAPL") -> Dict:
        """Run the complete forecasting pipeline.
        
        Args:
            symbol: Stock symbol to analyze.
            
        Returns:
            Dictionary with all results.
        """
        self.logger.info(f"Starting forecasting pipeline for {symbol}")
        
        # Load and prepare data
        data = self.load_and_prepare_data(symbol)
        
        # Split data
        train_data, val_data, test_data = self.splitter.split_data(
            data,
            self.config.training.train_split,
            self.config.training.val_split,
            self.config.training.test_split,
        )
        
        # Train models
        target_col = self.config.data.target.lower()
        trained_models = self.train_models(train_data, target_col)
        
        if not trained_models:
            self.logger.error("No models were successfully trained")
            return {}
        
        # Evaluate models
        evaluation_results = self.evaluate_models(trained_models, test_data, target_col)
        
        # Run backtests
        backtest_results = self.run_backtests(trained_models, test_data, target_col)
        
        # Create evaluation report
        evaluation_report = self.evaluator.create_evaluation_report(evaluation_results)
        
        # Save results
        results_dir = ensure_dir(Path("assets"))
        evaluation_report.to_csv(results_dir / "evaluation_report.csv", index=False)
        
        # Plot results
        self.evaluator.plot_evaluation_results(
            evaluation_results,
            save_path=str(results_dir / "evaluation_plots.png")
        )
        
        # Plot backtest results
        for name, backtest_result in backtest_results.items():
            if backtest_result:
                self.backtester.plot_backtest_results(
                    backtest_result,
                    save_path=str(results_dir / f"{name.lower()}_backtest.png")
                )
        
        self.logger.info("Pipeline completed successfully")
        
        return {
            "models": trained_models,
            "evaluation_results": evaluation_results,
            "backtest_results": backtest_results,
            "evaluation_report": evaluation_report,
            "data": {
                "train": train_data,
                "val": val_data,
                "test": test_data,
            },
        }


def main():
    """Main function to run the forecasting pipeline."""
    config_path = "configs/config.yaml"
    
    if not Path(config_path).exists():
        print(f"Configuration file not found: {config_path}")
        print("Please ensure the config file exists before running the pipeline.")
        return
    
    # Initialize and run pipeline
    pipeline = ForecastingPipeline(config_path)
    results = pipeline.run_pipeline("AAPL")
    
    if results:
        print("\n" + "="*50)
        print("FORECASTING PIPELINE COMPLETED")
        print("="*50)
        
        print(f"\nModels trained: {list(results['models'].keys())}")
        
        if "evaluation_report" in results:
            print("\nEvaluation Report:")
            print(results["evaluation_report"].to_string(index=False))
        
        print(f"\nResults saved to assets/ directory")
        print("\nTo view interactive results, run: streamlit run demo/app.py")
    else:
        print("Pipeline failed to complete successfully")


if __name__ == "__main__":
    main()
