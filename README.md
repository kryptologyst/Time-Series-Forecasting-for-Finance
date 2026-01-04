# Time Series Forecasting for Finance

A comprehensive, production-ready framework for time series forecasting in financial markets. This project demonstrates modern machine learning techniques for predicting stock prices and evaluating trading strategies with realistic backtesting.

## ⚠️ IMPORTANT DISCLAIMER

**THIS IS RESEARCH AND EDUCATIONAL SOFTWARE ONLY**

- **NOT INVESTMENT ADVICE**: All models, predictions, and backtests are for research and educational purposes only
- **NO GUARANTEES**: Past performance does not guarantee future results
- **HYPOTHETICAL RESULTS**: All backtests are hypothetical and may not reflect real trading conditions
- **RISK WARNING**: Financial markets are inherently risky and unpredictable
- **PROFESSIONAL CONSULTATION**: Always consult with qualified financial professionals before making investment decisions

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kryptologyst/Time-Series-Forecasting-for-Finance.git
   cd Time-Series-Forecasting-for-Finance
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the training pipeline**
   ```bash
   python scripts/train.py
   ```

4. **Launch the interactive demo**
   ```bash
   streamlit run demo/app.py
   ```

## 📁 Project Structure

```
├── src/                    # Source code
│   ├── data/              # Data handling and preprocessing
│   ├── models/             # Forecasting models
│   ├── evaluation/         # Model evaluation metrics
│   ├── backtest/           # Backtesting framework
│   └── utils/              # Utility functions
├── configs/                # Configuration files
├── scripts/                # Training and evaluation scripts
├── demo/                   # Streamlit demo application
├── tests/                  # Unit tests
├── assets/                 # Generated plots and results
├── data/                   # Data storage
├── notebooks/              # Jupyter notebooks for exploration
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project configuration
└── README.md              # This file
```

## Features

### Models Implemented

- **ARIMA**: AutoRegressive Integrated Moving Average with automatic parameter selection
- **LSTM**: Long Short-Term Memory neural networks with PyTorch
- **XGBoost**: Gradient boosting with advanced feature engineering
- **Ensemble**: Weighted combination of multiple models

### Data Pipeline

- **Real-time Data**: Yahoo Finance integration with fallback to synthetic data
- **Feature Engineering**: Technical indicators, lag features, rolling statistics
- **Time Series Splits**: Proper train/validation/test splits with leakage prevention
- **Walk-forward Validation**: Rolling window evaluation for robust testing

### Evaluation Framework

- **ML Metrics**: MAE, RMSE, MAPE, SMAPE, R², MASE
- **Financial Metrics**: Sharpe ratio, Sortino ratio, Calmar ratio, max drawdown
- **Trading Metrics**: Hit rate, win rate, total trades, transaction costs

### Backtesting Engine

- **Realistic Costs**: Transaction costs and slippage modeling
- **Position Sizing**: Multiple sizing strategies (fixed, Kelly, volatility targeting)
- **Risk Management**: Drawdown control and exposure limits
- **Benchmark Comparison**: Alpha, beta, information ratio calculations

## 🔧 Configuration

The project uses YAML configuration files for easy customization:

```yaml
# configs/config.yaml
data:
  symbols: ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
  start_date: "2015-01-01"
  end_date: "2023-12-31"
  target: "Close"

models:
  arima:
    order: [5, 1, 0]
    auto_arima: true
  
  lstm:
    sequence_length: 60
    hidden_size: 50
    epochs: 100
    learning_rate: 0.001
  
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1

backtesting:
  initial_capital: 100000
  transaction_cost: 0.001
  slippage: 0.0005
```

## Usage Examples

### Basic Training

```python
from src.utils import load_config
from scripts.train import ForecastingPipeline

# Load configuration
config = load_config("configs/config.yaml")

# Initialize pipeline
pipeline = ForecastingPipeline("configs/config.yaml")

# Run complete analysis
results = pipeline.run_pipeline("AAPL")
```

### Custom Model Training

```python
from src.models import ARIMAModel, XGBoostModel
from src.data import DataLoader, FeatureEngineer

# Load data
data_loader = DataLoader(config)
data = data_loader.load_data("AAPL")

# Add features
feature_engineer = FeatureEngineer(config)
data = feature_engineer.add_technical_indicators(data)

# Train models
arima_model = ARIMAModel(config)
arima_model.fit(data[features], data[target])

xgboost_model = XGBoostModel(config)
xgboost_model.fit(data[features], data[target])
```

### Backtesting

```python
from src.backtest import Backtester

# Initialize backtester
backtester = Backtester(config)

# Run backtest
results = backtester.run_backtest(
    prices=prices,
    signals=signals,
    benchmark_prices=benchmark_prices
)

# Plot results
backtester.plot_backtest_results(results)
```

## Interactive Demo

The Streamlit demo provides a user-friendly interface to:

- **Select stocks** from predefined symbols
- **Choose models** to compare (ARIMA, LSTM, XGBoost)
- **Configure parameters** for backtesting
- **View predictions** with interactive plots
- **Analyze performance** with comprehensive metrics
- **Explore feature importance** for model interpretability

Launch the demo:
```bash
streamlit run demo/app.py
```

## Testing

Run the test suite:
```bash
pytest tests/ -v --cov=src
```

Key test coverage:
- Data loading and preprocessing
- Model training and prediction
- Evaluation metrics calculation
- Backtesting functionality
- Utility functions

## Model Details

### ARIMA Model
- **Purpose**: Classical time series forecasting
- **Features**: Automatic parameter selection, confidence intervals
- **Best for**: Stationary time series with clear trends

### LSTM Model
- **Purpose**: Deep learning for complex patterns
- **Features**: Sequence learning, dropout regularization
- **Best for**: Non-linear patterns, long-term dependencies

### XGBoost Model
- **Purpose**: Gradient boosting with feature engineering
- **Features**: Technical indicators, lag features, rolling statistics
- **Best for**: Tabular data with engineered features

## Evaluation Metrics

### Machine Learning Metrics
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **MAPE**: Mean Absolute Percentage Error
- **SMAPE**: Symmetric Mean Absolute Percentage Error
- **R²**: Coefficient of Determination
- **MASE**: Mean Absolute Scaled Error

### Financial Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Calmar Ratio**: Return to max drawdown ratio
- **Max Drawdown**: Maximum peak-to-trough decline
- **Volatility**: Annualized standard deviation

### Trading Metrics
- **Hit Rate**: Percentage of profitable trades
- **Win Rate**: Percentage of winning trades
- **Total Trades**: Number of executed trades
- **Transaction Costs**: Total costs as percentage of capital

## 🛠️ Development

### Code Quality
- **Type Hints**: Full type annotation coverage
- **Documentation**: Google/NumPy style docstrings
- **Formatting**: Black code formatting
- **Linting**: Ruff for code quality checks

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Pre-commit Hooks
```bash
pip install pre-commit
pre-commit install
```

## Requirements

### Core Dependencies
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- scipy >= 1.10.0
- statsmodels >= 0.14.0

### Machine Learning
- xgboost >= 1.7.0
- lightgbm >= 4.0.0
- torch >= 2.0.0
- pytorch-lightning >= 2.0.0

### Time Series
- pmdarima >= 2.0.0
- prophet >= 1.1.0
- darts >= 0.24.0
- tsfresh >= 0.20.0

### Visualization & Demo
- matplotlib >= 3.7.0
- plotly >= 5.15.0
- streamlit >= 1.25.0

### Data Sources
- yfinance >= 0.2.0
- alpha-vantage >= 2.3.0

## Limitations and Considerations

### Data Limitations
- **Market Hours**: Data may not reflect after-hours trading
- **Corporate Actions**: Splits and dividends may affect historical data
- **Data Quality**: Free data sources may have gaps or errors

### Model Limitations
- **Overfitting**: Models may perform well on historical data but poorly on new data
- **Regime Changes**: Market conditions can change, making historical patterns irrelevant
- **Non-stationarity**: Financial time series often have changing statistical properties

### Backtesting Limitations
- **Survivorship Bias**: Only includes stocks that survived the entire period
- **Look-ahead Bias**: May inadvertently use future information
- **Transaction Costs**: Real trading costs may be higher than modeled
- **Liquidity**: Assumes all trades can be executed at market prices

## References

### Academic Papers
- Box, G.E.P. & Jenkins, G.M. (1970). Time Series Analysis: Forecasting and Control
- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System

### Financial Literature
- Bailey, D.H. & López de Prado, M. (2012). The Sharpe Ratio Efficient Frontier
- López de Prado, M. (2018). Advances in Financial Machine Learning
- Pardo, R. (2008). The Evaluation and Optimization of Trading Strategies

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the documentation
- Review the test cases for usage examples

## Version History

- **v1.0.0**: Initial release with ARIMA, LSTM, and XGBoost models
- **v1.1.0**: Added comprehensive backtesting framework
- **v1.2.0**: Implemented interactive Streamlit demo
- **v1.3.0**: Enhanced evaluation metrics and visualization

---

**Remember**: This software is for research and educational purposes only. Always consult with qualified financial professionals before making investment decisions.
# Time-Series-Forecasting-for-Finance
