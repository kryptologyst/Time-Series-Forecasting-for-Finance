"""
Streamlit demo application for time series forecasting.

This application provides an interactive interface to explore
forecasting models, view predictions, and analyze backtest results.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from omegaconf import DictConfig

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import DataLoader, FeatureEngineer, TimeSeriesSplitter
from models import ARIMAModel, LSTMForecaster, XGBoostModel
from evaluation import ModelEvaluator
from backtest import Backtester
from utils import load_config, setup_logging, set_seed, ensure_dir


# Page configuration
st.set_page_config(
    page_title="Time Series Forecasting for Finance",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.375rem;
        padding: 1rem;
        margin-bottom: 1rem;
        color: #856404;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.375rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ IMPORTANT DISCLAIMER:</strong><br>
    This is a research and educational demonstration only. The models and predictions shown here are 
    <strong>NOT investment advice</strong> and should not be used for actual trading decisions. 
    Past performance does not guarantee future results. All backtests are hypothetical and may not 
    reflect real trading conditions. Please consult with qualified financial professionals before 
    making any investment decisions.
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">📈 Time Series Forecasting for Finance</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")

# Load configuration
@st.cache_data
def load_app_config():
    """Load application configuration."""
    try:
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        return load_config(config_path)
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return None

config = load_app_config()
if config is None:
    st.error("Failed to load configuration. Please check the config file.")
    st.stop()

# Sidebar controls
st.sidebar.subheader("Data Settings")
symbol = st.sidebar.selectbox(
    "Select Stock Symbol",
    options=config.data.symbols,
    index=0
)

st.sidebar.subheader("Model Settings")
selected_models = st.sidebar.multiselect(
    "Select Models to Compare",
    options=["ARIMA", "LSTM", "XGBoost"],
    default=["ARIMA", "XGBoost"]
)

st.sidebar.subheader("Backtest Settings")
initial_capital = st.sidebar.number_input(
    "Initial Capital ($)",
    min_value=1000,
    max_value=1000000,
    value=config.backtesting.initial_capital,
    step=1000
)

transaction_cost = st.sidebar.slider(
    "Transaction Cost (%)",
    min_value=0.0,
    max_value=0.01,
    value=config.backtesting.transaction_cost,
    step=0.0001,
    format="%.4f"
)

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize pipeline components."""
    set_seed(config.seed)
    
    data_loader = DataLoader(config)
    feature_engineer = FeatureEngineer(config)
    splitter = TimeSeriesSplitter(config)
    evaluator = ModelEvaluator(config)
    backtester = Backtester(config)
    
    return data_loader, feature_engineer, splitter, evaluator, backtester

data_loader, feature_engineer, splitter, evaluator, backtester = initialize_components()

# Load and prepare data
@st.cache_data
def load_and_prepare_data(symbol: str):
    """Load and prepare data for the selected symbol."""
    try:
        # Try to load existing data
        data = data_loader.load_data(symbol)
        
        if data is None:
            # Download new data
            downloaded_data = data_loader.download_data(
                [symbol],
                config.data.start_date,
                config.data.end_date,
                save=True,
            )
            
            if symbol not in downloaded_data:
                # Create synthetic data
                data = data_loader.create_synthetic_data(
                    symbol,
                    config.data.start_date,
                    config.data.end_date,
                )
            else:
                data = downloaded_data[symbol]
        
        # Add technical indicators
        data = feature_engineer.add_technical_indicators(data)
        
        # Add lag features
        target_col = config.data.target.lower()
        lags = [1, 2, 3, 5, 10, 20]
        data = feature_engineer.add_lag_features(data, target_col, lags)
        
        # Add rolling features
        windows = [5, 10, 20]
        data = feature_engineer.add_rolling_features(
            data, target_col, windows, ["mean", "std", "min", "max"]
        )
        
        # Remove NaN values
        data = data.dropna()
        
        return data
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Main content
if st.button("🔄 Load Data and Run Analysis", type="primary"):
    with st.spinner("Loading data and running analysis..."):
        # Load data
        data = load_and_prepare_data(symbol)
        
        if data is None:
            st.error("Failed to load data")
            st.stop()
        
        st.success(f"Data loaded successfully! Shape: {data.shape}")
        
        # Split data
        train_data, val_data, test_data = splitter.split_data(
            data,
            config.training.train_split,
            config.training.val_split,
            config.training.test_split,
        )
        
        # Display data overview
        st.subheader("📊 Data Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", len(train_data))
        with col2:
            st.metric("Validation Samples", len(val_data))
        with col3:
            st.metric("Test Samples", len(test_data))
        
        # Plot price data
        st.subheader("📈 Price Data")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=train_data.index,
            y=train_data[config.data.target.lower()],
            mode='lines',
            name='Training Data',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=val_data.index,
            y=val_data[config.data.target.lower()],
            mode='lines',
            name='Validation Data',
            line=dict(color='orange')
        ))
        fig.add_trace(go.Scatter(
            x=test_data.index,
            y=test_data[config.data.target.lower()],
            mode='lines',
            name='Test Data',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title=f"{symbol} Stock Price",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Train models
        st.subheader("🤖 Model Training")
        
        target_col = config.data.target.lower()
        feature_cols = [col for col in train_data.columns if col != target_col]
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        
        trained_models = {}
        progress_bar = st.progress(0)
        
        for i, model_name in enumerate(selected_models):
            st.write(f"Training {model_name} model...")
            
            try:
                if model_name == "ARIMA":
                    model = ARIMAModel(config)
                elif model_name == "LSTM":
                    model = LSTMForecaster(config)
                elif model_name == "XGBoost":
                    model = XGBoostModel(config)
                else:
                    continue
                
                model.fit(X_train, y_train)
                trained_models[model_name] = model
                st.success(f"{model_name} model trained successfully!")
                
            except Exception as e:
                st.error(f"Error training {model_name} model: {e}")
                continue
            
            progress_bar.progress((i + 1) / len(selected_models))
        
        if not trained_models:
            st.error("No models were successfully trained")
            st.stop()
        
        # Make predictions
        st.subheader("🔮 Predictions")
        
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        
        predictions = {}
        for name, model in trained_models.items():
            try:
                pred = model.predict(X_test, steps=1)
                predictions[name] = pred
            except Exception as e:
                st.error(f"Error making predictions with {name}: {e}")
                continue
        
        # Plot predictions
        if predictions:
            fig = go.Figure()
            
            # Actual prices
            fig.add_trace(go.Scatter(
                x=test_data.index,
                y=y_test,
                mode='lines',
                name='Actual',
                line=dict(color='black', width=2)
            ))
            
            # Model predictions
            colors = ['red', 'green', 'blue', 'purple', 'orange']
            for i, (name, pred) in enumerate(predictions.items()):
                fig.add_trace(go.Scatter(
                    x=test_data.index,
                    y=pred,
                    mode='lines',
                    name=f'{name} Prediction',
                    line=dict(color=colors[i % len(colors)])
                ))
            
            fig.update_layout(
                title=f"{symbol} Price Predictions",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Model evaluation
        st.subheader("📊 Model Evaluation")
        
        evaluation_results = []
        for name, model in trained_models.items():
            if name in predictions:
                try:
                    result = evaluator.evaluate_model(
                        name,
                        y_test,
                        predictions[name],
                        test_data[target_col],
                    )
                    evaluation_results.append(result)
                except Exception as e:
                    st.error(f"Error evaluating {name}: {e}")
                    continue
        
        if evaluation_results:
            # Display evaluation metrics
            evaluation_report = evaluator.create_evaluation_report(evaluation_results)
            
            # ML Metrics
            st.subheader("🎯 Machine Learning Metrics")
            ml_cols = [col for col in evaluation_report.columns if col.startswith('ML_')]
            if ml_cols:
                ml_data = evaluation_report[['Model'] + ml_cols]
                st.dataframe(ml_data, use_container_width=True)
            
            # Finance Metrics
            st.subheader("💰 Financial Metrics")
            fin_cols = [col for col in evaluation_report.columns if col.startswith('FIN_')]
            if fin_cols:
                fin_data = evaluation_report[['Model'] + fin_cols]
                st.dataframe(fin_data, use_container_width=True)
        
        # Backtesting
        st.subheader("💼 Backtesting Results")
        
        backtest_results = {}
        for name, model in trained_models.items():
            if name in predictions:
                try:
                    # Generate trading signals
                    signals = np.where(predictions[name] > test_data[target_col].shift(1), 1, -1)
                    signals[0] = 0
                    
                    # Run backtest
                    results = backtester.run_backtest(
                        test_data[target_col],
                        pd.Series(signals, index=test_data.index),
                    )
                    
                    backtest_results[name] = results
                    
                except Exception as e:
                    st.error(f"Error running backtest for {name}: {e}")
                    continue
        
        # Display backtest results
        if backtest_results:
            st.subheader("📈 Portfolio Performance")
            
            # Create performance comparison
            performance_data = []
            for name, results in backtest_results.items():
                if results:
                    performance_data.append({
                        'Model': name,
                        'Total Return (%)': results.get('total_return', 0) * 100,
                        'Annualized Return (%)': results.get('annualized_return', 0) * 100,
                        'Volatility (%)': results.get('volatility', 0) * 100,
                        'Sharpe Ratio': results.get('sharpe_ratio', 0),
                        'Max Drawdown (%)': results.get('max_drawdown', 0) * 100,
                        'Total Trades': results.get('total_trades', 0),
                        'Win Rate (%)': results.get('win_rate', 0) * 100,
                    })
            
            if performance_data:
                perf_df = pd.DataFrame(performance_data)
                st.dataframe(perf_df, use_container_width=True)
                
                # Plot portfolio values
                st.subheader("💼 Portfolio Value Over Time")
                
                fig = go.Figure()
                
                for name, results in backtest_results.items():
                    if results and 'portfolio_values' in results:
                        portfolio_df = results['portfolio_values']
                        fig.add_trace(go.Scatter(
                            x=portfolio_df.index,
                            y=portfolio_df['portfolio_value'],
                            mode='lines',
                            name=f'{name} Portfolio',
                        ))
                
                fig.update_layout(
                    title="Portfolio Value Comparison",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        st.subheader("🔍 Feature Importance")
        
        for name, model in trained_models.items():
            try:
                importance = model.get_feature_importance()
                if importance:
                    st.write(f"**{name} Model Feature Importance:**")
                    
                    # Sort by importance
                    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                    
                    # Display top 10 features
                    top_features = sorted_importance[:10]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=[item[1] for item in top_features],
                            y=[item[0] for item in top_features],
                            orientation='h'
                        )
                    ])
                    
                    fig.update_layout(
                        title=f"{name} - Top 10 Features",
                        xaxis_title="Importance",
                        yaxis_title="Feature",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error getting feature importance for {name}: {e}")
                continue

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>This application is for research and educational purposes only.</p>
    <p>Not intended as investment advice. Please consult with qualified financial professionals.</p>
</div>
""", unsafe_allow_html=True)
