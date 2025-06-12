# Bitcoin Price Forecast App

This Streamlit application provides Bitcoin price forecasting using Support Vector Regression (SVR). It includes interactive visualizations and customizable model parameters.

## Features

- Historical Bitcoin price data visualization
- Price forecasting using SVR
- Interactive parameter tuning
- Technical indicators (Moving Averages, RSI, Volatility)
- Performance metrics
- Volume analysis

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app/main.py
```

2. The app will open in your default web browser
3. Use the sidebar to adjust:
   - Date range for historical data
   - Forecast horizon
   - SVR model parameters
   - Technical indicators

## Model Parameters

- **Kernel**: Choose between 'rbf', 'linear', or 'poly'
- **C**: Regularization parameter (0.1 to 10.0)
- **Epsilon**: Margin of tolerance (0.01 to 1.0)
- **Moving Average Periods**: Select multiple periods for technical analysis

## Data Source

The application uses Yahoo Finance (yfinance) to fetch real-time Bitcoin price data.

## Disclaimer

This tool is for educational purposes only. Cryptocurrency investments are subject to high market risks. Please make informed decisions and do your own research before making any investment decisions. 