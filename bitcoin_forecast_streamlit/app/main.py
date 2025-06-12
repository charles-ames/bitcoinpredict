import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, date
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Bitcoin Price Forecast",
    page_icon="₿",
    layout="wide"
)

# Title and description
st.title("Bitcoin Price Forecast")
st.markdown("""
This app uses Support Vector Regression (SVR) to forecast Bitcoin prices.
Adjust the parameters in the sidebar to customize the forecast.
""")

# Sidebar controls
st.sidebar.header("Model Parameters")

# Date range selection
end_date = datetime.now()
start_date = end_date - timedelta(days=90)  # 90 days max for CoinGecko
selected_start = st.sidebar.date_input("Start Date", start_date)
selected_end = st.sidebar.date_input("End Date", end_date)

# Convert to datetime.datetime if needed
if isinstance(selected_start, date) and not isinstance(selected_start, datetime):
    selected_start = datetime.combine(selected_start, datetime.min.time())
if isinstance(selected_end, date) and not isinstance(selected_end, datetime):
    selected_end = datetime.combine(selected_end, datetime.min.time())

# Warn if range is more than 90 days
if (selected_end - selected_start).days > 90:
    st.warning("CoinGecko only supports up to 90 days of data per request. Please select a shorter date range.")

# Model parameters
forecast_days = st.sidebar.slider("Forecast Horizon (days)", 7, 90, 30)
kernel = st.sidebar.selectbox("SVR Kernel", ["rbf", "linear", "poly"])
C = st.sidebar.slider("C (Regularization)", 0.1, 10.0, 1.0)
epsilon = st.sidebar.slider("Epsilon", 0.01, 1.0, 0.1)

# Feature engineering parameters
ma_periods = st.sidebar.multiselect(
    "Moving Average Periods",
    [7, 14, 30, 60, 90],
    default=[7, 30]
)

@st.cache_data
def fetch_bitcoin_data(start_date, end_date):
    """Fetch Bitcoin price data from CoinGecko API"""
    # Convert dates to Unix timestamps
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())
    
    # CoinGecko API endpoint
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
    params = {
        "vs_currency": "usd",
        "from": start_timestamp,
        "to": end_timestamp
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        market_data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(market_data['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.rename(columns={'price': 'Close'}, inplace=True)
        
        # Add volume data
        volume_df = pd.DataFrame(market_data['total_volumes'], columns=['timestamp', 'volume'])
        volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
        volume_df.set_index('timestamp', inplace=True)
        df['Volume'] = volume_df['volume']
        
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from CoinGecko: {str(e)}")
        return pd.DataFrame()

def create_features(df):
    """Create technical indicators and features"""
    # Calculate returns
    df['Returns'] = df['Close'].pct_change()
    
    # Calculate moving averages
    for period in ma_periods:
        df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
    
    # Calculate volatility
    df['Volatility'] = df['Returns'].rolling(window=14).std()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def prepare_data(df):
    """Prepare data for model training"""
    # Drop NaN values
    df = df.dropna()
    
    # Select features
    feature_columns = ['Returns', 'Volatility', 'RSI'] + [f'MA_{p}' for p in ma_periods]
    X = df[feature_columns]
    y = df['Close']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_model(X, y):
    """Train SVR model"""
    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    model.fit(X, y)
    return model

def make_predictions(model, X, scaler, last_features):
    """Make predictions for future dates"""
    # Scale the last known features
    last_features_scaled = scaler.transform(last_features)
    
    # Make predictions
    predictions = model.predict(last_features_scaled)
    return predictions

def plot_results(df, predictions, forecast_dates):
    """Create interactive plot with Plotly"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, 
                       subplot_titles=('Bitcoin Price', 'Volume'),
                       row_heights=[0.7, 0.3])

    # Add actual price
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Close'], name='Actual Price',
                  line=dict(color='blue')),
        row=1, col=1
    )

    # Add predictions
    fig.add_trace(
        go.Scatter(x=forecast_dates, y=predictions, name='Predictions',
                  line=dict(color='red', dash='dash')),
        row=1, col=1
    )

    # Add volume
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume',
               marker_color='rgba(0,0,255,0.3)'),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title='Bitcoin Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        height=800,
        showlegend=True
    )

    return fig

# Main execution
try:
    # Fetch data
    df = fetch_bitcoin_data(selected_start, selected_end)
    
    if df.empty:
        st.error("No data available for the selected date range.")
    else:
        # Create features
        df = create_features(df)
        
        # Prepare data
        X, y, scaler = prepare_data(df)
        
        # Train model
        model = train_model(X, y)
        
        # Make predictions
        last_features = df[['Returns', 'Volatility', 'RSI'] + [f'MA_{p}' for p in ma_periods]].iloc[-1:].values
        predictions = make_predictions(model, X, scaler, last_features)
        
        # Create forecast dates
        last_date = df.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        
        # Plot results
        fig = plot_results(df, predictions, forecast_dates)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics
        st.subheader("Model Performance")
        train_predictions = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, train_predictions))
        st.metric("Training RMSE", f"${rmse:.2f}")
        
        # Display latest prediction
        st.subheader("Latest Forecast")
        st.metric("Predicted Price", f"${predictions[-1]:.2f}")
        
except Exception as e:
    st.error(f"An error occurred: {str(e)}") 