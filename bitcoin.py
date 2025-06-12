import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')  # Try a different backend
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("Starting script...")
print("Matplotlib backend:", matplotlib.get_backend())

try:
    # Fetch 5 years of Bitcoin data
    print("Fetching 5 years of Bitcoin data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)  # 5 years of data
    btc = yf.download('BTC-USD', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    print(f"Downloaded {len(btc)} days of Bitcoin data")

    # Prepare features
    print("Preparing features...")
    btc['Returns'] = btc['Close'].pct_change()
    btc['MA5'] = btc['Close'].rolling(window=5).mean()
    btc['MA20'] = btc['Close'].rolling(window=20).mean()
    btc['MA50'] = btc['Close'].rolling(window=50).mean()
    btc['MA200'] = btc['Close'].rolling(window=200).mean()
    btc['Volatility'] = btc['Returns'].rolling(window=20).std()

    # Drop NaN values
    btc = btc.dropna()
    print(f"Data points after removing NaN: {len(btc)}")

    # Prepare features for prediction
    features = ['Returns', 'MA5', 'MA20', 'MA50', 'MA200', 'Volatility']
    X = btc[features].copy()  # Create a copy to avoid SettingWithCopyWarning
    y = btc['Close'].shift(-1)  # Predict next day's price
    
    # Remove the last row where y is NaN
    X = X[:-1]
    y = y[:-1]

    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    train_size = int(len(X_scaled) * 0.8)
    X_train = X_scaled[:train_size]
    X_test = X_scaled[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    train_idx = btc.index[:train_size]
    test_idx = btc.index[train_size:-1]  # -1 to match y_test length

    # Create and train SVM model
    print("Training SVM model...")
    svm = SVR(kernel='rbf', C=100, gamma=0.1)
    svm.fit(X_train, y_train)

    # Make predictions for historical data
    print("Making historical predictions...")
    train_predictions = svm.predict(X_train)
    test_predictions = svm.predict(X_test)

    # Print debug info for lengths
    print(f"train_predictions shape: {train_predictions.shape}, y_train shape: {y_train.shape}, train_idx: {len(train_idx)}")
    print(f"test_predictions shape: {test_predictions.shape}, y_test shape: {y_test.shape}, test_idx: {len(test_idx)}")

    # Prepare future dates (next 30 days)
    print("Preparing future predictions...")
    last_date = btc.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq='D')
    
    # Prepare future features
    future_features = pd.DataFrame(index=future_dates)
    future_features['Returns'] = 0  # We don't know future returns
    future_features['MA5'] = btc['MA5'].iloc[-1]
    future_features['MA20'] = btc['MA20'].iloc[-1]
    future_features['MA50'] = btc['MA50'].iloc[-1]
    future_features['MA200'] = btc['MA200'].iloc[-1]
    future_features['Volatility'] = btc['Volatility'].iloc[-1]
    
    # Scale future features
    future_features_scaled = scaler.transform(future_features[features])
    
    # Make future predictions
    future_predictions = svm.predict(future_features_scaled)

    # Calculate performance metrics
    print(f"train_predictions shape: {train_predictions.shape}, y_train shape: {y_train.shape}")
    print(f"test_predictions shape: {test_predictions.shape}, y_test shape: {y_test.shape}")
    y_train_np = np.array(y_train)
    y_test_np = np.array(y_test)
    train_rmse = np.sqrt(np.mean((y_train_np - train_predictions) ** 2))
    test_rmse = np.sqrt(np.mean((y_test_np - test_predictions) ** 2))

    print("\nModel Performance:")
    print("=" * 50)
    print(f'Training RMSE: ${train_rmse:.2f}')
    print(f'Testing RMSE: ${test_rmse:.2f}')

    # Create the plot
    print("\nCreating plot...")
    plt.figure(figsize=(15, 8))
    
    # Plot historical data
    plt.plot(btc.index, btc['Close'], label='Actual Price', color='blue', alpha=0.6)
    plt.plot(train_idx, train_predictions, label='Historical Predictions (Training)', color='green', alpha=0.4)
    plt.plot(test_idx, test_predictions, label='Historical Predictions (Testing)', color='red', alpha=0.4)
    
    # Plot future predictions
    plt.plot(future_dates, future_predictions, label='Future Predictions', color='purple', linestyle='--')
    
    # Add confidence interval for future predictions (using RMSE)
    plt.fill_between(future_dates, 
                    future_predictions - test_rmse,
                    future_predictions + test_rmse,
                    color='purple', alpha=0.2, label='Prediction Interval')
    
    # Add moving averages
    plt.plot(btc.index, btc['MA50'], label='50-day MA', color='orange', alpha=0.4)
    plt.plot(btc.index, btc['MA200'], label='200-day MA', color='red', alpha=0.4)
    
    # Customize the plot
    plt.title('Bitcoin Price: 5-Year History and 30-Day Forecast', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    
    # Format y-axis to show prices in USD
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add a vertical line to separate historical and future data
    plt.axvline(x=last_date, color='gray', linestyle='--', alpha=0.5)
    plt.text(last_date, plt.ylim()[0], 'Today', rotation=90, va='bottom')
    
    plt.tight_layout()
    
    # Show the plot
    print("Displaying plot...")
    plt.show(block=True)
    print("Plot should be displayed now")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback
    print(traceback.format_exc())