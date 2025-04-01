import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def prepare_data(df, target_col='Close', window=30):
    """
    Prepare data for prediction models
    
    Parameters:
    df (DataFrame): Historical price data
    target_col (str): Target column for prediction
    window (int): Lookback window for features
    
    Returns:
    tuple: X (features), y (target), scaler (for de-normalization)
    """
    # Select only the target column for simplicity
    data = df[target_col].values.reshape(-1, 1)
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create features and target
    X, y = [], []
    for i in range(window, len(scaled_data)):
        X.append(scaled_data[i-window:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler

def linear_regression_forecast(df, forecast_days):
    """
    Generate forecast using Linear Regression
    
    Parameters:
    df (DataFrame): Historical price data
    forecast_days (int): Number of days to forecast
    
    Returns:
    tuple: DataFrame with forecast, accuracy
    """
    # Prepare data
    X, y, scaler = prepare_data(df, 'Close', 30)
    
    # If not enough data, raise exception
    if len(X) < 2:
        raise ValueError("Not enough historical data for prediction")
    
    # Split into train and test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate on test data
    y_pred = model.predict(X_test)
    mape = (1 - mean_absolute_percentage_error(y_test, y_pred)) * 100
    
    # Generate future predictions
    last_window = X[-1]
    predictions = []
    
    for _ in range(forecast_days):
        # Predict next value
        next_pred = model.predict(last_window.reshape(1, -1))[0]
        predictions.append(next_pred)
        
        # Update window for next prediction
        last_window = np.append(last_window[1:], next_pred)
    
    # Scale predictions back to original range
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(
        np.concatenate([np.zeros((predictions.shape[0], 0)), predictions], axis=1)
    )[:, 0]
    
    # Create dates for forecast period
    last_date = df.index[-1]
    forecast_dates = []
    
    for i in range(1, forecast_days + 1):
        # Skip weekends for stock/forex markets if the asset is likely a stock/forex
        next_date = last_date + timedelta(days=i)
        # For simplicity, we're not skipping weekends
        forecast_dates.append(next_date)
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Predicted_Close': predictions
    })
    forecast_df = forecast_df.set_index('Date')
    
    return forecast_df, mape

def prophet_forecast(df, forecast_days):
    """
    Generate forecast using Facebook Prophet
    
    Parameters:
    df (DataFrame): Historical price data
    forecast_days (int): Number of days to forecast
    
    Returns:
    DataFrame: DataFrame with forecast
    """
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("Prophet is not installed. Please install it with 'pip install prophet'.")
    
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    prophet_df = df.reset_index()[['Date', 'Close']].rename(
        columns={'Date': 'ds', 'Close': 'y'}
    )
    
    # Create and fit model
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=forecast_days)
    
    # Generate forecast
    forecast = model.predict(future)
    
    # Filter to only get the future predictions
    forecast = forecast[forecast['ds'] > prophet_df['ds'].max()]
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'Date': forecast['ds'],
        'Predicted_Close': forecast['yhat'],
        'Predicted_Lower': forecast['yhat_lower'],
        'Predicted_Upper': forecast['yhat_upper']
    })
    forecast_df = forecast_df.set_index('Date')
    
    return forecast_df

def arima_forecast(df, forecast_days):
    """
    Generate forecast using ARIMA
    
    Parameters:
    df (DataFrame): Historical price data
    forecast_days (int): Number of days to forecast
    
    Returns:
    tuple: DataFrame with forecast, accuracy
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        raise ImportError("statsmodels is not installed. Please install it with 'pip install statsmodels'.")
    
    # Check stationarity
    adf_result = adfuller(df['Close'].diff().dropna())
    is_stationary = adf_result[1] < 0.05
    
    # Prepare data
    train_size = int(len(df) * 0.8)
    train_data = df['Close'][:train_size]
    test_data = df['Close'][train_size:]
    
    # Determine order based on stationarity
    if is_stationary:
        # For stationary data, we use lower differencing
        p, d, q = 2, 1, 2
    else:
        # For non-stationary data, we use higher differencing
        p, d, q = 2, 2, 2
    
    # Create and fit model
    model = ARIMA(train_data, order=(p, d, q))
    fitted_model = model.fit()
    
    # Forecast on test data to evaluate
    test_pred = fitted_model.forecast(steps=len(test_data))
    mape = (1 - mean_absolute_percentage_error(test_data, test_pred)) * 100
    
    # Create and fit model on all data
    full_model = ARIMA(df['Close'], order=(p, d, q))
    fitted_full_model = full_model.fit()
    
    # Generate forecast
    forecast_values = fitted_full_model.forecast(steps=forecast_days)
    
    # Create dates for forecast
    last_date = df.index[-1]
    forecast_dates = []
    
    for i in range(1, forecast_days + 1):
        next_date = last_date + timedelta(days=i)
        forecast_dates.append(next_date)
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Predicted_Close': forecast_values
    })
    forecast_df = forecast_df.set_index('Date')
    
    return forecast_df, mape
