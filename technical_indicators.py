import pandas as pd
import numpy as np

def add_sma(df, period1=20, period2=50):
    """
    Add Simple Moving Average to the dataframe
    
    Parameters:
    df (DataFrame): Price data
    period1 (int): First SMA period
    period2 (int): Second SMA period
    
    Returns:
    DataFrame: DataFrame with SMA columns added
    """
    df = df.copy()
    df[f'SMA_{period1}'] = df['Close'].rolling(window=period1).mean()
    df[f'SMA_{period2}'] = df['Close'].rolling(window=period2).mean()
    return df

def add_ema(df, period=20):
    """
    Add Exponential Moving Average to the dataframe
    
    Parameters:
    df (DataFrame): Price data
    period (int): EMA period
    
    Returns:
    DataFrame: DataFrame with EMA column added
    """
    df = df.copy()
    df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    return df

def add_rsi(df, period=14):
    """
    Add Relative Strength Index to the dataframe
    
    Parameters:
    df (DataFrame): Price data
    period (int): RSI period
    
    Returns:
    DataFrame: DataFrame with RSI column added
    """
    df = df.copy()
    
    # Calculate price changes
    delta = df['Close'].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    
    return df

def add_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """
    Add Moving Average Convergence Divergence to the dataframe
    
    Parameters:
    df (DataFrame): Price data
    fast_period (int): Fast EMA period
    slow_period (int): Slow EMA period
    signal_period (int): Signal line period
    
    Returns:
    DataFrame: DataFrame with MACD columns added
    """
    df = df.copy()
    
    # Calculate EMAs
    fast_ema = df['Close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df['Close'].ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD and signal line
    df['MACD'] = fast_ema - slow_ema
    df['MACD_Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    return df

def add_bollinger_bands(df, period=20, std_dev=2):
    """
    Add Bollinger Bands to the dataframe
    
    Parameters:
    df (DataFrame): Price data
    period (int): Moving average period
    std_dev (int): Number of standard deviations
    
    Returns:
    DataFrame: DataFrame with Bollinger Bands columns added
    """
    df = df.copy()
    
    # Calculate middle band (SMA)
    df['BB_Middle'] = df['Close'].rolling(window=period).mean()
    
    # Calculate standard deviation
    rolling_std = df['Close'].rolling(window=period).std()
    
    # Calculate upper and lower bands
    df['BB_Upper'] = df['BB_Middle'] + (rolling_std * std_dev)
    df['BB_Lower'] = df['BB_Middle'] - (rolling_std * std_dev)
    
    return df

def add_fibonacci_retracement(df):
    """
    Add Fibonacci Retracement levels to the dataframe
    
    Parameters:
    df (DataFrame): Price data
    
    Returns:
    DataFrame: DataFrame with Fibonacci levels added
    """
    df = df.copy()
    
    # Get high and low for the entire period
    price_min = df['Low'].min()
    price_max = df['High'].max()
    diff = price_max - price_min
    
    # Calculate Fibonacci levels
    df['Fib_0'] = price_min
    df['Fib_0.236'] = price_min + 0.236 * diff
    df['Fib_0.382'] = price_min + 0.382 * diff
    df['Fib_0.5'] = price_min + 0.5 * diff
    df['Fib_0.618'] = price_min + 0.618 * diff
    df['Fib_0.786'] = price_min + 0.786 * diff
    df['Fib_1'] = price_max
    
    return df

def add_stochastic_oscillator(df, k_period=14, d_period=3):
    """
    Add Stochastic Oscillator to the dataframe
    
    Parameters:
    df (DataFrame): Price data
    k_period (int): %K period
    d_period (int): %D period
    
    Returns:
    DataFrame: DataFrame with Stochastic Oscillator columns added
    """
    df = df.copy()
    
    # Calculate %K
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    
    df['%K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    
    # Calculate %D (SMA of %K)
    df['%D'] = df['%K'].rolling(window=d_period).mean()
    
    return df
