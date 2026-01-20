"""
Data Utilities for GoldenHour Project

Functions for:
- Data loading and validation
- Feature engineering
- Stationarity testing
- Business day frequency handling
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from statsmodels.tsa.stattools import adfuller


def load_gold_silver_data(
    filepath: str = '../data/gold_silver.csv',
    set_business_day_freq: bool = True,
    calculate_log_returns: bool = True,
    filter_complete_gprd: bool = False
) -> pd.DataFrame:
    """
    Load and prepare gold/silver dataset with standard transformations.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file
    set_business_day_freq : bool, default=True
        Set index frequency to Business Days ('B')
    calculate_log_returns : bool, default=True
        Calculate log returns for GOLD_PRICE and SILVER_PRICE
    filter_complete_gprd : bool, default=False
        Remove rows with missing GPRD values
    
    Returns
    -------
    pd.DataFrame
        Processed dataframe with DatetimeIndex
        
    Notes
    -----
    CRITICAL: Business day frequency is essential for correct
    seasonal parameter interpretation (m=5 for weekly cycle)
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Convert to datetime and set index
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE')
    df.set_index('DATE', inplace=True)
    
    # Set business day frequency
    if set_business_day_freq:
        df = df.asfreq('B')
    
    # Calculate log returns
    if calculate_log_returns:
        if 'GOLD_PRICE' in df.columns:
            df['GOLD_LOG_RETURN'] = np.log(df['GOLD_PRICE']) - np.log(df['GOLD_PRICE'].shift(1))
        if 'SILVER_PRICE' in df.columns:
            df['SILVER_LOG_RETURN'] = np.log(df['SILVER_PRICE']) - np.log(df['SILVER_PRICE'].shift(1))
    
    # Filter for complete GPRD data
    if filter_complete_gprd and 'GPRD' in df.columns:
        initial_len = len(df)
        df = df[df['GPRD'].notna()].copy()
        print(f"Filtered for complete GPRD: {initial_len} -> {len(df)} observations")
    
    return df


def validate_dataset(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> bool:
    """
    Validate dataset structure and completeness.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate
    required_columns : list of str, optional
        List of required column names
    
    Returns
    -------
    bool
        True if valid, raises ValueError otherwise
    """
    # Check index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex")
    
    # Check frequency
    if df.index.freq is None:
        print("⚠ WARNING: Index has no explicit frequency set. Consider using asfreq('B')")
    elif df.index.freq.name != 'B':
        print(f"⚠ WARNING: Frequency is '{df.index.freq.name}', not 'B' (Business Day)")
    
    # Check required columns
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    # Check for all-NaN columns
    all_nan_cols = df.columns[df.isna().all()].tolist()
    if all_nan_cols:
        print(f"⚠ WARNING: Columns with all NaN values: {all_nan_cols}")
    
    print(f"✓ Dataset validation passed: {len(df)} observations, {len(df.columns)} columns")
    return True


def test_stationarity(
    series: pd.Series,
    series_name: str = "Series",
    significance_level: float = 0.05
) -> dict:
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    
    Parameters
    ----------
    series : pd.Series
        Time series to test
    series_name : str
        Name for reporting
    significance_level : float, default=0.05
        Significance level for hypothesis test
    
    Returns
    -------
    dict
        Test results including statistic, p-value, and conclusion
    """
    # Remove NaN
    series_clean = series.dropna()
    
    # Perform ADF test
    result = adfuller(series_clean, autolag='AIC')
    
    # Extract results
    adf_stat = result[0]
    p_value = result[1]
    critical_values = result[4]
    
    # Conclusion
    is_stationary = p_value < significance_level
    
    results = {
        'series_name': series_name,
        'adf_statistic': adf_stat,
        'p_value': p_value,
        'critical_values': critical_values,
        'is_stationary': is_stationary,
        'n_obs': len(series_clean)
    }
    
    return results


def print_stationarity_results(results: dict):
    """Print formatted stationarity test results."""
    print("\n" + "="*70)
    print(f"ADF Test Results: {results['series_name']}")
    print("="*70)
    print(f"ADF Statistic:    {results['adf_statistic']:.6f}")
    print(f"p-value:          {results['p_value']:.6f}")
    print(f"Observations:     {results['n_obs']}")
    print("\nCritical Values:")
    for key, value in results['critical_values'].items():
        print(f"  {key:10}: {value:.6f}")
    
    if results['is_stationary']:
        print(f"\n✓ CONCLUSION: {results['series_name']} is STATIONARY (reject H0, p < 0.05)")
    else:
        print(f"\n✗ CONCLUSION: {results['series_name']} is NON-STATIONARY (fail to reject H0, p ≥ 0.05)")
    print("="*70)


def calculate_technical_indicators(
    df: pd.DataFrame,
    price_col: str = 'GOLD_PRICE',
    minimal_windows: bool = True
) -> pd.DataFrame:
    """
    Calculate technical indicators for feature engineering.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price data
    price_col : str, default='GOLD_PRICE'
        Column name for price
    minimal_windows : bool, default=True
        Use minimal windows to preserve data (7/10 days vs 14/20)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with additional indicator columns
        
    Notes
    -----
    Indicators calculated:
    - RSI (Relative Strength Index)
    - SMA (Simple Moving Average)
    - EMA (Exponential Moving Average)
    - Volatility (Rolling std of log returns)
    - MACD (Moving Average Convergence Divergence)
    """
    import ta
    
    df = df.copy()
    
    # Window sizes
    rsi_window = 7 if minimal_windows else 14
    ma_window = 10 if minimal_windows else 20
    vol_window = 10 if minimal_windows else 20
    
    # RSI
    df[f'RSI_{rsi_window}'] = ta.momentum.RSIIndicator(
        df[price_col], window=rsi_window
    ).rsi()
    
    # Simple Moving Average
    df[f'SMA_{ma_window}'] = df[price_col].rolling(window=ma_window).mean()
    
    # Exponential Moving Average
    df[f'EMA_{ma_window}'] = df[price_col].ewm(span=ma_window, adjust=False).mean()
    
    # Volatility (on log returns)
    log_return_col = f'{price_col.replace("_PRICE", "")}_LOG_RETURN'
    if log_return_col in df.columns:
        df[f'VOLATILITY_{vol_window}'] = df[log_return_col].rolling(window=vol_window).std()
    
    # MACD (minimal parameters if requested)
    if minimal_windows:
        macd = ta.trend.MACD(df[price_col], window_slow=10, window_fast=5, window_sign=5)
    else:
        macd = ta.trend.MACD(df[price_col], window_slow=26, window_fast=12, window_sign=9)
    
    df['MACD'] = macd.macd()
    df['MACD_SIGNAL'] = macd.macd_signal()
    
    return df


def create_sequences_for_lstm(
    data: np.ndarray,
    target_col_idx: int,
    lookback: int,
    forecast_horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.
    
    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_samples, n_features)
    target_col_idx : int
        Index of target column in data
    lookback : int
        Number of past timesteps to use as input
    forecast_horizon : int
        Number of future timesteps to predict
    
    Returns
    -------
    tuple of (X, y)
        X: shape (n_sequences, lookback, n_features)
        y: shape (n_sequences, forecast_horizon)
        
    Notes
    -----
    For multivariate input, X contains all features for lookback window.
    For output, y contains only the target variable for forecast horizon.
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    X, y = [], []
    n_samples = len(data)
    
    for i in range(n_samples - lookback - forecast_horizon + 1):
        # Input: all features for lookback window
        X.append(data[i:i+lookback, :])
        # Output: only target for forecast horizon
        y.append(data[i+lookback:i+lookback+forecast_horizon, target_col_idx])
    
    return np.array(X), np.array(y)


def calculate_naive_baseline(
    test_prices: pd.Series,
    forecast_horizon: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate naive random walk baseline forecast.
    
    Parameters
    ----------
    test_prices : pd.Series
        Test set prices
    forecast_horizon : int, default=5
        Number of steps ahead to forecast
    
    Returns
    -------
    tuple of (predictions, actuals)
        Naive predictions and corresponding actual values
        
    Notes
    -----
    Naive forecast: price_t+h = price_t for all h
    """
    naive_predictions = []
    naive_actuals = []
    
    for i in range(0, len(test_prices) - forecast_horizon, forecast_horizon):
        current_price = test_prices.iloc[i]
        actual_prices = test_prices.iloc[i+1:i+1+forecast_horizon]
        
        # Naive forecast: repeat current price
        for _ in range(len(actual_prices)):
            naive_predictions.append(current_price)
            naive_actuals.extend(actual_prices.values)
    
    return np.array(naive_predictions), np.array(naive_actuals)


def check_for_zero_or_negative_prices(df: pd.DataFrame, price_cols: List[str]) -> bool:
    """
    Check for zero or negative prices before log transformation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to check
    price_cols : list of str
        Price column names to check
    
    Returns
    -------
    bool
        True if all prices are positive, raises ValueError otherwise
    """
    for col in price_cols:
        if col not in df.columns:
            continue
        
        zero_or_neg = (df[col] <= 0).sum()
        if zero_or_neg > 0:
            bad_dates = df[df[col] <= 0].index.tolist()
            raise ValueError(
                f"Found {zero_or_neg} zero/negative values in {col}!\n"
                f"Dates: {bad_dates[:5]}... (showing first 5)\n"
                f"Cannot compute log returns with non-positive prices."
            )
    
    print(f"✓ All prices are positive in columns: {price_cols}")
    return True


def split_train_test_temporal(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/test chronologically (no shuffle).
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to split
    train_ratio : float, default=0.8
        Fraction of data for training
    verbose : bool, default=True
        Print split information
    
    Returns
    -------
    tuple of (train_df, test_df)
        Training and test dataframes
    """
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    if verbose:
        print(f"Train/Test Split (Temporal):")
        print(f"  Train: {len(train)} obs ({train.index.min()} to {train.index.max()})")
        print(f"  Test:  {len(test)} obs ({test.index.min()} to {test.index.max()})")
        print(f"  Ratio: {train_ratio:.0%} / {1-train_ratio:.0%}")
    
    return train, test


def create_advanced_features(
    df: pd.DataFrame,
    target_col: str = 'GOLD_CLOSE',
    gprd_col: str = 'GPRD',
    include_interactions: bool = True,
    lag_periods: List[int] = [1, 5, 20],
    rolling_windows: List[int] = [5, 20, 60],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create advanced features for time series forecasting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with price and GPRD data
    target_col : str, default='GOLD_CLOSE'
        Column name for target price
    gprd_col : str, default='GPRD'
        Column name for Geopolitical Risk Index
    include_interactions : bool, default=True
        Include interaction features (GPRD * Volatility, etc.)
    lag_periods : list, default=[1, 5, 20]
        Lag periods for GPRD (1=yesterday, 5=last week, 20=last month)
    rolling_windows : list, default=[5, 20, 60]
        Windows for rolling statistics (5=week, 20=month, 60=quarter)
    verbose : bool, default=True
        Print feature engineering summary
    
    Returns
    -------
    pd.DataFrame
        DataFrame with additional features
    
    Notes
    -----
    Features created:
    - Lagged GPRD: GPRD_lag1, GPRD_lag5, GPRD_lag20
    - Rolling GPRD: GPRD_MA5, GPRD_MA20, GPRD_volatility_20
    - Price features: Returns, Volatility, RSI
    - Interactions: GPRD * Volatility, GPRD * RSI (if enabled)
    
    All features are lagged by 1 to avoid data leakage!
    """
    df_features = df.copy()
    features_added = []
    
    # Calculate log returns if not present
    if 'log_return' not in df_features.columns:
        df_features['log_return'] = np.log(df_features[target_col]) - np.log(df_features[target_col].shift(1))
    
    # === GPRD Lag Features ===
    if gprd_col in df_features.columns:
        for lag in lag_periods:
            col_name = f'{gprd_col}_lag{lag}'
            df_features[col_name] = df_features[gprd_col].shift(lag)
            features_added.append(col_name)
        
        # === GPRD Rolling Features ===
        for window in rolling_windows:
            # Moving average
            col_name = f'{gprd_col}_MA{window}'
            df_features[col_name] = df_features[gprd_col].rolling(window).mean()
            features_added.append(col_name)
            
            # Volatility (std)
            col_name = f'{gprd_col}_volatility_{window}'
            df_features[col_name] = df_features[gprd_col].rolling(window).std()
            features_added.append(col_name)
        
        # GPRD Change (rate of change)
        df_features[f'{gprd_col}_change'] = df_features[gprd_col].pct_change()
        features_added.append(f'{gprd_col}_change')
    
    # === Price Volatility ===
    for window in rolling_windows:
        col_name = f'price_volatility_{window}'
        df_features[col_name] = df_features['log_return'].rolling(window).std()
        features_added.append(col_name)
    
    # === RSI (Relative Strength Index) ===
    rsi_period = 14
    delta = df_features[target_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df_features['RSI'] = 100 - (100 / (1 + rs))
    features_added.append('RSI')
    
    # === Simple Moving Averages ===
    for window in [20, 50]:
        col_name = f'SMA_{window}'
        df_features[col_name] = df_features[target_col].rolling(window).mean()
        features_added.append(col_name)
    
    # === Interaction Features ===
    if include_interactions and gprd_col in df_features.columns:
        # GPRD * Volatility (stress amplification)
        df_features['GPRD_x_volatility'] = df_features[gprd_col] * df_features['price_volatility_20']
        features_added.append('GPRD_x_volatility')
        
        # GPRD * RSI (risk in oversold/overbought)
        df_features['GPRD_x_RSI'] = df_features[gprd_col] * df_features['RSI']
        features_added.append('GPRD_x_RSI')
    
    # === Lag all features by 1 to avoid data leakage ===
    # (We predict t+1 using info known at time t)
    for feature in features_added:
        df_features[feature] = df_features[feature].shift(1)
    
    # Drop NaN rows
    df_features = df_features.dropna()
    
    if verbose:
        print(f"✅ Feature Engineering Complete:")
        print(f"   Original features: {df.shape[1]}")
        print(f"   New features added: {len(features_added)}")
        print(f"   Total features: {df_features.shape[1]}")
        print(f"   Rows after dropna: {len(df_features)} (from {len(df)})")
        print(f"\n   Features: {', '.join(features_added)}")
    
    return df_features

