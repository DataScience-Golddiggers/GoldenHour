"""
Forecast Utilities for GoldenHour Project

Standardized functions for:
- Multi-step forecast evaluation
- Walk-forward validation
- Forecast conversion (log returns -> prices)
- Exogenous variable alignment checks
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error


def convert_log_returns_to_prices(
    log_returns: np.ndarray,
    last_actual_price: float,
    method: str = 'direct'
) -> np.ndarray:
    """
    Convert log returns to prices using standardized method.
    
    Parameters
    ----------
    log_returns : np.ndarray
        Array of log returns to convert
    last_actual_price : float
        Last known actual price (before forecast period)
    method : str, default='direct'
        Conversion method:
        - 'direct': Compound all predicted returns from last actual price
                   (realistic multi-step forecast)
        - 'iterative': Update base price with each prediction
                      (less realistic, used for comparison)
    
    Returns
    -------
    np.ndarray
        Predicted prices
        
    Notes
    -----
    For multi-step forecasts, 'direct' method is recommended as it reflects
    real-world scenario where intermediate actual prices are unknown.
    """
    if method == 'direct':
        # Compound predicted returns from last known price
        prices = last_actual_price * np.exp(np.cumsum(log_returns))
    elif method == 'iterative':
        # Iteratively update base (legacy method, less realistic)
        prices = []
        current_price = last_actual_price
        for log_ret in log_returns:
            pred_price = current_price * np.exp(log_ret)
            prices.append(pred_price)
            current_price = pred_price
        prices = np.array(prices)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'direct' or 'iterative'")
    
    return prices


def evaluate_forecast_prices(
    actuals: np.ndarray,
    predictions: np.ndarray,
    return_dict: bool = True
) -> Union[Tuple[float, float], dict]:
    """
    Calculate standard forecast metrics on prices.
    
    Parameters
    ----------
    actuals : np.ndarray
        Actual prices
    predictions : np.ndarray
        Predicted prices
    return_dict : bool, default=True
        If True, return dictionary; else return tuple (rmse, mae)
    
    Returns
    -------
    dict or tuple
        RMSE and MAE metrics
    """
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    
    if return_dict:
        return {
            'rmse': rmse,
            'mae': mae,
            'n_predictions': len(predictions)
        }
    return rmse, mae


def check_exog_alignment(
    endog: pd.Series,
    exog: pd.DataFrame,
    raise_on_mismatch: bool = True
) -> bool:
    """
    Verify that endogenous and exogenous variables are properly aligned.
    
    Parameters
    ----------
    endog : pd.Series
        Endogenous time series
    exog : pd.DataFrame
        Exogenous variables
    raise_on_mismatch : bool, default=True
        Whether to raise error on misalignment
    
    Returns
    -------
    bool
        True if aligned, False otherwise
        
    Raises
    ------
    ValueError
        If misaligned and raise_on_mismatch=True
    """
    if not endog.index.equals(exog.index):
        msg = (
            f"Endogenous and exogenous indices do not match!\n"
            f"Endog: {len(endog)} obs, from {endog.index.min()} to {endog.index.max()}\n"
            f"Exog: {len(exog)} obs, from {exog.index.min()} to {exog.index.max()}"
        )
        if raise_on_mismatch:
            raise ValueError(msg)
        print(f"⚠ WARNING: {msg}")
        return False
    return True


def create_lagged_exog(
    exog_features: pd.DataFrame,
    lag: int = 1,
    dropna: bool = True
) -> pd.DataFrame:
    """
    Create lagged exogenous variables to avoid look-ahead bias.
    
    Parameters
    ----------
    exog_features : pd.DataFrame
        Original exogenous features
    lag : int, default=1
        Number of periods to lag
    dropna : bool, default=True
        Whether to drop NaN rows created by lagging
    
    Returns
    -------
    pd.DataFrame
        Lagged exogenous features
        
    Notes
    -----
    CRITICAL: Always lag exogenous variables by at least 1 period
    to ensure that at time t, you only use information available
    up to time t-1.
    """
    exog_lagged = exog_features.shift(lag)
    
    if dropna:
        exog_lagged = exog_lagged.dropna()
    
    return exog_lagged


def walk_forward_validation(
    model_fit_func,
    train_endog: pd.Series,
    test_endog: pd.Series,
    train_exog: Optional[pd.DataFrame] = None,
    test_exog: Optional[pd.DataFrame] = None,
    forecast_horizon: int = 5,
    retrain_frequency: int = 5,
    window_type: str = 'expanding',
    verbose: bool = True
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Perform walk-forward validation with configurable retraining.
    
    Parameters
    ----------
    model_fit_func : callable
        Function that takes (endog, exog) and returns fitted model with .forecast() method
    train_endog : pd.Series
        Training endogenous data
    test_endog : pd.Series
        Test endogenous data
    train_exog : pd.DataFrame, optional
        Training exogenous data
    test_exog : pd.DataFrame, optional
        Test exogenous data
    forecast_horizon : int, default=5
        Number of steps to forecast ahead
    retrain_frequency : int, default=5
        How often to retrain model (in steps)
    window_type : str, default='expanding'
        'expanding' or 'rolling' window
    verbose : bool, default=True
        Print progress
    
    Returns
    -------
    tuple of (predictions, actuals)
        Lists of forecast arrays
    """
    predictions = []
    actuals = []
    current_model = None
    last_retrain_idx = -retrain_frequency
    
    if verbose:
        print(f"Walk-forward validation:")
        print(f"  Forecast horizon: {forecast_horizon}")
        print(f"  Retrain frequency: {retrain_frequency}")
        print(f"  Test size: {len(test_endog)}")
    
    for i in range(0, len(test_endog) - forecast_horizon + 1, forecast_horizon):
        # Retrain if needed
        if i - last_retrain_idx >= retrain_frequency or current_model is None:
            # Combine train + test up to current point
            train_data_endog = pd.concat([train_endog, test_endog.iloc[:i]])
            
            if train_exog is not None and test_exog is not None:
                train_data_exog = pd.concat([train_exog, test_exog.iloc[:i]])
                current_model = model_fit_func(train_data_endog, train_data_exog)
            else:
                current_model = model_fit_func(train_data_endog, None)
            
            last_retrain_idx = i
            if verbose and i % (retrain_frequency * 10) == 0:
                print(f"  Retrained at step {i}/{len(test_endog)}")
        
        # Forecast
        if test_exog is not None:
            forecast_exog = test_exog.iloc[i:i+forecast_horizon]
            forecast = current_model.forecast(steps=forecast_horizon, exog=forecast_exog)
        else:
            forecast = current_model.forecast(steps=forecast_horizon)
        
        actual = test_endog.iloc[i:i+forecast_horizon]
        
        predictions.append(forecast.values if hasattr(forecast, 'values') else forecast)
        actuals.append(actual.values)
    
    if verbose:
        total_forecasts = sum(len(p) for p in predictions)
        print(f"  ✓ Generated {total_forecasts} forecasts")
    
    return predictions, actuals


def calculate_diebold_mariano(
    errors1: np.ndarray,
    errors2: np.ndarray,
    h: int = 1
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for comparing forecast accuracy.
    
    Parameters
    ----------
    errors1 : np.ndarray
        Forecast errors from model 1
    errors2 : np.ndarray
        Forecast errors from model 2
    h : int, default=1
        Forecast horizon (for HAC correction)
    
    Returns
    -------
    tuple of (dm_stat, p_value)
        DM test statistic and p-value
        
    Notes
    -----
    H0: Models have equal forecast accuracy
    H1: Model 1 is more accurate than Model 2
    
    Reject H0 if p-value < 0.05 (Model 1 significantly better)
    """
    from scipy import stats
    
    # Loss differential
    d = errors1**2 - errors2**2
    
    # Mean loss differential
    d_bar = np.mean(d)
    
    # Variance of loss differential (simple version, no HAC)
    d_var = np.var(d, ddof=1)
    
    # DM statistic
    dm_stat = d_bar / np.sqrt(d_var / len(d))
    
    # P-value (one-sided test)
    p_value = stats.t.cdf(dm_stat, df=len(d)-1)
    
    return dm_stat, p_value


def create_forecast_comparison_table(
    model_results: dict,
    baseline_name: str = 'Naive'
) -> pd.DataFrame:
    """
    Create comprehensive comparison table for multiple models.
    
    Parameters
    ----------
    model_results : dict
        Dictionary with model names as keys and results dicts as values
        Each result dict should contain 'rmse', 'mae', 'n_predictions'
    baseline_name : str, default='Naive'
        Name of baseline model for % improvement calculation
    
    Returns
    -------
    pd.DataFrame
        Comparison table sorted by RMSE
    """
    rows = []
    for model_name, results in model_results.items():
        rows.append({
            'Model': model_name,
            'RMSE': results['rmse'],
            'MAE': results['mae'],
            'N_Predictions': results['n_predictions']
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('RMSE')
    
    # Calculate % improvement over baseline
    if baseline_name in df['Model'].values:
        baseline_rmse = df[df['Model'] == baseline_name]['RMSE'].values[0]
        df['RMSE_Improvement_%'] = ((baseline_rmse - df['RMSE']) / baseline_rmse * 100).round(2)
    
    return df


def print_forecast_summary(
    model_name: str,
    rmse: float,
    mae: float,
    n_predictions: int,
    comparison_models: Optional[dict] = None
):
    """
    Print standardized forecast summary.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    rmse : float
        Root Mean Squared Error
    mae : float
        Mean Absolute Error
    n_predictions : int
        Number of predictions
    comparison_models : dict, optional
        Dictionary with comparison model results
    """
    print("="*70)
    print(f"FORECAST EVALUATION - {model_name}")
    print("="*70)
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAE:  ${mae:.2f}")
    print(f"  N Predictions: {n_predictions}")
    
    if comparison_models:
        print(f"\nComparison:")
        for name, metrics in comparison_models.items():
            rmse_comp = metrics['rmse']
            improvement = (1 - rmse/rmse_comp) * 100
            print(f"  vs {name}: {improvement:+.2f}% (RMSE: ${rmse_comp:.2f})")
    
    print("="*70)


def calculate_risk_adjusted_metrics(
    actual_prices: np.ndarray,
    predicted_prices: np.ndarray,
    trading_days_per_year: int = 252
) -> dict:
    """
    Calculate risk-adjusted performance metrics for financial forecasting.
    
    Parameters
    ----------
    actual_prices : np.ndarray
        Actual price values
    predicted_prices : np.ndarray
        Predicted price values
    trading_days_per_year : int, default=252
        Number of trading days per year (252 for US markets)
    
    Returns
    -------
    dict
        Dictionary with metrics:
        - sharpe_ratio: Annualized Sharpe ratio of predicted returns
        - max_drawdown: Maximum drawdown percentage
        - win_rate: Percentage of positive predicted returns
        - profit_factor: Ratio of gains to losses
        - var_95: Value at Risk at 95% confidence
        - expected_return: Annualized expected return from predictions
        - volatility: Annualized volatility of predicted returns
    
    Notes
    -----
    These metrics are more relevant for trading strategies than RMSE/MAE alone.
    A model with lower RMSE but poor directional accuracy may have worse
    risk-adjusted performance.
    """
    # Calculate returns
    actual_returns = np.diff(actual_prices) / actual_prices[:-1]
    predicted_returns = np.diff(predicted_prices) / predicted_prices[:-1]
    
    # Sharpe Ratio (annualized)
    if len(predicted_returns) > 0 and np.std(predicted_returns) > 0:
        sharpe_ratio = (np.mean(predicted_returns) / np.std(predicted_returns)) * np.sqrt(trading_days_per_year)
    else:
        sharpe_ratio = 0.0
    
    # Maximum Drawdown
    cumulative_returns = np.cumprod(1 + predicted_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdown) * 100  # as percentage
    
    # Win Rate (% positive returns)
    win_rate = (predicted_returns > 0).sum() / len(predicted_returns) * 100 if len(predicted_returns) > 0 else 0.0
    
    # Profit Factor
    gains = predicted_returns[predicted_returns > 0].sum()
    losses = abs(predicted_returns[predicted_returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else np.inf
    
    # Value at Risk (95% confidence)
    var_95 = np.percentile(predicted_returns, 5) * 100 if len(predicted_returns) > 0 else 0.0
    
    # Annualized metrics
    expected_return = np.mean(predicted_returns) * trading_days_per_year * 100
    volatility = np.std(predicted_returns) * np.sqrt(trading_days_per_year) * 100
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'var_95': var_95,
        'expected_return': expected_return,
        'volatility': volatility
    }


def print_risk_adjusted_summary(metrics: dict):
    """
    Print risk-adjusted metrics in a formatted way.
    
    Parameters
    ----------
    metrics : dict
        Dictionary from calculate_risk_adjusted_metrics()
    """
    print("\n" + "="*70)
    print("RISK-ADJUSTED PERFORMANCE METRICS")
    print("="*70)
    print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:.3f}")
    
    if metrics['sharpe_ratio'] > 1.0:
        print("    ✅ Excellent (>1.0) - Strong risk-adjusted returns")
    elif metrics['sharpe_ratio'] > 0.5:
        print("    ✓ Good (0.5-1.0) - Decent risk-adjusted returns")
    elif metrics['sharpe_ratio'] > 0:
        print("    ⚠ Marginal (0-0.5) - Weak risk-adjusted returns")
    else:
        print("    ❌ Poor (<0) - Negative risk-adjusted returns")
    
    print(f"\n  Expected Return:     {metrics['expected_return']:+.2f}% (annualized)")
    print(f"  Volatility:          {metrics['volatility']:.2f}% (annualized)")
    print(f"  Maximum Drawdown:    {metrics['max_drawdown']:.2f}%")
    
    if abs(metrics['max_drawdown']) > 20:
        print("    ⚠ High drawdown - Consider risk management")
    
    print(f"\n  Win Rate:            {metrics['win_rate']:.2f}%")
    
    if metrics['win_rate'] > 55:
        print("    ✅ Good (>55%) - More wins than losses")
    elif metrics['win_rate'] > 50:
        print("    ✓ Marginal (50-55%) - Slightly positive")
    else:
        print("    ⚠ Low (<50%) - More losses than wins")
    
    print(f"  Profit Factor:       {metrics['profit_factor']:.3f}")
    
    if metrics['profit_factor'] > 2.0:
        print("    ✅ Excellent (>2.0) - Gains >> Losses")
    elif metrics['profit_factor'] > 1.5:
        print("    ✓ Good (1.5-2.0) - Profitable")
    elif metrics['profit_factor'] > 1.0:
        print("    ⚠ Marginal (1.0-1.5) - Barely profitable")
    else:
        print("    ❌ Poor (<1.0) - Losses > Gains")
    
    print(f"\n  Value at Risk (95%): {metrics['var_95']:.2f}%")
    print("    (Expected loss in worst 5% of cases)")
    print("="*70)
