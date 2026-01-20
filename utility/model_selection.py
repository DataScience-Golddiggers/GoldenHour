"""
Advanced Model Selection Utilities for Time Series
==================================================

Provides sophisticated model selection strategies for ARIMA/SARIMAX models
beyond basic auto_arima, including:
- Custom grid search with cross-validation
- Information criteria comparison (AIC, BIC, HQIC)
- Out-of-sample validation during selection
- Residual diagnostic checks as selection criteria
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from itertools import product
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import pmdarima as pm
import warnings


def grid_search_arima(
    endog: pd.Series,
    exog: Optional[pd.DataFrame] = None,
    p_range: Tuple[int, int] = (0, 5),
    d_range: Tuple[int, int] = (0, 2),
    q_range: Tuple[int, int] = (0, 5),
    seasonal: bool = False,
    m: int = 5,
    P_range: Tuple[int, int] = (0, 2),
    D_range: Tuple[int, int] = (0, 1),
    Q_range: Tuple[int, int] = (0, 2),
    criterion: str = 'aic',
    test_size: float = 0.2,
    verbose: bool = True
) -> Dict:
    """
    Exhaustive grid search for ARIMA/SARIMAX with out-of-sample validation.
    
    Parameters
    ----------
    endog : pd.Series
        Endogenous variable (target)
    exog : pd.DataFrame, optional
        Exogenous variables
    p_range, d_range, q_range : tuple
        Min and max values for ARIMA orders (p, d, q)
    seasonal : bool
        Whether to include seasonal components
    m : int
        Seasonal period (5 for business days weekly)
    P_range, D_range, Q_range : tuple
        Min and max for seasonal orders (P, D, Q, m)
    criterion : str
        'aic', 'bic', 'hqic', or 'out_of_sample' (RMSE on holdout)
    test_size : float
        Proportion of data for out-of-sample validation
    verbose : bool
        Print progress
    
    Returns
    -------
    dict
        Best model info including order, metrics, fitted model, and diagnostics
    """
    # Split data for out-of-sample validation
    split_idx = int(len(endog) * (1 - test_size))
    train_endog = endog.iloc[:split_idx]
    test_endog = endog.iloc[split_idx:]
    
    if exog is not None:
        train_exog = exog.iloc[:split_idx]
        test_exog = exog.iloc[split_idx:]
    else:
        train_exog = None
        test_exog = None
    
    # Generate parameter combinations
    p_values = range(p_range[0], p_range[1] + 1)
    d_values = range(d_range[0], d_range[1] + 1)
    q_values = range(q_range[0], q_range[1] + 1)
    
    if seasonal:
        P_values = range(P_range[0], P_range[1] + 1)
        D_values = range(D_range[0], D_range[1] + 1)
        Q_values = range(Q_range[0], Q_range[1] + 1)
        param_combinations = list(product(p_values, d_values, q_values, P_values, D_values, Q_values))
    else:
        param_combinations = list(product(p_values, d_values, q_values))
    
    if verbose:
        print(f"Grid Search: Testing {len(param_combinations)} parameter combinations")
        print(f"Criterion: {criterion.upper()}")
        print(f"Train size: {len(train_endog)}, Test size: {len(test_endog)}\n")
    
    results = []
    best_score = np.inf
    best_model_info = None
    
    for idx, params in enumerate(param_combinations):
        if seasonal:
            p, d, q, P, D, Q = params
            order = (p, d, q)
            seasonal_order = (P, D, Q, m)
        else:
            p, d, q = params
            order = (p, d, q)
            seasonal_order = None
        
        try:
            # Fit model
            if seasonal:
                model = SARIMAX(
                    train_endog,
                    exog=train_exog,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                model = ARIMA(train_endog, exog=train_exog, order=order)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted_model = model.fit(method_kwargs={'maxiter': 200})
            
            # Calculate selection criterion
            if criterion == 'aic':
                score = fitted_model.aic
            elif criterion == 'bic':
                score = fitted_model.bic
            elif criterion == 'hqic':
                score = fitted_model.hqic
            elif criterion == 'out_of_sample':
                # Out-of-sample forecast RMSE
                forecast = fitted_model.forecast(steps=len(test_endog), exog=test_exog)
                score = np.sqrt(np.mean((test_endog.values - forecast.values) ** 2))
            else:
                raise ValueError(f"Unknown criterion: {criterion}")
            
            # Residual diagnostics
            residuals = fitted_model.resid
            ljung_box = acorr_ljungbox(residuals, lags=min(10, len(residuals) // 5), return_df=True)
            lb_pvalue = ljung_box['lb_pvalue'].iloc[-1]
            
            # ARCH test (heteroskedasticity)
            try:
                arch_test = het_arch(residuals, nlags=min(5, len(residuals) // 10))
                arch_pvalue = arch_test[1]
            except:
                arch_pvalue = np.nan
            
            # Store results
            result = {
                'order': order,
                'seasonal_order': seasonal_order if seasonal else None,
                criterion: score,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'hqic': fitted_model.hqic,
                'ljung_box_pvalue': lb_pvalue,
                'arch_pvalue': arch_pvalue,
                'n_params': len(fitted_model.params),
                'converged': fitted_model.mle_retvals['converged'] if hasattr(fitted_model, 'mle_retvals') else True
            }
            results.append(result)
            
            # Update best model
            if score < best_score:
                best_score = score
                best_model_info = {
                    **result,
                    'fitted_model': fitted_model,
                    'residuals': residuals
                }
            
            if verbose and (idx + 1) % 20 == 0:
                print(f"  Progress: {idx + 1}/{len(param_combinations)} models tested, "
                      f"best {criterion}: {best_score:.4f}")
        
        except Exception as e:
            if verbose and idx < 5:  # Only print first few errors
                print(f"  ✗ Failed for {order}: {str(e)[:50]}")
            continue
    
    if verbose:
        print(f"\n✓ Grid search complete!")
        print(f"  Best model: {'SARIMAX' if seasonal else 'ARIMA'}{best_model_info['order']}", end="")
        if seasonal:
            print(f"{best_model_info['seasonal_order']}", end="")
        print(f"\n  {criterion.upper()}: {best_score:.4f}")
        print(f"  AIC: {best_model_info['aic']:.4f}")
        print(f"  BIC: {best_model_info['bic']:.4f}")
        print(f"  Ljung-Box p-value: {best_model_info['ljung_box_pvalue']:.4f}")
        print(f"  ARCH p-value: {best_model_info.get('arch_pvalue', 'N/A')}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).sort_values(criterion)
    best_model_info['results_df'] = results_df
    
    return best_model_info


def compare_models_ic(
    endog: pd.Series,
    exog: Optional[pd.DataFrame] = None,
    candidate_orders: List[Tuple] = None,
    seasonal_orders: List[Tuple] = None
) -> pd.DataFrame:
    """
    Compare multiple model specifications using information criteria.
    
    Parameters
    ----------
    endog : pd.Series
        Endogenous variable
    exog : pd.DataFrame, optional
        Exogenous variables
    candidate_orders : list of tuples
        List of (p, d, q) orders to test
    seasonal_orders : list of tuples, optional
        List of (P, D, Q, m) seasonal orders to test
    
    Returns
    -------
    pd.DataFrame
        Comparison table with AIC, BIC, HQIC for each model
    """
    if candidate_orders is None:
        candidate_orders = [(1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 2), (5, 0, 0)]
    
    results = []
    
    for order in candidate_orders:
        if seasonal_orders is not None:
            for seas_order in seasonal_orders:
                try:
                    model = SARIMAX(endog, exog=exog, order=order, seasonal_order=seas_order,
                                   enforce_stationarity=False, enforce_invertibility=False)
                    fit = model.fit(disp=False, method_kwargs={'maxiter': 200})
                    
                    results.append({
                        'Model': f'SARIMAX{order}{seas_order}',
                        'Order': order,
                        'Seasonal': seas_order,
                        'AIC': fit.aic,
                        'BIC': fit.bic,
                        'HQIC': fit.hqic,
                        'Log-Likelihood': fit.llf,
                        'Params': len(fit.params)
                    })
                except:
                    continue
        else:
            try:
                model = ARIMA(endog, exog=exog, order=order)
                fit = model.fit()
                
                results.append({
                    'Model': f'ARIMA{order}',
                    'Order': order,
                    'AIC': fit.aic,
                    'BIC': fit.bic,
                    'HQIC': fit.hqic,
                    'Log-Likelihood': fit.llf,
                    'Params': len(fit.params)
                })
            except:
                continue
    
    df = pd.DataFrame(results).sort_values('AIC')
    return df


def validate_residuals(model_fit, lags: int = 20) -> Dict:
    """
    Comprehensive residual diagnostics for ARIMA/SARIMAX models.
    
    Parameters
    ----------
    model_fit : ARIMAResults or SARIMAXResults
        Fitted model
    lags : int
        Number of lags for diagnostic tests
    
    Returns
    -------
    dict
        Diagnostic test results
    """
    residuals = model_fit.resid
    
    # Ljung-Box test (autocorrelation)
    lb_test = acorr_ljungbox(residuals, lags=min(lags, len(residuals) // 5), return_df=True)
    
    # ARCH test (heteroskedasticity)
    try:
        arch_test = het_arch(residuals, nlags=min(5, len(residuals) // 10))
        arch_stat, arch_pvalue = arch_test[0], arch_test[1]
    except:
        arch_stat, arch_pvalue = np.nan, np.nan
    
    # Normality (Jarque-Bera)
    from scipy import stats
    jb_stat, jb_pvalue = stats.jarque_bera(residuals.dropna())
    
    # Summary statistics
    diagnostics = {
        'mean_residual': residuals.mean(),
        'std_residual': residuals.std(),
        'ljung_box_stat': lb_test['lb_stat'].iloc[-1],
        'ljung_box_pvalue': lb_test['lb_pvalue'].iloc[-1],
        'arch_stat': arch_stat,
        'arch_pvalue': arch_pvalue,
        'jarque_bera_stat': jb_stat,
        'jarque_bera_pvalue': jb_pvalue,
        'normality': 'Normal' if jb_pvalue > 0.05 else 'Non-Normal',
        'autocorrelation': 'No AC' if lb_test['lb_pvalue'].iloc[-1] > 0.05 else 'AC Present',
        'heteroskedasticity': 'Homoskedastic' if arch_pvalue > 0.05 else 'Heteroskedastic'
    }
    
    return diagnostics


def print_diagnostics_report(diagnostics: Dict):
    """Print formatted diagnostic report."""
    print("\n" + "="*70)
    print("RESIDUAL DIAGNOSTIC TESTS")
    print("="*70)
    print(f"Mean Residual:        {diagnostics['mean_residual']:.6f} (should be ~0)")
    print(f"Std Residual:         {diagnostics['std_residual']:.6f}")
    print(f"\nLjung-Box Test (Autocorrelation):")
    print(f"  Statistic: {diagnostics['ljung_box_stat']:.4f}")
    print(f"  p-value:   {diagnostics['ljung_box_pvalue']:.4f}")
    print(f"  Result:    {diagnostics['autocorrelation']}")
    if diagnostics['ljung_box_pvalue'] < 0.05:
        print("  ⚠ WARNING: Residuals show autocorrelation - model may be mis-specified")
    
    print(f"\nARCH Test (Heteroskedasticity/Volatility Clustering):")
    print(f"  Statistic: {diagnostics['arch_stat']:.4f}")
    print(f"  p-value:   {diagnostics['arch_pvalue']:.4f}")
    print(f"  Result:    {diagnostics['heteroskedasticity']}")
    if diagnostics['arch_pvalue'] < 0.05:
        print("  ⚠ WARNING: ARCH effects detected - consider GARCH model")
    
    print(f"\nJarque-Bera Test (Normality):")
    print(f"  Statistic: {diagnostics['jarque_bera_stat']:.4f}")
    print(f"  p-value:   {diagnostics['jarque_bera_pvalue']:.4f}")
    print(f"  Result:    {diagnostics['normality']}")
    if diagnostics['jarque_bera_pvalue'] < 0.05:
        print("  ⚠ Note: Non-normal residuals common in financial data (fat tails)")
    
    print("="*70)


def rolling_window_validation(
    endog: pd.Series,
    exog: Optional[pd.DataFrame],
    order: Tuple,
    seasonal_order: Optional[Tuple],
    window_size: int,
    forecast_horizon: int = 1,
    step_size: int = 1
) -> pd.DataFrame:
    """
    Rolling window cross-validation for time series.
    
    Parameters
    ----------
    endog : pd.Series
        Endogenous variable
    exog : pd.DataFrame, optional
        Exogenous variables
    order : tuple
        ARIMA order (p, d, q)
    seasonal_order : tuple, optional
        Seasonal order (P, D, Q, m)
    window_size : int
        Size of training window
    forecast_horizon : int
        Steps ahead to forecast
    step_size : int
        How many steps to move forward each iteration
    
    Returns
    -------
    pd.DataFrame
        DataFrame with actual vs predicted values and errors
    """
    results = []
    
    for i in range(window_size, len(endog) - forecast_horizon + 1, step_size):
        train_endog = endog.iloc[i - window_size:i]
        train_exog = exog.iloc[i - window_size:i] if exog is not None else None
        
        test_endog = endog.iloc[i:i + forecast_horizon]
        test_exog = exog.iloc[i:i + forecast_horizon] if exog is not None else None
        
        try:
            if seasonal_order is not None:
                model = SARIMAX(train_endog, exog=train_exog, order=order, 
                               seasonal_order=seasonal_order,
                               enforce_stationarity=False, enforce_invertibility=False)
            else:
                model = ARIMA(train_endog, exog=train_exog, order=order)
            
            fit = model.fit(disp=False, method_kwargs={'maxiter': 100})
            forecast = fit.forecast(steps=forecast_horizon, exog=test_exog)
            
            for j in range(len(forecast)):
                results.append({
                    'date': test_endog.index[j],
                    'actual': test_endog.iloc[j],
                    'predicted': forecast.iloc[j] if hasattr(forecast, 'iloc') else forecast[j],
                    'error': test_endog.iloc[j] - (forecast.iloc[j] if hasattr(forecast, 'iloc') else forecast[j]),
                    'abs_error': abs(test_endog.iloc[j] - (forecast.iloc[j] if hasattr(forecast, 'iloc') else forecast[j]))
                })
        except:
            continue
    
    return pd.DataFrame(results)


def granger_causality_test(
    df: pd.DataFrame,
    cause_col: str,
    effect_col: str,
    max_lag: int = 5,
    verbose: bool = True
) -> dict:
    """
    Test Granger causality: Does X cause Y?
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with time series
    cause_col : str
        Column name of potential cause (e.g., 'GPRD')
    effect_col : str
        Column name of effect (e.g., 'GOLD_CLOSE' or 'log_return')
    max_lag : int, default=5
        Maximum number of lags to test
    verbose : bool, default=True
        Print interpretation
    
    Returns
    -------
    dict
        Dictionary with results for each lag:
        - ssr_ftest: F-test statistic
        - ssr_chi2test: Chi-square test statistic
        - lrtest: Likelihood ratio test statistic
        - params_ftest: F-test on parameters
        - p_value: P-value from F-test
        - significant: Whether relationship is significant at 5%
    
    Notes
    -----
    Granger causality tests if past values of X help predict current Y
    beyond what past values of Y alone predict.
    
    Important: "Granger causality" ≠ true causality!
    It only tests predictive precedence, not causal mechanism.
    
    For safe-haven hypothesis: Test if GPRD Granger-causes Gold returns.
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    
    # Prepare data
    data = df[[effect_col, cause_col]].dropna()
    
    # Run test
    results = {}
    significant_lags = []
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gc_results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    
    for lag in range(1, max_lag + 1):
        test_result = gc_results[lag][0]
        p_value = test_result['ssr_ftest'][1]  # F-test p-value
        
        results[f'lag_{lag}'] = {
            'f_statistic': test_result['ssr_ftest'][0],
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        if p_value < 0.05:
            significant_lags.append(lag)
    
    if verbose:
        print("="*70)
        print(f"GRANGER CAUSALITY TEST: {cause_col} → {effect_col}")
        print("="*70)
        print(f"Null Hypothesis: {cause_col} does NOT Granger-cause {effect_col}")
        print()
        
        for lag in range(1, max_lag + 1):
            res = results[f'lag_{lag}']
            sig_marker = "✅ REJECT H0" if res['significant'] else "  Fail to reject"
            print(f"  Lag {lag}: F={res['f_statistic']:.3f}, p={res['p_value']:.4f}  {sig_marker}")
        
        print()
        if significant_lags:
            print(f"✅ SIGNIFICANT at lags: {significant_lags}")
            print(f"   → {cause_col} Granger-causes {effect_col} (predictive precedence)")
            print(f"   → Past {cause_col} values help predict current {effect_col}")
            print()
            if cause_col == 'GPRD' and 'return' in effect_col.lower():
                print("   💡 INTERPRETATION FOR SAFE-HAVEN HYPOTHESIS:")
                print("   → Geopolitical risk precedes gold price movements")
                print("   → Supports safe-haven hypothesis (risk → gold demand)")
                print("   → Including GPRD as exogenous variable is justified!")
        else:
            print(f"❌ NO SIGNIFICANT Granger causality detected")
            print(f"   → {cause_col} does NOT Granger-cause {effect_col}")
            print(f"   → Past {cause_col} values don't improve {effect_col} predictions")
            print()
            if cause_col == 'GPRD':
                print("   ⚠ WARNING: Weak evidence for safe-haven hypothesis")
                print("   → Consider: Different lag structure, non-linear effects,")
                print("   → or test during crisis periods specifically")
        
        print("="*70)
        print()
        print("⚠ CAUTION: Granger causality ≠ true causality!")
        print("   It only tests predictive precedence, not causal mechanism.")
        print("="*70)
    
    return results

