# GoldenHour: AI Agent Instructions

## Project Overview
**GoldenHour** is a financial time series analysis project focused on forecasting gold and silver prices using ARIMA/SARIMAX models with geopolitical risk as an exogenous variable. This is an **academic research project** by DataScience-Golddiggers analyzing the "safe-haven" hypothesis: precious metals increase in value during geopolitical uncertainty.

## Core Architecture

### Data Structure
- **Primary Dataset**: `data/gold_silver.csv` (10,573+ daily observations from 1985-2025)
- **Variables**: Gold/Silver prices (OPEN, HIGH, LOW, CLOSE, %CHANGE) + Geopolitical Risk Index (GPRD, GPRD_ACT, GPRD_THREAT)
- **Critical Temporal Feature**: Business day frequency only (no weekends) - markets closed Sat/Sun
- **Geopolitical Risk Index**: Must be validated against official Caldara & Iacoviello source from policyuncertainty.com before use

### Project Structure
```
data/              # Time series datasets
  gold_silver.csv  # Main dataset with prices + GPR index
  headers/         # Dataset metadata and documentation
docs/              # Comprehensive methodology documentation (Italian)
models/            # Reserved for trained model artifacts (empty)
notebooks/         # Jupyter notebooks for analysis (empty)
utility/           # Helper functions and utilities (empty)
```

## Critical Development Patterns

### Time Series Frequency Handling
**ALWAYS** set frequency to Business Days when working with this financial data:
```python
df.index = pd.to_datetime(df['DATE'])
df = df.asfreq('B')  # 'B' = Business Day frequency
```
**Why**: SARIMAX seasonal parameters (m=5 for weekly patterns) require correct frequency. Without 'B', lag calculations are wrong (t-5 from Monday would be Wednesday, not previous Monday).

### Stationarity and Target Variable
- **Never model raw prices directly** - they are non-stationary (confirmed by ADF test)
- **Always use log returns** as the target variable:
  ```python
  log_return = np.log(df['GOLD_PRICE']) - np.log(df['GOLD_PRICE'].shift(1))
  ```
- Set integration order `d=0` when using log returns (already differenced)
- Reverse transformation after forecasting: `price_forecast = price_t-1 * np.exp(log_return_forecast)`

### Exogenous Variable Lag Strategy
**Critical for valid predictions**: Exogenous features must be lagged by 1 period to avoid future data leakage:
```python
# Feature engineering
exog_features = pd.DataFrame({
    'GPR': df['GPRD'],
    'RSI': calculate_rsi(df['GOLD_PRICE']),
    'SMA': df['GOLD_PRICE'].rolling(20).mean(),
    'volatility': log_return.rolling(20).std()
})

# MANDATORY: Shift all exog variables by 1 period
exog_lagged = exog_features.shift(1).dropna()
endog = log_return.loc[exog_lagged.index]
```
**Rationale**: Models predict t+1 using information known at time t. Without lag, you'd be "training on the future."

### Model Selection Workflow
1. **Stationarity Testing**: Use ADF test on prices (expect p>0.05) and log returns (expect p<0.05)
2. **Parameter Selection**: Use `pmdarima.auto_arima()` with `seasonal=True, m=5` for weekly patterns
3. **Model Hierarchy**:
   - Baseline: Naive random walk (price_t = price_t-1)
   - Stage 1: ARIMA(p,0,q) endogenous only
   - Stage 2: SARIMAX(p,0,q)(P,D,Q,5) with lagged exogenous variables
   - Advanced: ARIMA-GARCH hybrid for volatility clustering
   - Advanced: LSTM for deep learning comparison

### Forecast Horizon
**Target**: 5-day ahead predictions (one trading week)
- Use `forecast(steps=5)` for multi-step forecasting
- For walk-forward validation, predict 5 days ahead at each step
- Exogenous variables: Need values for t+1 through t+5 (use lagged approach consistently)
- Alternative strategy: Iterative 1-step forecasts (predict t+1, use it to predict t+2, etc.)
   - Advanced: LSTM for advanced pattern capture

### Validation Requirements
**NEVER use standard train_test_split or K-Fold** - these shuffle data and violate temporal order.

**Use Walk-Forward Validation** (backtesting):
```python
# Rolling window approach
window_size = 500
for i in range(window_size, len(data)):
    train = data[i-window_size:i]
    test = data[i]
    model = SARIMAX(train).fit()
    forecast = model.forecast(steps=1)
    # Store prediction and continue
```

**Metrics**: Calculate RMSE and MAE on reconstructed prices (NOT on log returns). Avoid MAPE (fails when returns near zero).

## Technical Specifications

### Required Libraries
- `statsmodels` - SARIMAX implementation
- `pmdarima` - auto_arima for parameter selection
- `pandas` - Must support DatetimeIndex with 'B' frequency
- `arch` - GARCH models for volatility (advanced analysis)

### Key Statistical Concepts
- **Seasonality Parameter**: `m=5` (weekly cycle in 5-day trading week)
- **Business Day Logic**: Trading days only; weekends are not "missing data" but non-existent observations
- **Volatility Clustering**: Financial returns exhibit heteroskedasticity (ARCH effects) requiring GARCH modeling
- **Information Criteria**: Select models by minimizing AIC/BIC (balance fit vs. complexity)

## Documentation Reference
The `docs/` folder contains a **comprehensive 127-reference methodology document** in Italian covering:
- Dataset validation procedures (Section 1)
- Feature engineering rationale (Section 2)
- Box-Jenkins methodology (Section 3-4)
- Exogenous variable handling (Section 6)
- Walk-forward validation (Section 7)
- GARCH extension for volatility modeling (Section 8)

When implementing statistical procedures, cross-reference this document for theoretical justification.

## Common Pitfalls to Avoid
1. ❌ Forgetting to set `freq='B'` → Incorrect seasonal lag interpretation
2. ❌ Using contemporary exogenous variables → Future data leakage
3. ❌ Modeling prices instead of returns → Non-stationarity violations
4. ❌ Using shuffle-based CV → Temporal integrity violation
5. ❌ Calculating volatility on prices → Trend-inflated variance estimates
6. ❌ Using unvalidated GPR data → Risk of spurious correlations

## Development Workflow
1. **Data Loading**: Load CSV, validate GPR against official source, set DatetimeIndex with freq='B'
## Implementation Details
- **Language**: Python (primary implementation language)
- **Notebooks**: Store analysis notebooks in `notebooks/` directory
- **Utilities**: Place reusable functions in `utility/` directory (data loading, feature engineering, model evaluation)
- **Models**: Save trained model artifacts to `models/` directory
- **Data Frequency**: Daily prices only (no intraday/hourly data) - calendar dates don't affect analysis since only daily close prices are used
6. **Validation**: Walk-forward on test set, compute RMSE/MAE on prices
7. **Diagnostics**: Check residuals for ARCH effects (ACF of squared residuals)
8. **Extension**: If volatility clustering detected, implement ARIMA-GARCH hybrid

## Questions for Clarification
- Are there any existing Python implementations in the `notebooks/` or `utility/` folders that haven't been committed yet?
- Should models be trained on Italian market hours or assume standard US trading calendar?
- What is the target forecast horizon (1-day ahead, 5-day, longer)?
