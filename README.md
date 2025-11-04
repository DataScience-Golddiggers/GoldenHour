# GoldenHour

GoldenHour is Data Science's Course project focused on forecasting gold and silver prices using advanced time series models, including ARIMA, SARIMAX, ARIMA-GARCH, and LSTMs. The project investigates the "safe-haven" hypothesis, analyzing how precious metals respond to geopolitical risk and uncertainty.

## Project Structure

```
data/
	gold_silver.csv         # Main dataset: daily gold/silver prices + GPR index
	headers/                # Dataset metadata and documentation
models/
	arima-baseline/         # ARIMA baseline results
	arima-garch-hybrid/     # ARIMA-GARCH hybrid results
	lstm-deep-learning/     # LSTM model and results
	sarimax-exogenous/      # SARIMAX with exogenous variables
notebooks/
	01_data_exploration.ipynb
	02_arima_baseline.ipynb
	03_sarimax_exogenous.ipynb
	04_arima_garch_hybrid.ipynb
	05_lstm_deep_learning.ipynb
	06_lstm_walk_forward.ipynb
utility/                    # Helper functions (empty)
docs/                       # Methodology and documentation (Italian)
```

## Key Features
- **Time Series Forecasting**: ARIMA, SARIMAX, ARIMA-GARCH, and LSTM models
- **Geopolitical Risk Index**: Integrated as exogenous variable (Caldara & Iacoviello GPR)
- **Walk-Forward Validation**: Realistic backtesting for all models
- **Volatility Modeling**: ARIMA-GARCH hybrid for financial volatility
- **Deep Learning**: LSTM with expanding window and multi-step forecasting

## Methodology Highlights
- All models use log returns (not raw prices) to ensure stationarity
- Exogenous variables are lagged to prevent data leakage
- Business day frequency is enforced for all time series
- Model selection via AIC/BIC and walk-forward validation
- Metrics: RMSE and MAE on reconstructed prices

## Getting Started
1. Clone the repository
2. Install required Python packages (see notebooks for details)
3. Run analysis notebooks in order for full workflow
4. Review methodology in `docs/Progetto Serie Temporali Finanziarie_ ARIMA_SARIMAX.md`

## Data Sources
- Gold/Silver prices: [source documented in headers/]
- Geopolitical Risk Index: [policyuncertainty.com](https://www.policyuncertainty.com/gpr.html)

## Authors
- DataScience-Golddiggers (UnivPM)

## License
MIT License

