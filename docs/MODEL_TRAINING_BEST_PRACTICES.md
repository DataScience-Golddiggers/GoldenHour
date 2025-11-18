# Guida Completa: Best Practices per l'Addestramento dei Modelli

## 📋 Panoramica

Questo documento fornisce best practices avanzate per migliorare l'addestramento e la performance dei modelli ARIMA/SARIMAX e LSTM nel progetto GoldenHour.

---

## 🎯 ARIMA/SARIMAX: Selezione e Addestramento Ottimale

### 1. **Grid Search Intelligente**

**Problema**: `auto_arima` con stepwise=True può perdere combinazioni ottimali.

**Soluzione**: Usa `grid_search_arima()` per ricerca esaustiva con validazione out-of-sample

```python
from utility.model_selection import grid_search_arima

# Grid search con validazione out-of-sample
best_model_info = grid_search_arima(
    endog=train,
    exog=train_exog,
    p_range=(0, 5),
    q_range=(0, 5),
    d_range=(0, 1),
    seasonal=True,
    m=5,  # Business days weekly
    P_range=(0, 2),
    Q_range=(0, 2),
    D_range=(0, 1),
    criterion='out_of_sample',  # Usa RMSE su holdout invece di AIC
    test_size=0.2,
    verbose=True
)

# Accedi al modello migliore e ai risultati
fitted_model = best_model_info['fitted_model']
results_df = best_model_info['results_df']  # Tabella con tutti i modelli testati
```

**Vantaggi**:
- ✅ Testa tutte le combinazioni (nessuna ottimizzazione locale)
- ✅ Valida su dati out-of-sample (più realistico di AIC/BIC)
- ✅ Include diagnostici residui (Ljung-Box, ARCH test)
- ✅ Restituisce tabella comparativa completa

---

### 2. **Criterio di Selezione: AIC vs BIC vs Out-of-Sample**

**Regola Pratica**:
- **AIC**: Quando vuoi il miglior fit (piccoli dataset, <1000 obs)
- **BIC**: Quando vuoi parsimonia (evita overfitting, dataset medio-grandi)
- **Out-of-Sample RMSE**: **RACCOMANDATO** per forecasting (misura vera performance predittiva)

```python
# Confronta modelli con criteri multipli
from utility.model_selection import compare_models_ic

comparison_df = compare_models_ic(
    endog=train,
    exog=train_exog,
    candidate_orders=[(1,0,0), (1,0,1), (2,0,2), (5,0,0)],
    seasonal_orders=[(1,0,1,5), (0,0,1,5), (1,0,0,5)]
)

print(comparison_df.sort_values('AIC'))
```

**Output Example**:
```
         Model        Order  Seasonal    AIC      BIC     HQIC  Log-Likelihood  Params
SARIMAX(1,0,1)(1,0,1,5)  (1,0,1)  (1,0,1,5)  -1234.5  -1200.3  -1220.1    625.2      8
SARIMAX(2,0,2)(0,0,1,5)  (2,0,2)  (0,0,1,5)  -1220.1  -1190.5  -1208.7    618.0      7
```

---

### 3. **Diagnostici Residui Automatizzati**

**Critico**: Residui mal specificati → previsioni inaffidabili

```python
from utility.model_selection import validate_residuals, print_diagnostics_report

# Valida residui dopo fit
diagnostics = validate_residuals(fitted_model, lags=20)
print_diagnostics_report(diagnostics)
```

**Output Example**:
```
======================================================================
RESIDUAL DIAGNOSTIC TESTS
======================================================================
Mean Residual:        0.000012 (should be ~0)
Std Residual:         0.015234

Ljung-Box Test (Autocorrelation):
  Statistic: 18.4532
  p-value:   0.5634
  Result:    No AC
  ✓ OK: Residuals are white noise

ARCH Test (Heteroskedasticity/Volatility Clustering):
  Statistic: 45.2341
  p-value:   0.0023
  Result:    Heteroskedastic
  ⚠ WARNING: ARCH effects detected - consider GARCH model

Jarque-Bera Test (Normality):
  Statistic: 234.5123
  p-value:   0.0000
  Result:    Non-Normal
  ⚠ Note: Non-normal residuals common in financial data (fat tails)
======================================================================
```

**Azioni**:
- `Ljung-Box p < 0.05` → Modello mal specificato, incrementa p o q
- `ARCH p < 0.05` → Usa ARIMA-GARCH hybrid (volatility clustering)
- `Jarque-Bera p < 0.05` → Accettabile per dati finanziari (code pesanti)

---

### 4. **Cross-Validation Rolling Window**

**Problema**: Single train/test split può essere fuorviante

**Soluzione**: Valida su finestre multiple

```python
from utility.model_selection import rolling_window_validation

# Cross-validation con finestra rolling
cv_results = rolling_window_validation(
    endog=pd.concat([train, test]),
    exog=pd.concat([train_exog, test_exog]) if exog else None,
    order=(2, 0, 2),
    seasonal_order=(1, 0, 1, 5),
    window_size=500,  # 500 business days ~2 anni
    forecast_horizon=5,
    step_size=25  # Muovi finestra di 25 giorni (1 mese)
)

# Analizza performance nel tempo
print(f"Mean RMSE: {cv_results['abs_error'].mean():.4f}")
print(f"Std RMSE: {cv_results['abs_error'].std():.4f}")
print(f"Worst window: {cv_results['abs_error'].max():.4f}")
```

**Vantaggi**:
- Identifica periodi problematici (es. crisi 2008)
- Valuta stabilità del modello
- Evita overfit su singolo test set

---

## 🧠 LSTM: Strategie di Addestramento Avanzate

### 1. **Learning Rate Scheduling**

**Problema**: Learning rate fisso → convergenza lenta o oscillazioni

**Soluzione**: Usa scheduler adattivo

```python
from utility.deep_learning_utils import LearningRateScheduler, plot_training_history

# Strategia 1: Reduce on Plateau (RACCOMANDATO)
lr_scheduler = LearningRateScheduler(
    strategy='reduce_on_plateau',
    initial_lr=0.001,
    patience=10,  # Riduci dopo 10 epoch senza miglioramento
    factor=0.5,   # Riduci di 50%
    min_lr=1e-7
)

# Training loop manuale
for epoch in range(epochs):
    # Train epoch
    history = model.fit(X_train, y_train, epochs=1, verbose=0, validation_data=(X_val, y_val))
    val_loss = history.history['val_loss'][0]
    
    # Update learning rate
    new_lr = lr_scheduler.step(epoch, val_loss=val_loss)
    model.optimizer.learning_rate.assign(new_lr)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: LR = {new_lr:.6f}, Val Loss = {val_loss:.6f}")
```

**Strategie Disponibili**:
- `reduce_on_plateau`: Riduci quando val_loss si stabilizza **(BEST per time series)**
- `step_decay`: Riduci ogni N epoch
- `exponential_decay`: Decay esponenziale
- `cosine_annealing`: Cosine schedule con warm restarts

---

### 2. **Early Stopping Intelligente**

**Problema**: Early stopping standard può interrompere troppo presto

**Soluzione**: Early stopping con criteri multipli

```python
from utility.deep_learning_utils import TimeSeriesEarlyStopping

# Early stopping avanzato
early_stop = TimeSeriesEarlyStopping(
    patience=20,  # Aspetta 20 epoch senza miglioramento
    min_delta=0.0001,  # Miglioramento minimo richiesto
    restore_best_weights=True,
    check_directional_accuracy=False,  # Opzionale: valuta anche la direzione
    verbose=1
)

# Training loop
for epoch in range(max_epochs):
    history = model.fit(X_train, y_train, epochs=1, verbose=0, validation_data=(X_val, y_val))
    val_loss = history.history['val_loss'][0]
    
    # Check stopping
    if early_stop(epoch, val_loss, model.get_weights()):
        # Restore best weights
        if early_stop.restore_best_weights:
            model.set_weights(early_stop.get_best_weights())
        print(f"✓ Training stopped at epoch {epoch}, best epoch was {early_stop.best_epoch}")
        break
```

---

### 3. **Ottimizzazione Batch Size**

**Problema**: Batch size influenza sia velocità che convergenza

**Soluzione**: Test automatico

```python
from utility.deep_learning_utils import find_optimal_batch_size

def build_model():
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, n_features)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(forecast_horizon)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Trova batch size ottimale
optimal_batch = find_optimal_batch_size(
    X_train, y_train,
    model_builder=build_model,
    batch_sizes=[16, 32, 64, 128, 256],
    epochs=10,
    verbose=True
)

print(f"✓ Use batch_size={optimal_batch} for training")
```

**Regola Pratica**:
- Dataset < 10k: batch_size = 32
- Dataset 10k-50k: batch_size = 64-128
- Dataset > 50k: batch_size = 128-256

---

### 4. **Architecture Search**

**Problema**: Architettura LSTM scelta manualmente può essere subottimale

**Soluzione**: Test automatico di architetture

```python
from utility.deep_learning_utils import lstm_architecture_search

# Definisci configurazioni da testare
configs = [
    {'layers': [64, 32], 'dropout': 0.2, 'learning_rate': 0.001},
    {'layers': [128, 64], 'dropout': 0.2, 'learning_rate': 0.001},
    {'layers': [64, 64, 32], 'dropout': 0.3, 'learning_rate': 0.0005},
    {'layers': [128], 'dropout': 0.1, 'learning_rate': 0.001},
]

# Esegui search
search_results = lstm_architecture_search(
    X_train, y_train,
    X_val, y_val,
    input_shape=(lookback, n_features),
    output_dim=forecast_horizon,
    configs=configs,
    epochs=50,
    batch_size=optimal_batch,
    verbose=True
)

# Usa miglior modello
best_model = search_results['best_model']
best_config = search_results['best_config']
print(f"Best architecture: {best_config}")
```

---

### 5. **Directional Accuracy (Critical for Trading)**

**Problema**: RMSE non cattura accuratezza direzionale (su/giù)

**Soluzione**: Monitora directional accuracy

```python
from utility.deep_learning_utils import calculate_directional_accuracy

# Durante validazione
predictions = model.predict(X_test)
actuals = y_test

# Calcola accuracy direzionale
dir_acc = calculate_directional_accuracy(actuals, predictions)
print(f"Directional Accuracy: {dir_acc*100:.2f}%")

# Se > 50%: modello predice direzione meglio del caso
# Se < 50%: modello peggio del caso (problema serio!)
```

**Target**:
- **> 55%**: Buono per time series finanziarie
- **> 60%**: Ottimo (potenzialmente tradeable)
- **< 50%**: Problema - rivedi feature engineering

---

### 6. **Feature Importance per LSTM**

**Problema**: LSTM è black box, difficile interpretare contributo features

**Soluzione**: Permutation importance

```python
from utility.deep_learning_utils import analyze_lstm_feature_importance

# Analizza importanza features
feature_names = ['log_return', 'GPRD_lagged', 'RSI_lagged', 'SMA_lagged', 'volatility_lagged']

importance_df = analyze_lstm_feature_importance(
    model=trained_model,
    X_train=X_train,
    y_train=y_train,
    feature_names=feature_names,
    n_iterations=100
)

print("\nFeature Importance:")
print(importance_df)
```

**Output Example**:
```
              feature  importance       std
0        GPRD_lagged    0.002341  0.000234
1              RSI_lagged    0.001823  0.000198
2     volatility_lagged    0.001234  0.000145
3          SMA_lagged    0.000923  0.000112
4          log_return    0.000654  0.000089
```

**Interpretazione**: 
- Importance > 0: Feature utile (aumento RMSE quando permutata)
- GPRD_lagged più importante → Safe-haven hypothesis supportata

---

### 7. **Ensemble Models**

**Problema**: Singolo modello LSTM può overfittare

**Soluzione**: Ensemble di modelli con seed diversi

```python
from utility.deep_learning_utils import create_ensemble_predictions

# Train multiple models con seed diversi
models = []
for seed in [42, 123, 456, 789, 999]:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    model = build_lstm_model()
    model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=0, validation_split=0.2)
    models.append(model)

# Ensemble predictions
ensemble_preds = create_ensemble_predictions(
    models=models,
    X_test=X_test,
    method='mean'  # o 'median'
)

# Di solito: ensemble RMSE < singolo modello RMSE
```

---

## 🔧 Workflow Consigliato

### ARIMA/SARIMAX

1. **Esplora modelli candidati** con `compare_models_ic()`
2. **Grid search** con `grid_search_arima(criterion='out_of_sample')`
3. **Valida residui** con `validate_residuals()` e `print_diagnostics_report()`
4. **Se ARCH effects**: Passa ad ARIMA-GARCH
5. **Cross-validation** con `rolling_window_validation()`
6. **Walk-forward finale** per production

### LSTM

1. **Ottimizza batch size** con `find_optimal_batch_size()`
2. **Architecture search** con `lstm_architecture_search()`
3. **Training** con:
   - Learning rate scheduler (`reduce_on_plateau`)
   - Early stopping avanzato
   - Validation split 80/10/10 (train/val/test)
4. **Monitora**:
   - RMSE/MAE (forecast accuracy)
   - Directional accuracy (trading viability)
   - Feature importance (interpretability)
5. **Ensemble** di 5 modelli per robustezza
6. **Walk-forward finale** per production

---

## 📊 Metriche di Successo

### ARIMA/SARIMAX
✅ **Residui**:
- Ljung-Box p-value > 0.05 (no autocorrelation)
- Mean residual ≈ 0
- ARCH p-value > 0.05 (se non usi GARCH)

✅ **Forecast**:
- RMSE < baseline naive (random walk)
- MAE interpretabile in $ (es. MAE < $50 su oro)

### LSTM
✅ **Training**:
- Val loss converge senza overfitting (gap train/val < 20%)
- Early stopping dopo ~30-50 epoch (non 200+)
- Learning rate ridotta automaticamente 2-3 volte

✅ **Forecast**:
- RMSE < ARIMA/SARIMAX (altrimenti preferisci ARIMA)
- Directional accuracy > 55%
- Ensemble RMSE < singolo modello

---

## 🚨 Red Flags da Evitare

❌ **ARIMA/SARIMAX**:
- Auto_arima con stepwise=True senza verificare exhaustive grid
- Ignorare diagnostic test (Ljung-Box, ARCH)
- Usare solo AIC/BIC senza out-of-sample validation
- Non testare GARCH quando ARCH test fallisce

❌ **LSTM**:
- Learning rate fisso 0.001 per tutti i dataset
- Early stopping patience troppo basso (< 10)
- Batch size scelto a caso
- Non monitorare directional accuracy
- Training su singolo seed (instabile)
- Architettura enorme senza validation (overfitting garantito)

---

## 📚 References

### Papers
- Hyndman & Athanasopoulos (2021) - Forecasting: Principles and Practice
- Box, Jenkins, Reinsel (2015) - Time Series Analysis
- Bollerslev (1986) - Generalized GARCH
- Hochreiter & Schmidhuber (1997) - LSTM Networks

### Code Examples
- Vedi `notebooks/02_arima_baseline.ipynb` - Auto_arima vs Grid Search
- Vedi `notebooks/05_lstm_deep_learning.ipynb` - LSTM con LR scheduling
- Vedi `notebooks/07_lstm_multivariate_exogenous.ipynb` - Feature importance

---

**Documento creato**: 18 Novembre 2025  
**Versione**: 1.0  
**Progetto**: GoldenHour - DataScience-Golddiggers
