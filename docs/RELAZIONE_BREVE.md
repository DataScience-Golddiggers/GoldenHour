# Relazione Tecnica: Modelli Predittivi e Metriche di Valutazione

## 1. Introduzione
Questo documento descrive le metodologie adottate nel progetto **GoldenHour** per la previsione dei prezzi dell'oro, dettagliando le architetture dei modelli implementati e le metriche utilizzate per la valutazione delle performance. L'obiettivo è fornire una panoramica tecnica rigorosa per supportare l'analisi comparativa dei risultati.

## 2. Metriche di Valutazione
Per valutare l'accuratezza delle previsioni, sono state utilizzate due metriche standard per problemi di regressione. È fondamentale notare che, sebbene i modelli siano addestrati sui *log-returns* (rendimenti logaritmici) per garantire la stazionarietà, **tutte le metriche sono calcolate sui prezzi ricostruiti in Dollari ($)**. Questo garantisce che l'errore sia interpretabile dal punto di vista finanziario.

### 2.1 Root Mean Squared Error (RMSE)
Il RMSE (Radice dell'Errore Quadratico Medio) misura la deviazione standard dei residui di previsione. Penalizza maggiormente gli errori grandi rispetto a quelli piccoli, rendendolo sensibile agli outlier.

$$ RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} $$

Dove:
*   $y_i$: Prezzo reale dell'oro al tempo $i$.
*   $\hat{y}_i$: Prezzo previsto dell'oro al tempo $i$.
*   $n$: Numero totale di osservazioni nel test set.

### 2.2 Mean Absolute Error (MAE)
Il MAE (Errore Medio Assoluto) misura la media della grandezza assoluta degli errori. A differenza del RMSE, tratta tutti gli errori con lo stesso peso, fornendo una visione più lineare dell'accuratezza media.

$$ MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $$

### 2.3 Procedura di Calcolo (Ricostruzione dei Prezzi)
Poiché i modelli predicono il rendimento logaritmico $r_t = \ln(P_t) - \ln(P_{t-1})$, la conversione in prezzo per il calcolo delle metriche avviene ricorsivamente per l'orizzonte di previsione a 5 giorni:

$$ \hat{P}_{t+k} = \hat{P}_{t+k-1} \cdot e^{\hat{r}_{t+k}} $$

Dove $\hat{P}_{t}$ è l'ultimo prezzo noto (o previsto al passo precedente). Questo approccio "compounding" è essenziale per simulare realisticamente l'accumulo dell'errore su più giorni.

---

## 3. Descrizione dei Modelli
Sono state sviluppate e confrontate diverse famiglie di modelli, spaziando da approcci statistici classici a reti neurali profonde.

### 3.1 ARIMA Baseline (AutoRegressive Integrated Moving Average)
Il modello di riferimento (benchmark) per l'analisi.
*   **Tipo**: Univariato Lineare.
*   **Input**: Solo la serie storica dei rendimenti logaritmici dell'oro.
*   **Configurazione**: I parametri $(p, d, q)$ sono selezionati automaticamente tramite l'algoritmo `auto_arima` basato sul criterio informativo di Akaike (AIC).
*   **Ruolo**: Stabilire la performance base che qualsiasi modello più complesso deve superare per giustificare la sua adozione.

### 3.2 SARIMAX con Variabili Esogene
Estensione del modello ARIMA per includere stagionalità e fattori esterni.
*   **Tipo**: Multivariato Lineare.
*   **Componente Stagionale**: Include una stagionalità settimanale ($m=5$) per catturare i cicli dei giorni lavorativi di trading.
*   **Variabili Esogene**: Integra l'indice di rischio geopolitico (**GPRD**) e indicatori tecnici (RSI, SMA, Volatilità).
*   **Gestione Esogene**: Tutte le variabili esogene sono ritardate (*lagged*) di $t-1$ per evitare il *data leakage* (uso di informazioni future).
*   **Ipotesi**: Testa se l'inclusione del rischio geopolitico migliora linearmente la previsione.

### 3.3 ARIMA-GARCH Ibrido
Modello a due stadi per catturare la volatilità variabile nel tempo (eteroschedasticità), tipica delle serie finanziarie.
*   **Stadio 1 (Media)**: Un modello ARIMA modella il rendimento atteso.
*   **Stadio 2 (Varianza)**: Un modello GARCH(1,1) modella la varianza dei residui dell'ARIMA.
*   **Output**: Fornisce non solo la previsione puntuale del prezzo, ma anche un intervallo di confidenza dinamico basato sulla volatilità prevista. Utile per la gestione del rischio.

### 3.4 LSTM (Long Short-Term Memory)
Reti neurali ricorrenti (RNN) progettate per catturare dipendenze a lungo termine e pattern non lineari.

#### 3.4.1 LSTM Univariate
*   **Input**: Solo sequenze passate dei rendimenti dell'oro (Lookback window = 20 giorni).
*   **Obiettivo**: Verificare se una rete neurale può estrarre pattern temporali complessi meglio di un modello lineare ARIMA senza dati esterni.

#### 3.4.2 LSTM Multivariate (Technical Only)
*   **Input**: Rendimenti Oro + Indicatori Tecnici (RSI, SMA, Volatilità).
*   **Obiettivo**: Valutare il contributo informativo dell'analisi tecnica classica in un contesto non lineare.

#### 3.4.3 LSTM Multivariate (Exogenous/GPRD)
*   **Input**: Rendimenti Oro + Indicatori Tecnici + **Indice GPRD**.
*   **Obiettivo**: Il test definitivo per l'ipotesi "Safe Haven" in un contesto non lineare. Confronta direttamente le performance con la versione "Technical Only" per isolare il valore predittivo aggiunto dal rischio geopolitico.

## 4. Validazione
Tutti i modelli sono stati validati utilizzando una strategia **Walk-Forward Validation** (o Rolling Window) con riaddestramento periodico. Questo metodo rispetta rigorosamente l'ordine temporale dei dati, simulando uno scenario reale di trading in cui il modello viene aggiornato man mano che nuovi dati diventano disponibili, evitando qualsiasi forma di *look-ahead bias*.
