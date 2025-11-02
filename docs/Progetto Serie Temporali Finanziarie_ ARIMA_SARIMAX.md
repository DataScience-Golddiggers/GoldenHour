

# **Analisi e Previsione delle Serie Temporali dei Metalli Preziosi in Relazione al Rischio Geopolitico: Un Approccio Metodologico con ARIMA e SARIMAX**

## **Sezione 1: Preparazione e Contestualizzazione dei Dati Finanziari**

L'analisi delle serie temporali finanziarie, come i prezzi dei metalli preziosi, richiede un rigore metodologico che va oltre l'applicazione standard dei modelli statistici. Il dataset in esame 1 pone due sfide immediate: la validazione della variabile esogena (Geopolitical Risk) e la gestione della frequenza temporale specifica dei mercati finanziari.

### **1.1 Analisi del Dataset e delle Variabili Chiave**

Il progetto si basa sull'ipotesi fondamentale che i metalli preziosi, in particolare l'oro e l'argento, agiscano come "safe-haven assets", ovvero beni rifugio il cui valore aumenta in periodi di incertezza e tensione.1 Per verificare questa ipotesi, il dataset affianca le variabili endogene (i prezzi dei metalli) a una variabile esogena chiave: l'Indice di Rischio Geopolitico (GPR).

È fondamentale comprendere la natura di questo indice. Il GPR non è un indicatore generico, ma una misura accademica specifica sviluppata dagli economisti della Federal Reserve, Dario Caldara e Matteo Iacoviello.2 Questo indice viene costruito quantificando la frequenza di articoli in importanti testate giornalistiche che discutono di eventi geopolitici avversi, utilizzando parole chiave relative a minacce di guerra, atti terroristici e tensioni militari.2

La rilevanza di questo indice è stata convalidata in letteratura economica; la ricerca dimostra che un aumento del GPR preannuncia una riduzione degli investimenti e dell'occupazione ed è associato a maggiori rischi al ribasso (downside risks) nei mercati finanziari.2

Data la provenienza del dataset da una piattaforma collaborativa (Kaggle) 1 e il riferimento dell'utente a fonti di notizie come investing.com (che discute di rischio geopolitico ma non è la fonte dell'indice 7), emerge una criticità di validazione. Il fondamento teorico del modello SARIMAX si basa sulla correlazione tra i prezzi e l'indice *specifico* di Caldara & Iacoviello. Se la colonna GPR nel file Kaggle fosse un indicatore diverso, l'intero modello rischierebbe di analizzare una correlazione spuria.

Si raccomanda pertanto un passo preliminare obbligatorio:

1. Scaricare i dati ufficiali e aggiornati dell'indice GPR. La fonte primaria è il sito policyuncertainty.com 3 e il sito personale dell'autore (Matteo Iacoviello), che fornisce un link diretto al file Excel: data\_gpr\_export.xls.3  
2. Confrontare i dati ufficiali con la colonna GPR del dataset Kaggle.  
3. Se si riscontrano discrepanze, la colonna GPR del dataset Kaggle *deve* essere sostituita con i dati ufficiali per garantire la validità econometrica dell'analisi.

### **1.2 Gestione della Frequenza dei Dati di Borsa (Il Vincolo freq='B')**

La seconda sfida, identificata correttamente nella richiesta, è l'assenza di dati nei fine settimana. I modelli ARIMA e SARIMAX, implementati in statsmodels, richiedono serie temporali campionate a intervalli regolari per interpretare correttamente i ritardi (lags).11

Un approccio errato consisterebbe nel "riempire" i buchi del fine settimana, ad esempio inserendo valori NaN 13 o utilizzando l'interpolazione. Questo approccio è metodologicamente fallace: i dati non sono *mancanti* (missing), sono *inesistenti* nel dominio del problema, poiché i mercati sono chiusi. Sebbene il filtro di Kalman (il motore di stima di SARIMAX) possa tecnicamente gestire i NaN saltando la fase di aggiornamento 12, ciò distorcerebbe la dipendenza temporale della serie.

La soluzione corretta richiede la definizione esplicita della frequenza temporale. La libreria pandas permette di gestire questa struttura attraverso la frequenza 'B' (Business Day).11 L'implementazione prevede la conversione dell'indice in un DatetimeIndex 18 e la successiva impostazione della frequenza:

Python

\# Assumendo che 'df' sia il DataFrame caricato  
df \= pd.to\_datetime(df)  
df \= df.set\_index('Date')  
\# Imposta la frequenza a Business Day, escludendo i weekend  
df\_business\_freq \= df.asfreq('B')

Questo passaggio non è una semplice operazione di pulizia dei dati; è un presupposto fondamentale per la corretta modellazione stagionale. I dati finanziari giornalieri possono esibire una stagionalità settimanale (ad esempio, "l'effetto del lunedì" o "l'effetto del venerdì").19 Questo è un ciclo che si ripete ogni 5 giorni lavorativi.

Se la frequenza non fosse impostata su 'B', statsmodels assumerebbe una frequenza giornaliera standard ('D'). Di conseguenza, il lag $t-5$ (5 giorni prima) di un lunedì sarebbe il mercoledì della settimana precedente, non il lunedì precedente. Impostando freq='B', il modello interpreta correttamente $t-1$ di lunedì come venerdì, e $t-5$ come il lunedì della settimana prima, rendendo matematicamente possibile l'identificazione di una stagionalità settimanale $m=5$.20

## **Sezione 2: Sviluppo di Feature Finanziarie (Feature Engineering)**

Per arricchire il modello SARIMAX, è opportuno creare un set di predittori (variabili esogene) basati sull'analisi tecnica finanziaria. Questi serviranno ad affiancare l'indice GPR.

### **2.1 Trasformazioni Primarie (La Variabile Endogena)**

Il primo passo non riguarda una feature esogena, ma la trasformazione della variabile *endogena* (il target della previsione). I prezzi dei metalli preziosi, come la maggior parte delle serie finanziarie, sono palesemente non stazionari: mostrano un trend (deriva) nel tempo.21 Modellare i prezzi direttamente viola le assunzioni fondamentali dei modelli ARIMA.23

La soluzione standard in econometria è passare dai *livelli* (prezzi) ai *rendimenti logaritmici* (log returns).

$log\\\_return\_t \= \\log(Price\_t) \- \\log(Price\_{t-1})$

Questa trasformazione ha un duplice vantaggio:

1. **Trasformazione Logaritmica:** Aiuta a stabilizzare la varianza della serie.22  
2. **Differenziazione:** Calcolare il rendimento (una differenza) rimuove il trend, aiutando a stabilizzare la media.21

Di conseguenza, la variabile endogena ($Y$) che verrà modellata non sarà il prezzo, ma il suo rendimento logaritmico. La previsione finale del *prezzo* si otterrà solo al termine del processo, invertendo la trasformazione (esponenziando il rendimento previsto e applicandolo all'ultimo prezzo noto).

### **2.2 Indicatori Tecnici di Analisi (Le Variabili Esogene)**

Si procede ora alla creazione di feature esogene ($X$).

* **Indicatori di Trend (Medie Mobili):**  
  * Simple Moving Average (SMA): Calcolata come la media mobile dei prezzi su una finestra $N$.26  
    sma \= price.rolling(window=N).mean()  
  * Exponential Moving Average (EMA): Assegna un peso maggiore ai prezzi recenti.26  
    ema \= price.ewm(span=N).mean().28  
* **Indicatori di Volatilità (Rischio):**  
  * *Rolling Standard Deviation:* Misura la volatilità (rischio) dei rendimenti.  
  * Attenzione: Questo indicatore deve essere calcolato sui log-returns, non sui prezzi. Calcolare la deviazione standard sui prezzi (che hanno un trend) produrrebbe una misura di volatilità distorta e inflazionata dal trend stesso.30  
    rolling\_vol \= log\_return.rolling(window=N).std().32  
* **Indicatori di Momentum:**  
  * *Relative Strength Index (RSI):* Un oscillatore che misura la velocità e l'entità dei movimenti di prezzo, usato per identificare condizioni di ipercomprato o ipervenduto.34  
  * Moving Average Convergence Divergence (MACD): Un indicatore di momentum che segue il trend, basato sulla differenza tra due EMA (solitamente a 12 e 26 periodi).35  
    ema\_12 \= price.ewm(span=12).mean()  
    ema\_26 \= price.ewm(span=26).mean()  
    macd \= ema\_12 \- ema\_26

Questi indicatori possono essere calcolati efficientemente utilizzando pandas o librerie specializzate come ta.

Tuttavia, la creazione di queste feature introduce un paradosso critico per la *previsione*. Un modello SARIMAX, per prevedere il valore di domani ($t+1$), richiede i valori di *domani* ($t+1$) di tutte le variabili esogene.38 Ovviamente, l'RSI o il GPR di domani sono sconosciuti oggi. Questo problema, che verrà discusso in dettaglio nella Sezione 6, rende queste feature *inutilizzabili* nella loro forma attuale per la previsione out-of-sample.

La soluzione metodologicamente corretta è utilizzare i valori *ritardati (lagged)* di queste feature. Il modello deve essere costruito per prevedere $log\\\_return(t)$ utilizzando le informazioni note al tempo $t-1$.

log\_return(t) \\sim GPR(t-1) \+ RSI(t-1) \+ SMA(t-1)

In pratica, l'intera matrice delle feature esogene (exog) deve essere shiftata di un periodo prima di essere passata al modello.39

## **Sezione 3: Analisi di Stazionarietà (Metodologia Box-Jenkins, Parte 1\)**

La metodologia Box-Jenkins (l'approccio su cui si basano i modelli ARIMA) richiede un'analisi rigorosa della stazionarietà della serie.40

### **3.1 Fondamenti Statistici: Perché la Stazionarietà è Obbligatoria**

Un processo stazionario è una serie temporale le cui proprietà statistiche (come media e varianza) sono costanti nel tempo.22 I modelli ARIMA presuppongono questa stabilità per poter modellare le dipendenze temporali.23 Come già discusso, i prezzi finanziari non sono stazionari (mostrano trend evidenti).21

### **3.2 Test di Radice Unitaria: Augmented Dickey-Fuller (ADF)**

Per verificare formalmente la stazionarietà, si utilizza un test di radice unitaria. Il test Augmented Dickey-Fuller (ADF) è lo standard del settore.43 È implementato in statsmodels.tsa.stattools.adfuller.45

### **3.3 Interpretazione Rigorosa dei Risultati del Test ADF**

L'interpretazione dei risultati dell'ADF è spesso fonte di confusione, ma è fondamentale.47

* **Ipotesi Nulla ($H\_0$):** La serie temporale *possiede* una radice unitaria. Questo significa che è **non stazionaria**.43  
* **Ipotesi Alternativa ($H\_A$):** La serie temporale *non* possiede una radice unitaria. Questo significa che è **stazionaria**.48

La regola di decisione si basa sul p-value:

* **Se $p-value \< 0.05$:** Si rifiuta l'ipotesi nulla ($H\_0$). I dati forniscono una forte evidenza che la serie è stazionaria.47  
* **Se $p-value \> 0.05$:** Non si può rifiutare l'ipotesi nulla ($H\_0$). Si deve concludere che la serie è non stazionaria.51

### **3.4 Determinazione dell'Ordine di Integrazione (d)**

Il parametro $d$ nel modello ARIMA(p, **d**, q) rappresenta l'ordine di *Integrazione*, ovvero il numero di differenziazioni necessarie per rendere la serie stazionaria.25

La procedura pratica è la seguente:

1. Si esegue il test ADF sulla serie dei prezzi originali (es. Price\_Gold). Ci si attende un $p-value \> 0.05$, confermando la non stazionarietà.  
2. Si esegue il test ADF sulla serie trasformata (i log\_return, che rappresentano la prima differenza logaritmica).53  
3. Ci si attende un $p-value \< 0.05$, confermando che la serie differenziata è stazionaria.

Questa procedura dimostra che è necessaria *una* differenziazione per raggiungere la stazionarietà. Pertanto, l'ordine di integrazione è **$d=1$**.

A questo punto, l'analista ha due opzioni di implementazione:

* **Opzione A (Implicita):** Fornire i prezzi originali al modello e specificare order=(p, 1, q). statsmodels eseguirà la differenziazione internamente.55  
* **Opzione B (Esplicita):** Fornire i log\_return (già stazionari) al modello e specificare order=(p, 0, q).

Per l'analisi finanziaria, l'Opzione B è nettamente superiore. Permette un'analisi più pulita dei residui ed è un prerequisito fondamentale se si intende procedere con la modellazione della volatilità (GARCH), la quale viene applicata ai residui della serie *già stazionaria* (i rendimenti).57

**Pertanto, l'analisi successiva utilizzerà i log\_return come variabile endogena e imposterà $d=0$ nel modello.**

La tabella seguente (Tabella 1\) dovrebbe essere compilata come giustificazione formale di questa decisione.

---

**Tabella 1: Risultati del Test di Stazionarietà (Augmented Dickey-Fuller)**

| Serie (Variabile) | Statistica Test ADF | p-value | Valore Critico (5%) | Conclusione |
| :---- | :---- | :---- | :---- | :---- |
| Prezzo Oro (Originale) | *Valore (es. \-1.2)* | *Valore (es. 0.67)* | *\-2.86* | **Non Stazionaria** |
| Log-Return Oro (d=1) | *Valore (es. \-15.4)* | *Valore (es. \< 0.01)* | *\-2.86* | **Stazionaria** |
| Prezzo Argento (Originale) | *Valore (es. \-0.9)* | *Valore (es. 0.78)* | *\-2.86* | **Non Stazionaria** |
| Log-Return Argento (d=1) | *Valore (es. \-14.9)* | *Valore (es. \< 0.01)* | *\-2.86* | **Stazionaria** |
| Indice GPR (Originale) | *Valore (es. \-2.1)* | *Valore (es. 0.24)* | *\-2.86* | **Non Stazionaria** |
| Differenza GPR (d=1) | *Valore (es. \-18.2)* | *Valore (es. \< 0.01)* | *\-2.86* | **Stazionaria** |

## ***(Nota: I valori in corsivo sono ipotetici e devono essere sostituiti con i risultati effettivi dell'analisi sul dataset.)***

## **Sezione 4: Modellazione Endogena (ARIMA) e Identificazione dei Parametri (p, q)**

Dopo aver ottenuto una serie stazionaria ($log\\\_return$, $d=0$), il passo successivo della metodologia Box-Jenkins 40 è identificare i parametri $p$ (ordine AutoRegressivo) e $q$ (ordine di Media Mobile).

### **4.1 Identificazione Manuale: Analisi ACF e PACF**

Gli strumenti diagnostici tradizionali per determinare $p$ e $q$ sono i grafici della Funzione di Autocorrelazione (ACF) 59 e della Funzione di Autocorrelazione Parziale (PACF).60 Questi grafici devono essere generati sulla serie *stazionaria* (i $log\\\_return$).

* **ACF (plot\_acf):** Misura la correlazione tra $Y\_t$ e i suoi lag $Y\_{t-k}$ (incluso l'effetto indiretto dei lag intermedi).  
* **PACF (plot\_pacf):** Misura la correlazione *diretta* tra $Y\_t$ e $Y\_{t-k}$, rimuovendo l'effetto dei lag $t-1, t-2,..., t-k+1$.

Le regole euristiche per l'interpretazione (sulla serie stazionaria) sono 54:

* **Modello AR(p) (Autoregressivo):**  
  * ACF: Decade lentamente (geometricamente) o in modo sinusoidale ("tails off").  
  * PACF: Si interrompe bruscamente (taglia) dopo il lag $p$.  
* **Modello MA(q) (Media Mobile):**  
  * ACF: Si interrompe bruscamente dopo il lag $q$.  
  * PACF: Decade lentamente ("tails off").  
* **Modello ARMA(p, q) (Misto):**  
  * Sia ACF che PACF decadono lentamente.

### **4.2 Identificazione Automatica: auto\_arima**

L'ispezione visuale dei grafici ACF/PACF è notoriamente soggettiva.40 Un approccio moderno e più robusto consiste nell'automatizzare la ricerca dei parametri. La libreria pmdarima 66 offre la funzione auto\_arima 68, che implementa un processo di ricerca efficiente.

Questa funzione esegue una ricerca a passi (stepwise search) 71 per testare diverse combinazioni di (p, q) e (P, Q), selezionando il modello che minimizza un criterio informativo (Information Criterion).71 I criteri più comuni sono l'AIC (Akaike Information Criterion) e il BIC (Bayesian Information Criterion).71 Questi criteri bilanciano la bontà del fit (massimizzando la verosimiglianza) con la complessità del modello (penalizzando un numero eccessivo di parametri).

È importante notare un potenziale risultato dell'analisi dei rendimenti finanziari. Secondo l'Ipotesi dei Mercati Efficienti (Efficient Market Hypothesis, EMH), i rendimenti passati non dovrebbero avere alcun potere predittivo sui rendimenti futuri. Se ciò fosse vero, i $log\\\_return$ sarebbero essenzialmente "rumore bianco" (white noise) 75, ovvero una serie di shock casuali e incorrelati.

In questo scenario, non ci si deve sorprendere se auto\_arima dovesse restituire ordini molto bassi, come ARIMA(1, 0, 1), ARIMA(0, 0, 1), o addirittura ARIMA(0, 0, 0).75 Questo *non* significa che l'analisi è fallita; significa che la *media* dei rendimenti è imprevedibile. L'analisi quantitativa, infatti, sa bene che, sebbene la media dei rendimenti sia difficile da prevedere, la loro *varianza* (la volatilità) è spesso prevedibile (un fenomeno noto come volatility clustering).76 Questo risultato motiva il passaggio a modelli più avanzati (GARCH), discussi nella Sezione 8\.

---

**Tabella 2: Confronto Criteri Informativi (AIC/BIC) per Modelli ARMA Candidati (su $log\\\_return$)**

| Modello ARMA(p, q) | AIC | BIC | Log-Likelihood |
| :---- | :---- | :---- | :---- |
| ARMA(1, 0\) | *Valore* | *Valore* | *Valore* |
| ARMA(0, 1\) | *Valore* | *Valore* | *Valore* |
| ARMA(1, 1\) | *Valore* | *Valore* | *Valore* |
| ARMA(2, 2\) | *Valore* | *Valore* | *Valore* |
| **Modello auto\_arima** | *Valore Minimo* | *Valore Minimo* | *Valore* |

## **(Nota: La tabella deve essere compilata per identificare il modello con AIC/BIC più bassi, che sarà selezionato come modello base.**

71

## **Sezione 5: Modellazione Stagionale ed Esogena (SARIMAX)**

Il modello base ARIMA(p, 0, q) può ora essere esteso per includere le componenti stagionali (S) e le variabili esogene (X), portando al modello SARIMAX.52

### **5.1 Rilevamento della Stagionalità (P, D, Q, m)**

Il modello SARIMAX introduce un secondo set di parametri: seasonal\_order=(P, D, Q, m).79

* $(P, D, Q)$ sono gli ordini AR, di Integrazione e MA per la componente stagionale.  
* $m$ è il *periodo stagionale*, ovvero il numero di osservazioni in un ciclo.82

La funzione auto\_arima *non* determina $m$; deve essere specificato dall'utente.70 La scelta di $m$ dipende dalla comprensione del dominio dei dati.

Come discusso nella Sezione 1.2, l'uso di freq='B' (Business Day) 11 è stato fondamentale per allineare la serie temporale a un potenziale ciclo settimanale. L'ipotesi più plausibile per i dati finanziari giornalieri è "l'effetto giorno della settimana", ovvero un ciclo che si ripete ogni 5 giorni lavorativi.19

Pertanto, l'impostazione stagionale più logica da testare è **$m=5$**.20

Altre possibilità, sebbene meno probabili per i prezzi dei metalli, potrebbero includere cicli mensili (es. $m=21$, il numero approssimativo di giorni di trading in un mese) o annuali (es. $m=252$).86

La metodologia per testare $m=5$ è la seguente:

1. Analisi Visiva: Eseguire una decomposizione stagionale per ispezionare visivamente il ciclo settimanale.  
   statsmodels.tsa.seasonal.seasonal\_decompose(log\_return, period=5).70  
2. **Analisi ACF:** Ispezionare l'ACF della serie stazionaria per picchi significativi ai lag multipli di $m$ (es. lag 5, 10, 15).88  
3. Modellazione Automatica: Eseguire auto\_arima specificando $m=5$. La funzione determinerà automaticamente i migliori ordini $(P, D, Q)$ per quel ciclo.70  
   auto\_arima(log\_return, seasonal=True, m=5,...)

Se auto\_arima restituisce $(P=0, D=0, Q=0)$ anche con $m=5$, ciò costituisce una forte evidenza che non esiste una stagionalità settimanale significativa nei rendimenti 85, e il modello si riduce a un ARIMAX.

### **5.2 Integrazione delle Variabili Esogene (exog)**

Infine, si include il componente $X$ (eXogenous).79 Si costruisce un DataFrame pandas (denominato exog\_features) che contiene:

1. L'indice GPR (validato).5  
2. Le feature finanziarie (SMA, EMA, Volatilità, RSI, MACD).26

Come stabilito nella Sezione 2.2, per garantire la validità causale per la *previsione*, questa matrice deve essere ritardata di un periodo:

exog\_model \= exog\_features.shift(1).dropna()  
endog\_model \= log\_return.loc\[exog\_model.index\]  
Il modello finale viene quindi istanziato utilizzando statsmodels.tsa.statespace.sarimax.SARIMAX 79:

Python

\# p, q, P, D, Q sono determinati da auto\_arima  
model \= SARIMAX(endog=endog\_model,  
                exog=exog\_model,  
                order=(p, 0, q),  
                seasonal\_order=(P, D, Q, 5))  
sarimax\_fit \= model.fit()

## **Sezione 6: La Sfida Critica della Previsione con Variabili Esogene**

È imperativo dedicare una sezione specifica a quella che è la più grande trappola metodologica nell'uso di modelli ARIMAX o SARIMAX per la *previsione*: la gestione dei valori futuri delle variabili esogene.92

### **6.1 Il Paradosso delle Previsioni (exog Futuri Sconosciuti)**

Un analista potrebbe erroneamente addestrare un modello utilizzando dati contemporanei, ad esempio:  
model \= SARIMAX(log\_return(t), exog=GPR(t)).fit()  
Questo modello stima i coefficienti di regressione per le variabili exog 95 e produrrà un ottimo fit *in-sample*.

Il problema sorge al momento della previsione (forecasting). Quando si invoca sarimax\_fit.forecast(steps=H) per prevedere $H$ periodi futuri 96, statsmodels richiederà i valori futuri delle variabili esogene, $GPR(t+1),..., GPR(t+H)$.38 Poiché questi valori sono sconosciuti, il software solleverà un ValueError.38

Questo distingue fondamentalmente un **modello esplicativo** da un **modello predittivo**.

* *Modello Esplicativo (Contemporaneo):* Prezzo(t) \~ GPR(t). È utile per l'analisi economica (es. "un aumento del 10% del GPR oggi è correlato a un aumento del 0.5% dell'oro oggi").2  
* *Modello Predittivo (Ritardato):* Prezzo(t+1) \~ GPR(t). È l'unico modello utile per la previsione, poiché utilizza solo informazioni note al tempo $t$.93

Confondere i due è un errore metodologico grave.

### **6.2 Strategie Risolutive**

Esistono due strategie principali per affrontare questo problema, con diversi gradi di validità.

#### **Approccio 1 (Causale Ritardato \- Fortemente Raccomandato)**

Questo è l'approccio delineato nelle sezioni precedenti.

* **Logica:** Si assume che le notizie e gli indicatori (GPR, RSI) di *oggi* ($t$) influenzino le decisioni di trading e, quindi, il prezzo di *domani* ($t+1$). Questa è un'ipotesi causalmente valida.  
* Implementazione: L'intera matrice exog viene ritardata (shiftata) di un periodo prima dell'addestramento.39  
  exog\_model \= exog\_features.shift(1)  
* **Vantaggio:** Quando si chiama sarimax\_fit.forecast(steps=1), il modello richiede il valore exog al tempo $t+1$ (nel suo sistema di riferimento interno). Grazie allo shift, questo corrisponde al valore *originale* al tempo $t$, che è *noto* al momento della previsione. Questo metodo è robusto, causalmente valido e facile da implementare.

#### **Approccio 2 (Previsione a Due Stadi \- Non Raccomandato)**

* **Logica:**  
  1. Costruire un modello separato (es. un ARIMA) per ogni variabile esogena (es. un modello per prevedere il GPR).93  
  2. Prevedere i valori futuri del GPR per $t+1,..., t+H$.  
  3. Utilizzare questi valori *previsti* (e quindi incerti) come input exog nel modello SARIMAX principale per i prezzi.99  
* **Svantaggio:** Questo metodo introduce e propaga l'errore.93 L'incertezza della previsione del GPR si accumula e si combina con l'incertezza della previsione del prezzo dell'oro, rendendo le previsioni a lungo termine esponenzialmente inaffidabili.

## **Sezione 7: Strategie di Validazione Robuste e Valutazione del Modello**

La valutazione di un modello di serie temporali deve rispettare rigorosamente l'ordine cronologico dei dati.100

### **7.1 Validazione Cronologica (Il Pericolo dello Shuffle)**

L'uso di tecniche di validazione incrociata standard, come il K-Fold, o l'uso di sklearn.model\_selection.train\_test\_split 102 con l'impostazione predefinita shuffle=True, è *catastrofico* per le serie temporali. Ciò porterebbe inevitabilmente ad "addestrare sul futuro per prevedere il passato", producendo metriche di performance irrealisticamente ottimistiche e un modello inutile nel mondo reale.100

* Soluzione Semplice (Train-Test Split): Il metodo più semplice è un singolo split cronologico.103 Ad esempio, utilizzare l'80% dei dati per l'addestramento (train set) e il restante 20% per il test.  
  train\_size \= int(len(df) \* 0.8)  
  train, test \= df\[:train\_size\], df\[train\_size:\]  
* **Soluzione Avanzata (Time Series Cross-Validation):** Per una stima più robusta, si può utilizzare sklearn.model\_selection.TimeSeriesSplit.100 Questo iteratore crea K-fold *espandibili*:  
  * Fold 1: Train \[1...100\], Test \[101...120\]  
  * Fold 2: Train \[1...120\], Test \[121...140\]  
  * ...e così via. Questo approccio rispetta l'ordine temporale.

### **7.2 Validazione Walk-Forward (Backtesting Quantitativo)**

Per i dati finanziari, lo standard aureo è la validazione "Walk-Forward" (WFA), spesso chiamata backtesting.105 Questo metodo simula come un modello verrebbe utilizzato operativamente nel trading, dove viene ri-addestrato continuamente man mano che arrivano nuovi dati.107

La logica (utilizzando una finestra mobile o "rolling window") è 107:

1. Definire una dimensione della finestra di addestramento (es. 500 giorni).  
2. Addestrare il modello sui giorni \[1...500\].  
3. Prevedere il giorno . Salvare la previsione.  
4. Spostare la finestra in avanti: addestrare il modello sui giorni \[2...501\].  
5. Prevedere il giorno . Salvare la previsione.  
6. Ripetere fino alla fine del test set.

Questo processo è computazionalmente intensivo ma fornisce la stima più realistica della performance del modello e della sua capacità di adattarsi a cambiamenti nei regimi di mercato (concept drift).107

### **7.3 Metriche di Valutazione (Scegliere la Metrica Giusta)**

La scelta della metrica di errore è cruciale.109

* **RMSE (Root Mean Squared Error):** $RMSE \= \\sqrt{\\frac{1}{n} \\sum (y\_t \- \\hat{y}\_t)^2}$.109  
  * *Pro:* È la metrica più comune. Penalizza gli errori più grandi in modo quadratico, il che è desiderabile nella gestione del rischio finanziario, dove i grandi errori sono catastrofici.109  
  * *Unità:* Nelle stesse unità della variabile target.  
* **MAE (Mean Absolute Error):** $MAE \= \\frac{1}{n} \\sum |y\_t \- \\hat{y}\_t|$.109  
  * *Pro:* Meno sensibile agli outlier rispetto a RMSE.109 È più interpretabile (es. "il modello sbaglia in media di 15$").109  
* **MAPE (Mean Absolute Percentage Error):** $MAPE \= \\frac{1}{n} \\sum \\frac{|y\_t \- \\hat{y}\_t|}{y\_t}$.109  
  * **Controindicazione:** Questa metrica è **inutilizzabile** quando la variabile target (i $log\\\_return$) può essere zero o molto vicina allo zero. I rendimenti finanziari oscillano costantemente intorno allo zero, portando a divisioni per zero e valori MAPE che esplodono all'infinito, rendendo la metrica priva di significato.112

Il processo di valutazione deve quindi avvenire sui *prezzi* reinvertiti, non sui *log-returns*. Il flusso di validazione corretto per ogni previsione $t$ nel test set è:

1. Il modello SARIMAX prevede $log\\\_return\\\_forecast(t)$.  
2. Si inverte la trasformazione per ottenere il prezzo previsto:  
   $Price\\\_forecast(t) \= Price\\\_actual(t-1) \\times \\exp(log\\\_return\\\_forecast(t))$  
3. Si calcola l'errore confrontando $Price\\\_forecast(t)$ con $Price\\\_actual(t)$.  
4. Si calcola RMSE e MAE su questi errori di prezzo (in $).

---

**Tabella 3: Risultati della Validazione Walk-Forward (Backtest) \- Metriche di Errore sui Prezzi**

| Modello | RMSE (in $) | MAE (in $) |
| :---- | :---- | :---- |
| **Modello 1: Benchmark Naive (Random Walk)** | *Valore* | *Valore* |
| *(Previsione: Prezzo(t) \= Prezzo(t-1))* |  |  |
| **Modello 2: ARIMA Endogeno (p, 0, q)** | *Valore* | *Valore* |
| *(Solo su log-return storici)* |  |  |
| **Modello 3: SARIMAX (con exog ritardati)** | *Valore* | *Valore* |
| *(Include GPR, RSI, Volatilità, ecc. ritardati)* |  |  |

## ***(Nota: Questa tabella è il risultato cruciale del progetto. Se $RMSE(\\text{Modello 3}) \< RMSE(\\text{Modello 2})$, si può concludere che l'inclusione del GPR e delle altre feature esogene ha fornito un valore predittivo statisticamente significativo.)***

## **Sezione 8: Analisi Avanzata: Oltre la Previsione della Media (Modellazione della Volatilità)**

Un'analisi di livello esperto non si ferma alla previsione del valore (la media), ma affronta anche la previsione del *rischio* (la varianza). Qui emergono i limiti intrinseci dei modelli ARIMA.

### **8.1 Limiti dei Modelli (S)ARIMA**

I modelli ARIMA, sebbene potenti, hanno limitazioni significative per i dati finanziari:

1. **Assunzione di Linearità:** ARIMA è un modello lineare.23 Non è in grado di catturare dinamiche complesse e non lineari, come l'asimmetria delle reazioni ai crash di mercato rispetto ai rialzi.23  
2. **Assunzione di Omoschedasticità:** ARIMA assume che la varianza dei residui (gli errori del modello) sia costante nel tempo (omoschedastica).114

### **8.2 Identificazione del "Volatility Clustering" (Eteroschedasticità)**

L'assunzione di omoschedasticità è palesemente violata nei mercati finanziari. Questi esibiscono un fenomeno noto come "volatility clustering" (agglomerazione della volatilità): periodi di alta turbolenza sono seguiti da altra turbolenza, e periodi di calma sono seguiti da calma.58

Il modello (S)ARIMA non riesce a catturare questo comportamento.76 Lo si può diagnosticare facilmente:

1. Si estraggono i residui del modello SARIMAX finale: residuals \= sarimax\_fit.resid.  
2. Si analizzano i *quadrati dei residui* (residuals\*\*2).  
3. Se i quadrati dei residui mostrano autocorrelazione (es. picchi significativi nel loro grafico ACF), significa che la volatilità di oggi dipende dalla volatilità di ieri. Questo è l'effetto ARCH (Autoregressive Conditional Heteroskedasticity).117

### **8.3 Introduzione ai Modelli (G)ARCH**

Questo ci porta a una distinzione concettuale fondamentale:

* Il modello **SARIMAX** modella la **media condizionata** (la previsione del rendimento atteso).76  
* Il modello **GARCH** (Generalized Autoregressive Conditional Heteroskedasticity) modella la **varianza condizionata** (la previsione della volatilità).77

Dato che i residui del SARIMAX mostrano eteroschedasticità, è necessario un secondo modello per modellare questi residui.

### **8.4 Implementazione di un Modello Ibrido ARIMA-GARCH**

In Python, l'implementazione di un modello ibrido ARIMA-GARCH è tipicamente un processo in due fasi, poiché i modelli ARIMA (statsmodels) 56 e GARCH (arch) 58 risiedono in pacchetti separati.58

* **Passo 1: Modellare la Media (SARIMAX)**  
  * Si addestra il modello SARIMAX ottimale (dalla Sezione 5\) sui $log\\\_return$.57  
  * Si estraggono i residui (errori di previsione della media): residuals \= sarimax\_fit.resid.  
* **Passo 2: Modellare la Varianza (GARCH)**  
  * Si importa la libreria arch: from arch import arch\_model.114  
  * Si addestra un modello GARCH(1,1) (la configurazione più comune) *sui residui* del Passo 1\.

Python  
\# p=1, q=1 sono gli ordini standard per GARCH  
garch\_model \= arch\_model(residuals, vol='Garch', p=1, q=1)  
garch\_fit \= garch\_model.fit(disp='off')

* **Risultato:** Questo modello ibrido ora produce due previsioni:  
  1. Dall'ARIMA: la previsione del rendimento futuro.  
  2. Dal GARCH: la previsione della *volatilità* (varianza) futura di quel rendimento.

Questa seconda previsione è spesso più preziosa per la finanza, in quanto è un input diretto per la gestione del rischio, il calcolo del Value-at-Risk (VaR) e il pricing delle opzioni.

### **8.5 Confronto con Approcci Non Lineari (LSTM)**

Mentre ARIMA-GARCH è un approccio statistico classico, i modelli di deep learning come le reti LSTM (Long Short-Term Memory) 123 sono un'alternativa moderna.

* **Pro (LSTM):** Sono intrinsecamente non lineari e possono catturare pattern complessi che ARIMA ignora.123 Non richiedono la stazionarietà manuale (sebbene spesso migliori la performance).123  
* **Pro (ARIMA):** È più semplice, computazionalmente molto più veloce da addestrare 125, e altamente interpretabile (i coefficienti hanno un significato statistico chiaro).125 Spesso, ARIMA si rivela un benchmark molto difficile da battere.126

## **Sezione 9: Sintesi Operativa e Considerazioni Conclusive**

Questo rapporto ha delineato una metodologia rigorosa per l'analisi e la previsione dei prezzi di oro e argento utilizzando modelli ARIMA e SARIMAX, con un focus specifico sulla gestione dei dati finanziari e l'inclusione del Geopolitical Risk Index.

**Riepilogo dei Risultati Chiave (Attesi):**

1. **Preparazione dei Dati:** L'impostazione della frequenza su 'Business Day' (freq='B') è stata identificata come un prerequisito per testare la stagionalità settimanale $m=5$. La validazione della fonte dell'indice GPR 3 è stata definita come un passaggio critico per la validità del progetto.  
2. **Stazionarietà (Tabella 1):** I test ADF confermeranno che i prezzi sono non stazionari I(1) (integrati di ordine 1), mentre i rendimenti logaritmici sono stazionari I(0). Il modello predittivo deve targettizzare i $log\\\_return$ con $d=0$.  
3. **Selezione del Modello (Tabella 2):** L'analisi dei criteri AIC/BIC (tramite auto\_arima) determinerà gli ordini (p, q) ottimali per la media dei rendimenti e (P, D, Q) per l'eventuale ciclo stagionale $m=5$.  
4. **Gestione delle Variabili Esogene:** La sfida critica della previsione out-of-sample con exog è stata risolta imponendo un ritardo (lag) di 1 periodo sull'intera matrice delle feature (GPR, RSI, SMA, ecc.), garantendo la validità causale del modello predittivo.  
5. **Valutazione (Tabella 3):** I risultati del backtest (Validazione Walk-Forward) determineranno oggettivamente se l'inclusione delle variabili esogene (Modello SARIMAX) riduce l'errore di previsione (RMSE/MAE) rispetto a un modello ARIMA endogeno e a un benchmark naive.  
6. **Analisi della Volatilità:** L'analisi dei residui del SARIMAX rivelerà quasi certamente la presenza di volatility clustering (eteroschedasticità). Questo convalida la limitazione di ARIMA (che assume varianza costante) 76 e motiva l'implementazione di un modello ibrido **ARIMA-GARCH** per modellare e prevedere separatamente la media (rendimento) e la varianza (rischio).  
7. addestramento di un modello LSTM e valutare la sua efficacia rispetto ARIMA-Family

Considerazioni Finali:  
I modelli della famiglia ARIMA sono lineari e tendono a perdere efficacia su orizzonti di previsione lunghi.23 Presumono che le relazioni statistiche del passato rimangano valide in futuro.

#### **Bibliografia**

1. Gold-Silver Price VS Geopolitical Risk (1985–2025) \- Kaggle, accesso eseguito il giorno novembre 2, 2025, [https://www.kaggle.com/datasets/shreyanshdangi/gold-silver-price-vs-geopolitical-risk-19852025](https://www.kaggle.com/datasets/shreyanshdangi/gold-silver-price-vs-geopolitical-risk-19852025)  
2. Measuring Geopolitical Risk \- American Economic Association, accesso eseguito il giorno novembre 2, 2025, [https://www.aeaweb.org/articles?id=10.1257/aer.20191823](https://www.aeaweb.org/articles?id=10.1257/aer.20191823)  
3. Geopolitical Risk Index (GPR) \- Economic Policy Uncertainty Index, accesso eseguito il giorno novembre 2, 2025, [https://www.policyuncertainty.com/gpr.html](https://www.policyuncertainty.com/gpr.html)  
4. Measuring Geopolitical Risk \- Federal Reserve Board, accesso eseguito il giorno novembre 2, 2025, [https://www.federalreserve.gov/econres/ifdp/files/ifdp1222r1.pdf](https://www.federalreserve.gov/econres/ifdp/files/ifdp1222r1.pdf)  
5. accesso eseguito il giorno novembre 2, 2025, [https://en.macromicro.me/charts/55589/global-geopolitical-risk-index\#:\~:text=Geopolitical%20Risk%20Index%20(GPR)%20is,keywords%20used%20in%20the%20press.](https://en.macromicro.me/charts/55589/global-geopolitical-risk-index#:~:text=Geopolitical%20Risk%20Index%20\(GPR\)%20is,keywords%20used%20in%20the%20press.)  
6. Measuring Geopolitical Risk \- Matteo Iacoviello, accesso eseguito il giorno novembre 2, 2025, [https://www.matteoiacoviello.com/gpr\_files/GPR\_PAPER.pdf](https://www.matteoiacoviello.com/gpr_files/GPR_PAPER.pdf)  
7. Ample supply, subdued demand to curb oil prices despite geopolitical risks: Reuters poll By Reuters, accesso eseguito il giorno novembre 2, 2025, [https://www.investing.com/news/commodities-news/ample-supply-subdued-demand-to-curb-oil-prices-despite-geopolitical-risks-reuters-poll-4323196](https://www.investing.com/news/commodities-news/ample-supply-subdued-demand-to-curb-oil-prices-despite-geopolitical-risks-reuters-poll-4323196)  
8. The Energy Report: Trump-Putin Talks Ease Geopolitical Premium, accesso eseguito il giorno novembre 2, 2025, [https://www.investing.com/analysis/the-energy-report-trumpputin-talks-ease-geopolitical-premium-200668731](https://www.investing.com/analysis/the-energy-report-trumpputin-talks-ease-geopolitical-premium-200668731)  
9. Oil Market Repricing as Geopolitical Risk Premium Evaporates, accesso eseguito il giorno novembre 2, 2025, [https://www.investing.com/analysis/oil-market-repricing-as-geopolitical-risk-premium-evaporates-200669262](https://www.investing.com/analysis/oil-market-repricing-as-geopolitical-risk-premium-evaporates-200669262)  
10. Economic Policy Uncertainty Index, accesso eseguito il giorno novembre 2, 2025, [https://www.policyuncertainty.com/](https://www.policyuncertainty.com/)  
11. freq argument options in statsmodels tsa AR and ARMA models \- Stack Overflow, accesso eseguito il giorno novembre 2, 2025, [https://stackoverflow.com/questions/30089591/freq-argument-options-in-statsmodels-tsa-ar-and-arma-models](https://stackoverflow.com/questions/30089591/freq-argument-options-in-statsmodels-tsa-ar-and-arma-models)  
12. Fitting ARIMA to time series with missing values \- Cross Validated \- Stack Exchange, accesso eseguito il giorno novembre 2, 2025, [https://stats.stackexchange.com/questions/346225/fitting-arima-to-time-series-with-missing-values](https://stats.stackexchange.com/questions/346225/fitting-arima-to-time-series-with-missing-values)  
13. Python Pandas : Return the consecutive missing weekdays dates and assign rate next to missing dates in a dataframe \- Stack Overflow, accesso eseguito il giorno novembre 2, 2025, [https://stackoverflow.com/questions/60156375/python-pandas-return-the-consecutive-missing-weekdays-dates-and-assign-rate-ne](https://stackoverflow.com/questions/60156375/python-pandas-return-the-consecutive-missing-weekdays-dates-and-assign-rate-ne)  
14. Pandas datetime questions: How to Insert missing weekends into an existing dates column in a dataframe in python \- Stack Overflow, accesso eseguito il giorno novembre 2, 2025, [https://stackoverflow.com/questions/59382355/pandas-datetime-questions-how-to-insert-missing-weekends-into-an-existing-dates](https://stackoverflow.com/questions/59382355/pandas-datetime-questions-how-to-insert-missing-weekends-into-an-existing-dates)  
15. Missing values \- Arima model \- Stack Overflow, accesso eseguito il giorno novembre 2, 2025, [https://stackoverflow.com/questions/46496630/missing-values-arima-model](https://stackoverflow.com/questions/46496630/missing-values-arima-model)  
16. Setting the Frequency to Business Days | Automated hands-on \- CloudxLab, accesso eseguito il giorno novembre 2, 2025, [https://cloudxlab.com/assessment/displayslide/5472/setting-the-frequency-to-business-days](https://cloudxlab.com/assessment/displayslide/5472/setting-the-frequency-to-business-days)  
17. pandas.bdate\_range — pandas 2.3.3 documentation \- PyData |, accesso eseguito il giorno novembre 2, 2025, [https://pandas.pydata.org/docs/reference/api/pandas.bdate\_range.html](https://pandas.pydata.org/docs/reference/api/pandas.bdate_range.html)  
18. Time series / date functionality — pandas 2.3.3 documentation \- PyData |, accesso eseguito il giorno novembre 2, 2025, [https://pandas.pydata.org/docs/user\_guide/timeseries.html](https://pandas.pydata.org/docs/user_guide/timeseries.html)  
19. Tests for Stochastic Seasonality Applied to Daily Financial Time Series, accesso eseguito il giorno novembre 2, 2025, [http://www.stat.yale.edu/\~lc436/papers/test\_sto\_season.pdf](http://www.stat.yale.edu/~lc436/papers/test_sto_season.pdf)  
20. Time Series Part 2: Forecasting with SARIMAX models: An Intro \- JADS MKB Datalab, accesso eseguito il giorno novembre 2, 2025, [https://jadsmkbdatalab.nl/forecasting-with-sarimax-models/](https://jadsmkbdatalab.nl/forecasting-with-sarimax-models/)  
21. Time Series Analysis for Log Returns of S\&P500, accesso eseguito il giorno novembre 2, 2025, [https://ionides.github.io/531w18/midterm\_project/project38/Midterm\_proj.html](https://ionides.github.io/531w18/midterm_project/project38/Midterm_proj.html)  
22. 8.1 Stationarity and differencing | Forecasting: Principles and Practice (2nd ed) \- OTexts, accesso eseguito il giorno novembre 2, 2025, [https://otexts.com/fpp2/stationarity.html](https://otexts.com/fpp2/stationarity.html)  
23. Understanding the Limitations of ARIMA Forecasting \- Towards Data Science, accesso eseguito il giorno novembre 2, 2025, [https://towardsdatascience.com/understanding-the-limitations-of-arima-forecasting-899f8d8e5cf3/](https://towardsdatascience.com/understanding-the-limitations-of-arima-forecasting-899f8d8e5cf3/)  
24. What are the limitations of the time series models (ARIMA, etc) for Forecasting? \- Reddit, accesso eseguito il giorno novembre 2, 2025, [https://www.reddit.com/r/quant/comments/1826m0x/what\_are\_the\_limitations\_of\_the\_time\_series/](https://www.reddit.com/r/quant/comments/1826m0x/what_are_the_limitations_of_the_time_series/)  
25. ARIMA for Time Series Forecasting: A Complete Guide \- DataCamp, accesso eseguito il giorno novembre 2, 2025, [https://www.datacamp.com/tutorial/arima](https://www.datacamp.com/tutorial/arima)  
26. How to Calculate Moving Average in pandas? | by Amit Yadav \- Medium, accesso eseguito il giorno novembre 2, 2025, [https://medium.com/@amit25173/how-to-calculate-moving-average-in-pandas-62b9ececfc5c](https://medium.com/@amit25173/how-to-calculate-moving-average-in-pandas-62b9ececfc5c)  
27. How to calculate MOVING AVERAGE in a Pandas DataFrame? \- GeeksforGeeks, accesso eseguito il giorno novembre 2, 2025, [https://www.geeksforgeeks.org/pandas/how-to-calculate-moving-average-in-a-pandas-dataframe/](https://www.geeksforgeeks.org/pandas/how-to-calculate-moving-average-in-a-pandas-dataframe/)  
28. From SMA to TEMA: Coding Technical Indicators in Python — Building stocksimpy 2, accesso eseguito il giorno novembre 2, 2025, [https://python.plainenglish.io/from-sma-to-tema-coding-technical-indicators-in-python-building-stocksimpy-2-16e684091789](https://python.plainenglish.io/from-sma-to-tema-coding-technical-indicators-in-python-building-stocksimpy-2-16e684091789)  
29. Calculate Exponential Moving Average using Pandas DataFrame \- Stack Overflow, accesso eseguito il giorno novembre 2, 2025, [https://stackoverflow.com/questions/76317946/calculate-exponential-moving-average-using-pandas-dataframe](https://stackoverflow.com/questions/76317946/calculate-exponential-moving-average-using-pandas-dataframe)  
30. How to calculate volatility with Pandas? \- python \- Stack Overflow, accesso eseguito il giorno novembre 2, 2025, [https://stackoverflow.com/questions/52941128/how-to-calculate-volatility-with-pandas](https://stackoverflow.com/questions/52941128/how-to-calculate-volatility-with-pandas)  
31. How to compute volatility (standard deviation) in rolling window in Pandas \- Stack Overflow, accesso eseguito il giorno novembre 2, 2025, [https://stackoverflow.com/questions/43284304/how-to-compute-volatility-standard-deviation-in-rolling-window-in-pandas](https://stackoverflow.com/questions/43284304/how-to-compute-volatility-standard-deviation-in-rolling-window-in-pandas)  
32. Volatility And Measures Of Risk-Adjusted Return With Python \- QuantInsti Blog, accesso eseguito il giorno novembre 2, 2025, [https://blog.quantinsti.com/volatility-and-measures-of-risk-adjusted-return-based-on-volatility/](https://blog.quantinsti.com/volatility-and-measures-of-risk-adjusted-return-based-on-volatility/)  
33. pandas.core.window.rolling.Rolling.std — pandas 2.3.3 documentation \- PyData |, accesso eseguito il giorno novembre 2, 2025, [https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.std.html](https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.std.html)  
34. Momentum Indicators \- Technical Analysis Library in Python's documentation\!, accesso eseguito il giorno novembre 2, 2025, [https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html](https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html)  
35. Technical Indicators in Python for Trading \- PyQuant News, accesso eseguito il giorno novembre 2, 2025, [https://www.pyquantnews.com/free-python-resources/technical-indicators-in-python-for-trading](https://www.pyquantnews.com/free-python-resources/technical-indicators-in-python-for-trading)  
36. Building a MACD Indicator in Python \- Medium, accesso eseguito il giorno novembre 2, 2025, [https://medium.com/@financial\_python/building-a-macd-indicator-in-python-190b2a4c1777](https://medium.com/@financial_python/building-a-macd-indicator-in-python-190b2a4c1777)  
37. (3) Unlocking Trading Insights: Combining MACD and RSI Trading Strategies in Python | by Sercan Bugra Gultekin | Medium, accesso eseguito il giorno novembre 2, 2025, [https://medium.com/@bugragultekin/unlocking-trading-insights-macd-trading-strategy-with-python-50f9152016da](https://medium.com/@bugragultekin/unlocking-trading-insights-macd-trading-strategy-with-python-50f9152016da)  
38. Including exogenous variables in SARIMAX. Probably an easy solution. \#4284 \- GitHub, accesso eseguito il giorno novembre 2, 2025, [https://github.com/statsmodels/statsmodels/issues/4284](https://github.com/statsmodels/statsmodels/issues/4284)  
39. python \- ARIMAX With Multiple Lags \- Stack Overflow, accesso eseguito il giorno novembre 2, 2025, [https://stackoverflow.com/questions/70405718/arimax-with-multiple-lags](https://stackoverflow.com/questions/70405718/arimax-with-multiple-lags)  
40. Box-Jenkins Methodology | Columbia University Mailman School of Public Health, accesso eseguito il giorno novembre 2, 2025, [https://www.publichealth.columbia.edu/research/population-health-methods/box-jenkins-methodology](https://www.publichealth.columbia.edu/research/population-health-methods/box-jenkins-methodology)  
41. Box–Jenkins method \- Wikipedia, accesso eseguito il giorno novembre 2, 2025, [https://en.wikipedia.org/wiki/Box%E2%80%93Jenkins\_method](https://en.wikipedia.org/wiki/Box%E2%80%93Jenkins_method)  
42. log return of sp500. Stationary vs strictly stationary \- Quantitative Finance Stack Exchange, accesso eseguito il giorno novembre 2, 2025, [https://quant.stackexchange.com/questions/37962/log-return-of-sp500-stationary-vs-strictly-stationary](https://quant.stackexchange.com/questions/37962/log-return-of-sp500-stationary-vs-strictly-stationary)  
43. Augmented Dickey-Fuller (ADF) \- GeeksforGeeks, accesso eseguito il giorno novembre 2, 2025, [https://www.geeksforgeeks.org/machine-learning/augmented-dickey-fuller-adf/](https://www.geeksforgeeks.org/machine-learning/augmented-dickey-fuller-adf/)  
44. Stationarity and detrending (ADF/KPSS) \- statsmodels 0.14.4, accesso eseguito il giorno novembre 2, 2025, [https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity\_detrending\_adf\_kpss.html](https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html)  
45. statsmodels.tsa.stattools.adfuller, accesso eseguito il giorno novembre 2, 2025, [https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html)  
46. statsmodels.tsa.stattools.adfuller, accesso eseguito il giorno novembre 2, 2025, [https://www.statsmodels.org/0.6.1/generated/statsmodels.tsa.stattools.adfuller.html](https://www.statsmodels.org/0.6.1/generated/statsmodels.tsa.stattools.adfuller.html)  
47. Augmented Dickey Fuller Test (ADF Test) – Must Read Guide \- Machine Learning Plus, accesso eseguito il giorno novembre 2, 2025, [https://www.machinelearningplus.com/time-series/augmented-dickey-fuller-test/](https://www.machinelearningplus.com/time-series/augmented-dickey-fuller-test/)  
48. Augmented Dickey-Fuller Test in Python (With Example) \- Statology, accesso eseguito il giorno novembre 2, 2025, [https://www.statology.org/dickey-fuller-test-python/](https://www.statology.org/dickey-fuller-test-python/)  
49. The Augmented Dickey—Fuller (ADF) Test for Stationarity \- Victor Leung \- WordPress.com, accesso eseguito il giorno novembre 2, 2025, [https://victorleungtw.wordpress.com/2024/07/04/the-augmented-dickey-fuller-adf-test-for-stationarity/](https://victorleungtw.wordpress.com/2024/07/04/the-augmented-dickey-fuller-adf-test-for-stationarity/)  
50. Understanding Stationary Time Series Analysis \- Analytics Vidhya, accesso eseguito il giorno novembre 2, 2025, [https://www.analyticsvidhya.com/blog/2021/06/statistical-tests-to-check-stationarity-in-time-series-part-1/](https://www.analyticsvidhya.com/blog/2021/06/statistical-tests-to-check-stationarity-in-time-series-part-1/)  
51. Interpret all statistics and graphs for Augmented Dickey-Fuller Test \- Support \- Minitab, accesso eseguito il giorno novembre 2, 2025, [https://support.minitab.com/en-us/minitab/help-and-how-to/statistical-modeling/time-series/how-to/augmented-dickey-fuller-test/interpret-the-results/all-statistics-and-graphs/](https://support.minitab.com/en-us/minitab/help-and-how-to/statistical-modeling/time-series/how-to/augmented-dickey-fuller-test/interpret-the-results/all-statistics-and-graphs/)  
52. ARIMA and SARIMAX forecasting \- Skforecast Docs, accesso eseguito il giorno novembre 2, 2025, [https://skforecast.org/0.12.1/user\_guides/forecasting-sarimax-arima](https://skforecast.org/0.12.1/user_guides/forecasting-sarimax-arima)  
53. How to Difference a Time Series Dataset with Python \- MachineLearningMastery.com, accesso eseguito il giorno novembre 2, 2025, [https://machinelearningmastery.com/difference-time-series-dataset-python/](https://machinelearningmastery.com/difference-time-series-dataset-python/)  
54. Select ARIMA Model for Time Series Using Box-Jenkins Methodology \- MATLAB & Simulink, accesso eseguito il giorno novembre 2, 2025, [https://www.mathworks.com/help/econ/box-jenkins-model-selection.html](https://www.mathworks.com/help/econ/box-jenkins-model-selection.html)  
55. Python Statsmodel ARIMA start \[stationarity\] \- Stack Overflow, accesso eseguito il giorno novembre 2, 2025, [https://stackoverflow.com/questions/24316935/python-statsmodel-arima-start-stationarity](https://stackoverflow.com/questions/24316935/python-statsmodel-arima-start-stationarity)  
56. statsmodels.tsa.arima.model., accesso eseguito il giorno novembre 2, 2025, [https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)  
57. ARIMA-GARCH Model(Part 1\) \- TEJ, accesso eseguito il giorno novembre 2, 2025, [https://www.tejwin.com/en/insight/arima-garch-modelpart-1/](https://www.tejwin.com/en/insight/arima-garch-modelpart-1/)  
58. ARIMA-GARCH forecasting with Python | by Thomas Dierckx | Analytics Vidhya \- Medium, accesso eseguito il giorno novembre 2, 2025, [https://medium.com/analytics-vidhya/arima-garch-forecasting-with-python-7a3f797de3ff](https://medium.com/analytics-vidhya/arima-garch-forecasting-with-python-7a3f797de3ff)  
59. statsmodels.graphics.tsaplots.plot\_acf, accesso eseguito il giorno novembre 2, 2025, [https://www.statsmodels.org/stable/generated/statsmodels.graphics.tsaplots.plot\_acf.html](https://www.statsmodels.org/stable/generated/statsmodels.graphics.tsaplots.plot_acf.html)  
60. Understanding Autocorrelation and Partial Autocorrelation Functions (ACF and PACF) | by András Kis | Medium, accesso eseguito il giorno novembre 2, 2025, [https://medium.com/@kis.andras.nandor/understanding-autocorrelation-and-partial-autocorrelation-functions-acf-and-pacf-2998e7e1bcb5](https://medium.com/@kis.andras.nandor/understanding-autocorrelation-and-partial-autocorrelation-functions-acf-and-pacf-2998e7e1bcb5)  
61. Time Series: Interpreting ACF and PACF \- Kaggle, accesso eseguito il giorno novembre 2, 2025, [https://www.kaggle.com/code/iamleonie/time-series-interpreting-acf-and-pacf](https://www.kaggle.com/code/iamleonie/time-series-interpreting-acf-and-pacf)  
62. Interpreting ACF and PACF Plots for Time Series Forecasting \- Towards Data Science, accesso eseguito il giorno novembre 2, 2025, [https://towardsdatascience.com/interpreting-acf-and-pacf-plots-for-time-series-forecasting-af0d6db4061c/](https://towardsdatascience.com/interpreting-acf-and-pacf-plots-for-time-series-forecasting-af0d6db4061c/)  
63. Chapter 9: Box-Jenkins Methodology, accesso eseguito il giorno novembre 2, 2025, [http://business.baylor.edu/Tom\_kelly/Bails9.htm](http://business.baylor.edu/Tom_kelly/Bails9.htm)  
64. Choosing the best q and p from ACF and PACF plots in ARMA-type modeling \- Baeldung, accesso eseguito il giorno novembre 2, 2025, [https://www.baeldung.com/cs/acf-pacf-plots-arma-modeling](https://www.baeldung.com/cs/acf-pacf-plots-arma-modeling)  
65. Deciding p and q for an ARMA model based on ACF plot \- Cross Validated, accesso eseguito il giorno novembre 2, 2025, [https://stats.stackexchange.com/questions/492623/deciding-p-and-q-for-an-arma-model-based-on-acf-plot](https://stats.stackexchange.com/questions/492623/deciding-p-and-q-for-an-arma-model-based-on-acf-plot)  
66. Examples — pmdarima 2.0.4 documentation \- alkaline-ml, accesso eseguito il giorno novembre 2, 2025, [https://alkaline-ml.com/pmdarima/auto\_examples/index.html](https://alkaline-ml.com/pmdarima/auto_examples/index.html)  
67. Efficient Time-Series Using Python's Pmdarima Library \- Towards Data Science, accesso eseguito il giorno novembre 2, 2025, [https://towardsdatascience.com/efficient-time-series-using-pythons-pmdarima-library-f6825407b7f0/](https://towardsdatascience.com/efficient-time-series-using-pythons-pmdarima-library-f6825407b7f0/)  
68. Simple auto\_arima model — pmdarima 2.0.4 documentation \- alkaline-ml, accesso eseguito il giorno novembre 2, 2025, [https://alkaline-ml.com/pmdarima/auto\_examples/example\_simple\_fit.html](https://alkaline-ml.com/pmdarima/auto_examples/example_simple_fit.html)  
69. Fitting an auto\_arima model — pmdarima 2.0.4 documentation \- alkaline-ml, accesso eseguito il giorno novembre 2, 2025, [https://alkaline-ml.com/pmdarima/auto\_examples/arima/example\_auto\_arima.html](https://alkaline-ml.com/pmdarima/auto_examples/arima/example_auto_arima.html)  
70. Quick Intro: auto\_arima from pmdarima package | by Steven Kyle \- Medium, accesso eseguito il giorno novembre 2, 2025, [https://stevenkyle2013.medium.com/quick-intro-auto-arima-from-pmdarima-package-e7aab5e8dfb8](https://stevenkyle2013.medium.com/quick-intro-auto-arima-from-pmdarima-package-e7aab5e8dfb8)  
71. pmdarima.arima.auto\_arima \- alkaline-ml, accesso eseguito il giorno novembre 2, 2025, [https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto\_arima.html](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html)  
72. A Guide to Parameter Tuning in auto\_arima() Function for Time Series Forecasting \- Medium, accesso eseguito il giorno novembre 2, 2025, [https://medium.com/@aysuudemiir/a-guide-to-parameter-tuning-in-auto-arima-function-for-time-series-forecasting-aec50fb1523a](https://medium.com/@aysuudemiir/a-guide-to-parameter-tuning-in-auto-arima-function-for-time-series-forecasting-aec50fb1523a)  
73. Tuning ARIMA for Forecasting: An Easy Approach in Python | by Dr. Sandeep Singh Sandha, PhD | Medium, accesso eseguito il giorno novembre 2, 2025, [https://medium.com/@sandha.iitr/tuning-arima-for-forecasting-an-easy-approach-in-python-5f40d55184c4](https://medium.com/@sandha.iitr/tuning-arima-for-forecasting-an-easy-approach-in-python-5f40d55184c4)  
74. statsmodels.tsa.stattools.arma\_order\_select\_ic, accesso eseguito il giorno novembre 2, 2025, [https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.arma\_order\_select\_ic.html](https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.arma_order_select_ic.html)  
75. Introduction to time series analysis and forecasting, accesso eseguito il giorno novembre 2, 2025, [http://w3.cran.univ-lorraine.fr/perso/hugues.garnier/Enseignement/TSAF/C-TSAF-Box-Jenkins\_method.pdf](http://w3.cran.univ-lorraine.fr/perso/hugues.garnier/Enseignement/TSAF/C-TSAF-Box-Jenkins_method.pdf)  
76. GARCH vs ARIMA Explained – Which Time Series Model Should You Use? \- ResearchGate, accesso eseguito il giorno novembre 2, 2025, [https://www.researchgate.net/post/GARCH\_vs\_ARIMA\_Explained-Which\_Time\_Series\_Model\_Should\_You\_Use](https://www.researchgate.net/post/GARCH_vs_ARIMA_Explained-Which_Time_Series_Model_Should_You_Use)  
77. Application of Time Series Models (ARIMA, GARCH, and ARMA-GARCH) for Stock Market Forecasting \- Huskie Commons, accesso eseguito il giorno novembre 2, 2025, [https://huskiecommons.lib.niu.edu/cgi/viewcontent.cgi?article=1176\&context=studentengagement-honorscapstones](https://huskiecommons.lib.niu.edu/cgi/viewcontent.cgi?article=1176&context=studentengagement-honorscapstones)  
78. Time Series analysis tsa \- statsmodels 0.14.4, accesso eseguito il giorno novembre 2, 2025, [https://www.statsmodels.org/stable/tsa.html](https://www.statsmodels.org/stable/tsa.html)  
79. Python SARIMAX \- Statsmodels \- Codecademy, accesso eseguito il giorno novembre 2, 2025, [https://www.codecademy.com/resources/docs/python/statsmodels/sarimax](https://www.codecademy.com/resources/docs/python/statsmodels/sarimax)  
80. Three techniques to improve SARIMAX model for time series forecasting | by Birat Poudel, accesso eseguito il giorno novembre 2, 2025, [https://medium.com/@poudel.birat25/three-techniques-to-improve-sarimax-model-for-time-series-forecasting-5d48db984fbe](https://medium.com/@poudel.birat25/three-techniques-to-improve-sarimax-model-for-time-series-forecasting-5d48db984fbe)  
81. Time Series Forecasting: SARIMAX & Prophet \- Medium, accesso eseguito il giorno novembre 2, 2025, [https://medium.com/@ryassminh/test-c2ecbf558780](https://medium.com/@ryassminh/test-c2ecbf558780)  
82. ARIMA and SARIMAX models with Python \- Cienciadedatos.net, accesso eseguito il giorno novembre 2, 2025, [https://cienciadedatos.net/documentos/py51-arima-sarimax-models-python](https://cienciadedatos.net/documentos/py51-arima-sarimax-models-python)  
83. SARIMAX Model in Python \- Medium, accesso eseguito il giorno novembre 2, 2025, [https://medium.com/@rajukumardalimss/sarimax-model-in-python-393227d4409a](https://medium.com/@rajukumardalimss/sarimax-model-in-python-393227d4409a)  
84. SARIMAX \- Forecasting Economic Time Series \- Kaggle, accesso eseguito il giorno novembre 2, 2025, [https://www.kaggle.com/code/elasgustavoknaus/sarimax-forecasting-economic-time-series](https://www.kaggle.com/code/elasgustavoknaus/sarimax-forecasting-economic-time-series)  
85. Understanding the Seasonal Order of the SARIMA Model | by Angelica Lo Duca \- Medium, accesso eseguito il giorno novembre 2, 2025, [https://medium.com/data-science/understanding-the-seasonal-order-of-the-sarima-model-ebef613e40fa](https://medium.com/data-science/understanding-the-seasonal-order-of-the-sarima-model-ebef613e40fa)  
86. Stochastic Models for Radon Daily Time Series: Seasonality, Stationarity, and Long-Range Dependence Detection \- Frontiers, accesso eseguito il giorno novembre 2, 2025, [https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2020.575001/full](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2020.575001/full)  
87. How to Decompose Time Series Data into Trend and Seasonality \- MachineLearningMastery.com, accesso eseguito il giorno novembre 2, 2025, [https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/](https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/)  
88. Detecting Seasonality Through Autocorrelation \- Hex, accesso eseguito il giorno novembre 2, 2025, [https://hex.tech/blog/detecting-seasonality-through-autocorrelation/](https://hex.tech/blog/detecting-seasonality-through-autocorrelation/)  
89. Seasonality and SARIMAX \- Kaggle, accesso eseguito il giorno novembre 2, 2025, [https://www.kaggle.com/code/nholloway/seasonality-and-sarimax](https://www.kaggle.com/code/nholloway/seasonality-and-sarimax)  
90. ARIMA, SARIMA, and SARIMAX Explained | Zero To Mastery, accesso eseguito il giorno novembre 2, 2025, [https://zerotomastery.io/blog/arima-sarima-sarimax-explained/](https://zerotomastery.io/blog/arima-sarima-sarimax-explained/)  
91. How do I input multiple exogenous variables into a SARIMAX model in statsmodel?, accesso eseguito il giorno novembre 2, 2025, [https://stackoverflow.com/questions/44212127/how-do-i-input-multiple-exogenous-variables-into-a-sarimax-model-in-statsmodel](https://stackoverflow.com/questions/44212127/how-do-i-input-multiple-exogenous-variables-into-a-sarimax-model-in-statsmodel)  
92. SARIMAX.predict() and SARIMAX.forecast() exog? Does exog need to be preknown for predict()? \- Cross Validated \- Stats StackExchange, accesso eseguito il giorno novembre 2, 2025, [https://stats.stackexchange.com/questions/631000/sarimax-predict-and-sarimax-forecast-exog-does-exog-need-to-be-preknown-for](https://stats.stackexchange.com/questions/631000/sarimax-predict-and-sarimax-forecast-exog-does-exog-need-to-be-preknown-for)  
93. \[Q\] forecasting future values using ARIMAX? : r/statistics \- Reddit, accesso eseguito il giorno novembre 2, 2025, [https://www.reddit.com/r/statistics/comments/1in00eo/q\_forecasting\_future\_values\_using\_arimax/](https://www.reddit.com/r/statistics/comments/1in00eo/q_forecasting_future_values_using_arimax/)  
94. statsmodels ARIMA forecast without future values of exogenous variable \- Stack Overflow, accesso eseguito il giorno novembre 2, 2025, [https://stackoverflow.com/questions/61765604/statsmodels-arima-forecast-without-future-values-of-exogenous-variable](https://stackoverflow.com/questions/61765604/statsmodels-arima-forecast-without-future-values-of-exogenous-variable)  
95. statsmodels.tsa.statespace.sarimax., accesso eseguito il giorno novembre 2, 2025, [https://www.statsmodels.org/devel/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html](https://www.statsmodels.org/devel/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)  
96. SARIMAX and ARIMA: Frequently Asked Questions (FAQ) \- statsmodels 0.15.0 (+834), accesso eseguito il giorno novembre 2, 2025, [https://www.statsmodels.org/devel/examples/notebooks/generated/statespace\_sarimax\_faq.html](https://www.statsmodels.org/devel/examples/notebooks/generated/statespace_sarimax_faq.html)  
97. SARIMAX out of sample forecast with exogenous data \- Stack Overflow, accesso eseguito il giorno novembre 2, 2025, [https://stackoverflow.com/questions/72072931/sarimax-out-of-sample-forecast-with-exogenous-data](https://stackoverflow.com/questions/72072931/sarimax-out-of-sample-forecast-with-exogenous-data)  
98. Forecasting out-of-sample with exogenous variables using SARIMAX in Statsmodels \-python \- Stack Overflow, accesso eseguito il giorno novembre 2, 2025, [https://stackoverflow.com/questions/72681451/forecasting-out-of-sample-with-exogenous-variables-using-sarimax-in-statsmodels](https://stackoverflow.com/questions/72681451/forecasting-out-of-sample-with-exogenous-variables-using-sarimax-in-statsmodels)  
99. How to predict unseen data with auto arima using exogenous variables \- Stack Overflow, accesso eseguito il giorno novembre 2, 2025, [https://stackoverflow.com/questions/74496230/how-to-predict-unseen-data-with-auto-arima-using-exogenous-variables](https://stackoverflow.com/questions/74496230/how-to-predict-unseen-data-with-auto-arima-using-exogenous-variables)  
100. TimeSeriesSplit — scikit-learn 1.7.2 documentation, accesso eseguito il giorno novembre 2, 2025, [https://scikit-learn.org/stable/modules/generated/sklearn.model\_selection.TimeSeriesSplit.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)  
101. Walk forward validation : r/datascience \- Reddit, accesso eseguito il giorno novembre 2, 2025, [https://www.reddit.com/r/datascience/comments/18pxc6x/walk\_forward\_validation/](https://www.reddit.com/r/datascience/comments/18pxc6x/walk_forward_validation/)  
102. train\_test\_split — scikit-learn 1.7.2 documentation, accesso eseguito il giorno novembre 2, 2025, [https://scikit-learn.org/stable/modules/generated/sklearn.model\_selection.train\_test\_split.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)  
103. Trying to split data into train, test and validation sets (in chronological order) \- Stack Overflow, accesso eseguito il giorno novembre 2, 2025, [https://stackoverflow.com/questions/47040030/trying-to-split-data-into-train-test-and-validation-sets-in-chronological-orde](https://stackoverflow.com/questions/47040030/trying-to-split-data-into-train-test-and-validation-sets-in-chronological-orde)  
104. Time Series Splitting Techniques: Ensuring Accurate Model Validation | by Mouad En-nasiry, accesso eseguito il giorno novembre 2, 2025, [https://medium.com/@mouadenna/time-series-splitting-techniques-ensuring-accurate-model-validation-5a3146db3088](https://medium.com/@mouadenna/time-series-splitting-techniques-ensuring-accurate-model-validation-5a3146db3088)  
105. XGBoost Evaluate Model for Time Series using Walk-Forward Validation, accesso eseguito il giorno novembre 2, 2025, [https://xgboosting.com/xgboost-evaluate-model-for-time-series-using-walk-forward-validation/](https://xgboosting.com/xgboost-evaluate-model-for-time-series-using-walk-forward-validation/)  
106. Time Series Using Walk-Forward Validation \- Kaggle, accesso eseguito il giorno novembre 2, 2025, [https://www.kaggle.com/code/justozner/time-series-using-walk-forward-validation](https://www.kaggle.com/code/justozner/time-series-using-walk-forward-validation)  
107. Understanding Walk Forward Validation in Time Series Analysis: A Practical Guide, accesso eseguito il giorno novembre 2, 2025, [https://medium.com/@ahmedfahad04/understanding-walk-forward-validation-in-time-series-analysis-a-practical-guide-ea3814015abf](https://medium.com/@ahmedfahad04/understanding-walk-forward-validation-in-time-series-analysis-a-practical-guide-ea3814015abf)  
108. Implement Walk-Forward Optimization with XGBoost for Stock Price Prediction in Python, accesso eseguito il giorno novembre 2, 2025, [https://blog.quantinsti.com/walk-forward-optimization-python-xgboost-stock-prediction/](https://blog.quantinsti.com/walk-forward-optimization-python-xgboost-stock-prediction/)  
109. Evaluating forecast accuracy (MAE, RMSE, MAPE) | Intro to Time Series Class Notes, accesso eseguito il giorno novembre 2, 2025, [https://fiveable.me/intro-time-series/unit-8/evaluating-forecast-accuracy-mae-rmse-mape/study-guide/ijqkb0CAqRaHLBFi](https://fiveable.me/intro-time-series/unit-8/evaluating-forecast-accuracy-mae-rmse-mape/study-guide/ijqkb0CAqRaHLBFi)  
110. Metrics Evaluation: MSE, RMSE, MAE and MAPE | by Jonatasv \- Medium, accesso eseguito il giorno novembre 2, 2025, [https://medium.com/@jonatasv/metrics-evaluation-mse-rmse-mae-and-mape-317cab85a26b](https://medium.com/@jonatasv/metrics-evaluation-mse-rmse-mae-and-mape-317cab85a26b)  
111. MAE, MAPE, MASE and the Scaled RMSE \- Paul Morgan, accesso eseguito il giorno novembre 2, 2025, [https://www.pmorgan.com.au/tutorials/mae%2C-mape%2C-mase-and-the-scaled-rmse/](https://www.pmorgan.com.au/tutorials/mae%2C-mape%2C-mase-and-the-scaled-rmse/)  
112. High RMSE and MAE and low MAPE \- Data Science Stack Exchange, accesso eseguito il giorno novembre 2, 2025, [https://datascience.stackexchange.com/questions/37168/high-rmse-and-mae-and-low-mape](https://datascience.stackexchange.com/questions/37168/high-rmse-and-mae-and-low-mape)  
113. Navigating the Financial Landscape: The Power and Limitations of the ARIMA Model, accesso eseguito il giorno novembre 2, 2025, [https://drpress.org/ojs/index.php/HSET/article/view/19082](https://drpress.org/ojs/index.php/HSET/article/view/19082)  
114. How to Model Volatility with ARCH and GARCH for Time Series Forecasting in Python, accesso eseguito il giorno novembre 2, 2025, [https://machinelearningmastery.com/develop-arch-and-garch-models-for-time-series-forecasting-in-python/](https://machinelearningmastery.com/develop-arch-and-garch-models-for-time-series-forecasting-in-python/)  
115. Air State Analysis: Comparison of ARIMA and GARCH \- Kaggle, accesso eseguito il giorno novembre 2, 2025, [https://www.kaggle.com/code/vbmokin/air-state-analysis-comparison-of-arima-and-garch](https://www.kaggle.com/code/vbmokin/air-state-analysis-comparison-of-arima-and-garch)  
116. Advanced Stock Market Forecasting: A Comparative Analysis of ARIMA-GARCH, LSTM, and Integrated Wavelet-LSTM Models \- SHS Web of Conferences, accesso eseguito il giorno novembre 2, 2025, [https://www.shs-conferences.org/articles/shsconf/pdf/2024/16/shsconf\_edma2024\_02008.pdf](https://www.shs-conferences.org/articles/shsconf/pdf/2024/16/shsconf_edma2024_02008.pdf)  
117. 02 Stationary time series, accesso eseguito il giorno novembre 2, 2025, [https://web.vu.lt/mif/a.buteikis/wp-content/uploads/2018/02/Lecture\_02.pdf](https://web.vu.lt/mif/a.buteikis/wp-content/uploads/2018/02/Lecture_02.pdf)  
118. Financial Volatility Modelling, accesso eseguito il giorno novembre 2, 2025, [http://web.vu.lt/mif/a.buteikis/wp-content/uploads/2019/03/02\_GARCH.html](http://web.vu.lt/mif/a.buteikis/wp-content/uploads/2019/03/02_GARCH.html)  
119. Forecasting Volatility: Deep Dive into ARCH & GARCH Models | by Daniel Herrera | Medium, accesso eseguito il giorno novembre 2, 2025, [https://medium.com/@corredaniel1500/forecasting-volatility-deep-dive-into-arch-garch-models-46cd1945872b](https://medium.com/@corredaniel1500/forecasting-volatility-deep-dive-into-arch-garch-models-46cd1945872b)  
120. ARIMA, SARIMA, ARCH, GARCH, TARCH: A Brief Guide to Practical Application \- Medium, accesso eseguito il giorno novembre 2, 2025, [https://medium.com/@jlabs/arima-sarima-arch-garch-tarch-a-brief-guide-to-practical-application-ef054eb59078](https://medium.com/@jlabs/arima-sarima-arch-garch-tarch-a-brief-guide-to-practical-application-ef054eb59078)  
121. Can someone explain the main differences between ARIMA, ARCH and GARCH? \- Reddit, accesso eseguito il giorno novembre 2, 2025, [https://www.reddit.com/r/econometrics/comments/4dvtxi/can\_someone\_explain\_the\_main\_differences\_between/](https://www.reddit.com/r/econometrics/comments/4dvtxi/can_someone_explain_the_main_differences_between/)  
122. How to Model and Forecast Volatility with ARCH & GARCH Techniques | by Yilin Wu, accesso eseguito il giorno novembre 2, 2025, [https://python.plainenglish.io/how-to-model-and-forecast-volatility-with-arch-garch-techniques-980e57956ae5](https://python.plainenglish.io/how-to-model-and-forecast-volatility-with-arch-garch-techniques-980e57956ae5)  
123. ARIMA vs. LSTM \- GeeksforGeeks, accesso eseguito il giorno novembre 2, 2025, [https://www.geeksforgeeks.org/machine-learning/arima-vs-lstm/](https://www.geeksforgeeks.org/machine-learning/arima-vs-lstm/)  
124. \[D\] Complexity of Time Series Models: ARIMA vs. LSTM : r/statistics \- Reddit, accesso eseguito il giorno novembre 2, 2025, [https://www.reddit.com/r/statistics/comments/mv6he0/d\_complexity\_of\_time\_series\_models\_arima\_vs\_lstm/](https://www.reddit.com/r/statistics/comments/mv6he0/d_complexity_of_time_series_models_arima_vs_lstm/)  
125. ARIMA vs LSTM: A Comparative Study of Time Series Prediction Models, accesso eseguito il giorno novembre 2, 2025, [https://vivekupadhyay1.medium.com/arima-vs-lstm-a-comparative-study-of-time-series-prediction-models-91fa4219d9d9](https://vivekupadhyay1.medium.com/arima-vs-lstm-a-comparative-study-of-time-series-prediction-models-91fa4219d9d9)  
126. ARIMA vs Prophet vs LSTM for Time Series Prediction \- Neptune.ai, accesso eseguito il giorno novembre 2, 2025, [https://neptune.ai/blog/arima-vs-prophet-vs-lstm](https://neptune.ai/blog/arima-vs-prophet-vs-lstm)  
127. A Review of ARIMA vs. Machine Learning Approaches for Time Series Forecasting in Data Driven Networks \- MDPI, accesso eseguito il giorno novembre 2, 2025, [https://www.mdpi.com/1999-5903/15/8/255](https://www.mdpi.com/1999-5903/15/8/255)