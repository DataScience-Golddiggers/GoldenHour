# Risultati Chiave dell'Analisi LSTM Multivariata

**Data dell'analisi**: 03 Novembre 2025
**Modello di riferimento**: `notebooks/07_lstm_multivariate_exogenous.ipynb`

---

## Domanda di Ricerca Principale
L'obiettivo di questa analisi era testare l'ipotesi "safe-haven": **l'indice di rischio geopolitico (GPRD) ha un potere predittivo sul prezzo dell'oro?**

Per rispondere a questa domanda, è stato costruito un modello LSTM multivariato avanzato che utilizza come input:
-   **Endogeno**: Log-returns del prezzo dell'oro.
-   **Esogeni**: GPRD, RSI, SMA a 20 giorni e Volatilità a 20 giorni (tutti correttamente ritardati di un periodo per evitare look-ahead bias).

## Scoperta Fondamentale: il GPRD è Controproducente

Contrariamente alle aspettative, l'analisi ha rivelato che **l'indice di rischio geopolitico (GPRD) non solo non migliora le previsioni, ma le peggiora attivamente.**

L'analisi di *Permutation Feature Importance* ha assegnato al `GPRD_LAGGED` un'**importanza relativa negativa**.

### Cosa Significa un'Importanza Negativa?
Un valore di importanza negativo indica che il modello ha ottenuto performance **migliori** (un errore di previsione inferiore) quando i dati relativi al GPRD sono stati mescolati casualmente.

In termini pratici, il modello ha trovato il segnale del GPRD più **fuorviante che utile**, interpretandolo come rumore di fondo che confonde le previsioni invece di rafforzarle.

## Implicazioni per l'Ipotesi "Safe-Haven"

Questa scoperta mette in forte discussione la validità dell'ipotesi "safe-haven" nel contesto di un modello predittivo moderno. I risultati suggeriscono che:

1.  **Ridondanza dell'Informazione**: L'informazione contenuta nel GPRD potrebbe essere già implicitamente catturata da altre variabili, in particolare dalla **volatilità** (`VOLATILITY_LAGGED`), che è risultata essere la feature più importante. I mercati, reagendo agli eventi geopolitici, generano volatilità, rendendo quest'ultima un indicatore più diretto e utile per il modello.

2.  **GPRD come "Noise"**: Per un modello LSTM in grado di catturare complesse relazioni non lineari, il GPRD agisce come rumore, introducendo correlazioni spurie che danneggiano la capacità di generalizzazione del modello sul test set.

## Quali Feature sono Davvero Importanti?

L'analisi ha identificato un chiaro vincitore:
-   **`VOLATILITY_LAGGED`**: È di gran lunga la feature più predittiva. Questo conferma che i periodi di incertezza passata (misurati come volatilità dei rendimenti) sono il miglior indicatore per le previsioni future.
-   **`RSI_LAGGED`**: L'indicatore di momentum si è dimostrato il secondo più importante, confermando l'utilità degli indicatori tecnici classici.

## Conclusione e Raccomandazioni

Sebbene l'ipotesi "safe-haven" sia intuitivamente affascinante, questo rigoroso test basato su LSTM dimostra che, in un contesto multivariato, **il GPRD non offre un vantaggio predittivo**.

**Raccomandazioni per la ricerca accademica**:
-   Per un'analisi statistica formale, si raccomanda di utilizzare un modello **SARIMAX** con il GPRD come variabile esogena. Questo permetterà di ottenere coefficienti, p-value e intervalli di confidenza per testare statisticamente la significatività del GPRD, fornendo una prova complementare (e probabilmente confermando) i risultati ottenuti con l'LSTM.
-   Per la pura accuratezza predittiva, il modello LSTM multivariato rimane superiore, ma dovrebbe essere addestrato **escludendo il GPRD** per massimizzare le performance.
