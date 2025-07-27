# sentiment_ci_cd

## 1. Introduzione

L'obiettivo del progetto è la realizzazione di un modello di sentiment analysis su tweet

## 2. Scelte Progettuali

### 2.1 Modello di partenza
- **Modello base:** `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Motivazione:** Questo modello è stato scelto per la sua specializzazione nell’analisi del sentiment su dati provenienti da Twitter, garantendo una buona base di partenza per il fine-tuning su dati simili.

### 2.2 Dataset
- **Nome dataset:** `Ocelot02/tweet-sentiment-ita-eng`
- **Motivazione:** Il dataset contiene tweet in italiano e inglese, permettendo di addestrare un modello multilingue e di testare la robustezza su dati reali e rumorosi.

### 2.3 Tokenizzazione
- **Tokenizer:** AutoTokenizer associato al modello base, con `add_prefix_space=True` per gestire correttamente la tokenizzazione di frasi brevi tipiche dei tweet.

### 2.4 Scelte di implementazione
- **Librerie principali:** Transformers, Datasets, PyTorch, Evaluate.
- **Pipeline di inferenza:** Utilizzo di `transformers.pipeline` per semplificare l’inferenza e l’integrazione con una demo Gradio.
- **Gestione configurazione:** Parametri centralizzati in `src/config.py` per facilitare la modifica e la riproducibilità degli esperimenti.

## 3. Implementazione

### 3.1 Preprocessing e Tokenizzazione
- I dati sono stati caricati e suddivisi in train/validation/test.
- Ogni split è stato ridotto a un sottoinsieme (100/50/50) per velocizzare i test e non incorrere in errori su GitHub vista la mancanza di GPU.
- La tokenizzazione è stata applicata in modalità batch.

### 3.2 Addestramento
- **Ottimizzatore:** AdamW (`adamw_torch`) con betas=(0.9, 0.999), epsilon=1e-08.
- **Learning rate:** 2e-5
- **Batch size:** 32 (train ed eval)
- **Scheduler:** Linear
- **Epochs:** 1 (per test rapido)
- **Seed:** 42 per la riproducibilità

### 3.3 Valutazione
- **Metriche:** Accuracy e F1 macro, calcolate tramite la libreria Evaluate.
- **Risultati ottenuti:**
  - Loss: 0.9988
  - Accuracy: 0.54
  - F1: 0.5460

### 3.4 Inferenza e Demo
- Implementata una pipeline di inferenza con Gradio per testare il modello su nuovi tweet in tempo reale.

## 4. Risultati

- Il modello raggiunge una accuracy del 54% e un F1 macro di 0.5460 sul set di validazione.
- Le performance sono limitate dal numero ridotto di epoche e dalla dimensione del dataset usato per il test rapido.
- Il modello è adatto a una demo o a un prototipo, ma può essere migliorato aumentando i dati e le epoche di addestramento.

## 5. Limiti e Possibili Sviluppi

- **Limiti:**  
  - Dataset ridotto per motivi di test.
  - Solo 1 epoca di addestramento.
  - Possibile overfitting/underfitting non valutato a fondo.
- **Sviluppi futuri:**  
  - Addestramento su tutto il dataset.
  - Aumento delle epoche.
  - Analisi degli errori e tuning degli iperparametri.
  - Estensione a nuove lingue o domini.

## 6. Versioni e Requisiti

- **Transformers:** 4.54.0
- **PyTorch:** 2.7.1+cu126
- **Datasets:** 4.0.0
- **Tokenizers:** 0.21.2
