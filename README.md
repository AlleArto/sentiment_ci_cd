# Sentiment Analysis su Tweet con CI/CD
## 1. Introduzione
Questo progetto implementa un sistema completo di analisi del sentiment su tweet, basato su fine-tuning di un modello pre-addestrato (cardiffnlp/twitter-roberta-base-sentiment-latest) su un dataset multilingue.
L’intero ciclo di vita (training, deploy, monitoraggio, reportistica) è completamente automatizzato tramite pipeline CI/CD su GitHub Actions, con logging e monitoraggio delle performance.

## 2. Scelte Progettuali
## 2.1 Modello di partenza
- **Modello base**: cardiffnlp/twitter-roberta-base-sentiment-latest

- **Motivazione**: Specifico per il linguaggio dei social network, già pre-addestrato su tweet, ottimale per il dominio.

- **Vantaggi**: Fine-tuning più efficace, minor rischio di underfitting, maggiore robustezza.

## 2.2 Dataset
- **Nome**: Ocelot02/tweet-sentiment-ita-eng

- **Descrizione**: Tweet in italiano e inglese etichettati con sentiment (positivo, neutro, negativo).

- **Motivazione**: Dataset realistico e rumoroso, permette di valutare robustezza del modello.

- **Preprocessing**: Shuffle e selezione di subset ridotti per test rapidi (es. train: 150, validation: 50, test: 50), training completo su tutto il dataset nella CD.

## 2.3 Tokenizzazione
- **Tokenizer**: AutoTokenizer del modello base, con add_prefix_space=True per gestire correttamente l’inizio delle frasi.

- **Batching**: Tokenizzazione batch per massimizzare l’efficienza.

## 2.4 Scelte di implementazione
- **Librerie**: Transformers, Datasets, PyTorch, Evaluate, Gradio, Pandas, Matplotlib.

- **Configurazione**: Parametri centralizzati in src/config.py, modificabili tramite variabili d’ambiente.

- **Logging**: Ogni predizione di sentiment viene loggata in predictions_log.csv con timestamp, testo, label e confidence score.

- **Monitoraggio**: Script dedicati generano report e grafici sulle distribuzioni delle predizioni, confidence e drift.

## 2.5 Pipeline CI/CD
- **CI**: Linting (flake8) e test automatici (pytest) ad ogni push/PR tramite .github/workflows/ci.yml.

- **CD**: Training completo, push su Hugging Face Hub, inferenza batch e generazione automatica di report tramite .github/workflows/cd.yml.

## 3. Implementazione
## 3.1 Struttura dei file
- src/config.py – Parametri globali (modello, dataset, hyperparametri)

- src/data.py – Caricamento/tokenizzazione dataset

- src/metrics.py – Definizione metriche (accuracy, F1 macro)

- src/train.py – Logica training & salvataggio

- src/infer.py – Pipeline inferenza & logging predizioni

- src/generate_report.py – Script per report e dashboard di monitoraggio

- tests/test_training.py – Smoke test del training

- .github/workflows/ci.yml – Pipeline CI

- .github/workflows/cd.yml – Pipeline CD

## 3.2 Preprocessing e Tokenizzazione
- Caricamento dataset da Hugging Face Hub.

- Shuffle e sottoselezione per test rapidi.

- Tokenizzazione batch efficiente.

## 3.3 Training
- **Modello**: RoBERTa for Sequence Classification (3 classi)

- **Ottimizzatore**: adamw_torch (betas=(0.9, 0.999), epsilon=1e-08)

- **Learning rate**: 2e-5

- **Batch size**: 32

- **Scheduler**: Linear

- **Epochs**: 4 (configurabile)

- **Seed**: 42

- **Checkpoint**: Directory configurabile

## 3.4 Inferenza & Logging
- Pipeline di inferenza con transformers.pipeline.

- Logging automatico di predizione (predictions_log.csv).

## 3.5 Monitoraggio & Reportistica
Script per:

- Grafico distribuzione classi predette (report_distribuzione_classi.png)

- Grafico distribuzione confidence (report_distribuzione_confidence.png)

- Report testuale drift (report.txt, segnala se una classe supera l’80% delle predizioni)

## 4. Risultati
**Metriche**: Accuracy e F1 macro (evaluate)

- **Loss**: 0.9941

- **Accuracy**: 0.60

- **F1 macro**: 0.5994

**Osservazioni**:

- Valori ottenuti su validation/test dopo 4 epoche.

- Le performance possono essere migliorate con dataset più grandi e tuning avanzato.
