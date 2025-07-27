"""Configurazione centralizzata (modificabile via env)."""
import os
from pathlib import Path

# Modello base e destinazione su HF
BASE_MODEL = os.getenv("BASE_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")
HF_REPO_ID = os.getenv("HF_REPO_ID", "AlleArto/twitter-sentiment-roberta-ft")

DATASET_NAME = os.getenv("DATASET_NAME", "BAJIRAO/imdb_sentiment_3000")

# Addestramento
EPOCHS = int(os.getenv("EPOCHS", 3))
BATCH = int(os.getenv("BATCH", 32))
LR = float(os.getenv("LR", 2e-5))

# Percorsi
ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints"
