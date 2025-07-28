"""
Inferenza e logging predizioni sentiment.
Eseguibile sia come modulo (`python -m src.infer --batch test`)
sia come funzione importabile.
"""

import csv
import argparse
from datetime import datetime
from pathlib import Path

from transformers import pipeline
from datasets import load_dataset

from src.config import HF_REPO_ID, DATASET_NAME

LOG_FILE = Path("predictions_log.csv")


# ---------- logging ----------
def _write_header_if_needed():
    if not LOG_FILE.exists():
        with LOG_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "text", "label", "confidence"])


def log_prediction(text: str, predicted_label: str, confidence: float) -> None:
    _write_header_if_needed()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [datetime.utcnow().isoformat(), text, predicted_label, confidence]
        )


# ---------- singola predizione ----------
def predict_and_log(text: str, pipe):
    out = pipe(text)[0]
    log_prediction(text, out["label"], out["score"])
    return out["label"], out["score"]


# ---------- inferenza batch ----------
def batch_inference(split: str = "test", sample_size: int | None = None):
    """Esegue inferenza sullo split indicato del dataset e logga tutte le predizioni."""
    ds = load_dataset(DATASET_NAME)[split]
    if sample_size:
        ds = ds.shuffle(seed=0).select(range(sample_size))

    pipe = pipeline("sentiment-analysis", model=HF_REPO_ID)

    for row in ds:
        predict_and_log(row["text"], pipe)

    print(f"Batch inference completata su {len(ds)} esempi. Log salvato in {LOG_FILE}.")


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--batch",
        choices=["train", "validation", "test"],
        help="Esegue inferenza su uno split del dataset",
    )
    p.add_argument("--sample", type=int, help="Sub‑campione per inferenza rapida")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.batch:
        batch_inference(split=args.batch, sample_size=args.sample)
    else:
        txt = input("Testo da analizzare: ")
        pipe = pipeline("sentiment-analysis", model=HF_REPO_ID)
        label, score = predict_and_log(txt, pipe)
        print(f"{label} ({score:.3f})")
