"""
Script per la generazione di report grafici sulla distribuzione delle classi e delle confidence
delle predizioni effettuate dal modello di sentiment analysis.
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from src.infer import LOG_FILE

LOG_FILE = Path(LOG_FILE)
if not LOG_FILE.exists():
    print("predictions_log.csv non trovato: salto generazione report.")
    exit(0)

df = pd.read_csv(LOG_FILE, header=None,
                 names=["timestamp", "text", "label", "confidence"])
df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
df = df.dropna(subset=["confidence"])

OUT = Path("reports")
OUT.mkdir(exist_ok=True)

# distribuzione classi
plt.figure(figsize=(5, 4))
df["label"].value_counts().plot(kind="bar")
plt.title("Distribuzione Classi Predette")
plt.tight_layout()
plt.savefig(OUT / "report_distribuzione_classi.png")

# distribuzione confidence
plt.figure(figsize=(5, 4))
df["confidence"].plot(kind="hist", bins=20)
plt.title("Distribuzione Confidence")
plt.tight_layout()
plt.savefig(OUT / "report_distribuzione_confidence.png")

# drift & testo
counts = df["label"].value_counts(normalize=True)
with open(OUT / "report.txt", "w") as f:
    f.write("Distribuzione classi:\n")
    f.write(str(counts) + "\n\n")
    f.write("Potenziale drift (>80%)\n" if any(counts > 0.8)
            else "Nessun drift evidente.\n")

print(f"Report generato in {OUT.resolve()}")
