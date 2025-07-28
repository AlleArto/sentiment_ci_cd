"""
Script per la generazione di report grafici sulla distribuzione delle classi e delle confidence
delle predizioni effettuate dal modello di sentiment analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("predictions_log.csv", header=None,
                 names=["timestamp", "text", "label", "confidence"])

plt.figure(figsize=(5,4))
df["label"].value_counts().plot(kind="bar")
plt.title("Distribuzione Classi Predette")
plt.xlabel("Sentiment")
plt.ylabel("Conteggio")
plt.tight_layout()
plt.savefig("report_distribuzione_classi.png")

plt.figure(figsize=(5,4))
df["confidence"].plot(kind="hist", bins=20)
plt.title("Distribuzione Confidence")
plt.xlabel("Confidence")
plt.tight_layout()
plt.savefig("report_distribuzione_confidence.png")

counts = df["label"].value_counts(normalize=True)
with open("report.txt", "w") as f:
    f.write("Distribuzione classi:\n")
    f.write(str(counts) + "\n\n")
    if any(counts > 0.8):
        f.write("⚠️ Potenziale drift rilevato: una classe >80%\n")
    else:
        f.write("Nessun drift evidente.\n")

print("✅ Report generato: report.txt e immagini PNG!")
