"""
Script per la generazione di report di monitoring sulla distribuzione delle classi predette e sull'andamento della confidence nel tempo.
Utilizza i log delle predizioni per visualizzare possibili drift o anomalie.
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("predictions_log.csv", header=None,
                 names=["timestamp", "text", "label", "confidence"])

ax = df["label"].value_counts(normalize=True).plot.bar()
plt.title("Distribuzione delle classi predette")
plt.xlabel("Sentiment")
plt.ylabel("Percentuale")
plt.show()

df["timestamp"] = pd.to_datetime(df["timestamp"])
plt.figure()
df.set_index("timestamp")["confidence"].rolling(window=50).mean().plot()
plt.title("Confidence media (rolling 50 predizioni)")
plt.xlabel("Tempo")
plt.ylabel("Confidence media")
plt.show()

counts = df["label"].value_counts(normalize=True)
if any(counts > 0.8):
    print("Potenziale drift: una classe domina >80% delle predizioni")


df["true_label"] = ...
accuracy = (df["label"] == df["true_label"]).mean()
