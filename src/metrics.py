"""
Modulo per il calcolo delle metriche di valutazione (accuracy e f1) per la sentiment analysis.
Fornisce la funzione compute_metrics per l'integrazione con Trainer di HuggingFace.
"""
import evaluate
import numpy as np

_accuracy = evaluate.load("accuracy")
_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = _accuracy.compute(predictions=preds, references=labels)["accuracy"]
    f1 = _f1.compute(predictions=preds, references=labels, average="macro")["f1"]
    return {"accuracy": acc, "f1": f1}
