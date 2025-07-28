import csv
from datetime import datetime

LOG_FILE = "predictions_log.csv"

def log_prediction(text, predicted_label, confidence):
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), text, predicted_label, confidence])

def predict_and_log(text, pipe):
    output = pipe(text)[0]
    label = output['label']
    score = output['score']
    log_prediction(text, label, score)
    return label, score
