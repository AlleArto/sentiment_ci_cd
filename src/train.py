#!/usr/bin/env python
"""
Addestramento + (opz.) push to HuggingFace Hub.

Esegui:
$ python src/train.py --push
"""
import argparse
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from config import (
    BASE_MODEL,
    HF_REPO_ID,
    EPOCHS,
    BATCH,
    LR,
    CHECKPOINT_DIR,
)
from data import load_tokenized_dataset
from metrics import compute_metrics

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--push", action="store_true", help="Pubblica su HF Hub al termine")
    return p.parse_args()

def main():
    args = parse_args()
    ds_tok, tokenizer = load_tokenized_dataset()

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=3
    )

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        learning_rate=LR,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=args.push,
        hub_model_id=HF_REPO_ID,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    perf = trainer.evaluate(ds_tok["test"])
    print("Test metrics:", perf)

    if args.push:
        trainer.push_to_hub()
        print(f"✅  Modello pubblicato su https://huggingface.co/{HF_REPO_ID}")

if __name__ == "__main__":
    main()
