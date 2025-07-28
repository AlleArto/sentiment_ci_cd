"""
Modulo per il caricamento e la tokenizzazione del dataset per la sentiment analysis.
Fornisce la funzione load_tokenized_dataset per restituire dataset e tokenizer pronti all'uso.
"""

from datasets import load_dataset
from transformers import AutoTokenizer
from src.config import BASE_MODEL, DATASET_NAME

def load_tokenized_dataset(train_size=150, val_size=50, test_size=50):
    ds = load_dataset(DATASET_NAME)
    ds["train"] = ds["train"].shuffle(seed=0).select(range(train_size))
    ds["validation"] = ds["validation"].shuffle(seed=0).select(range(val_size))
    ds["test"] = ds["test"].shuffle(seed=0).select(range(test_size))
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_prefix_space=True)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True)

    ds_tok = ds.map(tok, batched=True)

    return ds_tok, tokenizer
