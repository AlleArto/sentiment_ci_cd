"""Gestione dataset: download, tokenizzazione, split."""
from datasets import load_dataset
from transformers import AutoTokenizer
from config import BASE_MODEL, DATASET_NAME, DATASET_CONFIG

def load_tokenized_dataset(sample_size: int | None = None):
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG)
    if sample_size:
        ds = ds.shuffle(seed=0).select(range(sample_size))
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_prefix_space=True)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True)

    ds_tok = ds.map(tok, batched=True)
    return ds_tok, tokenizer
