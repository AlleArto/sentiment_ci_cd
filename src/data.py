from datasets import load_dataset
from transformers import AutoTokenizer
from config import BASE_MODEL, DATASET_NAME

def load_tokenized_dataset():
    ds = load_dataset(DATASET_NAME)
    ds["train"] = ds["train"].shuffle(seed=0).select(range(200))
    ds["validation"] = ds["validation"].shuffle(seed=0).select(range(50))
    ds["test"] = ds["test"].shuffle(seed=0).select(range(50))
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_prefix_space=True)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True)

    ds_tok = ds.map(tok, batched=True)

    return ds_tok, tokenizer
