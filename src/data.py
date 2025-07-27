from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from config import BASE_MODEL, DATASET_NAME

def load_tokenized_dataset():
    ds = load_dataset(DATASET_NAME)

    split1 = ds.train_test_split(test_size=0.6, seed=42)
    train_ds = split1["train"]
    rest = split1["test"]

    split2 = rest.train_test_split(test_size=0.5, seed=42)
    val_ds = split2["train"]
    test_ds = split2["test"]

    ds_final = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds
    })
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_prefix_space=True)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True)

    ds_tok = ds_final.map(tok, batched=True)

    return ds_tok, tokenizer
