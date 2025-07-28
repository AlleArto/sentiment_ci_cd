"""
Test di smoke per la pipeline di training del modello di sentiment analysis.
Verifica che il training venga eseguito senza errori e che la directory di checkpoint venga creata.
"""

import tempfile
from pathlib import Path
import src.train, src.data
from argparse import Namespace
from src import config

def test_train_smoke(monkeypatch):
    tmp_ckpt = Path(tempfile.mkdtemp())
    monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_ckpt)
    monkeypatch.setattr(src.data, "load_tokenized_dataset", lambda: src.data.load_tokenized_dataset(train_size=50))
    monkeypatch.setattr(src.train, "parse_args", lambda: Namespace(epochs=1, push=False))
    src.train.main()
    assert tmp_ckpt.exists()
