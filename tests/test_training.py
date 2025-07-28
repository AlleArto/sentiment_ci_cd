"""Smoke‑test: verifica che train() restituisca metriche senza errori."""

import importlib

def test_train_smoke(tmp_path, monkeypatch):
    from src import config
    monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_path / "ckpt")

    train = importlib.import_module("src.train")
    from argparse import Namespace
    monkeypatch.setattr(train, "parse_args", lambda: Namespace(epochs=1, push=False))
    train.main()

    assert config.CHECKPOINT_DIR.exists()
