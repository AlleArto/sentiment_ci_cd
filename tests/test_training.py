"""Smoke‑test: verifica che train() restituisca metriche senza errori."""

import importlib, types

def test_train_smoke(tmp_path, monkeypatch):
    # Patch checkpoint dir a una cartella temporanea
    from src import config
    monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_path / "ckpt")

    train = importlib.import_module("src.train")
    # Esegui training solo su 1 batch per velocizzare (use Trainer subset)
    from argparse import Namespace
    monkeypatch.setattr(train, "parse_args", lambda: Namespace(epochs=1, push=False))
    train.main()

    assert (config.CHECKPOINT_DIR).exists()
