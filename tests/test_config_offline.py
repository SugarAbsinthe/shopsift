"""Explicit online/offline policy takes precedence over the cache heuristic."""

from src.config import ensure_hf_offline_if_needed


def test_explicit_online_allows_model_download_in_partial_cache(tmp_path, monkeypatch):
    (tmp_path / "hub").mkdir()
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    assert ensure_hf_offline_if_needed() is False
    import os
    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "0"


def test_explicit_offline_is_preserved_without_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert ensure_hf_offline_if_needed() is True
