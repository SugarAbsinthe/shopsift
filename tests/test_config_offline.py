"""Explicit network policy must not be inferred from unrelated cache contents."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.config import ensure_hf_offline_if_needed


@pytest.fixture(autouse=True)
def isolated_hf_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))


def test_explicit_online_allows_model_download_in_partial_cache(tmp_path, monkeypatch):
    (tmp_path / "hub").mkdir()
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    assert ensure_hf_offline_if_needed() is False
    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "0"


def test_explicit_offline_is_preserved_without_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert ensure_hf_offline_if_needed() is True


@pytest.mark.parametrize("contents", ["empty", "other-model", "partial-model"])
def test_existing_cache_does_not_force_offline(contents):
    hub = Path(os.environ["HF_HOME"]) / "hub"
    hub.mkdir(parents=True)
    if contents != "empty":
        model = "unrelated--model" if contents == "other-model" else "BAAI--bge-small-zh-v1.5"
        snapshot = hub / f"models--{model}" / "snapshots" / "revision"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")

    assert ensure_hf_offline_if_needed() is False
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ


def test_custom_cache_does_not_inherit_unrelated_default_cache_policy(tmp_path, monkeypatch):
    (Path(os.environ["HF_HOME"]) / "hub").mkdir(parents=True)
    custom = tmp_path / "custom-model-cache"
    custom.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(custom))

    assert ensure_hf_offline_if_needed() is False
    assert os.environ["HF_HUB_CACHE"] == str(custom)
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert list(custom.iterdir()) == []


@pytest.mark.parametrize(("hub_flag", "legacy_flag", "expected"), [
    (None, None, False),
    ("1", None, True),
    ("0", "1", False),
    ("TRUE", "0", True),
    (None, "1", True),
    ("", "yes", True),
    ("off", "1", False),
])
def test_explicit_flags_follow_sdk_precedence_without_rewriting(monkeypatch, hub_flag, legacy_flag, expected):
    for key, value in [("HF_HUB_OFFLINE", hub_flag), ("TRANSFORMERS_OFFLINE", legacy_flag)]:
        if value is not None:
            monkeypatch.setenv(key, value)
    before = {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}

    assert ensure_hf_offline_if_needed() is expected
    assert {key: os.environ.get(key) for key in before} == before


def test_backend_startup_leaves_custom_cache_and_network_policy_to_sdk(tmp_path):
    home = tmp_path / "hf-home"
    (home / "hub").mkdir(parents=True)
    cache = tmp_path / "custom-cache"
    cache.mkdir()
    env = {
        **os.environ,
        "HF_HOME": str(home),
        "HF_HUB_CACHE": str(cache),
        "HF_HUB_OFFLINE": "",
        "TRANSFORMERS_OFFLINE": "",
        "LANGSMITH_TRACING": "false",
    }
    script = """
import os, sys
from backend.main import app
from huggingface_hub import constants
assert app.title == 'ShopSift API'
assert constants.HF_HUB_CACHE == sys.argv[1]
assert constants.HF_HUB_OFFLINE is False
assert os.environ['HF_HUB_OFFLINE'] == ''
assert os.environ['TRANSFORMERS_OFFLINE'] == ''
"""
    subprocess.run(
        [sys.executable, "-c", script, str(cache)],
        cwd=Path(__file__).resolve().parents[1], env=env,
        capture_output=True, text=True, timeout=30, check=True,
    )
