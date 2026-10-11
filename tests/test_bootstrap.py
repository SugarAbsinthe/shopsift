"""Startup prerequisites, index reuse and data-preservation regressions."""

import json
from contextlib import closing
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from backend import bootstrap
from src.retrieval.index_manifest import activate_index_manifest, load_index_manifest


@pytest.fixture
def settings(tmp_path):
    products = tmp_path / "products.db"
    reviews = tmp_path / "product_reviews.db"
    with closing(sqlite3.connect(products)) as conn, conn:
        conn.execute(
            "CREATE TABLE products (product_id INTEGER, name TEXT, brand TEXT, "
            "category TEXT, subcategory TEXT, price REAL, rating REAL, sales_count INTEGER, "
            "release_date TEXT, description TEXT, specs TEXT)"
        )
        conn.execute(
            "INSERT INTO products VALUES (1, 'Test laptop', 'Test brand', '笔记本', '轻薄本', "
            "5000, 4.5, 1, '2025-01-01', 'Test description', ?)",
            (json.dumps({"内存": "16GB"}),),
        )
    with closing(sqlite3.connect(reviews)) as conn, conn:
        conn.execute(
            "CREATE TABLE product_reviews (review_id INTEGER, product_id INTEGER, "
            "sentiment TEXT, aspect TEXT, content TEXT)"
        )
        conn.execute("INSERT INTO product_reviews VALUES (1, 1, 'positive', 'weight', 'Test review')")
    return SimpleNamespace(
        OPENAI_API_KEY="test-key-do-not-log", OPENAI_BASE_URL="https://invalid.test/v1",
        LLM_MODEL="test-model", EMBEDDING_MODEL="fake-embedding",
        PRODUCT_DB_PATH=str(products), PRODUCT_CHROMA_DIR=str(tmp_path / "chroma"),
    )


def manifest(version="stable"):
    return {
        "schema_version": 1, "version": version, "embedding_model": "fake-embedding",
        "collections": {key: f"product_{key}_{version}" for key in ("descriptions", "specs", "reviews")},
        "counts": {"descriptions": 1, "specs": 1, "reviews": 1},
    }


class Collection:
    def __init__(self, name, count=1):
        self.name = name
        self.total = count

    def count(self):
        return self.total

    def add(self, *, ids, **kwargs):
        self.total += len(ids)


class Client:
    def __init__(self, names=()):
        self.collections = {name: Collection(name) for name in names}

    def list_collections(self):
        return list(self.collections.values())

    def get_collection(self, name):
        return self.collections[name]

    def create_collection(self, name):
        self.collections[name] = Collection(name, count=0)
        return self.collections[name]

    def delete_collection(self, name):
        self.collections.pop(name)


class Vector:
    def tolist(self):
        return [0.1, 0.2]


class Model:
    def encode(self, texts):
        return [Vector() for _ in texts]


def dependencies(monkeypatch, client):
    loaded = []
    monkeypatch.setattr(bootstrap, "_load_model", lambda name: loaded.append(name) or Model())
    monkeypatch.setattr(bootstrap, "_open_client", lambda path: client)
    return loaded


@pytest.mark.parametrize("key", ["", " ", "your-api-key-here", "sk-placeholder"])
def test_placeholder_key_stops_before_model_or_index(settings, monkeypatch, key):
    settings.OPENAI_API_KEY = key
    loaded = dependencies(monkeypatch, Client())
    with pytest.raises(bootstrap.BootstrapError, match="OPENAI_API_KEY"):
        bootstrap.prepare_runtime(settings)
    assert not loaded
    assert not Path(settings.PRODUCT_CHROMA_DIR).exists()


def test_missing_database_is_not_created_and_other_database_unchanged(settings, monkeypatch):
    reviews = Path(settings.PRODUCT_DB_PATH).parent / "product_reviews.db"
    reviews.unlink()
    before = Path(settings.PRODUCT_DB_PATH).read_bytes()
    loaded = dependencies(monkeypatch, Client())
    with pytest.raises(bootstrap.BootstrapError, match="Missing"):
        bootstrap.prepare_runtime(settings)
    assert not reviews.exists()
    assert Path(settings.PRODUCT_DB_PATH).read_bytes() == before
    assert not loaded


def test_database_schema_error_is_reported_before_model_load(settings, monkeypatch):
    with closing(sqlite3.connect(settings.PRODUCT_DB_PATH)) as conn, conn:
        conn.execute("ALTER TABLE products RENAME COLUMN specs TO wrong_specs")
    loaded = dependencies(monkeypatch, Client())
    with pytest.raises(bootstrap.BootstrapError, match="missing fields: specs"):
        bootstrap.prepare_runtime(settings)
    assert not loaded


def test_empty_catalog_is_rejected(settings, monkeypatch):
    with closing(sqlite3.connect(settings.PRODUCT_DB_PATH)) as conn, conn:
        conn.execute("DELETE FROM products")
    loaded = dependencies(monkeypatch, Client())
    with pytest.raises(bootstrap.BootstrapError, match="empty"):
        bootstrap.prepare_runtime(settings)
    assert not loaded


def test_invalid_manifest_is_preserved_and_not_rebuilt(settings, monkeypatch):
    persist = Path(settings.PRODUCT_CHROMA_DIR)
    persist.mkdir()
    path = persist / "retrieval_manifest.json"
    path.write_text("not valid json", encoding="utf-8")
    loaded = dependencies(monkeypatch, Client())
    with pytest.raises(bootstrap.BootstrapError, match="Invalid retrieval_manifest"):
        bootstrap.prepare_runtime(settings)
    assert path.read_text(encoding="utf-8") == "not valid json"
    assert not loaded


def test_embedding_mismatch_requires_explicit_rebuild(settings, monkeypatch):
    active = manifest()
    activate_index_manifest(settings.PRODUCT_CHROMA_DIR, active)
    settings.EMBEDDING_MODEL = "other-embedding"
    loaded = dependencies(monkeypatch, Client(active["collections"].values()))
    with pytest.raises(bootstrap.BootstrapError, match="EMBEDDING_MODEL differs"):
        bootstrap.prepare_runtime(settings)
    assert not loaded
    assert load_index_manifest(settings.PRODUCT_CHROMA_DIR) == active


def test_first_build_uses_runtime_paths_and_keeps_source_databases(settings, monkeypatch, capsys):
    client = Client()
    loaded = dependencies(monkeypatch, client)
    paths = [Path(settings.PRODUCT_DB_PATH), Path(settings.PRODUCT_DB_PATH).parent / "product_reviews.db"]
    before = [path.read_bytes() for path in paths]
    # Exercise the real versioned builder with controlled embeddings/storage.
    assert bootstrap.prepare_runtime(settings) == "built"
    active = load_index_manifest(settings.PRODUCT_CHROMA_DIR)
    assert active["embedding_model"] == settings.EMBEDDING_MODEL
    assert active["counts"] == {"descriptions": 1, "specs": 1, "reviews": 1}
    assert loaded == [settings.EMBEDDING_MODEL]
    assert [path.read_bytes() for path in paths] == before
    assert settings.OPENAI_API_KEY not in capsys.readouterr().out


def test_complete_index_is_reused_but_model_cache_is_warmed(settings, monkeypatch):
    active = manifest()
    activate_index_manifest(settings.PRODUCT_CHROMA_DIR, active)
    client = Client(active["collections"].values())
    loaded = dependencies(monkeypatch, client)
    monkeypatch.setattr(bootstrap, "_build_index", lambda *a, **kw: pytest.fail("must not rebuild"))
    assert bootstrap.prepare_runtime(settings) == "reused"
    assert loaded == [settings.EMBEDDING_MODEL]
    assert load_index_manifest(settings.PRODUCT_CHROMA_DIR) == active


@pytest.mark.parametrize("damage", ["missing", "wrong_count", "empty"])
def test_inconsistent_active_index_is_not_silently_replaced(settings, monkeypatch, damage):
    active = manifest()
    activate_index_manifest(settings.PRODUCT_CHROMA_DIR, active)
    client = Client(active["collections"].values())
    name = active["collections"]["specs"]
    if damage == "missing":
        client.collections.pop(name)
    else:
        client.collections[name].total = 2 if damage == "wrong_count" else 0
    dependencies(monkeypatch, client)
    monkeypatch.setattr(bootstrap, "_build_index", lambda *a, **kw: pytest.fail("must not rebuild"))
    with pytest.raises(bootstrap.BootstrapError, match="explicitly rebuild"):
        bootstrap.prepare_runtime(settings)
    assert load_index_manifest(settings.PRODUCT_CHROMA_DIR) == active


def test_explicit_rebuild_activates_new_version_without_deleting_old_collections(settings, monkeypatch):
    active = manifest()
    activate_index_manifest(settings.PRODUCT_CHROMA_DIR, active)
    client = Client(active["collections"].values())
    dependencies(monkeypatch, client)
    assert bootstrap.prepare_runtime(settings, rebuild_index=True) == "built"
    updated = load_index_manifest(settings.PRODUCT_CHROMA_DIR)
    assert updated["version"] != active["version"]
    assert set(active["collections"].values()) <= set(client.collections)


def test_legacy_collections_can_be_reused_with_model_caveat(settings, monkeypatch, capsys):
    client = Client(bootstrap.LEGACY_COLLECTIONS.values())
    dependencies(monkeypatch, client)
    assert bootstrap.prepare_runtime(settings) == "reused"
    assert "embedding model must match" in capsys.readouterr().out


def test_first_build_and_restart_with_real_chroma(settings, monkeypatch):
    """Actual persistent Chroma and SQLite, with no downloaded model or API."""
    monkeypatch.setattr(bootstrap, "_load_model", lambda name: Model())
    assert bootstrap.prepare_runtime(settings) == "built"
    active = load_index_manifest(settings.PRODUCT_CHROMA_DIR)
    monkeypatch.setattr(bootstrap, "_build_index", lambda *a, **kw: pytest.fail("must reuse persisted index"))
    assert bootstrap.prepare_runtime(settings) == "reused"
    assert load_index_manifest(settings.PRODUCT_CHROMA_DIR) == active


@pytest.mark.parametrize("empty_channel", ["specs", "reviews"])
def test_empty_optional_channel_can_be_built_and_reused(settings, monkeypatch, empty_channel):
    if empty_channel == "specs":
        with closing(sqlite3.connect(settings.PRODUCT_DB_PATH)) as conn, conn:
            conn.execute("UPDATE products SET specs = '{}'")
    else:
        reviews = Path(settings.PRODUCT_DB_PATH).parent / "product_reviews.db"
        with closing(sqlite3.connect(reviews)) as conn, conn:
            conn.execute("DELETE FROM product_reviews")
    dependencies(monkeypatch, Client())
    assert bootstrap.prepare_runtime(settings) == "built"
    assert load_index_manifest(settings.PRODUCT_CHROMA_DIR)["counts"][empty_channel] == 0
    assert bootstrap.prepare_runtime(settings) == "reused"
