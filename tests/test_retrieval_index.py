"""Versioned index activation and rollback tests."""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from src.embeddings.product_embedder import build_product_embeddings
from src.retrieval.index_manifest import (
    activate_index_manifest,
    load_index_manifest,
    resolve_collection_names,
)


class FakeVector:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


@pytest.mark.parametrize('size', (3, 67, 69))
def test_small_persistent_index_can_be_queried_after_builder_process_exits(tmp_path, size):
    products, reviews = _databases(tmp_path)
    # Also cover a prime count and a partial encoder batch. Count checks alone
    # cannot detect metadata IDs absent from the persisted HNSW segment.
    with sqlite3.connect(products) as conn:
        for identifier in range(2, size + 1):
            conn.execute("INSERT INTO products SELECT ?, name, brand, category, subcategory, price, rating, sales_count, release_date, description, specs FROM products WHERE product_id=1", (identifier,))
    with sqlite3.connect(reviews) as conn:
        for identifier in range(2, size + 1):
            conn.execute("INSERT INTO product_reviews SELECT ?, ?, sentiment, aspect, content FROM product_reviews WHERE review_id=1", (identifier, identifier))
    persist = str(tmp_path / "persistent")
    build = """
import sys
import numpy as np
from src.embeddings.product_embedder import build_product_embeddings
class Model:
    def encode(self, documents):
        return np.array([[1.0, 0.5] for _ in documents])
build_product_embeddings(*sys.argv[1:], 'local-test', model=Model())
"""
    query = """
import sys
import chromadb
from src.retrieval.index_manifest import load_index_manifest
manifest = load_index_manifest(sys.argv[1])
client = chromadb.PersistentClient(path=sys.argv[1])
for name in manifest['collections'].values():
    size = int(sys.argv[2])
    result = client.get_collection(name).query(query_embeddings=[[1.0, 0.5]], n_results=3,
        where={'product_id':{'$in':list(range(size-2,size+1))}})
    assert len(result['ids'][0]) == 3
"""
    root = Path(__file__).resolve().parents[1]
    for script, arguments in ((build, [products, reviews, persist]), (query, [persist, str(size)])):
        result = subprocess.run([sys.executable, "-c", script, *arguments], cwd=root, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr


class FakeModel:
    def encode(self, documents):
        return [FakeVector([float(index), 0.5]) for index, _ in enumerate(documents)]


class FakeBuildCollection:
    def __init__(self, fail_count=False):
        self.total = 0
        self.fail_count = fail_count
        self.metadata = []

    def add(self, **kwargs):
        self.total += len(kwargs["ids"])
        self.metadata.extend(kwargs["metadatas"])

    def count(self):
        return self.total + (1 if self.fail_count else 0)


class FakeBuildClient:
    def __init__(self, fail_name=""):
        self.fail_name = fail_name
        self.collections = {}
        self.deleted = []

    def create_collection(self, name, **kwargs):
        collection = FakeBuildCollection(name == self.fail_name)
        self.collections[name] = collection
        return collection

    def delete_collection(self, name):
        self.deleted.append(name)
        self.collections.pop(name, None)


def _databases(tmp_path):
    products = tmp_path / "products.db"
    conn = sqlite3.connect(products)
    conn.execute(
        "CREATE TABLE products (product_id INTEGER, name TEXT, brand TEXT, "
        "category TEXT, subcategory TEXT, price REAL, rating REAL, sales_count INTEGER, "
        "release_date TEXT, description TEXT, specs TEXT)"
    )
    conn.execute(
        "INSERT INTO products VALUES (1, 'Laptop', 'Brand', '电脑', '游戏本', "
        "7000, 4.8, 10, '2025-01-01', 'RTX laptop', ?)",
        (json.dumps({"GPU": "RTX 4060"}),),
    )
    conn.commit()
    conn.close()

    reviews = tmp_path / "reviews.db"
    conn = sqlite3.connect(reviews)
    conn.execute(
        "CREATE TABLE product_reviews (review_id INTEGER, product_id INTEGER, "
        "sentiment TEXT, aspect TEXT, content TEXT)"
    )
    conn.execute("INSERT INTO product_reviews VALUES (1, 1, 'positive', 'GPU', 'fast')")
    conn.commit()
    conn.close()
    return str(products), str(reviews)


def _manifest(version):
    return {
        "schema_version": 1,
        "version": version,
        "collections": {
            "descriptions": f"product_descriptions_{version}",
            "specs": f"product_specs_{version}",
            "reviews": f"product_reviews_{version}",
        },
    }


def test_manifest_falls_back_to_legacy_and_rejects_unsafe_names(tmp_path):
    names, version = resolve_collection_names(tmp_path)
    assert version == "legacy"
    assert names["reviews"] == "product_reviews"
    (tmp_path / "retrieval_manifest.json").write_text(
        json.dumps({**_manifest("v1"), "collections": {
            **_manifest("v1")["collections"], "reviews": "../reviews"
        }}),
        encoding="utf-8",
    )
    assert load_index_manifest(tmp_path) is None


def test_manifest_activation_replaces_previous_version(tmp_path):
    activate_index_manifest(tmp_path, _manifest("v1"))
    activate_index_manifest(tmp_path, _manifest("v2"))
    assert load_index_manifest(tmp_path)["version"] == "v2"
    assert list(tmp_path.glob(".manifest-*")) == []


def test_failed_build_keeps_old_manifest_active_and_cleans_new_collections(tmp_path):
    products, reviews = _databases(tmp_path)
    persist = tmp_path / "chroma"
    activate_index_manifest(persist, _manifest("stable"))
    client = FakeBuildClient(fail_name="product_specs_candidate")
    with pytest.raises(RuntimeError, match="count mismatch"):
        build_product_embeddings(
            products,
            reviews,
            str(persist),
            "fake-model",
            index_version="candidate",
            model=FakeModel(),
            client=client,
        )
    assert load_index_manifest(persist)["version"] == "stable"
    assert set(client.deleted) == {
        "product_descriptions_candidate",
        "product_specs_candidate",
        "product_reviews_candidate",
    }


def test_successful_build_activates_validated_counts(tmp_path):
    products, reviews = _databases(tmp_path)
    persist = tmp_path / "chroma"
    manifest = build_product_embeddings(
        products,
        reviews,
        str(persist),
        "fake-model",
        index_version="v2",
        batch_size=1,
        model=FakeModel(),
        client=FakeBuildClient(),
    )
    assert manifest["counts"] == {"descriptions": 1, "specs": 1, "reviews": 1}
    assert load_index_manifest(persist)["version"] == "v2"


@pytest.mark.parametrize("release_date", [None, ""])
def test_unknown_metadata_is_omitted_and_content_hash_is_recorded(tmp_path, release_date):
    products, reviews = _databases(tmp_path)
    with sqlite3.connect(products) as conn:
        conn.execute("UPDATE products SET rating=NULL, sales_count=NULL, release_date=?", (release_date,))
    client = FakeBuildClient()
    manifest = build_product_embeddings(products, reviews, str(tmp_path / "chroma"), "fake-model",
        index_version="unknowns", model=FakeModel(), client=client)
    metadata = client.collections[manifest["collections"]["descriptions"]].metadata[0]
    assert not any(field in metadata for field in ("rating", "sales_count", "release_date"))
    assert len(manifest["catalog_fingerprint"]) == 64
