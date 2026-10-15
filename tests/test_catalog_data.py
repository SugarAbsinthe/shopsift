"""Source/reference validation and non-destructive first-run import regressions."""

import json
import os
import sqlite3
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.data import prepare_catalog as preparation
from src.data.catalog import (
    CatalogError, ProductRecord, ReviewRecord, SourceRecord,
    _fingerprint_tables, canonical_json, catalog_fingerprint, dataset_counts,
    load_dataset, load_provenance, project_dataset, sha256_file,
)
from src.data.categories import LEGACY_CATEGORIES as CATEGORIES


def write_dataset(directory, products, reviews, sources, status="complete"):
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in (("products", products), ("reviews", reviews), ("sources", sources)):
        (directory / f"{name}.jsonl").write_text(
            "".join(canonical_json(row.model_dump()) + "\n" for row in rows), encoding="utf-8"
        )
    manifest = {
        "schema_version": 1, "dataset_version": "test-v1", "status": status,
        "files": {name: {"sha256": sha256_file(directory / name)} for name in ("products.jsonl", "reviews.jsonl", "sources.jsonl")},
        "counts": dataset_counts(products, reviews, sources),
        "catalog_fingerprint": _fingerprint_tables(project_dataset(products, reviews, sources)),
    }
    (directory / "manifest.json").write_text(canonical_json(manifest), encoding="utf-8")


@pytest.fixture
def sourced_dataset(tmp_path):
    source = SourceRecord(source_id="src_official", url="https://official.example/item", title="Test listing",
                          site="official.example", source_type="official", collected_at="2026-10-11T00:00:00+00:00")
    review_source = SourceRecord(source_id="src_review", url="https://review.example/test", title="Test review",
                                 site="review.example", source_type="editorial_review", collected_at="2026-10-11T00:00:00+00:00")
    products, reviews = [], []
    for category_index, category in enumerate(CATEGORIES):
        for index in range(50):
            pid = category_index * 100 + index + 1
            model = f"test-{category_index}-{index % 20}"
            memory = f"{16 + 16 * (index // 20)}GB"
            products.append(ProductRecord(
                product_id=pid, name=f"{model} {memory}", brand=f"Test brand {index % 4}",
                category=category, subcategory="Test subtype", price=1000 + index,
                description=f"{model} with {memory}", specs={"内存": memory},
                configuration={"内存": memory}, sku_key=f"sku-{pid}", model_key=model,
                price_source_id=source.source_id, spec_source_ids=[source.source_id],
            ))
            for point, sentiment in enumerate(("positive", "mixed")):
                reviews.append(ReviewRecord(
                    review_id=pid * 10 + point, product_id=pid, model_key=model,
                    sentiment=sentiment, aspect=("屏幕", "佩戴")[point], content=f"Test observation {point} for {model}",
                    summary_key=f"{model}-{point}", source_id=review_source.source_id,
                    source_type="editorial_review", scope="model",
                ))
    directory = tmp_path / "dataset"
    write_dataset(directory, products, reviews, [source, review_source])
    return directory, products, reviews, [source, review_source]


def test_release_validation_counts_distinct_models_and_summaries(sourced_dataset):
    directory, *_ = sourced_dataset
    _, _, report = load_dataset(directory, require_complete=True)
    assert report["counts"]["products"] == 200
    assert report["counts"]["unique_summaries"] == 160
    assert all(value["covered_models"] == 20 for value in report["counts"]["categories"].values())


@pytest.mark.parametrize("mutation", ["duplicate_sku", "missing_source", "wrong_review_model", "unsupported_rating", "different_summary"])
def test_invalid_source_and_identity_references_are_rejected(sourced_dataset, mutation):
    directory, products, reviews, sources = sourced_dataset
    if mutation == "duplicate_sku":
        products[1].sku_key = products[0].sku_key
    elif mutation == "missing_source":
        products[0].price_source_id = "src_missing"
        # Keep the original checksums/fingerprint to test the reference validator.
        path = directory / "products.jsonl"
        path.write_text("".join(canonical_json(p.model_dump()) + "\n" for p in products), encoding="utf-8")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        manifest["files"]["products.jsonl"]["sha256"] = sha256_file(path)
        (directory / "manifest.json").write_text(canonical_json(manifest), encoding="utf-8")
        with pytest.raises(CatalogError, match="Unknown product source"):
            load_dataset(directory)
        return
    elif mutation == "wrong_review_model":
        reviews[0].model_key = "wrong-model"
    elif mutation == "unsupported_rating":
        products[0].rating = 4.9
    else:
        reviews[40].content = "A different observation under the same summary key"
    write_dataset(directory, products, reviews, sources)
    with pytest.raises(CatalogError):
        load_dataset(directory)


def test_tampering_is_detected_before_database_creation(sourced_dataset, tmp_path):
    directory, *_ = sourced_dataset
    with (directory / "products.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    output = tmp_path / "output" / "products.db"
    with pytest.raises(CatalogError, match="checksum"):
        preparation.prepare_catalog(directory, output)
    assert not output.parent.exists()


def test_import_is_reproducible_idempotent_and_retains_provenance(sourced_dataset, tmp_path):
    directory, *_ = sourced_dataset
    output = tmp_path / "catalog" / "products.db"
    first = preparation.prepare_catalog(directory, output)
    reviews = output.parent / "product_reviews.db"
    before = (output.read_bytes(), reviews.read_bytes())
    second = preparation.prepare_catalog(directory, output)
    assert first["import_status"] == "imported" and second["import_status"] == "reused"
    assert before == (output.read_bytes(), reviews.read_bytes())
    assert catalog_fingerprint(output, reviews) == first["catalog_fingerprint"]
    assert load_provenance(output, "product_provenance", [1])[1]["price_source"]["url"].startswith("https://")
    with sqlite3.connect(output) as conn:
        assert conn.execute("SELECT rating, sales_count, release_date FROM products WHERE product_id=1").fetchone() == (None, None, "")


def test_existing_or_partial_user_data_is_preserved(sourced_dataset, tmp_path):
    directory, *_ = sourced_dataset
    output = tmp_path / "products.db"
    output.write_bytes(b"user-owned database")
    before = output.read_bytes()
    with pytest.raises(CatalogError, match="Only one database"):
        preparation.prepare_catalog(directory, output)
    assert output.read_bytes() == before
    reviews = tmp_path / "product_reviews.db"
    reviews.write_bytes(b"other user data")
    with pytest.raises(CatalogError, match="Existing catalog differs"):
        preparation.prepare_catalog(directory, output)
    assert output.read_bytes() == before and reviews.read_bytes() == b"other user data"


def test_second_publish_failure_rolls_back_owned_outputs(sourced_dataset, tmp_path, monkeypatch):
    directory, *_ = sourced_dataset
    output = tmp_path / "catalog" / "products.db"
    real_link = os.link
    calls = []
    def fail_second(source, target):
        calls.append(target)
        if len(calls) == 2:
            raise OSError("injected failure")
        real_link(source, target)
    monkeypatch.setattr(preparation.os, "link", fail_second)
    with pytest.raises(OSError, match="injected"):
        preparation.prepare_catalog(directory, output)
    assert not output.exists() and not (output.parent / "product_reviews.db").exists()
    assert not list(output.parent.glob(".catalog-stage-*"))
    assert not list(output.parent.glob("*.catalog-import.json"))


def test_default_first_run_and_custom_path_behavior(sourced_dataset, tmp_path, monkeypatch):
    directory, *_ = sourced_dataset
    output = tmp_path / "default" / "products.db"
    monkeypatch.setattr(preparation, "DEFAULT_PRODUCT_DB", output)
    monkeypatch.setattr(preparation, "DEFAULT_DATASET_DIR", directory)
    preparation.ensure_default_catalog(tmp_path / "custom" / "products.db")
    assert not (tmp_path / "custom").exists()
    preparation.ensure_default_catalog(output)
    before = output.read_bytes()
    preparation.ensure_default_catalog(output)
    assert output.read_bytes() == before


def test_interrupted_import_recovers_only_owned_unmodified_output(sourced_dataset, tmp_path):
    directory, *_ = sourced_dataset
    original = tmp_path / "source" / "products.db"
    preparation.prepare_catalog(directory, original)
    target = tmp_path / "interrupted"
    target.mkdir()
    stages = [target / ".catalog-stage-test-products.db", target / ".catalog-stage-test-reviews.db"]
    stages[0].write_bytes(original.read_bytes())
    stages[1].write_bytes((original.parent / "product_reviews.db").read_bytes())
    products, reviews = target / "products.db", target / "product_reviews.db"
    os.link(stages[0], products)
    marker = target / ".products.db.catalog-import.json"
    preparation._write_marker(marker, {"stages": [p.name for p in stages],
        "outputs": {products.name: sha256_file(stages[0]), reviews.name: sha256_file(stages[1])}})
    report = preparation.prepare_catalog(directory, products)
    assert report["import_status"] == "imported"
    assert products.exists() and reviews.exists() and not marker.exists()
    assert not any(p.exists() for p in stages)


def test_interrupted_import_preserves_both_outputs_if_one_was_modified(tmp_path):
    products, reviews = tmp_path / "products.db", tmp_path / "product_reviews.db"
    products.write_bytes(b"owned first database")
    reviews.write_bytes(b"owned second database")
    marker = tmp_path / ".products.db.catalog-import.json"
    preparation._write_marker(marker, {"stages": [".catalog-stage-test-products.db", ".catalog-stage-test-reviews.db"],
        "outputs": {products.name: sha256_file(products), reviews.name: sha256_file(reviews)}})
    reviews.write_bytes(b"user modified the second database")
    before = products.read_bytes(), reviews.read_bytes()
    with pytest.raises(CatalogError, match="was changed"):
        preparation._recover(marker, products, reviews)
    assert before == (products.read_bytes(), reviews.read_bytes())
    assert marker.exists()


def test_bundled_release_and_default_evaluation_are_bound():
    from collections import Counter
    from src.data.catalog import DEFAULT_DATASET_DIR
    from evals.retrieval_runner import REAL_DATASET_DIRECTORY, REAL_DATASET_ID
    from evals.retrieval_schema import load_retrieval_cases
    from evals.quality import validate_dataset_artifact
    manifest, tables, report = load_dataset(DEFAULT_DATASET_DIR, require_complete=True)
    assert report["counts"]["products"] == 549
    assert sum(v['models'] for v in report['counts']['categories'].values()) == 500
    assert {k: report["counts"]["categories"][k]["models"] for k in CATEGORIES} == dict(zip(CATEGORIES, (75, 75, 70, 70)))
    assert manifest['release_readiness']['status'] == 'passed'
    assert report["quality_policy"] == "source-preferred-v1"
    assert report["capability_status"] == "trial"
    cases_path = REAL_DATASET_DIRECTORY / "retrieval_cases.jsonl"
    _, dataset = validate_dataset_artifact(REAL_DATASET_DIRECTORY / "manifest.json", REAL_DATASET_ID, cases_path)
    assert dataset.catalog_fingerprint == manifest["catalog_fingerprint"]
    cases = load_retrieval_cases(cases_path)
    assert len(cases) == 76
    product_ids = {p["product_id"] for p in tables["products"]}
    for category in CATEGORIES:
        subset = [c for c in cases if c.constraints.category == category]
        assert len(subset) == 10
        assert Counter(tag for c in subset for tag in c.tags if tag in {"exact_config", "spec_budget", "brand_exclusion", "expected_empty"}) == {
            "exact_config": 3, "spec_budget": 5, "brand_exclusion": 1, "expected_empty": 1}
    assert all(set(c.relevant_ids + c.forbidden_ids) <= product_ids for c in cases)


def test_preflight_reports_catalog_mismatch_before_model_load(sourced_dataset, tmp_path):
    from evals.retrieval_runner import preflight_retrieval
    directory, *_ = sourced_dataset
    products = tmp_path / "catalog" / "products.db"
    preparation.prepare_catalog(directory, products)
    result = preflight_retrieval(products, tmp_path / "missing-index", expected_catalog_fingerprint="0" * 64)
    assert not result.passed
    assert "RETRIEVAL_CATALOG_DATASET_MISMATCH" in result.issues


def test_retriever_rejects_stale_index_before_loading_model(sourced_dataset, tmp_path, monkeypatch):
    from src.retrieval.index_manifest import activate_index_manifest
    from src.retrieval import product_retriever as module
    directory, *_ = sourced_dataset
    products = tmp_path / "catalog" / "products.db"
    preparation.prepare_catalog(directory, products)
    chroma = tmp_path / "index"
    activate_index_manifest(chroma, {"schema_version":1,"version":"test","embedding_model":"fake",
        "collections": {key:f"product_{key}_test" for key in ("descriptions","specs","reviews")},
        "counts":{"descriptions":200,"specs":200,"reviews":400},"catalog_fingerprint":"0"*64})
    monkeypatch.setattr(module, "SentenceTransformer", lambda *a, **kw: pytest.fail("must not load a model"))
    with pytest.raises(ValueError, match="Catalog content changed"):
        module.ProductRetriever(str(chroma), catalog_db=str(products))


def test_cli_does_not_import_embedding_or_model_clients(tmp_path):
    import subprocess
    import sys
    script = "from src.data.prepare_catalog import main; import sys; code=main(); assert code == 0; assert not any(m in sys.modules for m in ('sentence_transformers','chromadb','langchain_openai')); print('NO_MODEL_IMPORTS')"
    output = tmp_path / "cli" / "products.db"
    completed = subprocess.run([sys.executable,"-c",script,"--product-db",str(output)], capture_output=True, text=True, encoding="utf-8")
    assert completed.returncode == 0, completed.stderr
    assert "NO_MODEL_IMPORTS" in completed.stdout


def test_source_aware_tools_show_unknowns_and_combine_review_filters(sourced_dataset, tmp_path, monkeypatch):
    from src.agent.shopping_tools import create_get_product_detail, create_compare_products, create_get_reviews
    import chromadb
    directory, *_ = sourced_dataset
    output = tmp_path / "catalog" / "products.db"
    preparation.prepare_catalog(directory, output)
    detail = create_get_product_detail(str(output)).invoke({"product_id": 1})
    assert "销量: 未知" in detail and "价格快照:" in detail and "https://official.example/item" in detail
    compared = create_compare_products(str(output)).invoke({"product_ids": "1,2"})
    assert "未知" in compared and "价格快照:" in compared
    assert "| 销量 | 未知 " in compared and "| 发布日期 | 未知 " in compared
    collection = Mock()
    collection.get.return_value = {"ids": ["review_10"], "documents": ["ignored duplicate prefix"], "metadatas": [{
        "sentiment": "mixed", "aspect": "屏幕", "content": "A sourced point", "source_type": "editorial_review",
        "source_url": "https://review.example/test", "scope": "model",
    }]}
    client = Mock()
    client.get_collection.return_value = collection
    constructor = Mock(return_value=client)
    monkeypatch.setattr(chromadb, "PersistentClient", constructor)
    result = create_get_reviews(str(output.parent / "product_reviews.db"), "custom-index").invoke({"product_id": 1, "aspect": "屏幕"})
    constructor.assert_called_once_with(path="custom-index")
    assert collection.get.call_args.kwargs["where"] == {"$and": [{"product_id": 1}, {"aspect": "屏幕"}]}
    assert "褒贬并存" in result and "原创评测" in result and "https://review.example/test" in result
