"""Protect quality diagnostics against confusing valid variants with duplicates."""
from scripts.audit_catalog_quality import analyze_records, audit_cache, decision_specs
import hashlib


def product(pid=1, model="model-a", storage="256GB", **overrides):
    record = {"product_id": pid, "sku_key": f"sku-{pid}", "model_key": model,
              "name": f"{model} {storage}", "brand": "Example", "category": "手机", "subcategory": "手机",
              "price": 2000, "rating": None, "sales_count": None, "release_date": None,
              "description": "A catalog observation", "specs": {"CPU型号": "CPU-A", "存储": storage},
              "configuration": {"存储": storage}}
    record.update(overrides)
    return record


def test_storage_variants_are_not_duplicate_products():
    result = analyze_records([product(), product(2, storage="512GB")], [], [])
    assert result["duplicates"]["same_model_configuration"] == []
    assert result["categories"]["手机"]["storage_only_models"][0]["variant_count"] == 2
    assert result["categories"]["手机"]["field_coverage"]["处理器"] == 2


def test_identical_specs_on_distinct_models_are_information_collisions():
    result = analyze_records([product(), product(2, model="model-b")], [], [])
    assert result["duplicates"]["same_model_configuration"] == []
    assert len(result["duplicates"]["identical_specs_across_models"]) == 1


def test_missing_identity_key_is_not_a_conflicting_specification():
    row = product(configuration={"版本": "model-a", "存储": "256GB"})
    result = analyze_records([row], [], [])
    assert result["config_spec_disagreements"] == []
    assert result["configuration_keys_not_repeated_in_specs"][0]["keys"] == ["版本"]
    row["specs"]["存储"] = "512GB"
    assert len(analyze_records([row], [], [])["config_spec_disagreements"]) == 1


def test_absent_weight_is_not_inferred_from_lightweight_description():
    row = product(description="Lightweight phone, weighing 100g")
    assert "重量" not in decision_specs(row)


def test_cache_newlines_and_reserialization_are_distinguished(tmp_path):
    original = b'{"text":"test"}\n'
    (tmp_path / "test.html").write_bytes(original.replace(b"\n", b"\r\n"))
    source = {"source_id": "src_test", "page_sha256": hashlib.sha256(original).hexdigest(),
              "retrieval_method": "POST", "url": "https://example.org/reviews"}
    assert audit_cache([source], tmp_path)["counts"] == {"LF_restored_match": 1}
    (tmp_path / "test.html").write_bytes(b'{"text": "test"}')
    assert audit_cache([source], tmp_path)["counts"] == {"original_bytes_not_reproducible": 1}


def test_missing_evidence_is_reported_as_unresolved(tmp_path):
    source={'source_id':'src_missing','page_sha256':'a'*64,'url':'https://example.org/item'}
    assert len(audit_cache([source],tmp_path)['unresolved_sources'])==1


def test_release_check_fails_when_explicit_evidence_verification_fails(tmp_path, monkeypatch):
    import sys
    from scripts.audit_catalog_quality import main
    output = tmp_path / 'audit'
    monkeypatch.setattr(sys, 'argv', ['audit_catalog_quality', '--release-check',
                        '--source-cache', str(tmp_path / 'missing-cache'), '--output-dir', str(output)])
    assert main() == 2
    import json
    report = json.loads((output / 'audit.json').read_text(encoding='utf8'))
    assert report['release_readiness']['status'] == 'passed'
    assert len(report['source_cache']['unresolved_sources']) == 902
