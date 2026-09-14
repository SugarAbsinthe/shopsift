"""Tests for versioned evaluation manifests and quality gates."""

import json

import pytest
from pydantic import ValidationError

from evals.quality import (
    DatasetArtifact,
    QualityGateSpec,
    build_run_metadata,
    evaluate_quality_gate,
    load_dataset_manifest,
    validate_dataset_artifact,
)


def test_default_manifest_validates_case_count_and_checksum():
    manifest, context = validate_dataset_artifact(
        "evals/datasets/v1/manifest.json", "agent-behavior-v1"
    )
    assert manifest.dataset_version == "v1"
    assert context.case_count == 25
    assert len(context.sha256) == 64
    assert context.path == "evals/cases.jsonl"


@pytest.mark.parametrize("unsafe_path", ["../cases.jsonl", r"..\cases.jsonl", "C:/tmp/cases.jsonl"])
def test_manifest_rejects_paths_that_can_escape_repository(unsafe_path):
    with pytest.raises(ValidationError, match="stay within the repository"):
        DatasetArtifact(
            id="unsafe-case",
            path=unsafe_path,
            format="jsonl",
            case_count=1,
            sha256="0" * 64,
        )


def test_manifest_validation_rejects_checksum_changes(tmp_path):
    (tmp_path / "evals").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"id":"case"}\n', encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "dataset_version": "test",
        "created_at": "2026-09-14",
        "cases": [{
            "id": "test-cases",
            "path": "cases.jsonl",
            "format": "jsonl",
            "case_count": 1,
            "sha256": "0" * 64,
        }],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_dataset_artifact(manifest_path, "test-cases")


def test_quality_gate_reports_both_missing_and_regressed_metrics():
    spec = QualityGateSpec(
        name="strict",
        mode="deterministic",
        dataset_id="cases",
        minimums={"pass_rate": 1.0, "stage_accuracy": 1.0},
        maximums={"illegal_tool_calls": 0},
    )
    result = evaluate_quality_gate(
        {"pass_rate": 0.5, "illegal_tool_calls": 1}, spec
    )
    assert result.passed is False
    assert any("pass_rate" in failure for failure in result.failures)
    assert any("stage_accuracy" in failure for failure in result.failures)
    assert any("illegal_tool_calls" in failure for failure in result.failures)


def test_run_metadata_contains_versions_without_raw_conversation():
    metadata = build_run_metadata(
        mode="deterministic",
        started_at_monotonic=0,
        started_at="2026-09-14T00:00:00+00:00",
        dataset=None,
        cases_path="evals/cases.jsonl",
        model="scripted-eval",
        provider="local-double",
        external_access=False,
    )
    assert metadata["report_schema_version"] == 1
    assert metadata["runtime"]["external_access"] is False
    assert metadata["dataset"]["case_count"] == 25
    serialized = json.dumps(metadata, ensure_ascii=False)
    assert "用户" not in serialized
    assert "OPENAI_API_KEY" not in serialized
