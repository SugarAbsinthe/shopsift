"""Metric tests for the retrieval evaluation runner."""

import json

from evals.retrieval_runner import (
    RetrievalPreflight,
    evaluate_retrieval,
    preflight_retrieval,
    write_retrieval_reports,
)
from evals.retrieval_schema import RetrievalCase
from src.retrieval.models import ProductCandidate, RetrievalResult


class FakeRetriever:
    def search(self, query):
        products = {
            "good": [ProductCandidate(
                product_id=1,
                brand="联想",
                category="笔记本电脑",
                price=7000,
                sources=["description", "sparse"],
            )],
            "bad": [ProductCandidate(
                product_id=2,
                brand="华硕",
                category="笔记本电脑",
                price=9000,
                sources=["spec"],
            )],
            "empty": [],
        }[query.text]
        return RetrievalResult(products=products)


def test_runner_reports_quality_and_safety_metrics():
    cases = [
        RetrievalCase(
            id="good_case",
            query="good",
            relevant_ids=[1],
            required_sources=["description"],
            constraints={"max_price": 8000},
        ),
        RetrievalCase(
            id="bad_case",
            query="bad",
            relevant_ids=[1],
            forbidden_ids=[2],
            required_sources=["sparse"],
            constraints={"max_price": 8000},
        ),
        RetrievalCase(id="empty_case", query="empty"),
    ]
    snapshot = evaluate_retrieval(FakeRetriever(), cases).snapshot()
    assert snapshot["hit_rate"] == 0.5
    assert snapshot["recall_at_k"] == 0.5
    assert snapshot["source_coverage"] == 0.5
    assert snapshot["forbidden_violations"] == 1
    assert snapshot["constraint_violations"] == 1
    assert snapshot["empty_result_violations"] == 0
    assert snapshot["failures"][0]["id"] == "bad_case"


def test_preflight_reports_missing_dependencies_without_loading_model(tmp_path):
    result = preflight_retrieval(
        tmp_path / "missing-products.db",
        tmp_path / "missing-chroma",
    )
    assert result.passed is False
    assert result.issues == ["RETRIEVAL_DB_MISSING", "RETRIEVAL_INDEX_DIR_MISSING"]


def test_blocked_retrieval_report_is_structured_and_safe(tmp_path):
    preflight = RetrievalPreflight(
        passed=False,
        issues=["RETRIEVAL_DB_MISSING"],
    )
    json_path, markdown_path = write_retrieval_reports(
        None,
        tmp_path,
        status="blocked",
        preflight=preflight,
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["report_schema_version"] == 1
    assert payload["status"] == "blocked"
    assert payload["summary"] is None
    assert payload["preflight"]["issues"] == ["RETRIEVAL_DB_MISSING"]
    assert "blocked" in markdown_path.read_text(encoding="utf-8")
