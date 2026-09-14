"""Offline/live retrieval evaluation with constraint-safety metrics."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evals.retrieval_schema import RetrievalCase, load_retrieval_cases
from evals.quality import (
    DEFAULT_DATASET_MANIFEST_PATH,
    GateResult,
    build_run_metadata,
    evaluation_start,
    evaluate_quality_gate,
    get_quality_gate,
    validate_dataset_artifact,
)
from src.retrieval.models import RetrievalQuery
from src.retrieval.index_manifest import resolve_collection_names


@dataclass
class RetrievalPreflight:
    """Read-only checks required before loading the embedding model."""

    passed: bool
    issues: list[str] = field(default_factory=list)
    index_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": list(self.issues),
            "index_version": self.index_version,
        }


def preflight_retrieval(catalog_db: Path | str, chroma_dir: Path | str) -> RetrievalPreflight:
    """Check database and active Chroma collections without running retrieval."""
    issues: list[str] = []
    catalog_path = Path(catalog_db)
    chroma_path = Path(chroma_dir)

    if not catalog_path.is_file():
        issues.append("RETRIEVAL_DB_MISSING")
    else:
        try:
            conn = sqlite3.connect(f"file:{catalog_path.resolve()}?mode=ro", uri=True)
            try:
                conn.execute("SELECT 1 FROM products LIMIT 1").fetchone()
            finally:
                conn.close()
        except (OSError, sqlite3.Error):
            issues.append("RETRIEVAL_DB_INVALID")

    index_version: str | None = None
    if not chroma_path.is_dir():
        issues.append("RETRIEVAL_INDEX_DIR_MISSING")
    else:
        try:
            collection_names, index_version = resolve_collection_names(chroma_path)
            import chromadb

            client = chromadb.PersistentClient(path=str(chroma_path))
            for name in collection_names.values():
                try:
                    client.get_collection(name)
                except Exception:
                    issues.append("RETRIEVAL_INDEX_COLLECTION_MISSING")
                    break
        except Exception:
            issues.append("RETRIEVAL_INDEX_UNAVAILABLE")

    return RetrievalPreflight(
        passed=not issues,
        issues=list(dict.fromkeys(issues)),
        index_version=index_version,
    )


@dataclass
class RetrievalEvaluation:
    cases: int = 0
    relevant_cases: int = 0
    hit_cases: int = 0
    relevant_total: int = 0
    relevant_retrieved: int = 0
    forbidden_violations: int = 0
    constraint_violations: int = 0
    empty_result_violations: int = 0
    required_sources_total: int = 0
    required_sources_met: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "cases": self.cases,
            "hit_rate": round(self.hit_cases / self.relevant_cases, 4)
            if self.relevant_cases else None,
            "recall_at_k": round(self.relevant_retrieved / self.relevant_total, 4)
            if self.relevant_total else None,
            "source_coverage": round(
                self.required_sources_met / self.required_sources_total, 4
            ) if self.required_sources_total else None,
            "forbidden_violations": self.forbidden_violations,
            "constraint_violations": self.constraint_violations,
            "empty_result_violations": self.empty_result_violations,
            "failures": self.failures,
        }


def write_retrieval_reports(
    snapshot: dict[str, Any] | None,
    output_dir: Path | str,
    *,
    status: str,
    metadata: dict | None = None,
    gate: GateResult | None = None,
    preflight: RetrievalPreflight | None = None,
) -> tuple[Path, Path]:
    """Write a versioned retrieval report, including blocked preflight runs."""
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "retrieval.json"
    markdown_path = report_dir / "retrieval.md"
    payload = {
        "report_schema_version": 1,
        "mode": "retrieval",
        "status": status,
        "metadata": metadata or {},
        "preflight": preflight.to_dict() if preflight else None,
        "gate": gate.to_dict() if gate else None,
        "summary": snapshot,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Retrieval Evaluation",
        "",
        f"- Status: **{status}**",
    ]
    if metadata:
        dataset = metadata.get("dataset", {})
        lines.extend([
            f"- Dataset: `{dataset.get('dataset_id', 'ad-hoc')}` / `{dataset.get('dataset_version', 'unversioned')}`",
            f"- Dataset SHA-256: `{dataset.get('sha256', 'unknown')}`",
            f"- Code revision: `{metadata.get('code', {}).get('git_revision') or 'unknown'}`",
            f"- External access: `{metadata.get('runtime', {}).get('external_access')}`",
        ])
    if preflight:
        lines.extend(["", "## Preflight", "", f"- Passed: `{preflight.passed}`"])
        lines.extend(f"- Issue: `{issue}`" for issue in preflight.issues)
        if preflight.index_version:
            lines.append(f"- Index version: `{preflight.index_version}`")
    if gate:
        lines.extend([
            "",
            "## Quality gate",
            "",
            f"- `{gate.name}`: **{'passed' if gate.passed else 'failed'}**",
        ])
        lines.extend(f"- {failure}" for failure in gate.failures)
    if snapshot is not None:
        lines.extend([
            "",
            "## Metrics",
            "",
            "```json",
            json.dumps(snapshot, ensure_ascii=False, indent=2),
            "```",
        ])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def _violates_constraints(product, case: RetrievalCase) -> bool:
    constraints = case.constraints
    if constraints.min_price is not None and (
        product.price is None or product.price < constraints.min_price
    ):
        return True
    if constraints.max_price is not None and (
        product.price is None or product.price > constraints.max_price
    ):
        return True
    if constraints.category and product.category.casefold() != constraints.category.casefold():
        return True
    excluded = {brand.casefold() for brand in constraints.excluded_brands}
    return product.brand.casefold() in excluded


def evaluate_retrieval(retriever, cases: list[RetrievalCase]) -> RetrievalEvaluation:
    evaluation = RetrievalEvaluation(cases=len(cases))
    for case in cases:
        result = retriever.search(RetrievalQuery(
            text=case.query,
            top_k=case.top_k,
            constraints=case.constraints.model_dump(),
        ))
        returned_ids = {product.product_id for product in result.products}
        relevant = set(case.relevant_ids)
        retrieved_relevant = returned_ids & relevant
        case_failures: list[str] = []

        if relevant:
            evaluation.relevant_cases += 1
            evaluation.relevant_total += len(relevant)
            evaluation.relevant_retrieved += len(retrieved_relevant)
            if retrieved_relevant:
                evaluation.hit_cases += 1
            else:
                case_failures.append("miss")
        elif returned_ids:
            evaluation.empty_result_violations += 1
            case_failures.append("expected_empty")

        forbidden = returned_ids & set(case.forbidden_ids)
        evaluation.forbidden_violations += len(forbidden)
        if forbidden:
            case_failures.append("forbidden_product")

        violations = sum(
            _violates_constraints(product, case) for product in result.products
        )
        evaluation.constraint_violations += violations
        if violations:
            case_failures.append("constraint_violation")

        relevant_products = [
            product for product in result.products if product.product_id in relevant
        ]
        observed_sources = {
            source for product in relevant_products for source in product.sources
        }
        evaluation.required_sources_total += len(case.required_sources)
        met_sources = set(case.required_sources) & observed_sources
        evaluation.required_sources_met += len(met_sources)
        if len(met_sources) != len(case.required_sources):
            case_failures.append("required_source_missing")

        if case_failures:
            evaluation.failures.append({
                "id": case.id,
                "reasons": sorted(set(case_failures)),
                "returned_ids": sorted(returned_ids),
            })
    return evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the active retrieval index")
    parser.add_argument("--cases", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--dataset-id", default="retrieval-contract-v1")
    parser.add_argument("--output-dir", default="evals/results")
    args = parser.parse_args()

    from src.config import config
    from src.retrieval.product_retriever import ProductRetriever

    cases_path = Path(args.cases) if args.cases else Path(__file__).with_name("retrieval_cases.jsonl")
    default_cases = cases_path.resolve() == Path(__file__).with_name("retrieval_cases.jsonl").resolve()
    manifest_path = Path(args.manifest) if args.manifest else (
        DEFAULT_DATASET_MANIFEST_PATH if default_cases else None
    )
    dataset = None
    manifest = None
    if manifest_path is not None:
        try:
            manifest, dataset = validate_dataset_artifact(
                manifest_path, args.dataset_id, cases_path
            )
        except ValueError as exc:
            print(f"retrieval evaluation manifest unavailable: {exc}")
            return 2
    try:
        cases = load_retrieval_cases(cases_path)
    except ValueError as exc:
        print(f"retrieval evaluation cases unavailable: {exc}")
        return 2

    started_at_monotonic, started_at = evaluation_start()
    preflight = preflight_retrieval(config.PRODUCT_DB_PATH, config.PRODUCT_CHROMA_DIR)
    metadata = build_run_metadata(
        mode="retrieval",
        started_at_monotonic=started_at_monotonic,
        started_at=started_at,
        dataset=dataset,
        cases_path=cases_path,
        model="embedding-eval",
        provider="local",
        external_access=False,
    )
    metadata["dataset"]["selected_case_count"] = len(cases)
    if not preflight.passed:
        json_path, markdown_path = write_retrieval_reports(
            None,
            args.output_dir,
            status="blocked",
            metadata=metadata,
            preflight=preflight,
        )
        print(
            "retrieval: blocked by preconditions "
            f"({', '.join(preflight.issues)}); reports: {json_path}, {markdown_path}"
        )
        return 2

    try:
        retriever = ProductRetriever(
            chroma_dir=config.PRODUCT_CHROMA_DIR,
            catalog_db=config.PRODUCT_DB_PATH,
        )
        snapshot = evaluate_retrieval(retriever, cases).snapshot()
    except Exception:
        # Preflight and reports must not expose provider, filesystem, or model details.
        blocked = RetrievalPreflight(
            passed=False,
            issues=["RETRIEVAL_INDEX_UNAVAILABLE"],
            index_version=preflight.index_version,
        )
        json_path, markdown_path = write_retrieval_reports(
            None,
            args.output_dir,
            status="blocked",
            metadata=metadata,
            preflight=blocked,
        )
        print(
            "retrieval: blocked while loading index; "
            f"reports: {json_path}, {markdown_path}"
        )
        return 2

    gate_spec = (
        get_quality_gate(manifest, args.dataset_id, "retrieval")
        if manifest is not None
        else None
    )
    if gate_spec is None:
        from evals.quality import QualityGateSpec

        gate_spec = QualityGateSpec(
            name="retrieval-safety",
            mode="retrieval",
            dataset_id=args.dataset_id,
            maximums={
                "forbidden_violations": 0,
                "constraint_violations": 0,
                "empty_result_violations": 0,
            },
        )
    gate = evaluate_quality_gate(snapshot, gate_spec)
    status = "passed" if gate.passed else "failed"
    json_path, markdown_path = write_retrieval_reports(
        snapshot,
        args.output_dir,
        status=status,
        metadata=metadata,
        gate=gate,
        preflight=preflight,
    )
    print(
        f"retrieval: {status}; gate={'passed' if gate.passed else 'failed'}; "
        f"reports: {json_path}, {markdown_path}"
    )
    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
