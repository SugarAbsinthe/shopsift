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
from src.retrieval.models import RetrievalQuery, canonical_brand
from src.retrieval.index_manifest import resolve_collection_names, load_index_manifest, MANIFEST_FILENAME
from src.data.catalog import catalog_fingerprint


REAL_DATASET_ID = "retrieval-real-catalog-v4"
REAL_DATASET_DIRECTORY = Path(__file__).resolve().parent / "datasets" / "catalog"


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


def preflight_retrieval(catalog_db: Path | str, chroma_dir: Path | str, *,
                        expected_catalog_fingerprint: str | None = None,
                        embedding_model: str | None = None) -> RetrievalPreflight:
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
    active_manifest = load_index_manifest(chroma_path)
    if (chroma_path / MANIFEST_FILENAME).exists() and active_manifest is None:
        issues.append("RETRIEVAL_INDEX_MANIFEST_INVALID")
    if active_manifest and embedding_model and active_manifest.get("embedding_model") != embedding_model:
        issues.append("RETRIEVAL_EMBEDDING_MODEL_MISMATCH")
    expected_index = active_manifest.get("catalog_fingerprint") if active_manifest else None
    if catalog_path.is_file() and (expected_catalog_fingerprint or expected_index):
        try:
            observed = catalog_fingerprint(catalog_path, catalog_path.parent / "product_reviews.db")
            if expected_catalog_fingerprint and observed != expected_catalog_fingerprint.lower():
                issues.append("RETRIEVAL_CATALOG_DATASET_MISMATCH")
            if expected_index and observed != expected_index:
                issues.append("RETRIEVAL_CATALOG_INDEX_MISMATCH")
        except (OSError, ValueError, sqlite3.Error):
            issues.append("RETRIEVAL_CATALOG_INVALID")
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
    exact_cases: int = 0
    exact_hits: int = 0
    required_sources_total: int = 0
    required_sources_met: int = 0
    returned_products: int = 0
    sourced_products: int = 0
    diversity_violations: int = 0
    experience_violations: int = 0
    observations: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        other = [row for row in self.observations if "exact_config" not in row["tags"]]
        other_relevant = [row for row in other if row["relevant_ids"]]
        other_gold = sum(len(row["relevant_ids"]) for row in other_relevant)
        other_hits = sum(len(row["hit_ids"]) for row in other_relevant)
        other_sources = sum(len(row["required_sources"]) for row in other)
        other_met = sum(len(set(row["required_sources"]) & set(row["observed_sources"])) for row in other)
        return {
            "cases": self.cases,
            "diversity_violations": self.diversity_violations,
            "experience_violations": self.experience_violations,
            "hit_rate": round(self.hit_cases / self.relevant_cases, 4)
            if self.relevant_cases else None,
            "recall_at_k": round(self.relevant_retrieved / self.relevant_total, 4)
            if self.relevant_total else None,
            "source_coverage": round(
                self.required_sources_met / self.required_sources_total, 4
            ) if self.required_sources_total else None,
            "provenance_coverage": round(self.sourced_products / self.returned_products, 4)
            if self.returned_products else None,
            "forbidden_violations": self.forbidden_violations,
            "constraint_violations": self.constraint_violations,
            "empty_result_violations": self.empty_result_violations,
            "exact_cases": self.exact_cases,
            "exact_hits": self.exact_hits,
            "exact_hit_rate": round(self.exact_hits / self.exact_cases, 4) if self.exact_cases else None,
            "non_exact_metrics": {
                "cases": len(other), "nonempty_gold_cases": len(other_relevant),
                "hit_rate": round(sum(bool(row["hit_ids"]) for row in other_relevant) / len(other_relevant), 4) if other_relevant else None,
                "recall_at_k": round(other_hits / other_gold, 4) if other_gold else None,
                "source_coverage": round(other_met / other_sources, 4) if other_sources else None,
            },
            "failures": self.failures,
            "observations": self.observations,
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
    excluded = {canonical_brand(brand) for brand in constraints.excluded_brands}
    from src.data.categories import matches_spec_filters
    return canonical_brand(product.brand) in excluded or not matches_spec_filters(product.normalized_specs, constraints.spec_filters)


def evaluate_retrieval(retriever, cases: list[RetrievalCase]) -> RetrievalEvaluation:
    evaluation = RetrievalEvaluation(cases=len(cases))
    for case in cases:
        result = retriever.search(RetrievalQuery(
            text=case.query,
            top_k=case.top_k,
            constraints=case.constraints.model_dump(),
        ))
        returned_ids = {product.product_id for product in result.products}
        evaluation.returned_products += len(result.products)
        evaluation.sourced_products += sum(
            bool(product.provenance.get("price_source", {}).get("url"))
            and bool(product.provenance.get("price_source", {}).get("collected_at"))
            and bool(product.provenance.get("spec_sources"))
            for product in result.products
        )
        relevant = set(case.relevant_ids)
        retrieved_relevant = returned_ids & relevant
        if "exact_config" in case.tags:
            evaluation.exact_cases += 1
            evaluation.exact_hits += int(bool(retrieved_relevant))
        case_failures: list[str] = []
        distinct_models = len({p.provenance.get("model_key") or p.product_id for p in result.products})
        if case.min_distinct_models is not None and distinct_models < case.min_distinct_models:
            evaluation.diversity_violations += 1
            case_failures.append("model_diversity")
        if case.review_product_id is not None:
            from src.agent.shopping_tools import create_get_reviews
            from src.data.catalog import load_provenance
            with sqlite3.connect(Path(retriever.catalog_db).parent / "product_reviews.db") as conn:
                ids = [row[0] for row in conn.execute("SELECT review_id FROM product_reviews WHERE product_id=? AND aspect=?", (case.review_product_id, case.review_aspect))]
            provenance = load_provenance(Path(retriever.catalog_db).parent / "product_reviews.db", "review_provenance", ids)
            observed_summaries = {v["summary_key"] for v in provenance.values()}
            output = create_get_reviews(str(Path(retriever.catalog_db).parent / "product_reviews.db"), retriever.chroma_dir).invoke({"product_id":case.review_product_id,"aspect":case.review_aspect,"top_k":5})
            if observed_summaries != set(case.expected_summary_keys) or (observed_summaries and "https://" not in output):
                evaluation.experience_violations += 1
                case_failures.append("experience_source_or_scope")

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

        evaluation.observations.append({
            "id": case.id, "tags": case.tags,
            "returned_ids": [product.product_id for product in result.products],
            "relevant_ids": sorted(relevant), "hit_ids": sorted(retrieved_relevant),
            "recall_at_k": round(len(retrieved_relevant) / len(relevant), 4) if relevant else None,
            "required_sources": case.required_sources, "observed_sources": sorted(observed_sources),
            "distinct_models": distinct_models,
        })

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
    parser.add_argument("--dataset-id", default=REAL_DATASET_ID)
    parser.add_argument("--output-dir", default="evals/results")
    args = parser.parse_args()

    from src.config import config
    from src.retrieval.product_retriever import ProductRetriever

    bundled_real = REAL_DATASET_DIRECTORY / "retrieval_cases.jsonl"
    legacy_cases = Path(__file__).with_name("retrieval_cases.jsonl")
    default_path = bundled_real if args.dataset_id == REAL_DATASET_ID else legacy_cases
    cases_path = Path(args.cases) if args.cases else default_path
    manifest_path = Path(args.manifest) if args.manifest else (
        REAL_DATASET_DIRECTORY / "manifest.json" if cases_path.resolve() == bundled_real.resolve()
        else DEFAULT_DATASET_MANIFEST_PATH if cases_path.resolve() == legacy_cases.resolve() else None
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
    preflight = preflight_retrieval(
        config.PRODUCT_DB_PATH, config.PRODUCT_CHROMA_DIR,
        expected_catalog_fingerprint=dataset.catalog_fingerprint if dataset else None,
        embedding_model=config.EMBEDDING_MODEL,
    )
    metadata = build_run_metadata(
        mode="retrieval",
        started_at_monotonic=started_at_monotonic,
        started_at=started_at,
        dataset=dataset,
        cases_path=cases_path,
        model=config.EMBEDDING_MODEL,
        provider="local",
        external_access=False,
    )
    metadata["dataset"]["selected_case_count"] = len(cases)
    metadata["catalog"] = {"dataset_version": dataset.catalog_version if dataset else None,
                           "fingerprint": dataset.catalog_fingerprint if dataset else None,
                           "index_version": preflight.index_version,
                           "embedding_model": config.EMBEDDING_MODEL}
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

    retriever = None
    try:
        retriever = ProductRetriever(
            chroma_dir=config.PRODUCT_CHROMA_DIR,
            catalog_db=config.PRODUCT_DB_PATH,
            embedding_model=config.EMBEDDING_MODEL,
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
    finally:
        if retriever is not None:
            retriever.close()

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
