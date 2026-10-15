"""Catalog validation, provenance and deterministic logical content hashes."""

from __future__ import annotations

import hashlib
import math
import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator
from src.data.categories import CATEGORIES, LEGACY_CATEGORIES, FILTER_FIELDS, ENUM_VALUES


DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[2] / "datasets" / "real_catalog"
PRODUCT_FIELDS = (
    "product_id", "name", "brand", "category", "subcategory", "price", "rating",
    "sales_count", "release_date", "description", "specs",
)
REVIEW_FIELDS = ("review_id", "product_id", "sentiment", "aspect", "content")


class CatalogError(ValueError):
    """Invalid, incomplete or inconsistent catalog input."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def historical_dump(self) -> dict:
        return self.model_dump()


class SourceRecord(Record):
    source_id: str = Field(pattern=r"^src_[a-zA-Z0-9_-]+$")
    url: str
    title: str = Field(min_length=1)
    site: str = Field(min_length=1)
    source_type: Literal["official", "merchant", "user_post", "platform_review", "editorial_review"]
    collected_at: str
    published_at: str | None = None
    page_sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    retrieval_url: str | None = None
    retrieval_method: Literal["GET", "POST"] = "GET"
    retrieval_parameters: dict[str, str | int] = Field(default_factory=dict)
    facts: list[str] = Field(default_factory=list)
    platform: str | None = None
    seller_id: str | None = None
    seller_name: str | None = None
    capture_format: Literal["http_response", "browser_text"] | None = None
    locators: dict[str, str] = Field(default_factory=dict)
    disclosure: str | None = None

    def historical_dump(self) -> dict:
        result = self.model_dump()
        for field in ("platform", "seller_id", "seller_name", "capture_format", "disclosure"):
            if result[field] is None:
                result.pop(field)
        if not result["locators"]:
            result.pop("locators")
        return result

    @field_validator("url")
    @classmethod
    def public_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("source URL must be HTTPS without credentials")
        return value

    @field_validator("retrieval_url")
    @classmethod
    def public_retrieval_url(cls, value: str | None) -> str | None:
        return cls.public_url(value) if value is not None else None

    @field_validator("collected_at")
    @classmethod
    def timestamp(cls, value: str) -> str:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("collection timestamp must include a timezone")
        return value


class ProductRecord(Record):
    product_id: int = Field(gt=0, le=1_000_000_000, strict=True)
    name: str = Field(min_length=1)
    brand: str = Field(min_length=1)
    category: str
    subcategory: str = Field(min_length=1)
    price: float = Field(gt=0, allow_inf_nan=False)
    rating: float | None = Field(default=None, ge=0, le=5, allow_inf_nan=False)
    sales_count: int | None = Field(default=None, ge=0, strict=True)
    release_date: str | None = None
    description: str = Field(min_length=1)
    specs: dict[str, str] = Field(min_length=1)
    sku_key: str = Field(min_length=1)
    model_key: str = Field(min_length=1)
    configuration: dict[str, str] = Field(min_length=1)
    price_source_id: str
    spec_source_ids: list[str] = Field(min_length=1)
    field_source_ids: dict[str, list[str]] = Field(default_factory=dict)
    availability: Literal["listed", "in_stock", "out_of_stock", "unconfirmed"] = "listed"
    normalized_specs: dict[str, Any] = Field(default_factory=dict)

    def historical_dump(self) -> dict:
        result = self.model_dump()
        if not self.normalized_specs:
            result.pop("normalized_specs")
        return result

    @field_validator("category")
    @classmethod
    def registered_category(cls, value):
        if value not in CATEGORIES:
            raise ValueError("unregistered product category")
        return value

    @field_validator("normalized_specs")
    @classmethod
    def registered_specs(cls, value):
        for key, item in value.items():
            kind = FILTER_FIELDS.get(key)
            valid = (kind == "number" and isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item) and item >= 0
                     or kind == "string" and isinstance(item, str) and bool(item)
                     or kind == "bool" and isinstance(item, bool)
                     or kind == "set" and isinstance(item, list) and all(isinstance(v, str) for v in item))
            if not valid:
                raise ValueError(f"Invalid normalized specification: {key}")
            if key in ENUM_VALUES and (not set(item if isinstance(item,list) else [item]) <= ENUM_VALUES[key]):
                raise ValueError(f"Unregistered specification enum: {key}")
        return value

    @field_validator("name", "description", "brand", "subcategory", "sku_key", "model_key")
    @classmethod
    def plain_text(cls, value: str) -> str:
        if not value.strip() or "<" in value or ">" in value or "\x00" in value:
            raise ValueError("catalog text must be nonblank plain text")
        return value.strip()

    @field_validator("specs", "configuration")
    @classmethod
    def plain_values(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not key.strip() or not item.strip() or any(c in key + item for c in "<>\x00")
               for key, item in value.items()):
            raise ValueError("specifications and configurations must contain plain text")
        return value


class ReviewRecord(Record):
    review_id: int = Field(gt=0, strict=True)
    product_id: int = Field(gt=0, strict=True)
    sentiment: Literal["positive", "negative", "neutral", "mixed"]
    aspect: str = Field(min_length=1)
    content: str = Field(min_length=1, max_length=300)
    summary_key: str = Field(min_length=1)
    source_id: str
    source_type: Literal["user_post", "platform_review", "editorial_review"]
    scope: Literal["model", "sku"]
    model_key: str = Field(min_length=1)

    @field_validator("content", "aspect")
    @classmethod
    def plain_text(cls, value: str) -> str:
        if not value.strip() or "<" in value or ">" in value or "\x00" in value:
            raise ValueError("summary must be nonblank plain text")
        return value.strip()


def _jsonl(path: Path, model: type[Record]) -> list:
    rows = []
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    rows.append(model.model_validate_json(line))
                except ValueError as exc:
                    raise CatalogError(f"{path.name}:{number}: invalid {model.__name__}: {exc}") from exc
    except OSError as exc:
        raise CatalogError(f"Cannot read {path.name}") from exc
    return rows


def _unique(rows: list, attribute: str) -> None:
    values = [getattr(row, attribute) for row in rows]
    if len(values) != len(set(values)):
        raise CatalogError(f"Duplicate {attribute}")


def dataset_counts(products: list[ProductRecord], reviews: list[ReviewRecord], sources: list[SourceRecord]) -> dict:
    by_model: dict[str, set[str]] = {}
    for review in reviews:
        by_model.setdefault(review.model_key, set()).add(review.summary_key)
    categories = {}
    for category in CATEGORIES:
        selected = [p for p in products if p.category == category]
        if not selected and category not in LEGACY_CATEGORIES:
            continue
        models = {p.model_key for p in selected}
        covered = sum(len(by_model.get(key, set())) >= 2 for key in models)
        categories[category] = {
            "products": len(selected), "models": len(models),
            "brands": len({p.brand for p in selected}), "covered_models": covered,
            "experience_coverage": round(covered / len(models), 4) if models else 0.0,
        }
    return {
        "products": len(products), "reviews": len(reviews), "sources": len(sources),
        "unique_summaries": len({r.summary_key for r in reviews}), "categories": categories,
    }


def readiness_issues(counts: dict) -> list[str]:
    issues = []
    for category, value in counts["categories"].items():
        for field, minimum in (("products", 50), ("models", 20), ("brands", 4), ("experience_coverage", 0.8)):
            if value[field] < minimum:
                issues.append(f"{category}: {field}={value[field]}, required >= {minimum}")
    return issues


def _fingerprint_tables(tables: dict[str, list[dict]]) -> str:
    normalized = {}
    for table, rows in tables.items():
        normalized[table] = []
        for raw in rows:
            row = dict(raw)
            for key in ("specs", "metadata"):
                if key in row and isinstance(row[key], str):
                    row[key] = json.loads(row[key])
            normalized[table].append(row)
    return hashlib.sha256(canonical_json(normalized).encode("utf-8")).hexdigest()


def project_dataset(products: list[ProductRecord], reviews: list[ReviewRecord], sources: list[SourceRecord]) -> dict:
    source_map = {s.source_id: s.historical_dump() for s in sources}
    tables = {name: [] for name in ("products", "product_reviews", "product_provenance", "review_provenance", "catalog_sources")}
    for product in sorted(products, key=lambda p: p.product_id):
        row = {field: getattr(product, field) for field in PRODUCT_FIELDS}
        row["release_date"] = row["release_date"] or ""
        row["specs"] = canonical_json(row["specs"])
        tables["products"].append(row)
        metadata = {
            "sku_key": product.sku_key, "model_key": product.model_key,
            "configuration": product.configuration, "availability": product.availability,
            "price_source": source_map[product.price_source_id],
            "spec_sources": [source_map[key] for key in product.spec_source_ids],
            "field_sources": {field: [source_map[key] for key in keys]
                              for field, keys in product.field_source_ids.items()},
        }
        if product.normalized_specs:
            metadata["normalized_specs"] = product.normalized_specs
        tables["product_provenance"].append({
            "product_id": product.product_id, "sku_key": product.sku_key,
            "model_key": product.model_key, "metadata": canonical_json(metadata),
        })
    for review in sorted(reviews, key=lambda r: r.review_id):
        tables["product_reviews"].append({field: getattr(review, field) for field in REVIEW_FIELDS})
        source = source_map[review.source_id]
        metadata = {
            "summary_key": review.summary_key, "source_id": review.source_id,
            "source_url": source["url"], "source_title": source["title"],
            "source_type": review.source_type, "collected_at": source["collected_at"],
            "published_at": source["published_at"], "scope": review.scope, "model_key": review.model_key,
        }
        tables["review_provenance"].append({
            "review_id": review.review_id, "summary_key": review.summary_key,
            "metadata": canonical_json(metadata),
        })
    tables["catalog_sources"] = [{"source_id": s.source_id, "metadata": canonical_json(s.historical_dump())}
                                for s in sorted(sources, key=lambda s: s.source_id)]
    return tables


def load_dataset(directory: Path | str, *, require_complete: bool = False) -> tuple[dict, dict, dict]:
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["schema_version"] != 1 or manifest["status"] not in {"pilot", "complete"}:
            raise CatalogError("Unsupported catalog manifest")
        for name in ("products.jsonl", "reviews.jsonl", "sources.jsonl"):
            if sha256_file(directory / name) != manifest["files"][name]["sha256"].lower():
                raise CatalogError(f"Catalog checksum mismatch: {name}")
        for name in ("quarantine.jsonl", "rejected_reviews.jsonl"):
            if name in manifest["files"] and sha256_file(directory / name) != manifest["files"][name]["sha256"].lower():
                raise CatalogError(f"Catalog checksum mismatch: {name}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CatalogError(f"Invalid catalog manifest: {exc}") from exc
    products = _jsonl(directory / "products.jsonl", ProductRecord)
    reviews = _jsonl(directory / "reviews.jsonl", ReviewRecord)
    sources = _jsonl(directory / "sources.jsonl", SourceRecord)
    if not products:
        raise CatalogError("Product catalog is empty")
    for rows, attributes in ((products, ("product_id", "sku_key")), (reviews, ("review_id",)), (sources, ("source_id",))):
        for attribute in attributes:
            _unique(rows, attribute)
    product_map = {p.product_id: p for p in products}
    source_map = {s.source_id: s for s in sources}
    configurations = set()
    model_owners = {}
    for product in products:
        owner = (product.brand, product.category)
        if model_owners.setdefault(product.model_key, owner) != owner:
            raise CatalogError("A model key cannot span brands or categories")
        variant = (product.model_key, canonical_json(product.configuration))
        if variant in configurations:
            raise CatalogError(f"Duplicate hardware configuration: {product.sku_key}")
        configurations.add(variant)
        source_ids = [product.price_source_id, *product.spec_source_ids]
        source_ids.extend(key for keys in product.field_source_ids.values() for key in keys)
        if any(key not in source_map for key in source_ids):
            raise CatalogError(f"Unknown product source: {product.product_id}")
        if source_map[product.price_source_id].source_type not in {"official", "merchant"}:
            raise CatalogError("A price must come from an official or merchant listing")
        for field in ("rating", "sales_count", "release_date"):
            if getattr(product, field) is not None and not product.field_source_ids.get(field):
                raise CatalogError(f"{product.product_id}: {field} lacks a source")
    summaries = {}
    review_pairs = set()
    for review in reviews:
        product = product_map.get(review.product_id)
        source = source_map.get(review.source_id)
        if not product or product.model_key != review.model_key or not source or source.source_type != review.source_type:
            raise CatalogError(f"Invalid review product/model/source reference: {review.review_id}")
        pair = (review.product_id, review.summary_key)
        if pair in review_pairs:
            raise CatalogError("Duplicate summary for a product")
        review_pairs.add(pair)
        identity = (review.model_key, review.source_id, review.aspect, review.content, review.sentiment, review.scope)
        if review.summary_key in summaries and summaries[review.summary_key] != identity:
            raise CatalogError("A summary key cannot refer to different observations")
        summaries[review.summary_key] = identity
    counts = dataset_counts(products, reviews, sources)
    if counts != manifest.get("counts"):
        raise CatalogError("Catalog counts differ from manifest")
    policy = manifest.get("quality_policy")
    if policy is not None:
        from src.data.quality import POLICY, SOURCE_PREFERRED_POLICY, product_issues
        if policy not in {POLICY, "quality-first-v3", SOURCE_PREFERRED_POLICY}:
            raise CatalogError("Unsupported catalog quality policy")
        if policy == "quality-first-v3":
            from src.data.categories import REQUIRED
            for product in products:
                if product.category in LEGACY_CATEGORIES:
                    continue
                for field in ("price", "configuration", *("specs." + key for key in REQUIRED[product.category]), *("normalized_specs." + key for key in product.normalized_specs)):
                    if not product.field_source_ids.get(field):
                        raise CatalogError(f"{product.product_id}: {field} lacks field provenance")
                    if any(not source_map[sid].page_sha256 or not source_map[sid].locators.get(field) for sid in product.field_source_ids[field]):
                        raise CatalogError(f"{product.product_id}: {field} lacks hash or evidence locator")
                price = source_map[product.price_source_id]
                if not price.platform or not price.seller_id or not price.seller_name:
                    raise CatalogError(f"{product.product_id}: price lacks platform/seller identity")
            if manifest.get("category_capabilities"):
                from src.data.quality import category_capabilities
                observed = category_capabilities([p.historical_dump() for p in products], [r.model_dump() for r in reviews],
                                                [s.historical_dump() for s in sources],
                                                dialogue_passed=manifest.get("validation", {}).get("live_dialogue") == "passed",
                                                scene_matrix=manifest.get("scene_matrix"))
                if observed != manifest["category_capabilities"]:
                    raise CatalogError("Category capabilities differ from verified coverage")
        issues = [f"{p.product_id}: {issue}" for p in products
                  for issue in product_issues(p.model_dump(), policy=policy)]
        issues.extend(f"{category}: no qualified products" for category in LEGACY_CATEGORIES
                      if not counts["categories"][category]["products"])
    else:
        issues = readiness_issues(counts)
    if (require_complete or manifest["status"] == "complete") and (manifest["status"] != "complete" or issues):
        raise CatalogError("Catalog is not ready for first-run import: " + "; ".join(issues))
    tables = project_dataset(products, reviews, sources)
    fingerprint = _fingerprint_tables(tables)
    if fingerprint != manifest.get("catalog_fingerprint"):
        raise CatalogError("Catalog logical fingerprint mismatch")
    report = {"dataset_version": manifest["dataset_version"], "status": manifest["status"],
              "catalog_fingerprint": fingerprint, "counts": counts, "readiness_issues": issues}
    if policy:
        report.update({"quality_policy": policy, "capability_status": manifest.get("capability_status", "trial")})
    return manifest, tables, report


def catalog_fingerprint(products_db: Path | str, reviews_db: Path | str) -> str:
    """Hash logical rows, not SQLite files, so imports are reproducible."""
    tables = {}
    for path, names in ((products_db, ("products", "product_provenance", "catalog_sources")),
                        (reviews_db, ("product_reviews", "review_provenance"))):
        with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table in names:
                if table in existing:
                    # All identifiers here are constants, never external SQL input.
                    tables[table] = [dict(row) for row in conn.execute(f'SELECT * FROM "{table}" ORDER BY 1')]
                elif table in {"products", "product_reviews"}:
                    raise CatalogError(f"Missing {table} table")
                else:
                    tables[table] = []
    return _fingerprint_tables(tables)


def load_provenance(db_path: Path | str, table: str, ids: list[int]) -> dict[int, dict]:
    """Read optional source tables; legacy catalogs continue to work."""
    if table not in {"product_provenance", "review_provenance"}:
        raise ValueError("Unknown provenance table")
    if not ids:
        return {}
    key = "product_id" if table == "product_provenance" else "review_id"
    with closing(sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
            return {}
        placeholders = ",".join("?" for _ in ids)
        return {int(row[0]): json.loads(row[1]) for row in conn.execute(
            f'SELECT {key}, metadata FROM {table} WHERE {key} IN ({placeholders})', ids
        )}


def value_or_unknown(value: Any) -> str:
    return "未知" if value is None or value == "" else str(value)


def format_product_sources(provenance: dict) -> str:
    if not provenance:
        return ""
    price = provenance.get("price_source", {})
    lines = []
    if price:
        lines.append(f"价格快照: {price.get('collected_at', '未知')} | 来源: {price.get('url', '')}")
        if price.get("seller_name"):
            lines.append(f"销售商: {price['seller_name']} | 平台: {price.get('platform') or '未知'}")
    if provenance.get("availability"):
        labels = {"listed": "采集时在架，库存未确认", "in_stock": "采集时有库存", "out_of_stock": "采集时无库存", "unconfirmed": "可售状态未确认"}
        lines.append("可售状态: " + labels.get(provenance["availability"], "未知"))
    urls = dict.fromkeys(source.get("url", "") for source in provenance.get("spec_sources", []))
    if urls:
        lines.append("规格来源: " + " | ".join(url for url in urls if url))
    return "\n".join(lines)


def format_review_source(metadata: dict) -> str:
    labels = {"user_post": "用户帖子", "platform_review": "平台评价", "editorial_review": "原创评测"}
    label = labels.get(metadata.get("source_type"), "体验资料（旧数据，来源类型未知）")
    scope = {"model": "型号通用体验", "sku": "配置体验"}.get(metadata.get("scope"), "适用范围未知")
    url = metadata.get("source_url", "")
    return f"{label} / {scope}" + (f" | 来源: {url}" if url else "")
