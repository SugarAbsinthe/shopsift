"""Audit catalog information quality without changing records or calling an LLM."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.catalog import DEFAULT_DATASET_DIR, canonical_json, load_dataset, sha256_file
from src.data.categories import REQUIRED as CATEGORY_FIELDS
from src.data.release import evaluate_release


ALIASES = {
    "处理器": ("处理器", "CPU型号", "CPU"),
    "内存": ("内存", "运行内存"),
    "存储": ("存储", "存储容量"),
    "屏幕尺寸": ("屏幕尺寸",),
    "屏幕分辨率": ("屏幕分辨率", "分辨率"),
    "屏幕刷新率": ("屏幕刷新率", "刷新率"),
    "屏幕类型": ("屏幕类型", "屏幕"),
    "重量": ("重量", "机身重量", "质量"),
    "显卡": ("显卡", "GPU型号"),
    "电池容量": ("电池容量",),
    "快充": ("有线快充", "充电功率"),
    "后置摄像头": ("后置摄像头",),
    "网络版本": ("网络版本", "网络类型", "网络模式"),
    "手写笔支持": ("是否支持专用触控笔", "手写笔支持"),
    "键盘支持": ("是否支持专用键盘", "键盘支持"),
    "佩戴方式": ("佩戴方式",),
    "降噪说明": ("主动降噪", "降噪类型", "降噪"),
    "连接方式": ("连接方式", "连接版本", "通讯系统"),
    "单次续航": ("耳机单次续航", "单次续航", "续航时间"),
    "含盒续航": ("整体续航", "含盒续航"),
    "防护": ("防尘防水", "防护等级"),
    "单耳重量": ("单耳重量",),
}
DECISION_FIELDS = {
    "笔记本电脑": ("处理器", "内存", "存储", "屏幕尺寸", "显卡", "重量", "屏幕刷新率", "电池容量"),
    "手机": ("处理器", "存储", "屏幕尺寸", "屏幕刷新率", "后置摄像头", "电池容量", "重量", "快充"),
    "平板电脑": ("处理器", "存储", "屏幕尺寸", "屏幕刷新率", "屏幕类型", "电池容量", "重量", "手写笔支持", "键盘支持", "网络版本"),
    "无线耳机": ("佩戴方式", "降噪说明", "连接方式", "单次续航", "含盒续航", "防护", "单耳重量", "重量"),
}
PRICE_BANDS = {
    "笔记本电脑": (4500, 8000, 12000), "手机": (2000, 4000, 8000),
    "平板电脑": (2000, 4000, 8000), "无线耳机": (200, 500, 1000),
    "显示器": (1000, 2000, 5000), "键盘": (300, 700, 1500), "鼠标": (200, 500, 1000),
}
IDENTITY_FIELDS = {"版本", "型号", "地区版本", "特色简介"}


def normalized_value(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold().replace("®", "").replace("™", "")


def decision_specs(product: dict) -> dict:
    """Count explicit recorded information; never infer absent facts from names."""
    specs = product["specs"]
    result = {dimension: specs[key] for dimension, keys in ALIASES.items()
              for key in [next((key for key in keys if specs.get(key)), None)] if key is not None}
    if product['category'] in {'显示器','键盘','鼠标'}:
        result.update({key:specs[key] for key in CATEGORY_FIELDS[product['category']] if specs.get(key)})
    return result


def references(rows: list[dict]) -> list[dict]:
    return [{"product_id": row["product_id"], "name": row["name"], "model_key": row["model_key"]}
            for row in rows]


def duplicate_groups(rows: list[dict], key, *, across_models=False) -> list[list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return [references(group) for group in groups.values() if len(group) > 1
            and (not across_models or len({r["model_key"] for r in group}) > 1)]


def analyze_records(products: list[dict], reviews: list[dict], sources: list[dict]) -> dict:
    summaries = {r["summary_key"]: r for r in reviews}
    reviews_by_product = defaultdict(list)
    for review in reviews:
        reviews_by_product[review["product_id"]].append(review)
    categories = {}
    for category in dict.fromkeys(p["category"] for p in products):
        rows = [p for p in products if p["category"] == category]
        models = defaultdict(list)
        for row in rows:
            models[row["model_key"]].append(row)
        fields = DECISION_FIELDS.get(category, CATEGORY_FIELDS.get(category, ()))
        storage_only = []
        for model, variants in models.items():
            if len(variants) < 2:
                continue
            keys = set().union(*(p["configuration"].keys() for p in variants))
            differences = {key for key in keys if len({normalized_value(p["configuration"].get(key, "")) for p in variants}) > 1}
            if differences and differences <= {"存储", "存储容量"}:
                storage_only.append({"model_key": model, "products": references(variants), "variant_count": len(variants)})
        bands = []
        lower = 0
        for upper in (*PRICE_BANDS.get(category, ()), None):
            selected = [p for p in rows if p["price"] > lower and (upper is None or p["price"] <= upper)]
            bands.append({"min_exclusive": lower, "max_inclusive": upper, "products": len(selected),
                          "models": len({p["model_key"] for p in selected}), "brands": dict(Counter(p["brand"] for p in selected))})
            if upper is not None:
                lower = upper
        points = [r for r in summaries.values() if r["model_key"] in models]
        no_reviews = [p for p in rows if not reviews_by_product[p["product_id"]]]
        categories[category] = {
            "products": len(rows), "models": len(models), "brand_counts": dict(Counter(p["brand"] for p in rows)),
            "brand_model_counts": dict(Counter(group[0]["brand"] for group in models.values())),
            "variant_rows_above_one_per_model": len(rows) - len(models),
            "storage_only_models": storage_only,
            "field_coverage": {field: sum(field in decision_specs(p) for p in rows) for field in fields},
            "field_value_counts": {field: dict(Counter(decision_specs(p)[field] for p in rows if field in decision_specs(p))) for field in fields},
            "spec_key_counts": dict(Counter(key for p in rows for key in p["specs"])),
            "spec_count_distribution": dict(Counter(len(p["specs"]) for p in rows)),
            "identity_only_specs": references([p for p in rows if not set(p["specs"]) - IDENTITY_FIELDS]),
            "price_bands": bands,
            "products_with_reviews": len(rows) - len(no_reviews), "products_without_reviews": references(no_reviews),
            "unique_summary_points": len(points), "sentiments": dict(Counter(r["sentiment"] for r in points)),
            "review_aspects": dict(Counter(r["aspect"] for r in points)),
            "subcategories": dict(Counter(p["subcategory"] for p in rows)),
        }
    return {
        "duplicates": {
            "product_ids": duplicate_groups(products, lambda p: p["product_id"]),
            "sku_keys": duplicate_groups(products, lambda p: p["sku_key"]),
            "same_model_configuration": duplicate_groups(products, lambda p: (p["model_key"], canonical_json(p["configuration"]))),
            "normalized_same_model_configuration": duplicate_groups(products, lambda p: (p["model_key"], canonical_json({k: normalized_value(v) for k, v in p["configuration"].items()}))),
            "names": duplicate_groups(products, lambda p: normalized_value(p["name"])),
            "identical_specs_across_models": duplicate_groups(products, lambda p: (p["category"], canonical_json(p["specs"])), across_models=True),
        },
        "core_missing_keys": [{"product_id": p["product_id"], "keys": [k for k in
            ("product_id", "name", "brand", "category", "subcategory", "price", "rating", "sales_count", "release_date", "description", "specs") if k not in p]}
            for p in products if any(k not in p for k in ("product_id", "name", "brand", "category", "subcategory", "price", "rating", "sales_count", "release_date", "description", "specs"))],
        "unknown_optional_fields": {field: sum(p.get(field) is None for p in products) for field in ("rating", "sales_count", "release_date")},
        "release_month_only_in_specs": [{"product_id": p["product_id"], "name": p["name"], "recorded_date": p["specs"]["发布日期"]}
            for p in products if p.get("release_date") is None and p["specs"].get("发布日期")],
        "config_spec_disagreements": [{"product_id": p["product_id"], "key": key, "configuration": value, "spec": p["specs"][key]}
            for p in products for key, value in p["configuration"].items()
            if key in p["specs"] and normalized_value(p["specs"][key]) != normalized_value(value)],
        "configuration_keys_not_repeated_in_specs": [{"product_id": p["product_id"], "keys": [key for key in p["configuration"] if key not in p["specs"]]}
            for p in products if any(key not in p["specs"] for key in p["configuration"])],
        "coarse_cpu_models": references([p for p in products if re.fullmatch(r"(?:i[3579]|r[3579]|ultra\s*[3579])", decision_specs(p).get("处理器", ""), re.I)]),
        "categories": categories,
        "reviews": {"associations": len(reviews), "unique_points": len(summaries),
                    "shared_association_rows": len(reviews) - len(summaries),
                    "sentiments": dict(Counter(r["sentiment"] for r in summaries.values())),
                    "source_types": dict(Counter(r["source_type"] for r in summaries.values())),
                    "duplicate_point_contents": [text for text, count in Counter(r["content"] for r in summaries.values()).items() if count > 1]},
        "sources": {"count": len(sources), "types": dict(Counter(s["source_type"] for s in sources)),
                    "url_titles": [s["source_id"] for s in sources if s["title"] == s["url"]],
                    "price_capture_times": sorted({s["collected_at"] for s in sources if s["source_id"] in {p["price_source_id"] for p in products}})},
    }


def audit_cache(sources: list[dict], directory: Path) -> dict:
    counts = Counter()
    unresolved = []
    for source in sources:
        base = directory / source["source_id"].removeprefix("src_")
        raw = base.with_suffix(".response")
        saved = base.with_suffix(".html")
        expected = source.get("page_sha256")
        archived = directory / 'snapshots' / (str(expected) + '.response')
        browser = base.with_suffix(".browser.txt")
        if source.get("capture_format") == "browser_text":
            if browser.exists() and sha256_file(browser) == expected:
                counts["browser_text_match"] += 1
            else:
                counts["browser_text_missing_or_changed"] += 1
                unresolved.append({"source_id": source["source_id"], "url": source["url"]})
        elif raw.exists() and sha256_file(raw) == expected:
            counts["raw_response_match"] += 1
        elif archived.exists() and sha256_file(archived) == expected:
            counts["archived_response_match"] += 1
        elif not saved.exists():
            counts["missing_cache"] += 1
            unresolved.append({"source_id":source['source_id'],"url":source['url'],"reason":"missing_cache"})
        else:
            body = saved.read_bytes()
            if hashlib.sha256(body).hexdigest() == expected:
                counts["saved_bytes_match"] += 1
            elif hashlib.sha256(body.replace(b"\r\n", b"\n")).hexdigest() == expected:
                counts["LF_restored_match"] += 1
            else:
                counts["original_bytes_not_reproducible"] += 1
                unresolved.append({"source_id": source["source_id"], "method": source.get("retrieval_method", "GET"), "url": source["url"]})
    return {"counts": dict(counts), "unresolved_sources": unresolved,
            "interpretation": "Decoded/reserialized caches are not necessarily original response bytes; mismatch alone does not prove false facts."}


def retrieval_probes(products: list[dict]) -> list[dict]:
    """Probe real local retrieval; no remote chat API or scripted LLM."""
    from src.config import config
    from src.retrieval.models import RetrievalQuery
    from src.retrieval.product_retriever import ProductRetriever
    plans = [
        ("laptop_office", "笔记本电脑", 8000, "办公出差用的轻薄笔记本，8000元以内，推荐几款不同型号方便比较。", []),
        ("laptop_gaming", "笔记本电脑", 13000, "游戏本，13000元以内，比较显卡性能、散热和噪声。", []),
        ("laptop_exclusion", "笔记本电脑", 4500, "4500元以内办公笔记本，不要荣耀。", ["HONOR"]),
        ("phone_entry", "手机", 2000, "2000元以内手机，比较屏幕、续航和拍照。", []),
        ("phone_camera", "手机", 5500, "5500元以内拍照手机，希望比较不同型号。", []),
        ("phone_compact", "手机", 8000, "8000元以内小屏轻便手机，希望单手使用舒适。", []),
        ("tablet_study", "平板电脑", 3000, "3000元以内上网课记笔记平板，比较手写笔和屏幕。", []),
        ("tablet_office", "平板电脑", 8000, "8000元以内平板轻办公，比较键盘支持和电脑文件兼容性。", []),
        ("tablet_media", "平板电脑", 4500, "4500元以内大屏影音平板，比较扬声器和屏幕刷新率。", []),
        ("audio_headband", "无线耳机", 500, "500元以内头戴式主动降噪耳机，用于通勤。", []),
        ("audio_open", "无线耳机", 800, "800元以内开放式耳夹耳机，运动时稳固，比较舒适度。", []),
        ("audio_commute", "无线耳机", 1000, "1000元以内入耳降噪耳机，比较续航和多设备切换。", []),
    ]
    by_id = {p["product_id"]: p for p in products}
    probes = []
    retriever = ProductRetriever(config.PRODUCT_CHROMA_DIR, embedding_model=config.EMBEDDING_MODEL, catalog_db=config.PRODUCT_DB_PATH)
    try:
        for identifier, category, maximum, query, excluded in plans:
            result = retriever.search(RetrievalQuery(text=query, top_k=5,
                constraints={"category": category, "max_price": maximum, "excluded_brands": excluded}))
            rows = [by_id[p.product_id] for p in result.products]
            counts = Counter(p["model_key"] for p in rows)
            probes.append({"id": identifier, "query": query, "category": category,
                "constraints": result.constraints.model_dump(), "index_version": retriever.index_version,
                "returned": [{"product_id": p["product_id"], "name": p["name"], "price": p["price"],
                              "model_key": p["model_key"], "brand": p["brand"]} for p in rows],
                "returned_count": len(rows), "distinct_models": len(counts), "distinct_brands": len({p["brand"] for p in rows}),
                "largest_model_share": max(counts.values()) / len(rows) if rows else None})
    finally:
        retriever.close()
    return probes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("evals/results/catalog-quality"))
    parser.add_argument("--source-cache", type=Path)
    parser.add_argument("--with-retrieval", action="store_true")
    parser.add_argument("--as-of", default=date.today().isoformat())
    parser.add_argument("--release-check", action="store_true", help="Enforce the 500-model publication scale/distribution gate; does not affect local pilot import.")
    parser.add_argument("--allow-pilot", action="store_true", help="Audit a structurally complete collection draft without accepting it for initialization.")
    args = parser.parse_args()
    manifest, _, validation = load_dataset(args.dataset_dir, require_complete=not args.allow_pilot)
    records = {name: [json.loads(line) for line in (args.dataset_dir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
               for name in ("products", "reviews", "sources")}
    result = {"evaluated_on": args.as_of, "catalog_version": manifest["dataset_version"],
              "catalog_fingerprint": manifest["catalog_fingerprint"], "structural_validation": validation,
              "interpretation": "Field presence, duplication and retrieval diversity diagnostics; not product quality ratings or proof that every source is accurate.",
              "analysis": analyze_records(**records)}
    result["release_readiness"] = evaluate_release(**records, quality_policy=manifest.get('quality_policy'))
    if manifest.get("category_capabilities"):
        result["category_capabilities"] = manifest["category_capabilities"]
    if args.source_cache:
        result["source_cache"] = audit_cache(records["sources"], args.source_cache)
    if args.with_retrieval:
        result["retrieval_probes"] = retrieval_probes(records["products"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "audit.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Audited {len(records['products'])} configurations; report: {target}")
    failed = result["release_readiness"]["status"] == "blocked"
    if args.source_cache:
        failed = failed or bool(result["source_cache"]["unresolved_sources"])
    return 2 if args.release_check and failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
