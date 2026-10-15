"""Factual record checks and compatibility checks for historical catalog policies."""
from __future__ import annotations

import re
import math
from collections import Counter
from src.data.categories import REQUIRED

POLICY = "quality-first-v2"
SOURCE_PREFERRED_POLICY = "source-preferred-v1"


def product_issues(product: dict, *, policy: str | None = None) -> list[str]:
    # The accelerated catalog permits absent specifications. Absence must stay
    # absent (and fail hard filters); it is never filled with guessed facts.
    if policy == SOURCE_PREFERRED_POLICY:
        issues = []
        for field in ('name', 'brand', 'category', 'sku_key', 'model_key', 'configuration',
                      'specs', 'price_source_id', 'spec_source_ids'):
            if not product.get(field):
                issues.append(f'缺少商品基本字段：{field}')
        if product.get('category') not in REQUIRED:
            issues.append('未登记的商品品类')
        price = product.get('price')
        if (not isinstance(price, (int, float)) or isinstance(price, bool)
                or not math.isfinite(price) or price <= 0):
            issues.append('缺少有效正数人民币报价')
        return issues
    specs = product["specs"]
    issues = [f"缺少可核实的核心规格：{key}" for key in REQUIRED.get(product["category"], ())
              if not specs.get(key) or re.match(r"^(?:未知|未确认|未核实|不详|暂无|N/A)(?:$|[（(:：])", specs[key].strip(), re.I)]
    cpu = specs.get("处理器", "")
    if product["category"] == "笔记本电脑" and re.fullmatch(r"(?:i[3579]|r[3579]|ultra\s*[3579])", cpu, re.I):
        issues.append("处理器仅有家族简称，未确认完整型号")
    return issues


def category_capabilities(products: list[dict], reviews: list[dict], sources: list[dict], *, dialogue_passed: bool = False, scene_matrix: dict | None = None) -> dict:
    """Reproduce quality-first-v3 pilot metadata, including its original limits.

    Only the historical-policy loader calls this compatibility check. Current
    source-preferred catalogs use release.py for scale and manifest capability
    disclosures; the old 12--18 configuration limits do not apply to them.
    """
    from src.retrieval.models import canonical_brand
    source_map = {s['source_id']: s for s in sources}
    result = {}
    for category in ('显示器', '键盘', '鼠标'):
        rows = [p for p in products if p['category'] == category]
        if not rows:
            continue
        models = {p['model_key']: canonical_brand(p['brand']) for p in rows}
        model_brands = Counter(models.values())
        brands = Counter(canonical_brand(p['brand']) for p in rows)
        prices = [source_map[p['price_source_id']] for p in rows]
        sellers = Counter(s.get('seller_id') or 'unknown' for s in prices)
        platforms = Counter(s.get('platform') or 'unknown' for s in prices)
        points = {r['summary_key']: r for r in reviews if r['model_key'] in models}
        sites = Counter(source_map[r['source_id']]['site'] for r in points.values())
        def covered(key, value):
            return len({r['aspect'] for r in reviews if r.get(key) == value}) >= 2
        model_coverage = sum(covered('model_key', k) for k in models) / len(models)
        config_coverage = sum(covered('product_id', p['product_id']) for p in rows) / len(rows)
        gaps = []
        if len(models) < 12: gaps.append('型号不足12')
        if len(rows) > 18: gaps.append('配置超过本轮18条上限')
        if len(brands) < 4: gaps.append('品牌不足4')
        if max(model_brands.values()) / len(models) > .30 or max(brands.values()) / len(rows) > .30: gaps.append('品牌型号或配置占比超过30%')
        if max(sellers.values()) / len(rows) > .50: gaps.append('单一商家超过50%')
        if max(platforms.values()) / len(rows) > .60: gaps.append('单一平台超过60%')
        if sites and max(sites.values()) / len(points) > .50: gaps.append('单一体验站点超过50%')
        if min(model_coverage, config_coverage) < .80: gaps.append('两维度型号或配置体验覆盖不足80%')
        if any(not s.get('seller_id') for s in prices): gaps.append('商家身份缺口')
        if not dialogue_passed: gaps.append('真实多轮对话尚未全部通过')
        scenes = (scene_matrix or {}).get(category, {})
        scenes_valid = len(scenes) >= 3
        for keys in scenes.values():
            representatives = {p['model_key']: p for p in rows if p['model_key'] in keys}
            if len(representatives) < 3 or len({canonical_brand(p['brand']) for p in representatives.values()}) < 2:
                scenes_valid = False
                continue
            shared = set.intersection(*(set(p['specs']) for p in representatives.values()))
            differences = sum(len({p['specs'][field] for p in representatives.values()}) > 1 for field in shared)
            if differences < 2:
                scenes_valid = False
        if not scenes_valid: gaps.append('场景矩阵尚未达到每场景3型号2品牌及两项规格差异')
        result[category] = {'status': 'trial' if gaps else 'supported', 'products': len(rows), 'models': len(models),
                            'brand_models': dict(model_brands), 'brand_configurations': dict(brands),
                            'price_sellers': dict(sellers), 'price_platforms': dict(platforms),
                            'experience_sites': dict(sites), 'unique_summaries': len(points),
                            'model_experience_coverage': round(model_coverage,4),
                            'configuration_experience_coverage': round(config_coverage,4), 'gaps': gaps}
    return result
