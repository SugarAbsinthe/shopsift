"""Explicit publication gate; local pilots remain importable below this gate."""
from collections import Counter
from pathlib import Path
import json

from src.data.categories import CATEGORIES
from src.data.quality import SOURCE_PREFERRED_POLICY, product_issues
from src.retrieval.models import canonical_brand

DEFAULT_RELEASE_POLICY = Path(__file__).resolve().parents[2] / 'datasets/real_catalog/release_policy.json'


def evaluate_release(products: list[dict], reviews: list[dict], sources: list[dict], policy: dict | None = None,
                     *, quality_policy: str | None = None) -> dict:
    policy = policy or json.loads(DEFAULT_RELEASE_POLICY.read_text(encoding='utf8'))
    source_map = {s['source_id']: s for s in sources}
    qualified = [p for p in products if not product_issues(p, policy=quality_policy)]
    models = {p['model_key'] for p in qualified}
    issues = []
    if len(qualified) != len(products):
        issues.append('unqualified_core_fields')
    if len(models) < policy['minimum_models']:
        issues.append('minimum_models_not_met')
    categories = {}
    for category in CATEGORIES:
        rows = [p for p in qualified if p['category'] == category]
        model_brands = {p['model_key']: canonical_brand(p['brand']) for p in rows}
        brands = Counter(model_brands.values())
        configuration_brands = Counter(canonical_brand(p['brand']) for p in rows)
        prices = [source_map.get(p.get('price_source_id'), {}) for p in rows]
        sellers = Counter(s.get('seller_id') or 'unknown' for s in prices)
        platforms = Counter(s.get('platform') or 'unknown' for s in prices)
        summaries = {r['summary_key']:r for r in reviews if r['model_key'] in model_brands}
        sites = Counter(source_map.get(r['source_id'], {}).get('site') or 'unknown' for r in summaries.values())
        def share(counts):
            return max(counts.values(),default=0) / sum(counts.values()) if counts else 0
        failures = []
        if len(model_brands) < policy['minimum_models_per_category']: failures.append('category_balance_not_met')
        if len(brands) < policy['minimum_brands_per_category']: failures.append('brand_count_not_met')
        if max(share(brands),share(configuration_brands)) > policy['maximum_brand_share']: failures.append('brand_concentration')
        if share(sellers) > policy['maximum_seller_share']: failures.append('seller_concentration')
        if share(platforms) > policy['maximum_platform_share']: failures.append('platform_concentration')
        if sites and share(sites) > policy['maximum_experience_site_share']: failures.append('experience_site_concentration')
        if sellers.get('unknown') or platforms.get('unknown'): failures.append('price_channel_identity_unknown')
        warnings = []
        if quality_policy == SOURCE_PREFERRED_POLICY:
            diagnostic = {'seller_concentration', 'platform_concentration',
                          'experience_site_concentration', 'price_channel_identity_unknown'}
            warnings = [f for f in failures if f in diagnostic]
            failures = [f for f in failures if f not in diagnostic]
        categories[category] = {'qualified_models':len(model_brands),'configurations':len(rows),
                                'target_models':policy['target_models'][category],
                                'remaining_to_target':max(0,policy['target_models'][category]-len(model_brands)),
                                'brands':dict(brands),'failures':failures,'warnings':warnings}
        issues.extend(category+':'+f for f in failures)
    return {'policy_id':policy['policy_id'],'quality_policy':quality_policy,'status':'passed' if not issues else 'blocked',
            'qualified_models':len(models),'configurations':len(products),'minimum_models':policy['minimum_models'],
            'remaining_models':max(0,policy['minimum_models']-len(models)),'categories':categories,'issues':issues,
            'interpretation':'Scale, required record fields and category/brand distribution gate. Under source-preferred-v1, missing decision specifications are unknown and channel/experience concentration is diagnostic. Source traceability and dialogue acceptance are separate checks.'}
