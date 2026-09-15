"""Small deterministic checks for explicit product facts in Agent answers."""

from __future__ import annotations

import re

from src.retrieval.models import RetrievalConstraints, RetrievalResult, VerificationResult


_PRODUCT_ID_RE = re.compile(
    r"(?i)(?:产品\s*id|product\s*[_-]?\s*id)\s*[:：=#]?\s*(\d+)"
)
_PRICE_RE = re.compile(
    r"(?i)(?P<label>价格|售价|price)?\s*[:：]?\s*"
    r"(?P<symbol>[¥￥]|rmb\s*)?"
    r"(?P<amount>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    r"\s*(?P<unit>元|人民币|块)?"
)


def _explicit_prices(answer: str) -> list[tuple[int, int, float]]:
    prices = []
    for match in _PRICE_RE.finditer(answer):
        groups = match.groupdict()
        if not (groups["label"] or groups["symbol"] or groups["unit"]):
            continue
        prices.append((match.start(), match.end(), float(groups["amount"].replace(",", ""))))
    return prices


def _product_price_pairs(answer: str) -> list[tuple[int, float]]:
    ids = list(_PRODUCT_ID_RE.finditer(answer))
    prices = _explicit_prices(answer)
    if not ids or not prices:
        return []

    pairs: list[tuple[int, float]] = []
    # Only accept a price in the same local clause; ambiguous prices are left
    # unchecked instead of being guessed.
    for product_match in ids:
        nearby = [
            price
            for start, end, price in prices
            if abs(start - product_match.end()) <= 100
            or abs(product_match.start() - end) <= 100
        ]
        if len(nearby) == 1:
            pairs.append((int(product_match.group(1)), nearby[0]))
    return pairs


def _matches_constraints(product, constraints: RetrievalConstraints) -> bool:
    if constraints.min_price is not None and (
        product.price is None or product.price < constraints.min_price
    ):
        return False
    if constraints.max_price is not None and (
        product.price is None or product.price > constraints.max_price
    ):
        return False
    if constraints.category and product.category.casefold() != constraints.category.casefold():
        return False
    excluded = {brand.casefold() for brand in constraints.excluded_brands}
    return product.brand.casefold() not in excluded


def verify_answer(answer: str, retrieval_result: RetrievalResult | dict | None) -> VerificationResult:
    """Verify only explicit, deterministically parseable catalog assertions."""
    if retrieval_result is None:
        return VerificationResult(failure_codes=["no_structured_retrieval"])
    if not isinstance(retrieval_result, RetrievalResult):
        try:
            retrieval_result = RetrievalResult.model_validate(retrieval_result)
        except Exception:
            return VerificationResult(failure_codes=["invalid_structured_retrieval"])
    if not retrieval_result.evidence:
        return VerificationResult(failure_codes=["no_structured_evidence"])
    if not isinstance(answer, str) or not answer.strip():
        return VerificationResult(failure_codes=["empty_answer"])

    products_by_id = {product.product_id: product for product in retrieval_result.products}
    evidence_by_key = {
        (item.product_id, item.field): item for item in retrieval_result.evidence
    }
    mentioned_ids = list(dict.fromkeys(int(match.group(1)) for match in _PRODUCT_ID_RE.finditer(answer)))
    failures: list[str] = []
    evidence_ids: list[str] = []

    for product_id in mentioned_ids:
        product = products_by_id.get(product_id)
        if product is None:
            failures.append("product_id_not_retrieved")
            continue
        for field in ("product_id", "price", "brand", "category"):
            evidence = evidence_by_key.get((product_id, field))
            if evidence is not None:
                evidence_ids.append(evidence.evidence_id)

    for product_id, cited_price in _product_price_pairs(answer):
        product = products_by_id.get(product_id)
        if product is not None and product.price is not None and cited_price != float(product.price):
            failures.append("price_mismatch")

    if any(not _matches_constraints(product, retrieval_result.constraints) for product in retrieval_result.products):
        failures.append("constraint_violation")

    if failures:
        return VerificationResult(
            status="failed",
            failure_codes=list(dict.fromkeys(failures)),
            checked_product_ids=mentioned_ids,
            evidence_ids=list(dict.fromkeys(evidence_ids)),
        )
    if not mentioned_ids:
        return VerificationResult(failure_codes=["no_explicit_product_id"])
    return VerificationResult(
        status="verified",
        checked_product_ids=mentioned_ids,
        evidence_ids=list(dict.fromkeys(evidence_ids)),
    )
