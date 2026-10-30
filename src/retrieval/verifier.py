"""Small deterministic checks for explicit product facts in Agent answers."""

from __future__ import annotations

import re
from decimal import Decimal

from src.retrieval.models import RetrievalConstraints, RetrievalResult, VerificationResult, canonical_brand
from src.data.categories import matches_spec_filters


_PRODUCT_ID_RE = re.compile(
    r"(?i)(?:产品\s*id|product\s*[_-]?\s*id)\s*[:：=#]?\s*(\d+)"
)
_PRICE_RE = re.compile(
    r"(?i)(?P<label>价格|售价|price)?\s*[:：]?\s*"
    r"(?P<symbol>[¥￥]|rmb\s*)?"
    r"(?P<amount>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*(?P<unit>元|人民币|块|rmb\b)?"
)
_SOURCE_URL_RE = re.compile(r"https?://[^\s<>()\[\]`\"'，。；）]+", re.I)
_MONEY_ROLE_RE = re.compile(
    r"预算|差价|节省|优惠|立减|便宜|加价|月供|运费|定金|"
    r"售价|价格|报价|到手价|参考价|price", re.I,
)
_QUOTE_ROLES = {"售价", "价格", "报价", "到手价", "参考价", "price"}


def _without_source_urls(answer: str) -> str:
    # Merchant API product_id/price query parameters identify source records,
    # not catalog assertions. Preserve offsets for local-clause price pairing.
    return _SOURCE_URL_RE.sub(lambda match: " " * len(match.group()), answer)


def _explicit_prices(answer: str) -> list[tuple[int, int, Decimal]]:
    prices = []
    for match in _PRICE_RE.finditer(answer):
        groups = match.groupdict()
        if not (groups["label"] or groups["symbol"] or groups["unit"]):
            continue
        # A budget, saving, deposit or delivery fee is not a product quotation.
        clause = re.split(r"[，,。；;!?！？\n]", answer[:match.start("amount")])[-1]
        roles = list(_MONEY_ROLE_RE.finditer(clause))
        if roles and roles[-1].group().lower() not in _QUOTE_ROLES:
            continue
        prices.append((match.start(), match.end(), Decimal(groups["amount"].replace(",", ""))))
    return prices


def _product_price_pairs(answer: str) -> tuple[list[tuple[int, Decimal]], bool]:
    answer = _without_source_urls(answer)
    pairs: list[tuple[int, Decimal]] = []
    ambiguous = False
    # Single line breaks allow an ID followed by a price bullet. Sentence and
    # paragraph boundaries prevent a price leaking into another product block.
    for block in re.split(r"[。；;!?！？]|\n\s*\n", answer):
        ids = list(_PRODUCT_ID_RE.finditer(block))
        prices = _explicit_prices(block)
        for start, end, amount in prices:
            if len(ids) == 1:
                pairs.append((int(ids[0].group(1)), amount))
                continue
            preceding = [item for item in ids if item.end() <= start]
            if not preceding:
                ambiguous = True
                continue
            owner = preceding[-1]
            index = ids.index(owner)
            # "ID 1 and ID 2, respectively ..." needs a richer parser. Do not
            # silently mark it verified or guess the last ID owns both prices.
            grouped = index > 0 and re.fullmatch(
                r"[\s、,，和与及/]*(?:以及)?[\s、,，和与及/]*", block[ids[index-1].end():owner.start()]
            )
            if grouped:
                ambiguous = True
            else:
                pairs.append((int(owner.group(1)), amount))
    return pairs, ambiguous


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
    excluded = {canonical_brand(brand) for brand in constraints.excluded_brands}
    return canonical_brand(product.brand) not in excluded and matches_spec_filters(product.normalized_specs, constraints.spec_filters)


def verify_answer(answer: str, retrieval_result: RetrievalResult | dict | None) -> VerificationResult:
    """Verify only explicit, deterministically parseable catalog assertions."""
    if retrieval_result is None:
        return VerificationResult(failure_codes=["no_structured_retrieval"])
    if not isinstance(retrieval_result, RetrievalResult):
        try:
            retrieval_result = RetrievalResult.model_validate(retrieval_result)
        except Exception:
            return VerificationResult(failure_codes=["invalid_structured_retrieval"])
    if not isinstance(answer, str) or not answer.strip():
        return VerificationResult(failure_codes=["empty_answer"])

    products_by_id = {product.product_id: product for product in retrieval_result.products}
    evidence_by_key = {
        (item.product_id, item.field): item for item in retrieval_result.evidence
    }
    fact_text = _without_source_urls(answer)
    mentioned_ids = list(dict.fromkeys(int(match.group(1)) for match in _PRODUCT_ID_RE.finditer(fact_text)))
    failures: list[str] = []
    evidence_ids: list[str] = []
    unchecked: list[str] = []

    for product_id in mentioned_ids:
        product = products_by_id.get(product_id)
        if product is None:
            failures.append("product_id_not_retrieved")
            continue
        for field in ("product_id", "price", "brand", "category"):
            evidence = evidence_by_key.get((product_id, field))
            if evidence is not None:
                evidence_ids.append(evidence.evidence_id)

    price_pairs, ambiguous = _product_price_pairs(answer)
    if ambiguous:
        unchecked.append("price_attribution_ambiguous")
    for product_id, cited_price in price_pairs:
        product = products_by_id.get(product_id)
        evidence = evidence_by_key.get((product_id, "price"))
        if product is not None and (product.price is None or evidence is None):
            unchecked.append("price_evidence_missing")
        elif product is not None and cited_price != Decimal(str(product.price)):
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
    if not retrieval_result.evidence:
        return VerificationResult(failure_codes=["no_structured_evidence"])
    if not mentioned_ids:
        return VerificationResult(failure_codes=["no_explicit_product_id"])
    if unchecked:
        return VerificationResult(
            failure_codes=list(dict.fromkeys(unchecked)),
            checked_product_ids=mentioned_ids,
            evidence_ids=list(dict.fromkeys(evidence_ids)),
        )
    return VerificationResult(
        status="verified",
        checked_product_ids=mentioned_ids,
        evidence_ids=list(dict.fromkeys(evidence_ids)),
    )
