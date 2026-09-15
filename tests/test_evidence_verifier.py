"""Deterministic evidence and answer-verification contracts."""

from src.retrieval.models import (
    EvidenceItem,
    ProductCandidate,
    RetrievalConstraints,
    RetrievalResult,
    RetrievalStats,
)
from src.retrieval.verifier import verify_answer


def _result(*, constraints=None, products=None):
    products = products or [
        ProductCandidate(
            product_id=1,
            name="Test laptop",
            brand="Acme",
            category="laptop",
            price=7999,
        )
    ]
    return RetrievalResult(
        products=products,
        constraints=constraints or RetrievalConstraints(),
        stats=RetrievalStats(index_version="idx-test"),
        evidence=[
            EvidenceItem(
                evidence_id=f"catalog:idx-test:{product.product_id}:{field}",
                product_id=product.product_id,
                field=field,
                value=value,
                source_rank=1,
                index_version="idx-test",
            )
            for product in products
            for field, value in (
                ("product_id", product.product_id),
                ("price", product.price),
                ("brand", product.brand),
                ("category", product.category),
            )
            if value is not None
        ],
    )


def test_explicit_product_and_price_are_verified():
    result = verify_answer("产品 ID: 1，价格: 7999 元。", _result())
    assert result.status == "verified"
    assert result.failure_codes == []
    assert result.evidence_ids == [
        "catalog:idx-test:1:product_id",
        "catalog:idx-test:1:price",
        "catalog:idx-test:1:brand",
        "catalog:idx-test:1:category",
    ]


def test_unknown_product_and_wrong_price_are_blocked():
    unknown = verify_answer("产品 ID: 99，价格: 7999 元。", _result())
    assert unknown.status == "failed"
    assert unknown.failure_codes == ["product_id_not_retrieved"]

    wrong_price = verify_answer("产品 ID: 1，价格: 6999 元。", _result())
    assert wrong_price.status == "failed"
    assert wrong_price.failure_codes == ["price_mismatch"]


def test_constraint_violation_is_blocked_even_if_product_id_is_known():
    result = _result(constraints={"max_price": 7000})
    checked = verify_answer("产品 ID: 1。", result)
    assert checked.status == "failed"
    assert checked.failure_codes == ["constraint_violation"]


def test_missing_structured_result_is_not_reported_as_verified():
    result = verify_answer("我建议购买产品 ID: 1。", None)
    assert result.status == "not_applicable"
    assert result.failure_codes == ["no_structured_retrieval"]
