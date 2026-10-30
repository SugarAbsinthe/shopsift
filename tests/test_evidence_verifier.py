"""Deterministic evidence and answer-verification contracts."""

import pytest

from src.retrieval.models import (
    EvidenceItem,
    ProductCandidate,
    RetrievalConstraints,
    RetrievalResult,
    RetrievalStats,
)
from src.retrieval.verifier import verify_answer


def _result(*, constraints=None, products=None):
    products = products if products is not None else [
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


def test_source_url_ids_and_prices_are_not_catalog_assertions():
    checked = verify_answer("产品 ID: 1，价格: 7999 元。\n来源：https://example.com/view?product_id=24930&price=1234", _result())
    assert checked.status == "verified"
    assert checked.checked_product_ids == [1]
    wrong = verify_answer("来源：https://example.com/view?product_id=24930\n产品 ID: 99，价格: 7999 元。", _result())
    assert wrong.status == "failed"
    assert wrong.checked_product_ids == [99]


def test_chinese_brand_exclusion_blocks_english_brand_catalog():
    result = _result(products=[ProductCandidate(product_id=1, brand="Xiaomi", category="手机", price=7999)], constraints={"excluded_brands": ["小米"]})
    assert "constraint_violation" in verify_answer("产品 ID: 1。", result).failure_codes


@pytest.mark.parametrize("answer", [
    "产品 ID: 1，约 5999 元。",
    "产品 ID: 1，价格 ￥5,999.00。",
    "产品 ID: 1，价格 5999 RMB。",
    "产品 ID: 1\n- 价格：5999 元。",
    "预算 8000 元。产品 ID: 1，适合办公。",
    "预算 8000 元，产品 ID: 1，价格 5999 元。",
    "产品 ID: 1，价格 5999 元，比预算节省 2001 元。",
])
def test_price_checks_preserve_exact_quotes_and_ignore_budget_or_savings(answer):
    found = _result(products=[ProductCandidate(product_id=1, price=5999)])
    assert verify_answer(answer, found).status == "verified"


@pytest.mark.parametrize("separator", ["；", "，", "\n", "。", "\n\n"])
def test_each_product_quote_is_checked_independently(separator):
    found = _result(products=[ProductCandidate(product_id=1, price=5999),
                              ProductCandidate(product_id=2, price=7999)])
    valid = f"产品 ID: 1，价格 5999 元{separator}产品 ID: 2，价格 7999 元。"
    invalid = f"产品 ID: 1，价格 1 元{separator}产品 ID: 2，价格 2 元。"
    assert verify_answer(valid, found).status == "verified"
    assert verify_answer(invalid, found).failure_codes == ["price_mismatch"]


def test_distant_wrong_quote_in_the_same_product_block_is_not_skipped():
    found = _result()
    answer = "产品 ID: 1，" + "公开参数供参考，" * 30 + "价格 1 元。"
    assert verify_answer(answer, found).failure_codes == ["price_mismatch"]


def test_approximate_word_does_not_authorize_a_different_catalog_price():
    found = _result(products=[ProductCandidate(product_id=1, price=5999)])
    assert verify_answer("产品 ID: 1，约 6000 元。", found).failure_codes == ["price_mismatch"]


@pytest.mark.parametrize("answer", [
    "产品 ID: 1 和产品 ID: 2，价格分别为 5999 元和 7999 元。",
    "价格 5999 元和 7999 元，对应产品 ID: 1、产品 ID: 2。",
    "产品 ID: 1。另有商品售价 7999 元。",
])
def test_ambiguous_price_ownership_is_not_reported_as_verified(answer):
    found = _result(products=[ProductCandidate(product_id=1, price=5999),
                              ProductCandidate(product_id=2, price=7999)])
    checked = verify_answer(answer, found)
    assert checked.status == "not_applicable"
    assert "price_attribution_ambiguous" in checked.failure_codes


def test_unknown_catalog_price_cannot_verify_a_quote():
    found = _result(products=[ProductCandidate(product_id=1, price=None)])
    checked = verify_answer("产品 ID: 1，价格 5999 元。", found)
    assert checked.status == "not_applicable"
    assert checked.failure_codes == ["price_evidence_missing"]


def test_thousands_separator_preserves_fractional_price():
    found = _result(products=[ProductCandidate(product_id=1, price=5999.50)])
    assert verify_answer("产品 ID: 1，价格 ￥5,999.50。", found).status == "verified"
    assert verify_answer("产品 ID: 1，价格 ￥5,999.00。", found).status == "failed"


def test_empty_retrieval_allows_guidance_but_blocks_fabricated_product_facts():
    empty = _result(products=[])
    guidance = verify_answer("没有匹配结果，可以把预算调整到 5000 元后再查。", empty)
    assert guidance.status == "not_applicable"
    fabricated = verify_answer("推荐产品 ID: 99，价格 2000 元。", empty)
    assert fabricated.status == "failed"
    assert "product_id_not_retrieved" in fabricated.failure_codes
