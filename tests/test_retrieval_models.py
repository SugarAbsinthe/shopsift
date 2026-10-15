"""Unit tests for the runtime retrieval contracts."""

import pytest
from pydantic import ValidationError

from src.retrieval.models import RetrievalConstraints, RetrievalQuery


def test_query_normalizes_whitespace_and_brand_lists():
    query = RetrievalQuery(
        text="  RTX 4060\n gaming laptop  ",
        constraints={
            "preferred_brands": [" Lenovo ", "lenovo", "ASUS"],
        },
    )
    assert query.text == "RTX 4060 gaming laptop"
    assert query.constraints.preferred_brands == ["Lenovo", "ASUS"]


def test_constraints_reject_invalid_price_range_and_brand_conflict():
    with pytest.raises(ValidationError, match="min_price cannot exceed max_price"):
        RetrievalConstraints(min_price=9000, max_price=8000)
    with pytest.raises(ValidationError, match="both preferred and excluded"):
        RetrievalConstraints(
            preferred_brands=["Lenovo"], excluded_brands=["lenovo"]
        )


def test_query_rejects_blank_text_and_unsafe_result_size():
    with pytest.raises(ValidationError):
        RetrievalQuery(text=" \n ")
    with pytest.raises(ValidationError):
        RetrievalQuery(text="laptop", top_k=1000)


def test_brand_aliases_preserve_labels_and_cannot_conflict():
    normalized = RetrievalConstraints(excluded_brands=["小米", "Xiaomi", "REDMI"])
    assert normalized.excluded_brands == ["小米"]
    with pytest.raises(ValidationError, match="both preferred and excluded"):
        RetrievalConstraints(preferred_brands=["Apple"], excluded_brands=["苹果"])
