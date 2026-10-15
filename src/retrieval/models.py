"""Typed contracts shared by retrieval, caching, and evaluation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from src.data.categories import FILTER_FIELDS
import math


RetrievalSource = Literal["description", "spec", "sparse"]
EvidenceField = Literal["product_id", "price", "brand", "category"]


def canonical_brand(value: str) -> str:
    """Match Chinese/English official brand spellings in new and legacy data."""
    key = value.strip().casefold()
    aliases = {
        "苹果": "apple", "macbook": "apple", "联想": "lenovo", "thinkpad": "lenovo",
        "荣耀": "honor", "小米": "xiaomi", "红米": "xiaomi", "redmi": "xiaomi",
        "三星": "samsung", "索尼": "sony", "漫步者": "edifier", "花再": "edifier",
        "华为": "huawei", "华硕": "asus", "惠普": "hp", "戴尔": "dell", "宏碁": "acer",
        "罗技": "logitech", "logi": "logitech", "logitech g": "logitech",
        "雷蛇": "razer", "雷柏": "rapoo", "明基": "benq", "lg": "lg",
        "rog": "asus", "爱攻": "aoc", "樱桃": "cherry", "阿米洛": "varmilo",
        "纽曼": "newmine", "newmine纽曼": "newmine", "newmen": "新贵",
        "赛睿": "steelseries", "狼蛛": "aula", "黑爵": "ajazz",
        "a.艾酷": "akko", "航世": "b.o.w", "联想开天": "lenovo",
    }
    return aliases.get(key, key)


class SpecFilter(BaseModel):
    field: str
    op: Literal["eq", "gte", "lte", "contains_all"] = "eq"
    value: Any

    @model_validator(mode="after")
    def valid_predicate(self):
        kind = FILTER_FIELDS.get(self.field)
        if kind == "number":
            valid = self.op in {"eq", "gte", "lte"} and isinstance(self.value, (int, float)) and not isinstance(self.value, bool) and math.isfinite(self.value)
        elif kind == "set":
            valid = self.op == "contains_all" and isinstance(self.value, list) and bool(self.value) and all(isinstance(v, str) and v for v in self.value)
        elif kind == "bool":
            valid = self.op == "eq" and isinstance(self.value, bool)
        else:
            valid = kind == "string" and self.op == "eq" and isinstance(self.value, str) and bool(self.value)
        from src.data.categories import ENUM_VALUES
        if self.field in ENUM_VALUES:
            valid = valid and set(self.value if isinstance(self.value,list) else [self.value]) <= ENUM_VALUES[self.field]
        if not valid:
            raise ValueError("Unsupported specification field, operator or value")
        return self


class RetrievalConstraints(BaseModel):
    min_price: int | None = Field(default=None, ge=0)
    max_price: int | None = Field(default=None, ge=0)
    category: str = ""
    preferred_brands: list[str] = Field(default_factory=list)
    excluded_brands: list[str] = Field(default_factory=list)
    spec_filters: list[SpecFilter] = Field(default_factory=list)

    @field_validator("preferred_brands", "excluded_brands")
    @classmethod
    def normalize_brands(cls, value: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for brand in value:
            normalized = brand.strip()
            key = canonical_brand(normalized)
            if normalized and key not in seen:
                result.append(normalized)
                seen.add(key)
        return result

    @model_validator(mode="after")
    def validate_constraints(self) -> "RetrievalConstraints":
        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError("min_price cannot exceed max_price")
        overlap = {canonical_brand(item) for item in self.preferred_brands} & {
            canonical_brand(item) for item in self.excluded_brands
        }
        if overlap:
            raise ValueError("a brand cannot be both preferred and excluded")
        return self


class RetrievalQuery(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    constraints: RetrievalConstraints = Field(default_factory=RetrievalConstraints)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("query text cannot be blank")
        return normalized


class ProductCandidate(BaseModel):
    product_id: int
    name: str = ""
    brand: str = ""
    category: str = ""
    subcategory: str = ""
    price: float | None = None
    rating: float | None = None
    sales_count: int | None = None
    release_date: str = ""
    description: str = ""
    specs: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    normalized_specs: dict[str, Any] = Field(default_factory=dict)
    sources: list[RetrievalSource] = Field(default_factory=list)
    source_ranks: dict[RetrievalSource, int] = Field(default_factory=dict)
    score: float = 0.0


class RetrievalStats(BaseModel):
    source_hits: dict[RetrievalSource, int] = Field(default_factory=dict)
    fused_candidates: int = 0
    filtered_candidates: int = 0
    returned_candidates: int = 0
    sparse_fallback: bool = False
    index_version: str = "legacy"
    duration_ms: int = 0


class EvidenceItem(BaseModel):
    """A deterministic catalog fact used to audit a retrieval result."""

    evidence_id: str = Field(min_length=1, max_length=200)
    product_id: int
    source_type: Literal["catalog"] = "catalog"
    field: EvidenceField
    value: str | int | float
    source_rank: int | None = Field(default=None, ge=1)
    index_version: str = Field(default="legacy", min_length=1, max_length=100)


VerificationStatus = Literal["verified", "failed", "not_applicable"]


class VerificationResult(BaseModel):
    """Outcome of deterministic checks over an Agent answer."""

    status: VerificationStatus = "not_applicable"
    failure_codes: list[str] = Field(default_factory=list)
    checked_product_ids: list[int] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "verified"


class RetrievalResult(BaseModel):
    products: list[ProductCandidate] = Field(default_factory=list)
    reviews_by_product: dict[int, list[dict[str, str]]] = Field(default_factory=dict)
    stats: RetrievalStats = Field(default_factory=RetrievalStats)
    constraints: RetrievalConstraints = Field(default_factory=RetrievalConstraints)
    evidence: list[EvidenceItem] = Field(default_factory=list)
