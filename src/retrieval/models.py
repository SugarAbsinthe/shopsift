"""Typed contracts shared by retrieval, caching, and evaluation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


RetrievalSource = Literal["description", "spec", "sparse"]
EvidenceField = Literal["product_id", "price", "brand", "category"]


class RetrievalConstraints(BaseModel):
    min_price: int | None = Field(default=None, ge=0)
    max_price: int | None = Field(default=None, ge=0)
    category: str = ""
    preferred_brands: list[str] = Field(default_factory=list)
    excluded_brands: list[str] = Field(default_factory=list)

    @field_validator("preferred_brands", "excluded_brands")
    @classmethod
    def normalize_brands(cls, value: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for brand in value:
            normalized = brand.strip()
            key = normalized.casefold()
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
        overlap = {item.casefold() for item in self.preferred_brands} & {
            item.casefold() for item in self.excluded_brands
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
