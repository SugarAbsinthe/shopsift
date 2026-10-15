"""Objective contract for product-retrieval evaluation cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from src.retrieval.models import canonical_brand
from src.retrieval.models import SpecFilter


RetrievalSource = Literal["description", "spec", "sparse"]
DEFAULT_RETRIEVAL_CASES_PATH = Path(__file__).with_name("retrieval_cases.jsonl")


class RetrievalConstraints(BaseModel):
    min_price: int | None = Field(default=None, ge=0)
    max_price: int | None = Field(default=None, ge=0)
    category: str = ""
    preferred_brands: list[str] = Field(default_factory=list)
    excluded_brands: list[str] = Field(default_factory=list)
    spec_filters: list[SpecFilter] = Field(default_factory=list)

    @field_validator("preferred_brands", "excluded_brands")
    @classmethod
    def unique_brands(cls, value: list[str]) -> list[str]:
        normalized = [brand.strip() for brand in value if brand.strip()]
        if len(normalized) != len(set(normalized)):
            raise ValueError("brands must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_range(self) -> "RetrievalConstraints":
        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError("min_price cannot exceed max_price")
        overlap = {
            canonical_brand(brand) for brand in self.preferred_brands
        } & {canonical_brand(brand) for brand in self.excluded_brands}
        if overlap:
            raise ValueError("a brand cannot be both preferred and excluded")
        return self


class RetrievalCase(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    profile: dict[str, str] = Field(default_factory=dict)
    constraints: RetrievalConstraints = Field(default_factory=RetrievalConstraints)
    relevant_ids: list[int] = Field(default_factory=list)
    forbidden_ids: list[int] = Field(default_factory=list)
    required_sources: list[RetrievalSource] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    min_distinct_models: int | None = Field(default=None, ge=1, le=20)
    review_product_id: int | None = Field(default=None, gt=0)
    review_aspect: str = ""
    expected_summary_keys: list[str] = Field(default_factory=list)

    @field_validator("relevant_ids", "forbidden_ids", "required_sources", "tags")
    @classmethod
    def reject_duplicates(cls, value: list) -> list:
        if len(value) != len(set(value)):
            raise ValueError("list values must be unique")
        return value

    @model_validator(mode="after")
    def validate_ids(self) -> "RetrievalCase":
        overlap = set(self.relevant_ids) & set(self.forbidden_ids)
        if overlap:
            raise ValueError(
                f"product ids cannot be both relevant and forbidden: {sorted(overlap)}"
            )
        return self


def load_retrieval_cases(
    path: Path | str = DEFAULT_RETRIEVAL_CASES_PATH,
) -> list[RetrievalCase]:
    case_path = Path(path)
    cases: list[RetrievalCase] = []
    seen_ids: set[str] = set()
    with case_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                case = RetrievalCase.model_validate(json.loads(line))
            except Exception as exc:
                raise ValueError(
                    f"invalid retrieval case at {case_path}:{line_number}: {exc}"
                ) from exc
            if case.id in seen_ids:
                raise ValueError(
                    f"duplicate retrieval case id at {case_path}:{line_number}: {case.id}"
                )
            seen_ids.add(case.id)
            cases.append(case)
    if not cases:
        raise ValueError(f"no retrieval cases found in {case_path}")
    return cases
