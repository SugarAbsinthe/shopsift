"""Contracts for governed structured profile memory."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Literal


MEMORY_SCHEMA_VERSION = "1.0"
MemoryType = Literal["explicit", "inferred"]
MemoryStatus = Literal["pending", "confirmed", "rejected", "expired", "deleted"]

PROFILE_KEYS = frozenset(
    {
        "budget",
        "primary_use",
        "preferred_brand",
        "mobility",
        "must_have",
        "exclude_brand",
        "screen_preference",
        "battery_requirement",
        "product_category",
    }
)


def make_source_message_id(conv_id: str, message: str) -> str:
    """Create a stable source reference without using the full message as an id."""
    digest = hashlib.sha256(f"{conv_id}\0{message}".encode("utf-8")).hexdigest()[:20]
    return f"msg_{digest}"


@dataclass(frozen=True)
class MemoryCandidate:
    candidate_id: str
    conv_id: str
    key: str
    value: str
    type: MemoryType
    confidence: float
    source_message_id: str
    evidence_span: str
    created_at: str
    expires_at: str
    status: MemoryStatus
    schema_version: str = MEMORY_SCHEMA_VERSION

    def model_dump(self) -> dict:
        """Return a serialization shape compatible with other project models."""
        return asdict(self)
