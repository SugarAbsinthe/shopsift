"""Versioned, objective evaluation case contract.

The contract intentionally avoids exact-answer matching. It describes
observable execution properties that can be checked in both deterministic
and live-model runs: stage routing, tool boundaries, retrieval activation,
termination, and profile extraction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


StageName = Literal[
    "discovery",
    "needs_elicitation",
    "search",
    "comparison",
    "objection_handling",
    "recommendation",
    "summary",
]
ToolName = Literal[
    "get_product_detail",
    "get_reviews",
    "compare_products",
    "get_user_profile",
]
StopReason = Literal["completed", "tool_error", "max_tool_rounds"]

VALID_STAGES = frozenset(StageName.__args__)
VALID_TOOLS = frozenset(ToolName.__args__)
VALID_STOP_REASONS = frozenset(StopReason.__args__)
DEFAULT_CASES_PATH = Path(__file__).with_name("cases.jsonl")


class EvalMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class EvalCase(BaseModel):
    """One behavior-oriented evaluation case.

    ``scripted_tool_rounds`` and ``failing_tools`` are deterministic fixtures;
    live evaluation ignores them and lets the configured model choose tools.
    Expected fields are shared by both modes.
    """

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    question: str = Field(min_length=1, max_length=2000)
    history: list[EvalMessage] = Field(default_factory=list, max_length=20)
    expected_stages: list[StageName] = Field(min_length=1)
    required_tools: list[ToolName] = Field(default_factory=list)
    forbidden_tools: list[ToolName] = Field(default_factory=list)
    expected_stop_reasons: list[StopReason] = Field(
        default_factory=lambda: ["completed"], min_length=1
    )
    expected_retrieval: bool
    expected_profile_keys: list[str] = Field(default_factory=list)
    expected_pending_profile_keys: list[str] = Field(default_factory=list)
    max_tool_rounds: int = Field(default=3, ge=1, le=10)
    scripted_tool_rounds: list[list[ToolName]] = Field(default_factory=list)
    failing_tools: list[ToolName] = Field(default_factory=list)
    retriever_error: bool = False
    classifier_response: Optional[StageName] = None
    deterministic_answer: str = Field(default="已完成评测用例。", min_length=1)
    tags: list[str] = Field(default_factory=list)

    @field_validator(
        "expected_stages",
        "required_tools",
        "forbidden_tools",
        "expected_stop_reasons",
        "expected_profile_keys",
        "expected_pending_profile_keys",
        "failing_tools",
        "tags",
    )
    @classmethod
    def reject_duplicates(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("list values must be unique")
        return value

    @field_validator("scripted_tool_rounds")
    @classmethod
    def validate_tool_rounds(cls, rounds: list[list[str]]) -> list[list[str]]:
        for round_tools in rounds:
            if not round_tools:
                raise ValueError("scripted tool rounds cannot be empty")
            if len(round_tools) != len(set(round_tools)):
                raise ValueError("a scripted round cannot call the same tool twice")
        return rounds

    @model_validator(mode="after")
    def validate_expectations(self) -> "EvalCase":
        overlap = set(self.required_tools) & set(self.forbidden_tools)
        if overlap:
            raise ValueError(f"tools cannot be both required and forbidden: {sorted(overlap)}")

        executable_tools = {
            tool
            for round_tools in self.scripted_tool_rounds[: self.max_tool_rounds]
            for tool in round_tools
        }
        missing = set(self.required_tools) - executable_tools
        if missing:
            raise ValueError(
                "required tools must appear in an executable scripted round: "
                f"{sorted(missing)}"
            )

        scripted_tools = {
            tool for round_tools in self.scripted_tool_rounds for tool in round_tools
        }
        unknown_failures = set(self.failing_tools) - scripted_tools
        if unknown_failures:
            raise ValueError(
                f"failing tools must appear in the script: {sorted(unknown_failures)}"
            )

        if self.retriever_error and not self.expected_retrieval:
            raise ValueError("retriever_error requires expected_retrieval=true")
        return self


def load_cases(path: Path | str = DEFAULT_CASES_PATH) -> list[EvalCase]:
    """Load JSONL cases with line-aware validation errors."""
    case_path = Path(path)
    cases: list[EvalCase] = []
    seen_ids: set[str] = set()

    with case_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
                case = EvalCase.model_validate(payload)
            except Exception as exc:
                raise ValueError(
                    f"invalid evaluation case at {case_path}:{line_number}: {exc}"
                ) from exc
            if case.id in seen_ids:
                raise ValueError(
                    f"duplicate evaluation case id at {case_path}:{line_number}: {case.id}"
                )
            seen_ids.add(case.id)
            cases.append(case)

    if not cases:
        raise ValueError(f"no evaluation cases found in {case_path}")
    return cases
