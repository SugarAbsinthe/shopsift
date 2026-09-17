"""Minimal policy gateway for the tools used by the shopping graph.

The gateway is intentionally narrow.  It validates the tools that are actually
registered by the graph, applies stage and per-run limits, and returns stable
tool errors without exposing arguments or internal exceptions in user-visible
messages.  It is not a general sandbox and does not execute arbitrary code.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from langchain_core.messages import ToolMessage

from backend.logging_config import log, mark_timeout, record_failure, record_tool_policy
from src.profile.models import PROFILE_KEYS


KNOWN_STAGES = frozenset(
    {
        "discovery",
        "needs_elicitation",
        "search",
        "comparison",
        "objection_handling",
        "recommendation",
        "summary",
    }
)

@dataclass(frozen=True)
class ToolSpec:
    """Registration-time policy for one tool."""

    name: str
    access: str = "read_only"
    allowed_stages: frozenset[str] = field(default_factory=lambda: KNOWN_STAGES)
    argument_schema: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    timeout_ms: int = 1000
    max_calls_per_run: int = 3
    approval_required: bool = False

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", self.name):
            raise ValueError("invalid tool name")
        if self.access not in {"read_only", "profile_write"}:
            raise ValueError("invalid tool access")
        if not self.allowed_stages:
            raise ValueError("tool must allow at least one stage")
        if not isinstance(self.timeout_ms, int) or self.timeout_ms < 0:
            raise ValueError("timeout_ms must be a non-negative integer")
        if not isinstance(self.max_calls_per_run, int) or self.max_calls_per_run < 1:
            raise ValueError("max_calls_per_run must be positive")


def _known_spec(name: str) -> ToolSpec | None:
    """Return policy for a production tool, or ``None`` for a test/custom tool."""
    common = {
        "conv_id": {"type": str, "min_length": 1, "max_length": 128},
    }
    if name == "get_product_detail":
        return ToolSpec(
            name=name,
            argument_schema={"product_id": {"type": int, "min": 1, "max": 1_000_000_000}},
            allowed_stages=frozenset({"search", "comparison", "objection_handling", "recommendation", "summary"}),
            max_calls_per_run=4,
            timeout_ms=2000,
        )
    if name == "get_reviews":
        return ToolSpec(
            name=name,
            argument_schema={
                "product_id": {"type": int, "min": 1, "max": 1_000_000_000},
                "aspect": {"type": str, "max_length": 64},
                "top_k": {"type": int, "min": 1, "max": 8},
            },
            allowed_stages=frozenset({"search", "comparison", "objection_handling", "recommendation", "summary"}),
            max_calls_per_run=3,
            timeout_ms=5000,
        )
    if name == "compare_products":
        return ToolSpec(
            name=name,
            argument_schema={"product_ids": {"type": str, "max_length": 128}},
            allowed_stages=frozenset({"search", "comparison", "recommendation", "summary"}),
            max_calls_per_run=2,
            timeout_ms=3000,
        )
    if name == "get_user_profile":
        return ToolSpec(
            name=name,
            access="read_only",
            argument_schema=common,
            allowed_stages=KNOWN_STAGES,
            max_calls_per_run=2,
            timeout_ms=2000,
        )
    if name == "update_user_profile":
        return ToolSpec(
            name=name,
            access="profile_write",
            argument_schema={
                **common,
                "key": {"type": str, "choices": PROFILE_KEYS, "max_length": 64},
                "value": {"type": str, "min_length": 1, "max_length": 256},
            },
            allowed_stages=frozenset({"discovery", "needs_elicitation"}),
            max_calls_per_run=4,
            approval_required=True,
            # Do not pretend a thread timeout can cancel a write that already
            # started.  This local SQLite operation runs synchronously.
            timeout_ms=0,
        )
    return None


def build_tool_registry(tools: Iterable[Any]) -> dict[str, tuple[ToolSpec, Any]]:
    """Build a registry from the exact tool objects bound to the graph.

    Unknown-to-ShopSift test doubles are registered as read-only tools with a
    conservative call limit.  A tool name not present in ``tools`` remains
    unregistered and is denied by the gateway.
    """
    registry: dict[str, tuple[ToolSpec, Any]] = {}
    for tool in tools:
        name = str(getattr(tool, "name", ""))
        if not name:
            raise ValueError("tool name is required")
        if name in registry:
            raise ValueError(f"duplicate tool name: {name}")
        spec = _known_spec(name)
        if spec is None:
            spec = ToolSpec(name=name, argument_schema={}, timeout_ms=1500, max_calls_per_run=3)
        registry[name] = (spec, tool)
    return registry


class ToolGateway:
    """Execute registered tools only after policy and argument checks."""

    def __init__(self, tools: Iterable[Any], registry: Mapping[str, tuple[ToolSpec, Any]] | None = None):
        self.registry = dict(build_tool_registry(tools) if registry is None else registry)

    def _audit(self, *, tool: str, decision: str, reason: str, stage: str, access: str = "") -> None:
        record_tool_policy(
            tool=tool,
            decision=decision,
            reason=reason,
            stage=stage,
            access=access,
        )
        log("tool_policy", tool=tool, decision=decision, reason=reason, stage=stage, access=access)

    @staticmethod
    def _error(call_id: str, name: str, code: str) -> ToolMessage:
        return ToolMessage(
            content=f"TOOL_ERROR[{code}]",
            tool_call_id=call_id or "unknown",
            name=name or None,
            status="error",
            response_metadata={
                "tool_gateway": {"executed": False, "code": code},
            },
        )

    @staticmethod
    def _validate_args(spec: ToolSpec, tool: Any, args: Any) -> str | None:
        if not isinstance(args, dict):
            return "invalid_args"
        schema_fields = set(spec.argument_schema)
        if schema_fields:
            unknown = set(args) - schema_fields
            if unknown:
                return "invalid_args"
            for field_name, rule in spec.argument_schema.items():
                if field_name not in args:
                    # Optional arguments are represented by a default in the
                    # LangChain schema and can be omitted by the model.
                    fields = getattr(getattr(tool, "args_schema", None), "model_fields", {}) or {}
                    field = fields.get(field_name)
                    if field is not None and not field.is_required():
                        continue
                    return "invalid_args"
                value = args[field_name]
                expected_type = rule.get("type")
                if expected_type is int and (not isinstance(value, int) or isinstance(value, bool)):
                    return "invalid_args"
                if expected_type is str and not isinstance(value, str):
                    return "invalid_args"
                if isinstance(value, str):
                    if "min_length" in rule and len(value) < rule["min_length"]:
                        return "invalid_args"
                    if "max_length" in rule and len(value) > rule["max_length"]:
                        return "invalid_args"
                if isinstance(value, int):
                    if "min" in rule and value < rule["min"]:
                        return "invalid_args"
                    if "max" in rule and value > rule["max"]:
                        return "invalid_args"
                choices = rule.get("choices")
                if choices is not None and value not in choices:
                    return "invalid_args"

            if spec.name == "compare_products":
                values = [item.strip() for item in args["product_ids"].split(",")]
                if not 2 <= len(values) <= 4 or any(
                    not item.isdigit() or not 1 <= int(item) <= 1_000_000_000
                    for item in values
                ) or len({int(item) for item in values}) != len(values):
                    return "invalid_args"
        else:
            # Let custom/test tools keep their native schema validation, while
            # rejecting non-dict arguments before invocation.
            fields = getattr(getattr(tool, "args_schema", None), "model_fields", {}) or {}
            if fields:
                try:
                    tool.args_schema.model_validate(args)
                except Exception:
                    return "invalid_args"
        return None

    @staticmethod
    def _invoke_with_timeout(tool: Any, args: dict, timeout_ms: int) -> tuple[Any, str | None]:
        if timeout_ms <= 0:
            try:
                return tool.invoke(args), None
            except Exception as exc:  # pragma: no cover - caller records stable code
                return None, type(exc).__name__

        result: list[Any] = []
        error: list[BaseException] = []
        done = threading.Event()

        def worker() -> None:
            try:
                result.append(tool.invoke(args))
            except BaseException as exc:  # keep worker from leaking exceptions
                error.append(exc)
            finally:
                done.set()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        if not done.wait(timeout_ms / 1000):
            return None, "timeout"
        if error:
            return None, type(error[0]).__name__
        return result[0] if result else "", None

    def invoke(self, state: Mapping[str, Any]) -> dict[str, Any]:
        stage = str(state.get("stage", "discovery"))
        counts = dict(state.get("tool_call_counts", {}) or {})
        messages: list[ToolMessage] = []
        last_message = (state.get("messages") or [])[-1] if state.get("messages") else None
        calls = getattr(last_message, "tool_calls", []) or []

        for call in calls:
            name = str(call.get("name", ""))
            call_id = str(call.get("id", "unknown"))
            entry = self.registry.get(name)
            if entry is None:
                self._audit(tool=name, decision="deny", reason="unregistered_tool", stage=stage)
                messages.append(self._error(call_id, name, "unregistered_tool"))
                continue

            spec, tool = entry
            if stage not in spec.allowed_stages:
                self._audit(tool=name, decision="deny", reason="stage_not_allowed", stage=stage, access=spec.access)
                messages.append(self._error(call_id, name, "stage_not_allowed"))
                continue
            if counts.get(name, 0) >= spec.max_calls_per_run:
                self._audit(tool=name, decision="deny", reason="call_limit", stage=stage, access=spec.access)
                messages.append(self._error(call_id, name, "call_limit"))
                continue

            args = call.get("args", {})
            reason = self._validate_args(spec, tool, args)
            if (
                reason is None
                and "conv_id" in spec.argument_schema
                and args.get("conv_id") != state.get("conv_id")
            ):
                reason = "conversation_mismatch"
            if reason:
                self._audit(tool=name, decision="deny", reason=reason, stage=stage, access=spec.access)
                messages.append(self._error(call_id, name, reason))
                continue
            if spec.approval_required and call_id not in set(state.get("approved_tool_calls", []) or []):
                self._audit(tool=name, decision="deny", reason="approval_required", stage=stage, access=spec.access)
                messages.append(self._error(call_id, name, "approval_required"))
                continue

            counts[name] = counts.get(name, 0) + 1
            self._audit(tool=name, decision="allow", reason="policy_pass", stage=stage, access=spec.access)
            result, error = self._invoke_with_timeout(tool, args, spec.timeout_ms)
            if error == "timeout":
                mark_timeout("tool", "timeout")
                self._audit(tool=name, decision="timeout", reason="timeout", stage=stage, access=spec.access)
                message = self._error(call_id, name, "timeout")
                message.response_metadata["tool_gateway"]["executed"] = True
                messages.append(message)
            elif error:
                record_failure("tool", error)
                self._audit(tool=name, decision="error", reason="tool_error", stage=stage, access=spec.access)
                message = self._error(call_id, name, "tool_error")
                message.response_metadata["tool_gateway"]["executed"] = True
                messages.append(message)
            else:
                messages.append(
                    ToolMessage(
                        content=str(result) if result is not None else "",
                        tool_call_id=call_id,
                        name=name,
                        status="success",
                        response_metadata={
                            "tool_gateway": {"executed": True, "code": "allowed"},
                        },
                    )
                )

        return {"messages": messages, "tool_call_counts": counts}
