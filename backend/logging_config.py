"""Structured, privacy-aware telemetry for Agent requests and runs."""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional


_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)
_run_ctx: contextvars.ContextVar[Optional["RunTelemetry"]] = contextvars.ContextVar(
    "run_telemetry", default=None
)
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_SECRET_PATTERNS = (
    re.compile(r"(?i)\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)\bghp_[A-Za-z0-9]{12,}\b"),
    re.compile(r"(?i)\bgithub_pat_[A-Za-z0-9_]{12,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)(api[_-]?key|password|token)\s*[=:]\s*[^\s,;]+"),
)
_MAX_LOG_TEXT = 500

_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(logging.Formatter("%(message)s"))

logger = logging.getLogger("shopsift")
logger.setLevel(logging.INFO)
logger.handlers = [_stream_handler]
logger.propagate = False


@dataclass
class RunTelemetry:
    run_id: str
    request_id: str = ""
    provider: str = "unknown"
    model: str = "unknown"
    prompt_version: str = "unknown"
    pricing_version: str = "unconfigured"
    input_cost_per_million: Optional[float] = None
    output_cost_per_million: Optional[float] = None
    started_at: float = field(default_factory=time.perf_counter)
    llm_calls: int = 0
    llm_latency_ms: int = 0
    llm_retries: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    retrieval_triggered: bool = False
    cache_hit: Optional[bool] = None
    requested_tools: list[str] = field(default_factory=list)
    executed_tools: list[str] = field(default_factory=list)
    tool_errors: int = 0
    retrieval_stats: dict[str, Any] = field(default_factory=dict)
    node_latency_ms: dict[str, int] = field(default_factory=dict)
    failure_counts: dict[str, int] = field(default_factory=dict)
    fallbacks: list[str] = field(default_factory=list)
    tool_policy: list[dict[str, str]] = field(default_factory=list)
    timeouts: int = 0
    cancelled: bool = False
    budget_stop: Optional[str] = None
    checkpoint_restored: bool = False

    def estimated_cost_usd(self) -> Optional[float]:
        if (
            self.input_cost_per_million is None
            or self.output_cost_per_million is None
            or self.input_tokens is None
            or self.output_tokens is None
        ):
            return None
        cost = (
            self.input_tokens * self.input_cost_per_million
            + self.output_tokens * self.output_cost_per_million
        ) / 1_000_000
        return round(cost, 8)

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "request_id": self.request_id,
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "pricing_version": self.pricing_version,
            "estimated_cost_usd": self.estimated_cost_usd(),
            "latency_ms": round((time.perf_counter() - self.started_at) * 1000),
            "llm_calls": self.llm_calls,
            "llm_latency_ms": self.llm_latency_ms,
            "llm_retries": self.llm_retries,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "retrieval_triggered": self.retrieval_triggered,
            "cache_hit": self.cache_hit,
            "requested_tools": list(self.requested_tools),
            "executed_tools": list(self.executed_tools),
            "tool_errors": self.tool_errors,
            "retrieval_stats": dict(self.retrieval_stats),
            "node_latency_ms": dict(self.node_latency_ms),
            "failure_counts": dict(self.failure_counts),
            "fallbacks": list(self.fallbacks),
            "tool_policy": [dict(item) for item in self.tool_policy],
            "timeouts": self.timeouts,
            "cancelled": self.cancelled,
            "budget_stop": self.budget_stop,
            "checkpoint_restored": self.checkpoint_restored,
        }


def set_request_id(rid: str = "") -> contextvars.Token:
    """Bind a validated request id and return the token needed to reset it."""
    normalized = rid if _REQUEST_ID_PATTERN.fullmatch(rid or "") else ""
    return _request_id_ctx.set(normalized or uuid.uuid4().hex[:12])


def reset_request_id(token: contextvars.Token) -> None:
    _request_id_ctx.reset(token)


def get_request_id() -> str:
    return _request_id_ctx.get("")


def hash_identifier(value: str) -> str:
    """Create a stable correlation key without logging the source identifier."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def create_run_telemetry(
    run_id: str = "",
    *,
    provider: str = "unknown",
    model: str = "unknown",
    prompt_version: str = "unknown",
    pricing_version: str = "unconfigured",
    input_cost_per_million: Optional[float] = None,
    output_cost_per_million: Optional[float] = None,
) -> RunTelemetry:
    return RunTelemetry(
        run_id=run_id or uuid.uuid4().hex,
        request_id=get_request_id(),
        provider=provider,
        model=model,
        prompt_version=prompt_version,
        pricing_version=pricing_version,
        input_cost_per_million=input_cost_per_million,
        output_cost_per_million=output_cost_per_million,
    )


@contextmanager
def run_context(telemetry: RunTelemetry):
    token = _run_ctx.set(telemetry)
    try:
        yield telemetry
    finally:
        _run_ctx.reset(token)


def get_run_telemetry() -> Optional[RunTelemetry]:
    return _run_ctx.get()


def record_llm_retry() -> None:
    telemetry = get_run_telemetry()
    if telemetry:
        telemetry.llm_retries += 1


def record_failure(category: str, code: str) -> None:
    """Record a bounded failure taxonomy without exception messages."""
    telemetry = get_run_telemetry()
    if not telemetry:
        return
    allowed_categories = {
        "model", "tool", "retrieval", "verification", "infrastructure",
        "cancellation", "budget",
    }
    normalized_category = category if category in allowed_categories else "infrastructure"
    normalized_code = re.sub(r"[^A-Za-z0-9_.-]", "_", str(code))[:64] or "unknown"
    key = f"{normalized_category}:{normalized_code}"
    telemetry.failure_counts[key] = telemetry.failure_counts.get(key, 0) + 1


def record_fallback(name: str) -> None:
    telemetry = get_run_telemetry()
    normalized = re.sub(r"[^A-Za-z0-9_.-]", "_", str(name))[:64]
    if telemetry and normalized and normalized not in telemetry.fallbacks:
        telemetry.fallbacks.append(normalized)


def record_tool_policy(
    *, tool: str, decision: str, reason: str, stage: str, access: str = ""
) -> None:
    telemetry = get_run_telemetry()
    if not telemetry:
        return
    telemetry.tool_policy.append({
        "tool": str(tool)[:64],
        "decision": str(decision)[:32],
        "reason": str(reason)[:64],
        "stage": str(stage)[:32],
        "access": str(access)[:32],
    })


def mark_timeout(category: str = "infrastructure", code: str = "timeout") -> None:
    telemetry = get_run_telemetry()
    if telemetry:
        telemetry.timeouts += 1
        record_failure(category, code)


def mark_cancelled() -> None:
    telemetry = get_run_telemetry()
    if telemetry:
        telemetry.cancelled = True
        record_failure("cancellation", "client_or_request")


def mark_budget_stop(reason: str) -> None:
    telemetry = get_run_telemetry()
    if telemetry:
        telemetry.budget_stop = reason
        record_failure("budget", reason)


def mark_checkpoint_restored(restored: bool) -> None:
    telemetry = get_run_telemetry()
    if telemetry:
        telemetry.checkpoint_restored = bool(restored)


def _add_optional_count(current: Optional[int], value: Any) -> Optional[int]:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return current
    return (current or 0) + value


def record_llm_response(response: Any) -> None:
    """Record provider usage only when integer counters are available."""
    telemetry = get_run_telemetry()
    if not telemetry:
        return

    usage = getattr(response, "usage_metadata", None) or {}
    if not usage:
        metadata = getattr(response, "response_metadata", None) or {}
        usage = metadata.get("token_usage") or metadata.get("usage") or {}

    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    total_tokens = usage.get("total_tokens")
    if (
        total_tokens is None
        and isinstance(input_tokens, int)
        and not isinstance(input_tokens, bool)
        and input_tokens >= 0
        and isinstance(output_tokens, int)
        and not isinstance(output_tokens, bool)
        and output_tokens >= 0
    ):
        total_tokens = input_tokens + output_tokens
    telemetry.input_tokens = _add_optional_count(telemetry.input_tokens, input_tokens)
    telemetry.output_tokens = _add_optional_count(telemetry.output_tokens, output_tokens)
    telemetry.total_tokens = _add_optional_count(telemetry.total_tokens, total_tokens)


def mark_retrieval() -> None:
    telemetry = get_run_telemetry()
    if telemetry:
        telemetry.retrieval_triggered = True
        if telemetry.cache_hit is None:
            telemetry.cache_hit = False


def mark_cache_hit() -> None:
    telemetry = get_run_telemetry()
    if telemetry:
        telemetry.retrieval_triggered = True
        telemetry.cache_hit = True


def record_retrieval_stats(stats: Any) -> None:
    """Record aggregate retrieval counters without query or profile content."""
    telemetry = get_run_telemetry()
    if not telemetry:
        return
    payload = stats.model_dump() if hasattr(stats, "model_dump") else dict(stats)
    allowed = {
        "source_hits",
        "fused_candidates",
        "filtered_candidates",
        "returned_candidates",
        "sparse_fallback",
        "index_version",
        "duration_ms",
    }
    telemetry.retrieval_stats = {
        key: payload[key] for key in allowed if key in payload
    }


def record_requested_tools(tool_names: Iterable[str]) -> None:
    telemetry = get_run_telemetry()
    if telemetry:
        telemetry.requested_tools.extend(name for name in tool_names if name)


def record_executed_tools(tool_names: Iterable[str], statuses: Iterable[str]) -> None:
    telemetry = get_run_telemetry()
    if not telemetry:
        return
    names = [name for name in tool_names if name]
    status_list = list(statuses)
    telemetry.executed_tools.extend(names)
    telemetry.tool_errors += sum(status == "error" for status in status_list)


def _sanitize_text(value: str) -> str:
    sanitized = value
    for pattern in _SECRET_PATTERNS:
        sanitized = pattern.sub("[REDACTED]", sanitized)
    if len(sanitized) > _MAX_LOG_TEXT:
        sanitized = sanitized[:_MAX_LOG_TEXT] + "…"
    return sanitized


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _sanitize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_value(item) for item in value]
    return _sanitize_text(str(value))


def log(event: str, **kwargs: Any) -> None:
    telemetry = get_run_telemetry()
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "level": "INFO",
        "event": _sanitize_text(event),
    }
    request_id = get_request_id() or (telemetry.request_id if telemetry else "")
    if request_id:
        payload["request_id"] = request_id
    if telemetry:
        payload["run_id"] = telemetry.run_id
    payload.update({key: _sanitize_value(value) for key, value in kwargs.items()})
    logger.info(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


class Timer:
    """Time a block, record LLM totals, and emit one structured event."""

    def __init__(self, event: str, **kwargs: Any):
        self.event = event
        self.kwargs = kwargs
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        duration_ms = round((time.perf_counter() - self._start) * 1000)
        telemetry = get_run_telemetry()
        if telemetry and self.event == "llm_call":
            telemetry.llm_calls += 1
            telemetry.llm_latency_ms += duration_ms
        if telemetry and self.event == "graph_node":
            node = str(self.kwargs.get("node", "unknown"))[:64]
            telemetry.node_latency_ms[node] = (
                telemetry.node_latency_ms.get(node, 0) + duration_ms
            )
        fields = dict(self.kwargs)
        if args and args[0] is not None:
            fields["status"] = "error"
            fields["error_type"] = args[0].__name__
        else:
            fields["status"] = "success"
        log(self.event, duration_ms=duration_ms, **fields)
