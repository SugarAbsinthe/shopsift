"""Privacy, context isolation, telemetry, and optional tracing tests."""

import asyncio
import json
import logging
from types import SimpleNamespace

from backend.logging_config import (
    Timer,
    create_run_telemetry,
    get_request_id,
    hash_identifier,
    log,
    logger,
    mark_cache_hit,
    mark_retrieval,
    record_failure,
    record_fallback,
    record_executed_tools,
    record_llm_response,
    record_llm_retry,
    record_retrieval_stats,
    record_requested_tools,
    record_tool_policy,
    reset_request_id,
    run_context,
    set_request_id,
)
from src.config import config, init_langsmith


class ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_structured_log_is_json_and_redacts_credentials():
    handler = ListHandler()
    logger.addHandler(handler)
    request_token = set_request_id("req-safe")
    telemetry = create_run_telemetry("run-safe")
    try:
        with run_context(telemetry):
            log(
                "provider_error",
                error="Bearer secret-token-value sk-abcdefghijk password=hunter2",
            )
    finally:
        reset_request_id(request_token)
        logger.removeHandler(handler)

    payload = json.loads(handler.messages[-1])
    serialized = json.dumps(payload)
    assert payload["event"] == "provider_error"
    assert payload["request_id"] == "req-safe"
    assert payload["run_id"] == "run-safe"
    assert "secret-token-value" not in serialized
    assert "sk-abcdefghijk" not in serialized
    assert "hunter2" not in serialized
    assert serialized.count("[REDACTED]") >= 3


def test_request_context_is_isolated_between_async_tasks():
    async def worker(request_id):
        token = set_request_id(request_id)
        try:
            await asyncio.sleep(0)
            return get_request_id()
        finally:
            reset_request_id(token)

    async def collect():
        return await asyncio.gather(worker("req-one"), worker("req-two"))

    assert asyncio.run(collect()) == ["req-one", "req-two"]
    assert get_request_id() == ""


def test_invalid_request_id_is_replaced_and_identifier_hash_is_stable():
    token = set_request_id("bad\nrequest")
    try:
        generated = get_request_id()
    finally:
        reset_request_id(token)
    assert generated != "bad\nrequest"
    assert len(generated) == 12
    assert hash_identifier("conversation-1") == hash_identifier("conversation-1")
    assert hash_identifier("conversation-1") != hash_identifier("conversation-2")


def test_run_telemetry_records_only_available_usage():
    telemetry = create_run_telemetry("run-metrics")
    response = SimpleNamespace(
        usage_metadata={
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
        },
        response_metadata={},
    )
    with run_context(telemetry):
        with Timer("llm_call", attempt=1):
            pass
        record_llm_retry()
        record_llm_response(response)
        mark_retrieval()
        mark_cache_hit()
        record_retrieval_stats({
            "source_hits": {"description": 5, "spec": 4, "sparse": 2},
            "returned_candidates": 3,
            "index_version": "v2",
            "query": "must-not-be-recorded",
        })
        record_requested_tools(["search_products"])
        record_executed_tools(["search_products", "get_reviews"], ["success", "error"])

    snapshot = telemetry.snapshot()
    assert snapshot["llm_calls"] == 1
    assert snapshot["llm_retries"] == 1
    assert snapshot["input_tokens"] == 11
    assert snapshot["output_tokens"] == 7
    assert snapshot["total_tokens"] == 18
    assert snapshot["retrieval_triggered"] is True
    assert snapshot["cache_hit"] is True
    assert snapshot["requested_tools"] == ["search_products"]
    assert snapshot["executed_tools"] == ["search_products", "get_reviews"]
    assert snapshot["tool_errors"] == 1
    assert snapshot["retrieval_stats"]["returned_candidates"] == 3
    assert "query" not in snapshot["retrieval_stats"]


def test_run_telemetry_records_versioned_cost_and_bounded_diagnostics():
    telemetry = create_run_telemetry(
        "run-diagnostics",
        provider="compatible-provider",
        model="test-model",
        prompt_version="prompt-v1",
        pricing_version="pricing-v1",
        input_cost_per_million=1.0,
        output_cost_per_million=2.0,
    )
    response = SimpleNamespace(
        usage_metadata={"input_tokens": 1000, "output_tokens": 500},
        response_metadata={},
    )
    with run_context(telemetry):
        with Timer("graph_node", node="agent"):
            pass
        record_llm_response(response)
        record_failure("model", "TimeoutError")
        record_fallback("retrieval_unavailable")
        record_tool_policy(
            tool="get_product_detail",
            decision="allow",
            reason="policy_pass",
            stage="search",
            access="read_only",
        )

    snapshot = telemetry.snapshot()
    assert snapshot["provider"] == "compatible-provider"
    assert snapshot["model"] == "test-model"
    assert snapshot["prompt_version"] == "prompt-v1"
    assert snapshot["pricing_version"] == "pricing-v1"
    assert snapshot["total_tokens"] == 1500
    assert snapshot["estimated_cost_usd"] == 0.002
    assert "agent" in snapshot["node_latency_ms"]
    assert snapshot["failure_counts"] == {"model:TimeoutError": 1}
    assert snapshot["fallbacks"] == ["retrieval_unavailable"]
    assert snapshot["tool_policy"] == [{
        "tool": "get_product_detail",
        "decision": "allow",
        "reason": "policy_pass",
        "stage": "search",
        "access": "read_only",
    }]


def test_langsmith_is_disabled_by_default(monkeypatch):
    monkeypatch.setattr(config, "LANGSMITH_TRACING", False)
    environ = {"LANGSMITH_TRACING": "true"}
    assert init_langsmith(environ) is False
    assert config.LANGSMITH_TRACING is False
    assert environ["LANGSMITH_TRACING"] == "false"


def test_langsmith_requires_an_explicit_key(monkeypatch):
    monkeypatch.setattr(config, "LANGSMITH_TRACING", True)
    monkeypatch.setattr(config, "LANGSMITH_API_KEY", "")
    environ = {}
    assert init_langsmith(environ) is False
    assert environ["LANGSMITH_TRACING"] == "false"


def test_langsmith_can_be_enabled_without_network_access(monkeypatch):
    monkeypatch.setattr(config, "LANGSMITH_TRACING", True)
    monkeypatch.setattr(config, "LANGSMITH_API_KEY", "test-placeholder")
    monkeypatch.setattr(config, "LANGSMITH_PROJECT", "eval-project")
    environ = {}
    assert init_langsmith(environ) is True
    assert environ["LANGSMITH_TRACING"] == "true"
    assert environ["LANGSMITH_API_KEY"] == "test-placeholder"
    assert environ["LANGSMITH_PROJECT"] == "eval-project"
