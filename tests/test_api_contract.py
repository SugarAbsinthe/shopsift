"""Regression tests for chat metadata and conversation cleanup contracts."""

import asyncio
import json

import pytest
from fastapi import HTTPException

from backend.schemas import ChatRequest
from backend.logging_config import get_request_id, reset_request_id, set_request_id
from backend.routers import chat as chat_router
from backend.routers import conversations as conversation_router


class FakeAgent:
    def __init__(self, error=None):
        self.error = error

    def run(self, **kwargs):
        if self.error:
            raise self.error
        return {
            "answer": "ok",
            "stage": "search",
            "product_context": "context",
            "user_profile": "profile",
            "tool_rounds": 1,
            "agent_rounds": 2,
            "stop_reason": "completed",
            "run_id": "run-test",
            "request_id": get_request_id(),
            "latency_ms": 25,
            "llm_calls": 2,
            "llm_latency_ms": 20,
            "llm_retries": 0,
            "input_tokens": 12,
            "output_tokens": 8,
            "total_tokens": 20,
            "retrieval_triggered": True,
            "cache_hit": False,
            "requested_tools": ["search_products"],
            "executed_tools": ["search_products"],
            "tool_errors": 0,
            "retrieval_stats": {"returned_candidates": 3},
            "provider": "compatible-provider",
            "model": "test-model",
            "prompt_version": "prompt-v1",
            "pricing_version": "pricing-v1",
            "estimated_cost_usd": 0.001,
            "node_latency_ms": {"agent": 20},
            "failure_counts": {},
            "fallbacks": [],
            "tool_policy": [],
            "timeouts": 0,
            "cancelled": False,
            "budget_stop": None,
            "checkpoint_restored": True,
        }


def test_chat_response_includes_execution_metadata(monkeypatch):
    monkeypatch.setattr(chat_router, "get_agent", lambda: FakeAgent())
    token = set_request_id("request-test")
    try:
        response = asyncio.run(chat_router.chat(ChatRequest(
            conv_id="conv-1", question="hello", chat_history=[]
        )))
    finally:
        reset_request_id(token)
    assert response.agent_rounds == 2
    assert response.stop_reason == "completed"
    assert response.run_id == "run-test"
    assert response.request_id == "request-test"
    assert response.llm_calls == 2
    assert response.total_tokens == 20
    assert response.executed_tools == ["search_products"]
    assert response.retrieval_stats == {"returned_candidates": 3}
    assert response.provider == "compatible-provider"
    assert response.model == "test-model"
    assert response.prompt_version == "prompt-v1"
    assert response.estimated_cost_usd == 0.001
    assert response.node_latency_ms == {"agent": 20}
    assert response.checkpoint_restored is True


def test_chat_error_does_not_expose_internal_exception(monkeypatch):
    monkeypatch.setattr(
        chat_router, "get_agent", lambda: FakeAgent(ValueError("secret detail"))
    )
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(chat_router.chat(ChatRequest(
            conv_id="conv-1", question="hello", chat_history=[]
        )))
    assert exc_info.value.detail == "Agent execution failed"
    assert "secret" not in exc_info.value.detail


def test_chat_timeout_sets_cooperative_cancellation(monkeypatch):
    class BlockingAgent:
        def __init__(self):
            self.cancellation_event = None

        def run(self, **kwargs):
            self.cancellation_event = kwargs["cancellation_event"]
            self.cancellation_event.wait(1)
            return {}

    agent = BlockingAgent()
    monkeypatch.setattr(chat_router, "get_agent", lambda: agent)
    monkeypatch.setattr(chat_router, "_AGENT_TIMEOUT", 0.01)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(chat_router.chat(ChatRequest(
            conv_id="conv-timeout", question="hello", chat_history=[]
        )))

    assert exc_info.value.status_code == 504
    assert exc_info.value.detail == "Agent response timed out, please try again"
    assert agent.cancellation_event is not None
    assert agent.cancellation_event.is_set()


def test_stream_keeps_request_context_after_response_creation(monkeypatch):
    class FakeStreamAgent:
        async def run_stream(self, **kwargs):
            payload = json.dumps({"request_id": get_request_id()})
            yield f"event: done\ndata: {payload}\n\n"

    monkeypatch.setattr(chat_router, "get_agent", lambda: FakeStreamAgent())

    async def execute():
        token = set_request_id("stream-request")
        response = await chat_router.chat_stream(ChatRequest(
            conv_id="conv-stream", question="hello", chat_history=[]
        ))
        reset_request_id(token)
        chunks = [chunk async for chunk in response.body_iterator]
        return "".join(chunks)

    body = asyncio.run(execute())
    payload = json.loads(body.split("data: ", 1)[1])
    assert payload["request_id"] == "stream-request"
    assert get_request_id() == ""


def test_delete_conversation_clears_runtime_state(monkeypatch):
    deleted = []
    cleared = []

    class FakeStore:
        def get_conversation(self, conv_id):
            return {"id": conv_id}

        def delete_conversation(self, conv_id):
            deleted.append(conv_id)

    monkeypatch.setattr(conversation_router, "get_conv_store", lambda: FakeStore())
    monkeypatch.setattr(
        conversation_router, "clear_conversation_runtime", lambda conv_id: cleared.append(conv_id)
    )
    result = asyncio.run(conversation_router.delete_conversation("conv-1"))
    assert result == {"deleted": "conv-1"}
    assert deleted == ["conv-1"]
    assert cleared == ["conv-1"]


def test_clear_messages_also_clears_runtime_state(monkeypatch):
    cleared_messages = []
    cleared_runtime = []

    class FakeStore:
        def get_conversation(self, conv_id):
            return {"id": conv_id}

        def clear_messages(self, conv_id):
            cleared_messages.append(conv_id)

    monkeypatch.setattr(conversation_router, "get_conv_store", lambda: FakeStore())
    monkeypatch.setattr(
        conversation_router,
        "clear_conversation_runtime",
        lambda conv_id: cleared_runtime.append(conv_id),
    )
    result = asyncio.run(conversation_router.clear_messages("conv-1"))
    assert result == {"cleared": "conv-1"}
    assert cleared_messages == ["conv-1"]
    assert cleared_runtime == ["conv-1"]
