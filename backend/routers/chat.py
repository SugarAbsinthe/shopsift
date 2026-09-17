"""Chat endpoint — wraps the existing ShoppingGuideAgent as an HTTP API."""

from __future__ import annotations

import asyncio
import contextvars
import json as _json
import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage

from backend.dependencies import get_agent
from backend.schemas import ChatRequest, ChatResponse
from backend.logging_config import (
    get_request_id,
    hash_identifier,
    log,
    reset_request_id,
    set_request_id,
    Timer,
)
from src.config import config

router = APIRouter(tags=["chat"])


def _build_chat_history(chat_history: list) -> list:
    """Convert frontend ChatMessage list to LangChain message objects."""
    msgs = []
    for m in chat_history:
        if m.role == "user":
            msgs.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            msgs.append(AIMessage(content=m.content))
    return msgs[-20:]


def _format_sse(event: str, data: dict | str) -> str:
    """Format a single SSE message."""
    payload = _json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else data
    return f"event: {event}\ndata: {payload}\n\n"


_AGENT_TIMEOUT = config.AGENT_REQUEST_TIMEOUT_SECONDS


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the shopping guide Agent (non-streaming)."""
    agent = get_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    log(
        "request_start",
        conversation_hash=hash_identifier(request.conv_id),
        msg_len=len(request.question),
    )
    history = _build_chat_history(request.chat_history)
    cancellation_event = threading.Event()

    try:
        with Timer("agent_run"):
            context = contextvars.copy_context()
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: context.run(
                        agent.run,
                        question=request.question,
                        conv_id=request.conv_id,
                        chat_history=history,
                        cancellation_event=cancellation_event,
                    )
                ),
                timeout=_AGENT_TIMEOUT,
            )
        log(
            "request_end",
            run_id=result.get("run_id"),
            stage=result.get("stage"),
            tool_rounds=result.get("tool_rounds"),
            stop_reason=result.get("stop_reason"),
            latency_ms=result.get("latency_ms"),
        )
    except asyncio.TimeoutError:
        log(
            "request_timeout",
            conversation_hash=hash_identifier(request.conv_id),
            timeout=_AGENT_TIMEOUT,
        )
        raise HTTPException(status_code=504, detail="Agent response timed out, please try again")
    except Exception as e:
        log("request_error", error_type=type(e).__name__)
        raise HTTPException(status_code=500, detail="Agent execution failed")
    finally:
        cancellation_event.set()

    return ChatResponse(
        answer=result.get("answer", ""),
        stage=result.get("stage", "discovery"),
        product_context=result.get("product_context", ""),
        user_profile=result.get("user_profile", ""),
        tool_rounds=result.get("tool_rounds", 0),
        agent_rounds=result.get("agent_rounds", 0),
        stop_reason=result.get("stop_reason", "completed"),
        run_id=result.get("run_id", ""),
        request_id=result.get("request_id", ""),
        latency_ms=result.get("latency_ms", 0),
        llm_calls=result.get("llm_calls", 0),
        llm_latency_ms=result.get("llm_latency_ms", 0),
        llm_retries=result.get("llm_retries", 0),
        input_tokens=result.get("input_tokens"),
        output_tokens=result.get("output_tokens"),
        total_tokens=result.get("total_tokens"),
        retrieval_triggered=result.get("retrieval_triggered", False),
        cache_hit=result.get("cache_hit"),
        requested_tools=result.get("requested_tools", []),
        executed_tools=result.get("executed_tools", []),
        tool_errors=result.get("tool_errors", 0),
        retrieval_stats=result.get("retrieval_stats", {}),
        provider=result.get("provider", "unknown"),
        model=result.get("model", "unknown"),
        prompt_version=result.get("prompt_version", "unknown"),
        pricing_version=result.get("pricing_version", "unconfigured"),
        estimated_cost_usd=result.get("estimated_cost_usd"),
        node_latency_ms=result.get("node_latency_ms", {}),
        failure_counts=result.get("failure_counts", {}),
        fallbacks=result.get("fallbacks", []),
        tool_policy=result.get("tool_policy", []),
        timeouts=result.get("timeouts", 0),
        cancelled=result.get("cancelled", False),
        budget_stop=result.get("budget_stop"),
        checkpoint_restored=result.get("checkpoint_restored", False),
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Send a message to the Agent and stream the response via SSE."""
    agent = get_agent()
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    log(
        "stream_start",
        conversation_hash=hash_identifier(request.conv_id),
        msg_len=len(request.question),
    )
    history = _build_chat_history(request.chat_history)
    cancellation_event = threading.Event()
    stream_request_id = get_request_id()

    async def _event_generator():
        request_token = set_request_id(stream_request_id)
        try:
            async for sse_msg in agent.run_stream(
                question=request.question,
                conv_id=request.conv_id,
                chat_history=history,
                cancellation_event=cancellation_event,
            ):
                yield sse_msg
            log("stream_end")
        except Exception as exc:
            log("stream_error", error_type=type(exc).__name__)
            yield _format_sse("error", {"message": "Agent execution failed"})
        finally:
            cancellation_event.set()
            reset_request_id(request_token)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
