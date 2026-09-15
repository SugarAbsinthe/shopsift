"""LangGraph state machine for the shopping guide Agent.

Five-node graph:
  analyze  — load profile, classify stage, extract new profile signals
  retrieve — profile-augmented product search
  agent    — LLM with tools and stage-specific prompt
  tools    — ToolNode executes tool calls
  finalize — tool-free answer when the loop reaches its configured limit

Flow: analyze → retrieve → agent ⇄ tools → END, or agent → finalize → END

Why separate nodes instead of flattening everything into one:
  - analyze and retrieve are rule-driven, deterministic steps — keeping them
    separate makes behavior predictable and debuggable
  - agent and tools are the LLM-driven loop — isolating them prevents the
    deterministic steps from being re-executed every tool round
  - The conditional edge after agent is the key control point: the model can
    request tools, finish normally, or be routed to a safe final response
"""

import asyncio
import sqlite3
import threading
from pathlib import Path
from typing import Annotated, TypedDict, Literal

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from src.retrieval.models import RetrievalResult, VerificationResult
from src.retrieval.verifier import verify_answer

from backend.logging_config import (
    Timer,
    create_run_telemetry,
    hash_identifier,
    log,
    mark_retrieval,
    record_executed_tools,
    record_llm_response,
    record_llm_retry,
    record_requested_tools,
    run_context,
)


class ShoppingState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    conv_id: str
    stage: str
    product_context: str
    user_profile: str
    tool_rounds: int
    agent_rounds: int
    stop_reason: str
    profile_constraints: dict
    retrieval_result: dict
    evidence: list[dict]
    verification: dict


class ShoppingGuideGraph:
    """LangGraph state machine for shopping guide conversations.

    Five-node graph:
      analyze  — load profile, classify stage, extract new profile signals
      retrieve — profile-augmented product search
      agent    — LLM with tools
                  dynamically selects per-stage prompt to stay focused
      tools    — ToolNode executes tool calls
      finalize — tool-free answer after a forced loop stop

    Flow: analyze → retrieve → agent ⇄ tools → END, or agent → finalize → END
    """

    def __init__(self, llm, tools: list, product_retriever, profile_store,
                 system_prompt: str, stage_classifier_prompt: str,
                 max_tool_rounds: int = 3, stage_prompts: dict = None,
                 checkpoint_db_path: str = None):
        self.llm = llm
        self.llm_with_tools = llm.bind_tools(tools)
        self.tools = tools
        self.product_retriever = product_retriever
        self.profile_store = profile_store
        self.system_prompt = system_prompt
        self.stage_classifier_prompt = stage_classifier_prompt
        self.max_tool_rounds = max_tool_rounds
        self.stage_prompts = stage_prompts or {}
        self.tool_node = ToolNode(self.tools, handle_tool_errors=True)

        self._checkpoint_conn = None
        self.checkpointer = None
        if checkpoint_db_path:
            checkpoint_path = Path(checkpoint_db_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self._checkpoint_conn = sqlite3.connect(
                str(checkpoint_path), check_same_thread=False
            )
            self.checkpointer = SqliteSaver(self._checkpoint_conn)

        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(ShoppingState)

        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tools_node)
        workflow.add_node("finalize", self._finalize_node)

        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "retrieve")
        workflow.add_edge("retrieve", "agent")
        workflow.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {"tools": "tools", "finalize": "finalize", "end": END}
        )
        workflow.add_edge("tools", "agent")
        workflow.add_edge("finalize", END)

        return workflow.compile(checkpointer=self.checkpointer)

    # ---- Nodes ----

    def _analyze_node(self, state: ShoppingState) -> dict:
        conv_id = state["conv_id"]
        messages = state["messages"]

        # Load current profile
        user_profile = self.profile_store.serialize_profile(conv_id)

        # Get last user message for stage classification
        last_user_msg = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                last_user_msg = m.content
                break

        # Classify stage
        stage = self._classify_stage(last_user_msg, state.get("stage", "discovery"))

        # Extract profile signals from user message (lightweight extraction)
        if last_user_msg:
            self._extract_profile_signals(conv_id, last_user_msg)

        # Reload profile after extraction
        user_profile = self.profile_store.serialize_profile(conv_id)
        structured_getter = getattr(self.profile_store, "get_structured", None)
        structured_profile = structured_getter(conv_id) if structured_getter else {}

        return {
            "stage": stage,
            "user_profile": user_profile,
            "profile_constraints": build_retrieval_constraints(structured_profile),
        }

    def _retrieve_node(self, state: ShoppingState) -> dict:
        stage = state.get("stage", "discovery")
        user_profile = state.get("user_profile", "")
        profile_constraints = state.get("profile_constraints", {})
        messages = state["messages"]

        # Only search products in relevant stages
        if stage not in ("search", "comparison", "recommendation", "objection_handling"):
            return {"product_context": "", "retrieval_result": {}, "evidence": []}

        # Build profile-augmented query from last user message
        last_user_msg = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                last_user_msg = m.content
                break

        if not last_user_msg:
            return {"product_context": "", "retrieval_result": {}, "evidence": []}

        mark_retrieval()

        # Augment query with profile context
        augmented_query = last_user_msg
        if user_profile and user_profile != "(暂无画像)":
            augmented_query = f"{last_user_msg}\n用户画像: {user_profile}"

        try:
            retrieve_kwargs = {"top_k": 5}
            if profile_constraints:
                retrieve_kwargs["filters"] = profile_constraints
            structured_retriever = getattr(self.product_retriever, "retrieve_result", None)
            if structured_retriever is not None:
                retrieval_result = structured_retriever(augmented_query, **retrieve_kwargs)
                if not isinstance(retrieval_result, RetrievalResult):
                    retrieval_result = RetrievalResult.model_validate(retrieval_result)
                formatter = getattr(self.product_retriever, "format_result", None)
                product_context = (
                    formatter(retrieval_result)
                    if formatter is not None
                    else self.product_retriever.retrieve(augmented_query, **retrieve_kwargs)
                )
                return {
                    "product_context": product_context,
                    "retrieval_result": retrieval_result.model_dump(
                        exclude={"reviews_by_product"}
                    ),
                    "evidence": [item.model_dump() for item in retrieval_result.evidence],
                }
            product_context = self.product_retriever.retrieve(augmented_query, **retrieve_kwargs)
        except Exception:
            product_context = "(产品检索暂时不可用)"

        return {"product_context": product_context, "retrieval_result": {}, "evidence": []}

    def _invoke_with_retry(self, messages: list, max_retries: int = 3, model=None):
        """Invoke LLM with exponential backoff on transient failures.

        Only retries on infrastructure errors (timeout, rate limit, connection
        reset, 502/503). Does NOT retry on model-level errors (bad request,
        context too long) — those need code or prompt fixes, not retries.
        Delay: 1s → 2s → 4s (3 attempts max).
        """
        import time as _time
        target_model = model or self.llm_with_tools
        last_exc = None
        for attempt in range(max_retries):
            try:
                with Timer("llm_call", attempt=attempt + 1):
                    result = target_model.invoke(messages)
                record_llm_response(result)
                return result
            except Exception as e:
                last_exc = e
                err_str = str(e).lower()
                if not any(kw in err_str for kw in ("timeout", "rate limit", "429", "connection", "reset", "503", "502")):
                    raise
                if attempt < max_retries - 1:
                    delay = 2 ** attempt  # 1s, 2s, 4s
                    record_llm_retry()
                    log("llm_retry", attempt=attempt + 2, delay=delay)
                    _time.sleep(delay)
        log("llm_fail", attempts=max_retries, error_type=type(last_exc).__name__)
        raise last_exc

    @staticmethod
    def _verify_response(response: AIMessage, state: ShoppingState) -> tuple[AIMessage, dict, str | None]:
        """Verify only explicit catalog facts when structured retrieval exists."""
        if getattr(response, "tool_calls", None):
            result = VerificationResult(failure_codes=["tool_calls_pending"])
            return response, result.model_dump(), None
        result = verify_answer(response.content, state.get("retrieval_result"))
        if result.status != "failed":
            return response, result.model_dump(), None
        safe_response = AIMessage(content="当前信息不足以确认产品或价格，请调整条件后重试。")
        return safe_response, result.model_dump(), "verification_failed"

    def _agent_node(self, state: ShoppingState) -> dict:
        stage = state.get("stage", "discovery")
        user_profile = state.get("user_profile", "(暂无画像)")
        product_context = state.get("product_context", "")
        conv_id = state.get("conv_id", "")
        agent_rounds = state.get("agent_rounds", 0)

        # Select per-stage prompt, fall back to default system prompt
        prompt = self.stage_prompts.get(stage, self.system_prompt)
        system_text = prompt.format(
            conv_id=conv_id,
            stage=stage,
            user_profile=user_profile,
            product_context=product_context or "(尚未搜索产品，请先挖掘用户需求)",
        )

        # Prepare messages for LLM: system + conversation
        full_messages = [SystemMessage(content=system_text)] + list(state["messages"])

        response = self._invoke_with_retry(full_messages)
        response, verification, verification_stop = self._verify_response(response, state)
        record_requested_tools(
            call.get("name", "") for call in (response.tool_calls or [])
        )

        return {
            "messages": [response],
            "agent_rounds": agent_rounds + 1,
            "verification": verification,
            "stop_reason": (
                verification_stop or
                (state.get("stop_reason") or "completed")
                if not response.tool_calls else ""
            ),
        }

    def _tools_node(self, state: ShoppingState) -> dict:
        """Execute requested tools and count actual tool-node rounds."""
        try:
            result = self.tool_node.invoke(state)
            tool_messages = result.get("messages", [])
            has_error = any(
                isinstance(message, ToolMessage)
                and getattr(message, "status", "success") == "error"
                for message in tool_messages
            )
            record_executed_tools(
                [getattr(message, "name", "") or "tool" for message in tool_messages],
                [getattr(message, "status", "success") for message in tool_messages],
            )
            return {
                "messages": tool_messages,
                "tool_rounds": state.get("tool_rounds", 0) + 1,
                "stop_reason": "tool_error" if has_error else "",
            }
        except Exception as exc:
            last = state.get("messages", [])[-1] if state.get("messages") else None
            tool_messages = []
            for call in getattr(last, "tool_calls", []) or []:
                tool_messages.append(ToolMessage(
                    content="工具执行失败，请基于已有信息回答。",
                    tool_call_id=call.get("id", "unknown"),
                    status="error",
                ))
            record_executed_tools(
                [call.get("name", "") or "tool" for call in getattr(last, "tool_calls", []) or []],
                ["error"] * len(tool_messages),
            )
            return {
                "messages": tool_messages,
                "tool_rounds": state.get("tool_rounds", 0) + 1,
                "stop_reason": "tool_error",
            }

    def _finalize_node(self, state: ShoppingState) -> dict:
        """Produce a user-facing answer after a forced loop termination."""
        stage = state.get("stage", "discovery")
        user_profile = state.get("user_profile", "(暂无画像)")
        product_context = state.get("product_context", "")
        conv_id = state.get("conv_id", "")
        prompt = self.stage_prompts.get(stage, self.system_prompt)
        system_text = prompt.format(
            conv_id=conv_id,
            stage=stage,
            user_profile=user_profile,
            product_context=product_context or "(尚未搜索产品)",
        )
        conversation = list(state.get("messages", []))
        skipped_tool_messages = []
        last_message = conversation[-1] if conversation else None
        for call in getattr(last_message, "tool_calls", []) or []:
            skipped_tool_messages.append(ToolMessage(
                content="工具调用因达到最大轮次而跳过。",
                tool_call_id=call.get("id", "unknown"),
                status="error",
            ))
        full_messages = [
            SystemMessage(content=(
                system_text
                + "\n工具调用已停止。不得再调用任何工具，请严格基于现有对话、"
                  "检索结果和工具返回生成一个非空的最终答复；信息不足时明确说明。"
            )),
            *conversation,
            *skipped_tool_messages,
        ]
        response = self._invoke_with_retry(full_messages, model=self.llm)
        if not getattr(response, "content", ""):
            response = AIMessage(content="已达到工具调用上限，现有信息不足以形成可靠结论，请补充需求后重试。")
        response, verification, verification_stop = self._verify_response(response, state)
        return {
            "messages": [*skipped_tool_messages, response],
            "agent_rounds": state.get("agent_rounds", 0) + 1,
            "verification": verification,
            "stop_reason": verification_stop or state.get("stop_reason") or "max_tool_rounds",
        }

    # ---- Routing ----

    def _route_after_agent(self, state: ShoppingState) -> Literal["tools", "finalize", "end"]:
        """Route after agent node: continue to tools if LLM requested tool calls
        and we haven't hit the limit. Otherwise end the turn.

        The max_tool_rounds cap (default 3) prevents infinite agent-tool loops.
        When exceeded, pending tool calls are skipped and the graph routes to
        a tool-free finalize node that must produce a user-facing response.
        """
        messages = state["messages"]
        tool_rounds = state.get("tool_rounds", 0)

        last_msg = messages[-1] if messages else None
        if last_msg and isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            if tool_rounds >= self.max_tool_rounds:
                return "finalize"
            return "tools"
        return "end"

    # ---- Helpers ----

    def _classify_stage(self, user_message: str, current_stage: str) -> str:
        """Classify the conversation stage via lightweight LLM call."""
        return classify_stage(user_message, current_stage, self.llm, self.stage_classifier_prompt)

    def _extract_profile_signals(self, conv_id: str, user_message: str) -> None:
        """Lightweight profile signal extraction from user message."""
        extract_profile_signals(conv_id, user_message, self.profile_store)

    # ---- Public API ----

    @staticmethod
    def _run_config(conv_id: str, telemetry) -> dict:
        """Build trace metadata without exposing the raw conversation id."""
        conv_hash = hash_identifier(conv_id)
        return {
            "configurable": {"thread_id": conv_id},
            "tags": ["shopsift"],
            "metadata": {
                "run_id": telemetry.run_id,
                "request_id": telemetry.request_id,
                "conversation_hash": conv_hash,
            },
        }

    async def run_stream(self, user_message: str, conv_id: str,
                         chat_history: list = None):
        """Stream real model chunks plus graph lifecycle events as SSE."""
        import json as _json
        telemetry = create_run_telemetry()
        run_id = telemetry.run_id
        config = self._run_config(conv_id, telemetry)
        has_checkpoint = bool(self.checkpointer and self.checkpointer.get_tuple(config))
        initial_state = {
            "messages": ([HumanMessage(content=user_message)] if has_checkpoint else
                         (chat_history or []) + [HumanMessage(content=user_message)]),
            "conv_id": conv_id,
            "product_context": "",
            "user_profile": "",
            "tool_rounds": 0,
            "agent_rounds": 0,
            "stop_reason": "",
        }
        if not has_checkpoint:
            initial_state["stage"] = "discovery"

        def _emit(event_type, data):
            return f"event: {event_type}\ndata: {_json.dumps(data, ensure_ascii=False)}\n\n"

        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        cancelled = threading.Event()

        def _push(item):
            if not loop.is_closed():
                loop.call_soon_threadsafe(queue.put_nowait, item)

        def _worker():
            stream = None
            latest_state = dict(initial_state)
            try:
                with run_context(telemetry):
                    stream = self.graph.stream(
                        initial_state,
                        config=config,
                        stream_mode=["updates", "messages"],
                    )
                    for mode, payload in stream:
                        if cancelled.is_set():
                            break
                        if mode == "updates":
                            for node_output in payload.values():
                                if node_output:
                                    latest_state.update({
                                        key: value for key, value in node_output.items()
                                        if key != "messages"
                                    })
                        _push(("chunk", mode, payload))
                    if not cancelled.is_set():
                        final_state = (
                            dict(self.graph.get_state(config).values)
                            if self.checkpointer else latest_state
                        )
                        log(
                            "run_end",
                            stage=final_state.get("stage", "discovery"),
                            tool_rounds=final_state.get("tool_rounds", 0),
                            stop_reason=final_state.get("stop_reason", "completed"),
                            **telemetry.snapshot(),
                        )
                        _push(("done", final_state))
            except Exception as exc:
                with run_context(telemetry):
                    log("stream_worker_error", error_type=type(exc).__name__)
                _push(("error", exc))
            finally:
                close = getattr(stream, "close", None)
                if close:
                    close()

        worker_task = asyncio.create_task(asyncio.to_thread(_worker))
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=60)
                except asyncio.TimeoutError:
                    yield _emit("error", {
                        "run_id": run_id,
                        "message": "Agent 响应超时，请稍后重试",
                    })
                    break
                kind = item[0]
                if kind == "error":
                    yield _emit("error", {
                        "run_id": run_id,
                        "message": "Agent 执行失败，请稍后重试",
                    })
                    break
                if kind == "done":
                    result = item[1]
                    yield _emit("done", {
                        "stage": result.get("stage", "discovery"),
                        "tool_rounds": result.get("tool_rounds", 0),
                        "agent_rounds": result.get("agent_rounds", 0),
                        "stop_reason": result.get("stop_reason", "completed"),
                        "user_profile": result.get("user_profile", ""),
                        "product_context": result.get("product_context", ""),
                        **telemetry.snapshot(),
                    })
                    break

                _, mode, payload = item
                if mode == "messages":
                    message, metadata = payload
                    if metadata.get("langgraph_node") not in {"agent", "finalize"}:
                        continue
                    content = getattr(message, "content", "")
                    if isinstance(content, str) and content:
                        yield _emit("token", {"content": content})
                    continue

                if mode != "updates":
                    continue
                for node_name, node_output in payload.items():
                    node_output = node_output or {}
                    if node_name == "analyze":
                        yield _emit("stage", {"stage": node_output.get("stage", "discovery")})
                    elif node_name == "retrieve" and node_output.get("product_context"):
                        yield _emit("status", {"message": "已找到相关产品"})
                    elif node_name == "agent":
                        for message in node_output.get("messages", []):
                            calls = getattr(message, "tool_calls", []) or []
                            if calls:
                                yield _emit("tool_start", {
                                    "tools": [call.get("name", "") for call in calls],
                                })
                    elif node_name == "tools":
                        tool_messages = [
                            message for message in node_output.get("messages", [])
                            if isinstance(message, ToolMessage)
                        ]
                        yield _emit("tool_end", {
                            "tools": [getattr(message, "name", "") or "tool" for message in tool_messages],
                            "statuses": [getattr(message, "status", "success") for message in tool_messages],
                        })
        finally:
            cancelled.set()
            if not worker_task.done():
                worker_task.cancel()

    def run(self, user_message: str, conv_id: str,
            chat_history: list = None) -> dict:
        """Run the graph for one conversation turn.

        Args:
            user_message: The user's latest message.
            conv_id: Conversation ID for profile persistence.
            chat_history: Optional list of prior LangChain messages.

        Returns:
            dict with keys: messages, stage, product_context, user_profile, tool_rounds
        """
        telemetry = create_run_telemetry()
        config = self._run_config(conv_id, telemetry)
        has_checkpoint = bool(self.checkpointer and self.checkpointer.get_tuple(config))
        initial_state = {
            "messages": ([HumanMessage(content=user_message)] if has_checkpoint else
                         (chat_history or []) + [HumanMessage(content=user_message)]),
            "conv_id": conv_id,
            "product_context": "",
            "user_profile": "",
            "tool_rounds": 0,
            "agent_rounds": 0,
            "stop_reason": "",
        }
        if not has_checkpoint:
            initial_state["stage"] = "discovery"

        with run_context(telemetry):
            try:
                result = self.graph.invoke(initial_state, config=config)
            except Exception as exc:
                log("run_error", error_type=type(exc).__name__)
                raise
            log(
                "run_end",
                stage=result.get("stage", "discovery"),
                tool_rounds=result.get("tool_rounds", 0),
                stop_reason=result.get("stop_reason", "completed"),
                **telemetry.snapshot(),
            )

        return {
            "messages": result["messages"],
            "stage": result.get("stage", "discovery"),
            "product_context": result.get("product_context", ""),
            "user_profile": result.get("user_profile", ""),
            "tool_rounds": result.get("tool_rounds", 0),
            "agent_rounds": result.get("agent_rounds", 0),
            "stop_reason": result.get("stop_reason", "completed"),
            "retrieval_result": result.get("retrieval_result", {}),
            "evidence": result.get("evidence", []),
            "verification": result.get("verification", {}),
            **telemetry.snapshot(),
        }

    def clear_thread(self, conv_id: str) -> None:
        """Delete persisted graph state for one conversation."""
        if self.checkpointer:
            self.checkpointer.delete_thread(conv_id)

    def close(self) -> None:
        if self._checkpoint_conn is not None:
            self._checkpoint_conn.close()
            self._checkpoint_conn = None


# ---- Standalone helpers (usable by both old and new architecture) ----


def build_retrieval_constraints(structured_profile: dict) -> dict:
    """Map trusted profile fields to retrieval filters; ignore unknown keys."""
    import re

    def value(key: str) -> str:
        item = structured_profile.get(key, {})
        return str(item.get("value", "")) if isinstance(item, dict) else ""

    constraints: dict = {}
    budget = value("budget").strip()
    if budget:
        canonical_budget = parse_budget_expression(f"预算{budget}") or budget
        upper_match = re.fullmatch(r"<=\s*(\d+)", canonical_budget)
        range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", canonical_budget)
        single_match = re.fullmatch(r"(\d+)", canonical_budget)
        if upper_match:
            constraints["max_price"] = int(upper_match.group(1))
        elif range_match:
            minimum, maximum = map(int, range_match.groups())
            if minimum <= maximum:
                constraints["min_price"] = minimum
                constraints["max_price"] = maximum
        elif single_match:
            constraints["max_price"] = int(single_match.group(1))

    category = value("product_category").strip()
    if category:
        constraints["category"] = category
    preferred = value("preferred_brand").strip()
    excluded = value("exclude_brand").strip()
    if preferred:
        constraints["preferred_brands"] = [preferred]
    if excluded:
        constraints["excluded_brands"] = [excluded]
        if preferred.casefold() == excluded.casefold():
            constraints.pop("preferred_brands", None)
    return constraints


BUDGET_AROUND_TOLERANCE = 0.10
_PRICE_TOKEN_PATTERN = (
    r"(?:\d+(?:\.\d+)?\s*(?:[kKwW]|万|千)?|"
    r"[一二两三四五六七八九]万[一二两三四五六七八九]?|"
    r"[一二两三四五六七八九]千[一二两三四五六七八九]?)"
)


def _parse_price_amount(raw: str) -> int | None:
    """Parse a bounded set of shopping-price expressions without guessing."""
    import re

    token = raw.replace(",", "").replace("，", "").replace("元", "").strip()
    arabic = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kKwW]|万|千)?", token)
    if arabic:
        value = float(arabic.group(1))
        unit = arabic.group(2) or ""
        multiplier = 1000 if unit in {"k", "K", "千"} else 10000 if unit in {"w", "W", "万"} else 1
        return int(round(value * multiplier))

    digits = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9}
    chinese = re.fullmatch(r"([一二两三四五六七八九])(万|千)([一二两三四五六七八九])?", token)
    if chinese:
        leading = digits[chinese.group(1)]
        unit = chinese.group(2)
        trailing = digits.get(chinese.group(3), 0)
        if unit == "万":
            return leading * 10000 + trailing * 1000
        return leading * 1000 + trailing * 100
    return None


def parse_budget_expression(message: str) -> str | None:
    """Normalize explicit budget language into a profile-safe string.

    Canonical forms remain compatible with the existing string profile store:
    ``<=6000`` for a hard upper bound and ``5400-6600`` for a range.
    Approximate budgets use a configurable +/-10 percent interval.
    """
    import re

    text = message.replace(",", "").replace("，", "")
    flags = re.IGNORECASE

    range_match = re.search(
        rf"(?P<low>{_PRICE_TOKEN_PATTERN})\s*元?\s*(?:-|—|~|～|到|至)\s*"
        rf"(?P<high>{_PRICE_TOKEN_PATTERN})\s*元?",
        text,
        flags,
    )
    if range_match:
        low = _parse_price_amount(range_match.group("low"))
        high = _parse_price_amount(range_match.group("high"))
        if low is not None and high is not None and low <= high:
            return f"{low}-{high}"
        return None

    upper_match = re.search(
        rf"(?P<amount>{_PRICE_TOKEN_PATTERN})\s*元?\s*(?:以内|以下|之内)",
        text,
        flags,
    ) or re.search(
        rf"(?:不超过|最多|上限(?:是|为)?)\s*(?P<amount>{_PRICE_TOKEN_PATTERN})\s*元?",
        text,
        flags,
    )
    if upper_match:
        amount = _parse_price_amount(upper_match.group("amount"))
        return f"<={amount}" if amount is not None else None

    around_match = re.search(
        rf"(?P<amount>{_PRICE_TOKEN_PATTERN})\s*元?\s*(?:左右|上下)",
        text,
        flags,
    ) or re.search(
        rf"(?:大约|大概|约)\s*(?P<amount>{_PRICE_TOKEN_PATTERN})\s*元?",
        text,
        flags,
    )
    if around_match:
        amount = _parse_price_amount(around_match.group("amount"))
        if amount is None:
            return None
        lower = int(round(amount * (1 - BUDGET_AROUND_TOLERANCE)))
        upper = int(round(amount * (1 + BUDGET_AROUND_TOLERANCE)))
        return f"{lower}-{upper}"

    bare_match = re.search(
        rf"预算\s*[:：]?\s*(?P<amount>{_PRICE_TOKEN_PATTERN})\s*元?",
        text,
        flags,
    )
    if bare_match:
        amount = _parse_price_amount(bare_match.group("amount"))
        return f"<={amount}" if amount is not None else None
    return None


def classify_stage(user_message: str, current_stage: str, llm=None,
                   stage_classifier_prompt: str = "") -> str:
    """Classify the conversation stage.

    Rule-first strategy: regex keywords cover ~70% of real-world inputs
    (zero latency, zero cost). LLM only invoked for ambiguous cases.
    This is a cost-latency-accuracy tradeoff: rules are fast and predictable
    but brittle; LLM is flexible but costs a call. The right balance depends
    on how well your keywords match your actual user input patterns.
    """
    if not user_message:
        return current_stage or "discovery"

    msg_lower = user_message.lower()

    # Short greeting → discovery
    if len(user_message) < 10 and any(kw in msg_lower for kw in ["你好", "hi", "hello", "在吗"]):
        return "discovery"

    # Comparison keywords → comparison
    if any(kw in msg_lower for kw in ["对比", "比较", "区别", "哪个好", "选哪个", "vs"]):
        return "comparison"

    # Objection/concern keywords → objection_handling
    if any(kw in msg_lower for kw in ["质量", "售后", "靠谱吗", "行不行", "问题多", "会不会",
                                        "散热", "卡不卡", "耐用", "翻车", "差评"]):
        return "objection_handling"

    # Search intent → search
    if any(kw in msg_lower for kw in ["推荐", "找", "搜索", "有没有", "买什么", "选一个",
                                        "有什么", "哪些"]):
        return "search"

    # Needs keywords → needs_elicitation. Explicit search intent wins when a
    # message contains both requirements and a request for recommendations.
    if any(kw in msg_lower for kw in ["预算", "打游戏", "办公", "出差", "学生", "轻薄",
                                        "画图", "剪视频", "编程", "做图", "渲染"]):
        return "needs_elicitation"

    # Summary/closing
    if any(kw in msg_lower for kw in ["谢谢", "好的", "了解了", "就这个", "下单", "买了"]):
        return "summary"

    # Fallback: use LLM for ambiguous cases
    if llm is not None and stage_classifier_prompt:
        try:
            prompt = stage_classifier_prompt.format(
                current_stage=current_stage,
                user_message=user_message,
            )
            result = llm.invoke(prompt)
            stage = result.content.strip().lower()
            valid_stages = {"discovery", "needs_elicitation", "search", "comparison",
                            "objection_handling", "recommendation", "summary"}
            if stage in valid_stages:
                return stage
        except Exception:
            pass

    return current_stage or "discovery"


def extract_profile_signals(conv_id: str, user_message: str, profile_store) -> None:
    """Lightweight profile signal extraction from user message.

    Extracts: budget, product_category, primary_use, mobility,
    preferred_brand, exclude_brand. Callable from both
    ShoppingGuideGraph and standalone callers.
    """
    import re
    msg = user_message

    budget = parse_budget_expression(msg)
    if budget:
        try:
            profile_store.update(
                conv_id, "budget", budget, confidence=0.8, source="deduced"
            )
        except Exception:
            pass

    # Product category detection
    category_map = {
        "手机": "手机", "iPhone": "手机", "华为mate": "手机", "小米14": "手机",
        "笔记本": "笔记本电脑", "电脑": "笔记本电脑", "游戏本": "笔记本电脑",
        "轻薄本": "笔记本电脑", "macbook": "笔记本电脑", "thinkpad": "笔记本电脑",
        "平板": "平板电脑", "iPad": "平板电脑", "pad": "平板电脑",
        "耳机": "无线耳机", "airpods": "无线耳机", "降噪耳机": "无线耳机",
        "手表": "智能手表", "手环": "智能手表", "watch": "智能手表",
    }
    msg_lower = msg.lower()
    for keyword, cat_val in category_map.items():
        if keyword.lower() in msg_lower:
            profile_store.update(conv_id, "product_category", cat_val, confidence=0.75, source="deduced")
            break

    # Primary use detection
    use_map = {
        "游戏": "gaming", "打游戏": "gaming", "吃鸡": "gaming", "3a": "gaming",
        "办公": "office", "文档": "office", "ppt": "office", "excel": "office",
        "编程": "coding", "代码": "coding", "开发": "coding",
        "设计": "design", "ps": "design", "pr": "design", "剪视频": "design",
        "上课": "student", "学生": "student", "作业": "student",
        "出差": "office", "携带": "office",
    }
    for keyword, use_val in use_map.items():
        if keyword in msg_lower:
            profile_store.update(conv_id, "primary_use", use_val, confidence=0.75, source="deduced")
            break

    # Mobility detection
    if any(kw in msg for kw in ["出差", "携带", "通勤", "带去", "轻便", "轻薄", "经常带"]):
        profile_store.update(conv_id, "mobility", "high", confidence=0.8, source="deduced")

    # Brand preference
    brands = ["联想", "华硕", "苹果", "华为", "惠普", "戴尔", "小米", "宏碁", "thinkpad", "macbook"]
    for brand in brands:
        if brand.lower() in msg_lower:
            profile_store.update(conv_id, "preferred_brand", brand, confidence=0.7, source="deduced")
            break

    # Brand exclusion
    for brand in brands:
        if any(kw in msg for kw in [f"不要{brand}", f"排除{brand}", f"不买{brand}", f"除{brand}"]):
            profile_store.update(conv_id, "exclude_brand", brand, confidence=0.8, source="deduced")
            break
