"""Shopping Guide Agent — wraps LLM + tools + retrievers + profile store.

LangGraph pipeline (analyze → retrieve → agent ⇄ tools → finalize) with
per-stage prompt injection, durable conversation state, and bounded tool loops.
"""

import threading
from typing import Optional
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage

from src.agent.langgraph_engine import ShoppingGuideGraph
from src.agent.shopping_tools import (
    create_get_product_detail,
    create_get_reviews,
    create_compare_products,
    create_get_user_profile,
)
from src.agent.shopping_prompts import (
    SHOPPING_SYSTEM_PROMPT,
    STAGE_CLASSIFIER_PROMPT,
    DISCOVERY_AGENT_PROMPT,
    SEARCH_AGENT_PROMPT,
    COMPARE_AGENT_PROMPT,
    RECOMMEND_AGENT_PROMPT,
    SHOPPING_PROMPT_VERSION,
)
from src.config import config


class ShoppingGuideAgent:
    """Complete shopping guide Agent with per-stage prompt injection.

    Wraps: LLM, ProductRetriever, ProfileStore, ShoppingGuideGraph (5-node).
    """

    def __init__(
        self,
        llm,
        product_retriever,
        profile_store,
        catalog_db: Optional[str] = None,
        reviews_db: Optional[str] = None,
        system_prompt: str = SHOPPING_SYSTEM_PROMPT,
        max_tool_rounds: int = 3,
        checkpoint_db_path: Optional[str] = None,
    ):
        self.llm = llm
        self.product_retriever = product_retriever
        self.profile_store = profile_store
        self.catalog_db = catalog_db or config.PRODUCT_DB_PATH
        self.reviews_db = reviews_db or str(Path(config.PRODUCT_DB_PATH).parent / "product_reviews.db")
        self.max_tool_rounds = max_tool_rounds
        self._system_prompt = system_prompt  # kept for run_simple fallback

        # Create tools via factory functions (no module globals)
        tools = [
            create_get_product_detail(self.catalog_db),
            create_get_reviews(self.reviews_db),
            create_compare_products(self.catalog_db),
            create_get_user_profile(profile_store),
        ]

        # Per-stage prompt mapping: the agent node dynamically selects
        # the right prompt based on the current conversation stage.
        stage_prompts = {
            "discovery": DISCOVERY_AGENT_PROMPT,
            "needs_elicitation": DISCOVERY_AGENT_PROMPT,
            "search": SEARCH_AGENT_PROMPT,
            "comparison": COMPARE_AGENT_PROMPT,
            "objection_handling": SEARCH_AGENT_PROMPT,
            "recommendation": RECOMMEND_AGENT_PROMPT,
            "summary": RECOMMEND_AGENT_PROMPT,
        }

        self.graph = ShoppingGuideGraph(
            llm=llm,
            tools=tools,
            product_retriever=product_retriever,
            profile_store=profile_store,
            system_prompt=system_prompt,
            stage_classifier_prompt=STAGE_CLASSIFIER_PROMPT,
            max_tool_rounds=max_tool_rounds,
            stage_prompts=stage_prompts,
            checkpoint_db_path=checkpoint_db_path or config.AGENT_CHECKPOINT_DB_PATH,
            max_llm_calls=config.MAX_LLM_CALLS,
            max_total_tokens=config.MAX_TOTAL_TOKENS,
            max_run_seconds=config.MAX_RUN_SECONDS,
            provider=config.LLM_PROVIDER,
            model_name=config.LLM_MODEL,
            prompt_version=SHOPPING_PROMPT_VERSION,
            pricing_version=config.MODEL_PRICING_VERSION,
            input_cost_per_million=config.MODEL_INPUT_COST_PER_1M,
            output_cost_per_million=config.MODEL_OUTPUT_COST_PER_1M,
        )

    def run(
        self,
        question: str,
        conv_id: str = "default",
        chat_history: list = None,
        cancellation_event: threading.Event | None = None,
    ) -> dict:
        """Run the shopping guide Agent for one conversation turn.

        Args:
            question: The user's latest message.
            conv_id: Conversation ID for profile persistence.
            chat_history: Optional list of prior LangChain messages.

        Returns:
            dict with keys: answer, stage, product_context, user_profile,
                           messages, tool_rounds
        """
        result = self.graph.run(
            user_message=question,
            conv_id=conv_id,
            chat_history=chat_history,
            cancellation_event=cancellation_event,
        )

        # Extract final AI response
        answer = ""
        for m in reversed(result["messages"]):
            if isinstance(m, AIMessage) and m.content:
                answer = m.content
                break

        return {
            "answer": answer,
            "stage": result["stage"],
            "product_context": result["product_context"],
            "user_profile": result["user_profile"],
            "messages": result["messages"],
            "tool_rounds": result["tool_rounds"],
            "agent_rounds": result["agent_rounds"],
            "stop_reason": result["stop_reason"],
            "run_id": result["run_id"],
            "request_id": result["request_id"],
            "latency_ms": result["latency_ms"],
            "llm_calls": result["llm_calls"],
            "llm_latency_ms": result["llm_latency_ms"],
            "llm_retries": result["llm_retries"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "total_tokens": result["total_tokens"],
            "retrieval_triggered": result["retrieval_triggered"],
            "cache_hit": result["cache_hit"],
            "requested_tools": result["requested_tools"],
            "executed_tools": result["executed_tools"],
            "tool_errors": result["tool_errors"],
            "retrieval_stats": result["retrieval_stats"],
            "provider": result["provider"],
            "model": result["model"],
            "prompt_version": result["prompt_version"],
            "pricing_version": result["pricing_version"],
            "estimated_cost_usd": result["estimated_cost_usd"],
            "node_latency_ms": result["node_latency_ms"],
            "failure_counts": result["failure_counts"],
            "fallbacks": result["fallbacks"],
            "tool_policy": result["tool_policy"],
            "timeouts": result["timeouts"],
            "cancelled": result["cancelled"],
            "budget_stop": result["budget_stop"],
            "checkpoint_restored": result["checkpoint_restored"],
        }

    async def run_stream(
        self,
        question: str,
        conv_id: str = "default",
        chat_history: list = None,
        cancellation_event: threading.Event | None = None,
    ):
        """Async generator yielding SSE-formatted events for streaming chat."""
        async for event in self.graph.run_stream(
            user_message=question,
            conv_id=conv_id,
            chat_history=chat_history,
            cancellation_event=cancellation_event,
        ):
            yield event

    def clear_conversation_state(self, conv_id: str) -> None:
        errors = []
        try:
            self.graph.clear_thread(conv_id)
        except Exception as exc:
            errors.append(exc)
        try:
            self.profile_store.clear_conv(conv_id)
        except Exception as exc:
            errors.append(exc)
        if errors:
            raise RuntimeError("Failed to clear all conversation runtime state") from errors[0]

    def close(self) -> None:
        self.graph.close()

    def run_simple(self, question: str, conv_id: str = "default") -> dict:
        """Simplified single-shot mode: no LangGraph, one LLM call with tools.

        Useful for quick testing or when LangGraph is unavailable.
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        user_profile = self.profile_store.serialize_profile(conv_id)

        # Quick product search
        retrieval_result = {}
        try:
            structured_retriever = getattr(self.product_retriever, "retrieve_result", None)
            if structured_retriever is not None:
                retrieval_result = structured_retriever(question, top_k=5)
                formatter = getattr(self.product_retriever, "format_result", None)
                product_context = (
                    formatter(retrieval_result)
                    if formatter is not None
                    else self.product_retriever.retrieve(question, top_k=5)
                )
            else:
                product_context = self.product_retriever.retrieve(question, top_k=5)
        except Exception:
            product_context = "(产品检索暂不可用)"

        prompt = ChatPromptTemplate.from_messages([
            ("system", SHOPPING_SYSTEM_PROMPT),
            ("user", "{question}"),
        ])

        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({
            "conv_id": conv_id,
            "stage": "search",
            "user_profile": user_profile,
            "product_context": product_context,
            "question": question,
        })
        from src.retrieval.verifier import verify_answer

        verification = verify_answer(answer, retrieval_result)
        if verification.status == "failed":
            answer = "当前信息不足以确认产品或价格，请调整条件后重试。"

        return {
            "answer": answer,
            "stage": "search",
            "product_context": product_context,
            "user_profile": user_profile,
            "messages": [],
            "tool_rounds": 0,
            "retrieval_result": (
                retrieval_result.model_dump()
                if hasattr(retrieval_result, "model_dump")
                else retrieval_result
            ),
            "verification": verification.model_dump(),
        }
