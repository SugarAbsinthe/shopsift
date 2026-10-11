"""Integration checks for structured retrieval flowing through the graph."""

from langchain_core.messages import AIMessage, HumanMessage
import asyncio
import json
import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from src.agent.langgraph_engine import ShoppingGuideGraph
from src.retrieval.models import EvidenceItem, ProductCandidate, RetrievalResult, RetrievalStats


class _LLM:
    def __init__(self, content):
        self.content = content

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return AIMessage(content=self.content)


class _Retriever:
    def __init__(self):
        product = ProductCandidate(
            product_id=1, name="Test laptop", brand="Acme", category="laptop", price=7999
        )
        self.result = RetrievalResult(
            products=[product],
            stats=RetrievalStats(index_version="idx-test"),
            evidence=[
                EvidenceItem(
                    evidence_id=f"catalog:idx-test:1:{field}",
                    product_id=1,
                    field=field,
                    value=value,
                    source_rank=1,
                    index_version="idx-test",
                )
                for field, value in (
                    ("product_id", 1),
                    ("price", 7999),
                    ("brand", "Acme"),
                    ("category", "laptop"),
                )
            ],
        )

    def retrieve_result(self, query, top_k=5, filters=None):
        return self.result

    def format_result(self, result):
        return "Product ID: 1, price: 7999 RMB"


def _state():
    return {
        "messages": [HumanMessage(content="search")],
        "conv_id": "graph-evidence",
        "stage": "search",
        "user_profile": "",
        "profile_constraints": {},
    }


def _graph(answer):
    return ShoppingGuideGraph(
        llm=_LLM(answer),
        tools=[],
        product_retriever=_Retriever(),
        profile_store=object(),
        system_prompt="{product_context}",
        stage_classifier_prompt="",
    )


def test_graph_blocks_explicit_unretrieved_product():
    graph = _graph("Product ID: 99, price: 7999 RMB")
    state = _state()
    state.update(graph._retrieve_node(state))
    assert "reviews_by_product" not in state["retrieval_result"]
    output = graph._agent_node(state)
    assert output["stop_reason"] == "verification_failed"
    assert output["verification"]["failure_codes"] == ["product_id_not_retrieved"]
    assert output["messages"][0].content == "当前信息不足以确认产品或价格，请调整条件后重试。"


def test_graph_keeps_verified_explicit_product_fact():
    graph = _graph("Product ID: 1, price: 7999 RMB")
    state = _state()
    state.update(graph._retrieve_node(state))
    output = graph._agent_node(state)
    assert output["stop_reason"] == "completed"
    assert output["verification"]["status"] == "verified"


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize(("answer", "failed"), [
    ("Product ID: 99, price: 7999 RMB", True),
    ("Product ID: 1, price: 8999 RMB", True),
    ("Product ID: 1, price: 7999 RMB", False),
])
def test_stream_only_emits_verified_answer_and_done_matches_state(tmp_path, checkpoint, answer, failed):
    from tests.test_graph import StreamingFakeLLM, FakeProfileStore
    graph = ShoppingGuideGraph(
        llm=StreamingFakeLLM(chunks=[answer[:15], answer[15:]]),
        tools=[], product_retriever=_Retriever(), profile_store=FakeProfileStore(),
        system_prompt="{product_context}", stage_classifier_prompt="",
        checkpoint_db_path=str(tmp_path / "stream.db") if checkpoint else None,
    )

    async def collect():
        return [event async for event in graph.run_stream("推荐产品", "stream-evidence")]

    try:
        events = asyncio.run(collect())
        tokens = "".join(json.loads(event.split("data: ", 1)[1])["content"]
                         for event in events if event.startswith("event: token"))
        done = json.loads(next(event for event in events if event.startswith("event: done")).split("data: ", 1)[1])
        expected = "当前信息不足以确认产品或价格，请调整条件后重试。" if failed else answer
        assert tokens == done["answer"] == expected
        assert done["stop_reason"] == ("verification_failed" if failed else "completed")
        if checkpoint:
            state = graph.graph.get_state({"configurable": {"thread_id": "stream-evidence"}}).values
            assert state["messages"][-1].content == expected
    finally:
        graph.close()


def test_stream_drops_unverified_chunks_when_model_fails(tmp_path):
    from tests.test_graph import StreamingFakeLLM, FakeProfileStore

    class FailingLLM(StreamingFakeLLM):
        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            yield ChatGenerationChunk(message=AIMessageChunk(content="Product ID: 99"))
            raise ValueError("private model failure")

    graph = ShoppingGuideGraph(
        llm=FailingLLM(chunks=[]), tools=[], product_retriever=_Retriever(),
        profile_store=FakeProfileStore(), system_prompt="{product_context}",
        stage_classifier_prompt="", checkpoint_db_path=str(tmp_path / "failure.db"),
    )

    async def collect():
        return [event async for event in graph.run_stream("推荐产品", "failed-stream")]

    try:
        events = asyncio.run(collect())
        assert not any(event.startswith("event: token") for event in events)
        assert not any(event.startswith("event: done") for event in events)
        assert any(event.startswith("event: error") for event in events)
        assert "private model failure" not in "".join(events)
    finally:
        graph.close()


@pytest.mark.parametrize("max_tool_rounds", [1, 3])
def test_stream_discards_tool_preambles_and_verifies_finalization(tmp_path, max_tool_rounds):
    from langchain_core.tools import tool
    from tests.test_graph import StreamingFakeLLM, FakeProfileStore

    class ToolLLM(StreamingFakeLLM):
        calls: int = 0

        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            index = self.calls
            self.calls += 1
            if index < 2:
                yield ChatGenerationChunk(message=AIMessageChunk(
                    content="Product ID: 99 (unverified tool preamble)",
                    tool_call_chunks=[{
                        "name": "get_product_detail", "args": '{"product_id": 1}',
                        "id": f"call-{index}", "index": 0,
                    }],
                ))
            else:
                yield ChatGenerationChunk(message=AIMessageChunk(content="Product ID: 99, price: 7999 RMB"))

    @tool
    def get_product_detail(product_id: int) -> str:
        """Return controlled product details."""
        return "Product ID: 1, price: 7999 RMB"

    graph = ShoppingGuideGraph(
        llm=ToolLLM(chunks=[]), tools=[get_product_detail], product_retriever=_Retriever(),
        profile_store=FakeProfileStore(), system_prompt="{product_context}",
        stage_classifier_prompt="", checkpoint_db_path=str(tmp_path / "tools.db"),
        max_tool_rounds=max_tool_rounds,
    )

    async def collect():
        return [event async for event in graph.run_stream("推荐产品", "tool-stream")]

    try:
        events = asyncio.run(collect())
        tokens = "".join(json.loads(event.split("data: ", 1)[1])["content"]
                         for event in events if event.startswith("event: token"))
        done = json.loads(next(event for event in events if event.startswith("event: done")).split("data: ", 1)[1])
        assert tokens == done["answer"] == "当前信息不足以确认产品或价格，请调整条件后重试。"
        assert done["stop_reason"] == "verification_failed"
        assert done["tool_rounds"] == (1 if max_tool_rounds == 1 else 2)
        assert any(event.startswith("event: tool_start") for event in events)
    finally:
        graph.close()
