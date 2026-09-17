"""Test LangGraph routing and node logic without external LLM calls."""
import asyncio
import json
import pytest
from unittest.mock import Mock, MagicMock
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import tool
from src.agent.langgraph_engine import (
    ShoppingGuideGraph,
    ShoppingState,
    build_retrieval_constraints,
    classify_stage,
    parse_budget_expression,
)


def _make_tool_call_msg(tool_call_id="call_1", tool_name="search"):
    """Create an AIMessage with tool_calls using the dict format accepted by AIMessage."""
    return AIMessage(
        content="",
        tool_calls=[{
            "name": tool_name,
            "args": {"query": "test"},
            "id": tool_call_id,
            "type": "tool_call",
        }],
    )


class TestClassifyStage:
    """Stage classification logic — deterministic paths."""

    def test_greeting_returns_discovery(self):
        assert classify_stage("你好", "discovery") == "discovery"
        assert classify_stage("hi", "discovery") == "discovery"

    def test_comparison_keywords(self):
        assert classify_stage("联想和华硕哪个好", "search") == "comparison"
        assert classify_stage("帮我对比一下", "search") == "comparison"

    def test_search_intent(self):
        assert classify_stage("推荐游戏本", "discovery") == "search"
        assert classify_stage("有哪些笔记本", "discovery") == "search"
        assert classify_stage("推荐预算6k以内的办公本", "discovery") == "search"

    def test_needs_keywords(self):
        assert classify_stage("预算8000打游戏", "discovery") == "needs_elicitation"

    def test_objection_keywords(self):
        assert classify_stage("这个质量靠谱吗", "search") == "objection_handling"
        assert classify_stage("散热行不行", "search") == "objection_handling"

    def test_summary_keywords(self):
        assert classify_stage("谢谢，就这个了", "recommendation") == "summary"

    def test_empty_message_returns_current_stage(self):
        assert classify_stage("", "search") == "search"


class TestGraphRouting:
    """Routing logic — no LLM needed."""

    def test_route_with_tool_calls_under_limit(self):
        graph = _make_graph(max_tool_rounds=3)
        state = {
            "messages": [_make_tool_call_msg()],
            "tool_rounds": 1,
        }
        result = graph._route_after_agent(state)
        assert result == "tools"

    def test_route_with_tool_calls_at_limit(self):
        graph = _make_graph(max_tool_rounds=3)
        state = {
            "messages": [_make_tool_call_msg()],
            "tool_rounds": 3,
        }
        result = graph._route_after_agent(state)
        assert result == "finalize"

    def test_route_without_tool_calls(self):
        graph = _make_graph()
        state = {
            "messages": [AIMessage(content="推荐完了")],
            "tool_rounds": 1,
        }
        result = graph._route_after_agent(state)
        assert result == "end"

    def test_route_empty_messages(self):
        graph = _make_graph()
        state = {"messages": [], "tool_rounds": 0}
        result = graph._route_after_agent(state)
        assert result == "end"


class TestExtractProfileSignals:
    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("预算6k以内", "<=6000"),
            ("预算5000到6500", "5000-6500"),
            ("预算6k左右", "5400-6600"),
            ("预算1.2w以下", "<=12000"),
            ("预算一万一以内", "<=11000"),
            ("预算6000元", "<=6000"),
        ],
    )
    def test_budget_extraction(self, message, expected):
        from src.agent.langgraph_engine import extract_profile_signals
        mock_store = Mock()
        extract_profile_signals("test_conv", message, mock_store)
        mock_store.update.assert_called_once_with(
            "test_conv", "budget", expected, confidence=0.95, source="explicit"
        )

    def test_reversed_budget_range_is_not_silently_reordered(self):
        assert parse_budget_expression("预算8000到5000") is None


def _make_graph(max_tool_rounds=3):
    """Minimal graph for testing routing logic."""
    return ShoppingGuideGraph(
        llm=Mock(),
        tools=[],
        product_retriever=Mock(),
        profile_store=Mock(),
        system_prompt="test",
        stage_classifier_prompt="test",
        max_tool_rounds=max_tool_rounds,
    )


class FakeToolLLM:
    """Deterministic model double for graph lifecycle tests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


class StreamingFakeLLM(BaseChatModel):
    chunks: list[str]

    @property
    def _llm_type(self):
        return "streaming-fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[
            ChatGeneration(message=AIMessage(content="".join(self.chunks)))
        ])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        for chunk in self.chunks:
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))


class FakeRetriever:
    def retrieve(self, query, top_k=5, filters=None):
        return "context"


class FakeProfileStore:
    def serialize_profile(self, conv_id):
        return "(暂无画像)"

    def update(self, *args, **kwargs):
        return None


def test_profile_constraints_are_structured_and_exclusion_wins_conflict():
    constraints = build_retrieval_constraints({
        "budget": {"value": "5000-8000"},
        "product_category": {"value": "笔记本电脑"},
        "preferred_brand": {"value": "联想"},
        "exclude_brand": {"value": "联想"},
        "untrusted_field": {"value": "ignored"},
    })
    assert constraints == {
        "min_price": 5000,
        "max_price": 8000,
        "category": "笔记本电脑",
        "excluded_brands": ["联想"],
    }


def test_search_turn_retrieves_once_with_structured_hard_constraints():
    class RecordingRetriever:
        def __init__(self):
            self.calls = []

        def retrieve(self, query, top_k=5, filters=None):
            self.calls.append({"query": query, "top_k": top_k, "filters": filters})
            return "context"

    class StructuredProfileStore:
        def __init__(self):
            self.values = {}

        def update(self, conv_id, key, value, **kwargs):
            self.values[key] = value

        def serialize_profile(self, conv_id):
            return "\n".join(f"- {key}: {value}" for key, value in self.values.items())

        def get_structured(self, conv_id):
            return {
                key: {"value": value, "confidence": 0.8, "source": "deduced"}
                for key, value in self.values.items()
            }

    retriever = RecordingRetriever()
    graph = ShoppingGuideGraph(
        llm=FakeToolLLM([AIMessage(content="final")]),
        tools=[],
        product_retriever=retriever,
        profile_store=StructuredProfileStore(),
        system_prompt="test {conv_id} {stage} {user_profile} {product_context}",
        stage_classifier_prompt="",
    )
    result = graph.run("推荐预算6k以内的办公笔记本，不要戴尔", "constraint-conv")
    assert result["stage"] == "search"
    assert len(retriever.calls) == 1
    assert retriever.calls[0]["top_k"] == 5
    assert retriever.calls[0]["filters"] == {
        "max_price": 6000,
        "category": "笔记本电脑",
        "excluded_brands": ["戴尔"],
    }
    graph.close()


def test_turn_local_constraints_do_not_leak_into_the_next_checkpointed_turn(tmp_path):
    class RecordingRetriever:
        def __init__(self):
            self.filters = []

        def retrieve(self, query, top_k=5, filters=None):
            self.filters.append(filters or {})
            return "context"

    class ProfileStore:
        def __init__(self):
            self.values = {}

        def update(self, conv_id, key, value, **kwargs):
            self.values[key] = value

        def serialize_profile(self, conv_id):
            return "\n".join(f"- {key}: {value}" for key, value in self.values.items())

        def get_structured(self, conv_id):
            return {key: {"value": value} for key, value in self.values.items()}

    retriever = RecordingRetriever()
    graph = ShoppingGuideGraph(
        llm=FakeToolLLM([AIMessage(content="final")]),
        tools=[],
        product_retriever=retriever,
        profile_store=ProfileStore(),
        system_prompt="test {conv_id} {stage} {user_profile} {product_context}",
        stage_classifier_prompt="",
        checkpoint_db_path=str(tmp_path / "session-checkpoints.db"),
    )

    graph.run("这次预算6000元，推荐笔记本", "session-conv")
    graph.run("再推荐几款笔记本", "session-conv")

    assert retriever.filters[0] == {"max_price": 6000, "category": "笔记本电脑"}
    assert retriever.filters[1] == {"category": "笔记本电脑"}
    graph.close()


@pytest.mark.parametrize(
    ("budget", "expected"),
    [
        ("<=6000", {"max_price": 6000}),
        ("5000-6500", {"min_price": 5000, "max_price": 6500}),
        ("6k以内", {"max_price": 6000}),
        ("6k左右", {"min_price": 5400, "max_price": 6600}),
        ("6000", {"max_price": 6000}),
    ],
)
def test_budget_profile_values_map_to_expected_constraints(budget, expected):
    assert build_retrieval_constraints({"budget": {"value": budget}}) == expected


def _lifecycle_graph(responses, max_tool_rounds=3):
    llm = FakeToolLLM(responses)
    graph = ShoppingGuideGraph(
        llm=llm,
        tools=[],
        product_retriever=FakeRetriever(),
        profile_store=FakeProfileStore(),
        system_prompt="test {conv_id} {stage} {user_profile} {product_context}",
        stage_classifier_prompt="",
        max_tool_rounds=max_tool_rounds,
    )
    return graph, llm


def test_graph_completes_without_tools():
    graph, _ = _lifecycle_graph([AIMessage(content="final")])
    result = graph.run("hello", "conv-1")
    assert result["messages"][-1].content == "final"
    assert result["tool_rounds"] == 0
    assert result["stop_reason"] == "completed"


def test_graph_finalizes_when_tool_round_limit_is_reached():
    first_tool_call = _make_tool_call_msg("call_1")
    second_tool_call = _make_tool_call_msg("call_2")
    graph, llm = _lifecycle_graph(
        [first_tool_call, second_tool_call, AIMessage(content="final after limit")],
        max_tool_rounds=1,
    )
    result = graph.run("find a laptop", "conv-2")
    assert result["messages"][-1].content == "final after limit"
    assert result["tool_rounds"] == 1
    assert result["stop_reason"] == "max_tool_rounds"
    assert llm.calls == 3
    assert isinstance(result["messages"][-2], ToolMessage)
    assert result["messages"][-2].tool_call_id == "call_2"


def test_graph_reports_tool_error_and_still_finalizes():
    tool_call = _make_tool_call_msg()
    graph, _ = _lifecycle_graph([tool_call, AIMessage(content="fallback")], max_tool_rounds=1)
    result = graph.run("find a laptop", "conv-3")
    assert result["messages"][-1].content == "fallback"
    assert result["stop_reason"] == "tool_error"


def test_checkpoint_preserves_stage_across_graph_instances(tmp_path):
    checkpoint_path = str(tmp_path / "agent-checkpoints.db")
    first_llm = FakeToolLLM([AIMessage(content="first")])
    first = ShoppingGuideGraph(
        llm=first_llm,
        tools=[],
        product_retriever=FakeRetriever(),
        profile_store=FakeProfileStore(),
        system_prompt="test {conv_id} {stage} {user_profile} {product_context}",
        stage_classifier_prompt="",
        checkpoint_db_path=checkpoint_path,
    )
    first_result = first.run("推荐游戏本", "persistent-conv")
    assert first_result["stage"] == "search"
    first.close()

    second_llm = FakeToolLLM([AIMessage(content="second")])
    second = ShoppingGuideGraph(
        llm=second_llm,
        tools=[],
        product_retriever=FakeRetriever(),
        profile_store=FakeProfileStore(),
        system_prompt="test {conv_id} {stage} {user_profile} {product_context}",
        stage_classifier_prompt="",
        checkpoint_db_path=checkpoint_path,
    )
    second_result = second.run("这个呢", "persistent-conv")
    assert second_result["stage"] == "search"
    second.clear_thread("persistent-conv")
    second.close()

    third_llm = FakeToolLLM([AIMessage(content="third")])
    third = ShoppingGuideGraph(
        llm=third_llm,
        tools=[],
        product_retriever=FakeRetriever(),
        profile_store=FakeProfileStore(),
        system_prompt="test {conv_id} {stage} {user_profile} {product_context}",
        stage_classifier_prompt="",
        checkpoint_db_path=checkpoint_path,
    )
    reset_result = third.run("这个呢", "persistent-conv")
    assert reset_result["stage"] == "discovery"
    third.close()


def test_run_stream_emits_real_model_chunks_and_complete_metadata(tmp_path):
    llm = StreamingFakeLLM(chunks=["first", "-second"])
    graph = ShoppingGuideGraph(
        llm=llm,
        tools=[],
        product_retriever=FakeRetriever(),
        profile_store=FakeProfileStore(),
        system_prompt="test {conv_id} {stage} {user_profile} {product_context}",
        stage_classifier_prompt="",
        checkpoint_db_path=str(tmp_path / "stream-checkpoints.db"),
    )

    async def collect():
        return [event async for event in graph.run_stream("hello", "stream-conv")]

    events = asyncio.run(collect())
    token_payloads = [
        json.loads(event.split("data: ", 1)[1])
        for event in events if event.startswith("event: token")
    ]
    done_payload = json.loads(
        next(event for event in events if event.startswith("event: done")).split("data: ", 1)[1]
    )
    assert [payload["content"] for payload in token_payloads] == ["first", "-second"]
    assert done_payload["stop_reason"] == "completed"
    assert done_payload["agent_rounds"] == 1
    assert done_payload["run_id"]
    assert done_payload["latency_ms"] >= 0
    assert done_payload["llm_calls"] == 1
    assert done_payload["llm_retries"] == 0
    assert done_payload["retrieval_triggered"] is False
    assert done_payload["requested_tools"] == []
    assert done_payload["executed_tools"] == []
    assert "user_profile" in done_payload
    assert "product_context" in done_payload
    graph.close()


def test_sync_and_stream_paths_produce_the_same_answer(tmp_path):
    sync_graph = ShoppingGuideGraph(
        llm=StreamingFakeLLM(chunks=["same", " answer"]),
        tools=[],
        product_retriever=FakeRetriever(),
        profile_store=FakeProfileStore(),
        system_prompt="test {conv_id} {stage} {user_profile} {product_context}",
        stage_classifier_prompt="",
        checkpoint_db_path=str(tmp_path / "sync.db"),
    )
    sync_result = sync_graph.run("hello", "sync-conv")

    stream_graph = ShoppingGuideGraph(
        llm=StreamingFakeLLM(chunks=["same", " answer"]),
        tools=[],
        product_retriever=FakeRetriever(),
        profile_store=FakeProfileStore(),
        system_prompt="test {conv_id} {stage} {user_profile} {product_context}",
        stage_classifier_prompt="",
        checkpoint_db_path=str(tmp_path / "stream.db"),
    )

    async def collect():
        return [event async for event in stream_graph.run_stream("hello", "stream-conv")]

    events = asyncio.run(collect())
    stream_answer = "".join(
        json.loads(event.split("data: ", 1)[1])["content"]
        for event in events if event.startswith("event: token")
    )
    assert stream_answer == sync_result["messages"][-1].content == "same answer"
    sync_graph.close()
    stream_graph.close()


def test_run_stream_emits_tool_lifecycle_events(tmp_path):
    @tool
    def search(query: str) -> str:
        """Search a test catalog."""
        return "found"

    llm = FakeToolLLM([
        _make_tool_call_msg(tool_name="search"),
        AIMessage(content="done"),
    ])
    graph = ShoppingGuideGraph(
        llm=llm,
        tools=[search],
        product_retriever=FakeRetriever(),
        profile_store=FakeProfileStore(),
        system_prompt="test {conv_id} {stage} {user_profile} {product_context}",
        stage_classifier_prompt="",
        checkpoint_db_path=str(tmp_path / "tool-events.db"),
    )

    async def collect():
        return [event async for event in graph.run_stream("hello", "tool-conv")]

    events = asyncio.run(collect())
    assert any(event.startswith("event: tool_start") for event in events)
    assert any(event.startswith("event: tool_end") for event in events)
    done = json.loads(
        next(event for event in events if event.startswith("event: done")).split("data: ", 1)[1]
    )
    assert done["tool_rounds"] == 1
    assert done["stop_reason"] == "completed"
    assert done["requested_tools"] == ["search"]
    assert done["executed_tools"] == ["search"]
    assert done["tool_errors"] == 0
    graph.close()
