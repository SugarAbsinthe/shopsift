"""User-visible revisions tested against real governed SQLite memory."""

import pytest

from langchain_core.messages import AIMessage
from src.agent.langgraph_engine import ShoppingGuideGraph, build_retrieval_constraints, extract_profile_signals
from src.profile.commands import process_profile_commands
from src.profile.models import make_source_message_id
from tests.test_graph import FakeToolLLM, DialogueRetriever
from tests.test_profile_store import _store


def graph_with_store(tmp_path):
    store = _store(tmp_path)
    retriever = DialogueRetriever()
    graph = ShoppingGuideGraph(
        llm=FakeToolLLM([AIMessage(content="目录候选")]), tools=[],
        profile_store=store, product_retriever=retriever,
        system_prompt="{user_profile} {product_context}", stage_classifier_prompt="",
        checkpoint_db_path=str(tmp_path / "checkpoints.db"),
    )
    return graph, store, retriever


@pytest.mark.parametrize("message,key", [
    ("预算不限", "budget"), ("不再限制预算", "budget"),
    ("其实不一定要联想的", "preferred_brand"), ("品牌无所谓", "preferred_brand"),
])
def test_natural_clear_changes_only_the_requested_conversation_key(tmp_path, message, key):
    store = _store(tmp_path)
    for conv in ("target", "other"):
        extract_profile_signals(conv, "预算6000元，首选联想笔记本", store)
    result = process_profile_commands("target", message, store)
    assert result.handled
    profile = store.get_structured("target")
    assert key not in profile
    assert "product_category" in profile
    assert key in store.get_structured("other")


@pytest.mark.parametrize("message", ["不要删除预算", "不删除预算", "暂时不取消预算", "不要帮我删除预算", "如果删除预算会怎样", "例如删除预算", "“删除预算”是什么意思"])
def test_negated_hypothetical_or_quoted_commands_do_not_delete(tmp_path, message):
    store = _store(tmp_path)
    extract_profile_signals("target", "预算6000元", store)
    assert not process_profile_commands("target", message, store).handled
    assert store.get_structured("target")["budget"]["value"] == "<=6000"


def test_combined_clear_and_search_preserves_requirements_and_provenance(tmp_path):
    graph, store, retriever = graph_with_store(tmp_path)
    try:
        graph.run("预算6000元，推荐笔记本，不要联想", "target")
        message = "取消品牌排除，预算8000元，推荐笔记本"
        result = graph.run(message, "target")
        assert result["stage"] == "search"
        assert retriever.calls[-1][1] == {"max_price": 8000, "category": "笔记本电脑"}
        candidate_id = store.get_structured("target")["budget"]["candidate_id"]
        candidate = next(c for c in store.list_candidates("target") if c.candidate_id == candidate_id)
        assert candidate.source_message_id == make_source_message_id("target", message)
    finally:
        graph.close()


def test_unspecified_relaxation_asks_before_changing_budget_and_accepts_reply(tmp_path):
    graph, store, retriever = graph_with_store(tmp_path)
    try:
        graph.run("预算6000元，推荐笔记本", "target")
        result = graph.run("算了，预算不卡那么死", "target")
        assert "预算上限调整到多少" in result["messages"][-1].content
        assert store.get_structured("target")["budget"]["value"] == "<=6000"
        assert len(retriever.calls) == 1
        graph.run("8000元", "target")
        assert store.get_structured("target")["budget"]["value"] == "<=8000"
        assert retriever.calls[-1][1]["max_price"] == 8000
    finally:
        graph.close()


def test_temporary_clear_is_not_persisted(tmp_path):
    graph, store, retriever = graph_with_store(tmp_path)
    try:
        graph.run("预算6000元，推荐笔记本", "target")
        graph.run("这次预算不限，推荐笔记本", "target")
        assert "max_price" not in retriever.calls[-1][1]
        assert store.get_structured("target")["budget"]["value"] == "<=6000"
        graph.run("再推荐几款笔记本", "target")
        assert retriever.calls[-1][1]["max_price"] == 6000
    finally:
        graph.close()


def test_turn_scope_applies_to_a_later_clear_clause(tmp_path):
    store = _store(tmp_path)
    extract_profile_signals("target", "预算6000元，首选联想笔记本", store)
    result = process_profile_commands("target", "这次推荐几款，取消预算并取消品牌偏好", store)
    assert set(result.session_cleared) == {"budget", "preferred_brand"}
    assert "budget" in store.get_structured("target")
    assert "preferred_brand" in store.get_structured("target")


def test_explicit_budget_in_the_same_message_resolves_a_relaxation(tmp_path):
    graph, store, retriever = graph_with_store(tmp_path)
    try:
        graph.run("预算6000元，推荐笔记本", "target")
        result = graph.run("预算放宽一点，预算8000元，推荐笔记本", "target")
        assert result["stage"] == "search"
        assert retriever.calls[-1][1]["max_price"] == 8000
    finally:
        graph.close()


@pytest.mark.parametrize("temporary", [False, True])
@pytest.mark.parametrize("clauses,expected", [("预算8000元，取消预算", None), ("取消预算，预算8000元", 8000)])
def test_clear_and_replacement_follow_user_clause_order(tmp_path, temporary, clauses, expected):
    graph, store, retriever = graph_with_store(tmp_path)
    try:
        graph.run("预算6000元，推荐笔记本", "target")
        graph.run(("这次" if temporary else "") + clauses + "，推荐笔记本", "target")
        assert retriever.calls[-1][1].get("max_price") == expected
        if temporary:
            assert store.get_structured("target")["budget"]["value"] == "<=6000"
    finally:
        graph.close()


def test_explicit_numeric_revision_replaces_the_old_opposing_bound(tmp_path):
    store = _store(tmp_path)
    extract_profile_signals("target", "推荐笔记本，至少32GB内存，1TB存储", store)
    extract_profile_signals("target", "内存改成不超过16GB内存", store)
    filters = build_retrieval_constraints(store.get_structured("target"))["spec_filters"]
    assert {"field": "memory_gb", "op": "lte", "value": 16.0} in filters
    assert not any(p["field"] == "memory_gb" and p["op"] == "gte" for p in filters)
    assert {"field": "storage_gb", "op": "eq", "value": 1024.0} in filters


def test_purchase_price_is_extracted_but_a_product_quote_is_not():
    from src.agent.langgraph_engine import parse_budget_expression
    assert parse_budget_expression("帮我挑个300块钱的无线鼠标") == "<=300"
    assert parse_budget_expression("这款售价7999元，推荐吗") is None
    assert parse_budget_expression("预算6,000元") == "<=6000"
    assert parse_budget_expression("预算放宽到8000元") == "<=8000"
