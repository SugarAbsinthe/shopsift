"""Category isolation, lifecycle safety and migration of real SQLite memory."""

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src.agent.langgraph_engine import build_retrieval_constraints, extract_profile_signals
from tests.test_profile_store import _store, EmptyMemoryCollection
from tests.test_profile_refinements import graph_with_store
from src.profile.profile_store import ProfileStore


def values(store, category=None):
    return {k: v["value"] for k, v in store.get_structured("target", category_scope=category).items()}


def test_category_switch_isolates_and_restores_all_purchase_requirements(tmp_path):
    store = _store(tmp_path)
    extract_profile_signals("target", "预算6000元，首选联想笔记本，至少32GB内存，主要办公", store)
    laptop = values(store)
    extract_profile_signals("target", "电脑定下了，帮我挑个300块钱的无线鼠标，不要苹果", store)
    assert values(store) == {"product_category": "鼠标", "budget": "<=300", "exclude_brand": "苹果"}
    extract_profile_signals("target", "继续推荐笔记本", store)
    assert values(store) == laptop
    reopened = _store(tmp_path)
    assert values(reopened) == laptop
    extract_profile_signals("target", "再看鼠标", reopened)
    assert values(reopened)["budget"] == "<=300"
    assert values(reopened)["exclude_brand"] == "苹果"


def test_graph_filters_and_prompt_do_not_inherit_old_category(tmp_path):
    graph, store, retriever = graph_with_store(tmp_path)
    try:
        graph.run("预算300元，推荐鼠标，不要苹果", "target")
        result = graph.run("再帮我推荐笔记本", "target")
        assert retriever.calls[-1][1] == {"category": "笔记本电脑"}
        assert "<=300" not in result["user_profile"]
        assert "苹果" not in result["user_profile"]
        graph.run("继续推荐鼠标", "target")
        assert retriever.calls[-1][1] == {"category": "鼠标", "max_price": 300, "excluded_brands": ["苹果"]}
    finally:
        graph.close()


def test_budget_given_before_category_binds_once(tmp_path):
    store = _store(tmp_path)
    extract_profile_signals("target", "预算6000元，主要办公", store)
    extract_profile_signals("target", "笔记本", store)
    assert values(store)["budget"] == "<=6000"
    extract_profile_signals("target", "推荐鼠标", store)
    assert values(store) == {"product_category": "鼠标"}


def test_temporary_category_switch_neither_leaks_nor_mutates_persistent_needs(tmp_path):
    graph, store, retriever = graph_with_store(tmp_path)
    try:
        graph.run("预算6000元，首选联想笔记本", "target")
        graph.run("预算300元，推荐鼠标，不要苹果", "target")
        graph.run("继续推荐笔记本", "target")
        result = graph.run("这次先看鼠标，取消预算", "target")
        assert retriever.calls[-1][1] == {"category": "鼠标", "excluded_brands": ["苹果"]}
        assert "联想" not in result["user_profile"]
        assert "6000" not in result["user_profile"]
        assert "300" not in result["user_profile"]
        assert values(store)["product_category"] == "笔记本电脑"
        assert values(store, "鼠标")["budget"] == "<=300"
        graph.run("继续推荐笔记本", "target")
        assert retriever.calls[-1][1]["max_price"] == 6000
    finally:
        graph.close()


def test_switch_with_clear_only_deletes_the_target_category_budget(tmp_path):
    graph, store, _ = graph_with_store(tmp_path)
    try:
        graph.run("预算300元，推荐鼠标", "target")
        graph.run("预算6000元，推荐笔记本", "target")
        graph.run("推荐鼠标，取消预算", "target")
        assert "budget" not in values(store)
        assert values(store, "笔记本电脑")["budget"] == "<=6000"
    finally:
        graph.close()


def test_inactive_candidate_confirmation_and_deletion_cannot_touch_active_category(tmp_path):
    store = _store(tmp_path)
    extract_profile_signals("target", "预算6000元，首选苹果笔记本", store)
    pending = store.add_candidate("target", "budget", "<=300", memory_type="inferred", confidence=.8,
                                  source_message_id="user-mouse", evidence_span="鼠标预算", category_scope="鼠标")
    confirmed = store.confirm_candidate("target", pending.candidate_id)
    assert confirmed.category_scope == "鼠标"
    assert values(store)["budget"] == "<=6000"
    store.delete_candidate("target", pending.candidate_id)
    assert values(store)["budget"] == "<=6000"
    extract_profile_signals("target", "推荐鼠标，不要苹果", store)
    assert "preferred_brand" not in values(store)
    assert values(store, "笔记本电脑")["preferred_brand"] == "苹果"


def test_category_view_cannot_access_another_conversation(tmp_path):
    store = _store(tmp_path)
    view = store.for_category("target", "鼠标")
    with pytest.raises(ValueError):
        view.get_structured("other")
    with pytest.raises(ValueError):
        view.delete_profile_key("other", "budget")


def test_expiring_an_inactive_category_does_not_remove_another_category_value(tmp_path):
    store = _store(tmp_path)
    extract_profile_signals("target", "预算300元，推荐鼠标", store)
    mouse_id = store.get_structured("target")["budget"]["candidate_id"]
    extract_profile_signals("target", "预算6000元，推荐笔记本", store)
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("UPDATE memory_candidates SET expires_at = '2000-01-01T00:00:00+00:00' WHERE candidate_id = ?", (mouse_id,))
    assert values(store)["budget"] == "<=6000"
    assert "budget" not in values(store, "鼠标")
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("SELECT profile_value FROM user_profiles WHERE conv_id = 'target' "
                            "AND category_scope = '笔记本电脑' AND profile_key = 'budget'").fetchone()[0] == "<=6000"


def test_profile_tool_reads_the_effective_turn_not_the_persistent_category(tmp_path):
    from langchain_core.messages import AIMessage
    from src.agent.langgraph_engine import ShoppingGuideGraph
    from src.agent.shopping_tools import create_get_user_profile
    from tests.test_graph import FakeToolLLM, DialogueRetriever
    store = _store(tmp_path)
    extract_profile_signals("target", "预算6000元，首选联想笔记本", store)
    graph = ShoppingGuideGraph(llm=FakeToolLLM([AIMessage(content="done")]),
                               tools=[create_get_user_profile(store)], profile_store=store,
                               product_retriever=DialogueRetriever(), system_prompt="{user_profile}", stage_classifier_prompt="")
    try:
        effective = "## 本轮有效需求\n- product_category: 鼠标\n- budget: <=300"
        state = {"conv_id": "target", "stage": "search", "user_profile": effective,
                 "messages": [AIMessage(content="", tool_calls=[{"name":"get_user_profile", "args":{"conv_id":"target"}, "id":"profile-read"}])]}
        output = graph._tools_node(state)
        assert output["messages"][0].content == effective
        assert output["messages"][0].status == "success"
    finally:
        graph.close()


def legacy_store(tmp_path):
    path = tmp_path / "profiles.db"
    now = datetime.now(timezone.utc)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE user_profiles(id INTEGER PRIMARY KEY, conv_id TEXT, profile_key TEXT, "
                     "profile_value TEXT, confidence REAL, last_updated TEXT, source TEXT, candidate_id TEXT, UNIQUE(conv_id,profile_key))")
        conn.executemany("INSERT INTO user_profiles VALUES(?,?,?,?,?,?,?,?)", [
            (1,"target","product_category","笔记本电脑",.95,now.isoformat(),"explicit",None),
            (2,"target","budget","<=6000",.95,now.isoformat(),"explicit","mem_original"),
        ])
        conn.execute("CREATE TABLE memory_candidates(candidate_id TEXT PRIMARY KEY, conv_id TEXT, profile_key TEXT, "
                     "profile_value TEXT, memory_type TEXT, confidence REAL, source_message_id TEXT, evidence_span TEXT, "
                     "created_at TEXT, expires_at TEXT, status TEXT, schema_version TEXT)")
        conn.execute("INSERT INTO memory_candidates VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (
            "mem_original","target","budget","<=6000","explicit",.95,"original-source","original evidence",
            now.isoformat(),(now+timedelta(days=180)).isoformat(),"confirmed","1.0"))
    store = ProfileStore.__new__(ProfileStore)
    store.db_path, store.decay_lambda, store.memory_col = str(path), 0., EmptyMemoryCollection()
    return store


def test_v1_migration_preserves_identity_evidence_and_is_idempotent(tmp_path):
    store = legacy_store(tmp_path)
    store._init_db()
    original = next(c for c in store.list_candidates("target") if c.candidate_id == "mem_original")
    assert original.category_scope == "笔记本电脑"
    assert original.source_message_id == "original-source"
    assert original.evidence_span == "original evidence"
    assert original.schema_version == "1.0"
    extract_profile_signals("target", "预算300元，推荐鼠标", store)
    ids = {c.candidate_id for c in store.list_candidates("target")}
    store._init_db()
    assert {c.candidate_id for c in store.list_candidates("target")} == ids
    assert values(store)["budget"] == "<=300"
    extract_profile_signals("target", "推荐笔记本", store)
    assert values(store)["budget"] == "<=6000"


def test_failed_migration_rolls_back_schema_and_releases_database(tmp_path, monkeypatch):
    store = legacy_store(tmp_path)
    monkeypatch.setattr(store, "_migrate_legacy_profiles", lambda conn: (_ for _ in ()).throw(RuntimeError("simulated migration failure")))
    with pytest.raises(RuntimeError):
        store._init_db()
    with sqlite3.connect(store.db_path, timeout=.2) as conn:
        assert "category_scope" not in {r[1] for r in conn.execute("PRAGMA table_info(user_profiles)")}
        assert conn.execute("SELECT COUNT(*) FROM user_profiles").fetchone()[0] == 2
        conn.execute("BEGIN IMMEDIATE")
        conn.rollback()
