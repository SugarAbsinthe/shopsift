"""Integration checks for structured retrieval flowing through the graph."""

from langchain_core.messages import AIMessage, HumanMessage

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
