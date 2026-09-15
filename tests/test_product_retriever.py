"""Deterministic regression tests for hybrid product retrieval."""

import json
import sqlite3

from src.retrieval.models import RetrievalQuery
from src.retrieval.product_retriever import ProductRetriever, reciprocal_rank_fusion


class FakeEmbedding:
    def tolist(self):
        return [0.1, 0.2]


class FakeModel:
    def encode(self, text):
        return FakeEmbedding()


class FakeCollection:
    def __init__(self, product_ids, reviews=False):
        self.product_ids = product_ids
        self.reviews = reviews
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        if self.reviews:
            allowed = kwargs["where"]["product_id"]["$in"]
            product_ids = [pid for pid in self.product_ids if pid in allowed]
            return {
                "ids": [[f"review-{pid}" for pid in product_ids]],
                "documents": [[f"review {pid}" for pid in product_ids]],
                "metadatas": [[
                    {"product_id": pid, "aspect": "quality", "sentiment": "positive"}
                    for pid in product_ids
                ]],
            }
        return {
            "ids": [[str(pid) for pid in self.product_ids]],
            "metadatas": [[{"product_id": pid} for pid in self.product_ids]],
        }


class FakeClient:
    def __init__(self):
        self.collections = {
            "product_descriptions": FakeCollection([1, 2, 3]),
            "product_specs": FakeCollection([2, 3, 1]),
            "product_reviews": FakeCollection([1, 2, 3], reviews=True),
        }

    def get_collection(self, name):
        return self.collections[name]


def _catalog(tmp_path):
    path = tmp_path / "products.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE products (product_id INTEGER PRIMARY KEY, name TEXT, "
        "brand TEXT, category TEXT, subcategory TEXT, price REAL, rating REAL, "
        "sales_count INTEGER, release_date TEXT, description TEXT, specs TEXT)"
    )
    rows = [
        (1, "联想拯救者 Pro 5", "联想", "笔记本电脑", "游戏本", 7999, 4.8, 100, "2025-01-01", "高性能游戏本", {"GPU": "RTX 4060"}),
        (2, "华硕天选", "华硕", "笔记本电脑", "游戏本", 8999, 4.7, 90, "2025-01-02", "游戏笔记本", {"GPU": "RTX 4070"}),
        (3, "联想小新", "联想", "笔记本电脑", "轻薄本", 5999, 4.6, 80, "2025-01-03", "便携办公电脑", {"内存": "32GB"}),
    ]
    conn.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [row[:-1] + (json.dumps(row[-1], ensure_ascii=False),) for row in rows],
    )
    conn.commit()
    conn.close()
    return str(path)


def _retriever(tmp_path):
    client = FakeClient()
    retriever = ProductRetriever(
        chroma_dir="unused",
        catalog_db=_catalog(tmp_path),
        model=FakeModel(),
        client=client,
    )
    return retriever, client


def test_rrf_deduplicates_each_source_and_rewards_cross_source_hits():
    ordered, scores, ranks = reciprocal_rank_fusion({
        "description": [1, 1, 2],
        "spec": [2, 1],
        "sparse": [1],
    })
    assert ordered[0] == 1
    assert scores[1] > scores[2]
    assert ranks[1] == {"description": 1, "spec": 2, "sparse": 1}


def test_search_always_runs_three_recall_channels_and_returns_full_fields(tmp_path):
    retriever, client = _retriever(tmp_path)
    result = retriever.search("RTX 4060", top_k=2)
    assert len(client.collections["product_descriptions"].calls) == 1
    assert len(client.collections["product_specs"].calls) == 1
    assert result.stats.source_hits["sparse"] >= 1
    assert result.products[0].product_id == 1
    assert result.products[0].price == 7999
    assert result.products[0].specs["GPU"] == "RTX 4060"
    assert result.constraints == RetrievalQuery(text="RTX 4060", top_k=2).constraints
    assert [item.evidence_id for item in result.evidence] == [
        "catalog:legacy:1:product_id",
        "catalog:legacy:1:price",
        "catalog:legacy:1:brand",
        "catalog:legacy:1:category",
        "catalog:legacy:2:product_id",
        "catalog:legacy:2:price",
        "catalog:legacy:2:brand",
        "catalog:legacy:2:category",
    ]
    assert all("RTX 4060" not in str(item.model_dump()) for item in result.evidence)


def test_hard_constraints_override_rank_and_brand_preference(tmp_path):
    retriever, _ = _retriever(tmp_path)
    result = retriever.search(RetrievalQuery(
        text="游戏本",
        top_k=5,
        constraints={
            "max_price": 8000,
            "category": "笔记本电脑",
            "preferred_brands": ["华硕"],
            "excluded_brands": ["联想"],
        },
    ))
    assert result.products == []
    assert result.stats.filtered_candidates == 3


def test_sparse_query_is_parameterized_and_degrades_safely(tmp_path):
    retriever, _ = _retriever(tmp_path)
    ids, fallback = retriever._sparse_recall('" OR 1=1 --', 5)
    assert ids == []
    assert fallback in {False, True}
    ids, fallback = retriever._sparse_recall(
        "想要 RTX 4060 游戏本\n用户画像: 出差", 5
    )
    assert ids[0] == 1
    assert fallback is False
    retriever._sparse_conn = None
    assert retriever._sparse_recall("RTX 4060", 5) == ([], True)


def test_retrieve_keeps_string_api_and_empty_result_message(tmp_path):
    retriever, _ = _retriever(tmp_path)
    text = retriever.retrieve("游戏本", filters={"max_price": 100})
    assert isinstance(text, str)
    assert "未找到满足当前约束的商品" in text
