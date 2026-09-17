"""LangChain tools for the shopping guide Agent.

Six tools mapped to the shopping guide domain:
  1. search_products     — semantic search over product catalog
  2. get_product_detail  — full spec sheet for one product
  3. get_reviews         — review snippets by product + aspect
  4. compare_products    — side-by-side comparison of 2-4 products
  5. update_user_profile — legacy compatibility write (approval-gated)
  6. get_user_profile    — read current effective profile
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional

from langchain.tools import tool

from src.config import config
from src.retrieval.index_manifest import resolve_collection_names

# Module-level state
_product_retriever = None
_profile_store = None
_catalog_db = None
_reviews_db = None


def init_shopping_tools(product_retriever, profile_store, catalog_db: str, reviews_db: str):
    """Initialize module-level dependencies. Called once at agent creation."""
    global _product_retriever, _profile_store, _catalog_db, _reviews_db
    _product_retriever = product_retriever
    _profile_store = profile_store
    _catalog_db = catalog_db
    _reviews_db = reviews_db


def _query_catalog(product_id: int) -> Optional[dict]:
    conn = sqlite3.connect(_catalog_db)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM products WHERE product_id = ?", (product_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


@tool
def search_products(query: str, top_k: int = 5) -> str:
    """Search the product catalog using semantic search.

    Use this tool when the user asks for product recommendations, comparisons,
    or wants to browse what's available. Provide a descriptive query that
    includes category, budget, use case, and any constraints.

    Args:
        query: Natural language search query (e.g. '轻薄游戏本 RTX4060 预算8000')
        top_k: Number of products to return (default 5, max 8)
    """
    if _product_retriever is None:
        return "错误: 产品检索器未初始化"
    try:
        return _product_retriever.retrieve(query, top_k=min(top_k, 8))
    except Exception as e:
        return f"产品检索出错: {e}"


def _format_specs(specs: dict) -> str:
    """Format a specs dict into readable lines."""
    lines = []
    for key, value in specs.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


@tool
def get_product_detail(product_id: int) -> str:
    """Get the complete specification sheet for a specific product.

    Use this tool when the user asks about a specific product's details or
    when you need to verify a spec before making a recommendation.

    Args:
        product_id: The product's unique identifier from search results.
    """
    p = _query_catalog(product_id)
    if not p:
        return f"未找到产品 ID={product_id}"

    specs = json.loads(p["specs"]) if p["specs"] else {}

    return (
        f"【{p['name']}】\n"
        f"  品牌: {p['brand']}  品类: {p['category']} / {p['subcategory']}\n"
        f"  价格: ¥{p['price']}\n"
        f"  评分: {p['rating']} 分  销量: {p['sales_count']}\n"
        f"  发布日期: {p['release_date']}\n"
        f"  详细参数:\n{_format_specs(specs)}\n"
        f"  简介: {p['description']}"
    )


@tool
def get_reviews(product_id: int, aspect: str = "", top_k: int = 5) -> str:
    """Get user reviews for a specific product, optionally filtered by aspect.

    Use this tool when the user asks about product quality, reliability,
    or specific concerns (performance, build quality, battery, photo quality, etc.).

    Args:
        product_id: The product's unique identifier.
        aspect: Optional filter. Leave empty to get reviews from all aspects.
        top_k: Max reviews to return (default 5).
    """
    import chromadb
    persist_dir = str(Path(_reviews_db).parent / "product_chroma_db")
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        collection_names, _ = resolve_collection_names(persist_dir)
        col = client.get_collection(collection_names["reviews"])
    except Exception:
        return "评价数据暂时不可用"

    where = {"product_id": product_id}
    if aspect:
        where["aspect"] = aspect

    try:
        results = col.get(where=where, limit=top_k, include=["documents", "metadatas"])
    except Exception:
        return "查阅评价时出错"

    if not results["ids"]:
        return f"该产品暂无{'关于' + aspect if aspect else '相关'}评价"

    lines = [f"共 {len(results['ids'])} 条评价:\n"]
    for i, doc in enumerate(results["documents"]):
        meta = results["metadatas"][i]
        emoji = "[绿]" if meta.get("sentiment") == "positive" else "[红]"
        lines.append(f"  {emoji} {meta.get('aspect', '')}: {doc}")
    return "\n".join(lines)


@tool
def compare_products(product_ids: str) -> str:
    """Compare 2-4 products side by side across all key specifications.

    Use this tool when the user wants to compare specific products or
    is deciding between a few options.

    Args:
        product_ids: Comma-separated list of product IDs, e.g. '1,5,12'.
                     Provide 2 to 4 IDs for a meaningful comparison.
    """
    ids = [int(x.strip()) for x in product_ids.split(",") if x.strip()]
    if len(ids) < 2:
        return "请提供至少 2 个产品 ID 进行对比"
    if len(ids) > 4:
        ids = ids[:4]

    products = []
    for pid in ids:
        p = _query_catalog(pid)
        if p:
            p["_specs"] = json.loads(p["specs"]) if p["specs"] else {}
            products.append(p)

    if not products:
        return "未找到指定的产品"

    # Common fields
    common_fields = [
        ("品牌", "brand"), ("品类", "category"), ("子类", "subcategory"),
        ("价格", "price"), ("评分", "rating"),
    ]

    lines = ["## 产品对比\n"]
    header = "| 属性 |" + "|".join(f" {p['name'][:15]} " for p in products) + "|"
    lines.append(header)
    lines.append("|---|" + "|".join("---" for _ in products) + "|")

    for label, key in common_fields:
        vals = []
        for p in products:
            v = p.get(key, "?")
            if key == "price" and v != "?":
                v = f"¥{v}"
            vals.append(str(v))
        lines.append(f"| {label} |" + "|".join(f" {v} " for v in vals) + "|")

    # Collect all spec keys from all products
    all_spec_keys = []
    for p in products:
        for k in p.get("_specs", {}):
            if k not in all_spec_keys:
                all_spec_keys.append(k)

    for key in all_spec_keys:
        vals = []
        for p in products:
            v = p.get("_specs", {}).get(key, "-")
            vals.append(str(v)[:25])
        lines.append(f"| {key} |" + "|".join(f" {v} " for v in vals) + "|")

    return "\n".join(lines)


@tool
def update_user_profile(conv_id: str, key: str, value: str) -> str:
    """Update the user's preference profile with a new key-value pair.

    Call this tool whenever the user reveals new preferences, constraints,
    or requirements that should be remembered for this conversation.
    Key examples: budget, primary_use, preferred_brand, product_category, mobility.

    Args:
        conv_id: The current conversation ID.
        key: Profile key (e.g. budget, primary_use, preferred_brand, product_category).
        value: Profile value (e.g. '8000-12000', 'gaming', '联想', '手机').
    """
    if _profile_store is None:
        return "画像存储未初始化"
    _profile_store.update(conv_id, key, value, confidence=0.85, source="explicit")
    return f"已更新用户画像: {key} = {value}"


@tool
def get_user_profile(conv_id: str) -> str:
    """Get the current user profile including all known preferences and constraints.

    Call this tool at the beginning of a conversation to check what you already
    know about the user, and after profile updates to get the latest state.

    Args:
        conv_id: The current conversation ID.
    """
    if _profile_store is None:
        return "画像存储未初始化"
    return _profile_store.serialize_profile(conv_id)


SHOPPING_TOOLS = [
    search_products,
    get_product_detail,
    get_reviews,
    compare_products,
    update_user_profile,
    get_user_profile,
]


# ---- Factory functions ----
# Each factory returns an @tool whose dependencies (retriever, db path, etc.)
# are captured via closure rather than stored in module-level globals.
# This eliminates the init_shopping_tools() pattern and makes tools testable
# with mock dependencies without monkey-patching.

def create_search_products(product_retriever):
    """Factory: returns a search_products tool closing over a specific retriever."""
    @tool("search_products")
    def _search_products(query: str, top_k: int = 5) -> str:
        """Search the product catalog using semantic search.

        Use this tool when the user asks for product recommendations, comparisons,
        or wants to browse what's available. Provide a descriptive query that
        includes category, budget, use case, and any constraints.

        Args:
            query: Natural language search query (e.g. '轻薄游戏本 RTX4060 预算8000')
            top_k: Number of products to return (default 5, max 8)
        """
        if product_retriever is None:
            return "错误: 产品检索器未初始化"
        try:
            return product_retriever.retrieve(query, top_k=min(top_k, 8))
        except Exception as e:
            return f"产品检索出错: {e}"
    return _search_products


def create_get_product_detail(catalog_db: str):
    """Factory: returns a get_product_detail tool closing over a specific catalog path."""
    @tool("get_product_detail")
    def _get_product_detail(product_id: int) -> str:
        """Get the complete specification sheet for a specific product.

        Use this tool when the user asks about a specific product's details or
        when you need to verify a spec before making a recommendation.

        Args:
            product_id: The product's unique identifier from search results.
        """
        p = _query_catalog_static(catalog_db, product_id)
        if not p:
            return f"未找到产品 ID={product_id}"

        specs = json.loads(p["specs"]) if p["specs"] else {}

        return (
            f"【{p['name']}】\n"
            f"  品牌: {p['brand']}  品类: {p['category']} / {p['subcategory']}\n"
            f"  价格: ¥{p['price']}\n"
            f"  评分: {p['rating']} 分  销量: {p['sales_count']}\n"
            f"  发布日期: {p['release_date']}\n"
            f"  详细参数:\n{_format_specs(specs)}\n"
            f"  简介: {p['description']}"
        )
    return _get_product_detail


def create_get_reviews(reviews_db: str):
    """Factory: returns a get_reviews tool closing over a specific reviews path."""
    @tool("get_reviews")
    def _get_reviews(product_id: int, aspect: str = "", top_k: int = 5) -> str:
        """Get user reviews for a specific product, optionally filtered by aspect.

        Use this tool when the user asks about product quality, reliability,
        or specific concerns (performance, build quality, battery, photo quality, etc.).

        Args:
            product_id: The product's unique identifier.
            aspect: Optional filter. Leave empty to get reviews from all aspects.
            top_k: Max reviews to return (default 5).
        """
        import chromadb
        from pathlib import Path as _Path
        persist_dir = str(_Path(reviews_db).parent / "product_chroma_db")
        client = chromadb.PersistentClient(path=persist_dir)
        try:
            collection_names, _ = resolve_collection_names(persist_dir)
            col = client.get_collection(collection_names["reviews"])
        except Exception:
            return "评价数据暂时不可用"

        where = {"product_id": product_id}
        if aspect:
            where["aspect"] = aspect

        try:
            results = col.get(where=where, limit=top_k, include=["documents", "metadatas"])
        except Exception:
            return "查阅评价时出错"

        if not results["ids"]:
            return f"该产品暂无{'关于' + aspect if aspect else '相关'}评价"

        lines = [f"共 {len(results['ids'])} 条评价:\n"]
        for i, doc in enumerate(results["documents"]):
            meta = results["metadatas"][i]
            emoji = "[绿]" if meta.get("sentiment") == "positive" else "[红]"
            lines.append(f"  {emoji} {meta.get('aspect', '')}: {doc}")
        return "\n".join(lines)
    return _get_reviews


def create_compare_products(catalog_db: str):
    """Factory: returns a compare_products tool closing over a specific catalog path."""
    @tool("compare_products")
    def _compare_products(product_ids: str) -> str:
        """Compare 2-4 products side by side across all key specifications.

        Use this tool when the user wants to compare specific products or
        is deciding between a few options.

        Args:
            product_ids: Comma-separated list of product IDs, e.g. '1,5,12'.
                         Provide 2 to 4 IDs for a meaningful comparison.
        """
        ids = [int(x.strip()) for x in product_ids.split(",") if x.strip()]
        if len(ids) < 2:
            return "请提供至少 2 个产品 ID 进行对比"
        if len(ids) > 4:
            ids = ids[:4]

        products = []
        for pid in ids:
            p = _query_catalog_static(catalog_db, pid)
            if p:
                p["_specs"] = json.loads(p["specs"]) if p["specs"] else {}
                products.append(p)

        if not products:
            return "未找到指定的产品"

        common_fields = [
            ("品牌", "brand"), ("品类", "category"), ("子类", "subcategory"),
            ("价格", "price"), ("评分", "rating"),
        ]

        lines = ["## 产品对比\n"]
        header = "| 属性 |" + "|".join(f" {p['name'][:15]} " for p in products) + "|"
        lines.append(header)
        lines.append("|---|" + "|".join("---" for _ in products) + "|")

        for label, key in common_fields:
            vals = []
            for p in products:
                v = p.get(key, "?")
                if key == "price" and v != "?":
                    v = f"¥{v}"
                vals.append(str(v))
            lines.append(f"| {label} |" + "|".join(f" {v} " for v in vals) + "|")

        all_spec_keys = []
        for p in products:
            for k in p.get("_specs", {}):
                if k not in all_spec_keys:
                    all_spec_keys.append(k)

        for key in all_spec_keys:
            vals = []
            for p in products:
                v = p.get("_specs", {}).get(key, "-")
                vals.append(str(v)[:25])
            lines.append(f"| {key} |" + "|".join(f" {v} " for v in vals) + "|")

        return "\n".join(lines)
    return _compare_products


def create_get_user_profile(profile_store):
    """Factory: returns a get_user_profile tool closing over a specific profile store."""
    @tool("get_user_profile")
    def _get_user_profile(conv_id: str) -> str:
        """Get the current user profile including all known preferences and constraints.

        Call this tool at the beginning of a conversation to check what you already
        know about the user, and after profile updates to get the latest state.

        Args:
            conv_id: The current conversation ID.
        """
        if profile_store is None:
            return "画像存储未初始化"
        return profile_store.serialize_profile(conv_id)
    return _get_user_profile


def create_update_user_profile(profile_store):
    """Factory: returns an update_user_profile tool closing over a specific profile store."""
    @tool("update_user_profile")
    def _update_user_profile(conv_id: str, key: str, value: str) -> str:
        """Update the user's preference profile with a new key-value pair.

        Call this tool whenever the user reveals new preferences, constraints,
        or requirements that should be remembered for this conversation.
        Key examples: budget, primary_use, preferred_brand, product_category, mobility.

        Args:
            conv_id: The current conversation ID.
            key: Profile key (e.g. budget, primary_use, preferred_brand, product_category).
            value: Profile value (e.g. '8000-12000', 'gaming', '联想', '手机').
        """
        if profile_store is None:
            return "画像存储未初始化"
        profile_store.update(conv_id, key, value, confidence=0.85, source="explicit")
        return f"已更新用户画像: {key} = {value}"
    return _update_user_profile


def _query_catalog_static(catalog_db: str, product_id: int) -> Optional[dict]:
    """Stateless SQLite catalog lookup for use by factory-created tools."""
    conn = sqlite3.connect(catalog_db)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM products WHERE product_id = ?", (product_id,)).fetchone()
    conn.close()
    return dict(row) if row else None
