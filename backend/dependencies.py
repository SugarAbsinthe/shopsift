"""Shared dependencies for FastAPI routes.

Provides singleton Agent and ConversationStore instances with lazy init.
The first request triggers initialization; subsequent requests reuse.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---- Lazy-loaded singletons ----

_agent = None
_conv_store = None
_profile_store_for_api = None
_product_retriever_for_health = None

BASE_DIR = Path(__file__).resolve().parent.parent


def _init_agent():
    """Create the ShoppingGuideAgent singleton (called once on first request)."""
    global _agent

    import sys
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))

    from src.config import config, create_llm
    from src.cache.rag_cache import RAGCache
    from src.retrieval.product_retriever import ProductRetriever
    from src.profile.profile_store import ProfileStore
    from src.agent.shopping_agent import ShoppingGuideAgent

    # Shared cache instance for health check
    global _product_retriever_for_health

    cache = RAGCache(
        redis_url=config.REDIS_URL,
        ttl=config.RAG_CACHE_TTL,
    ) if config.REDIS_ENABLED else None

    retriever = ProductRetriever(
        chroma_dir=config.PRODUCT_CHROMA_DIR,
        embedding_model=config.EMBEDDING_MODEL,
        catalog_db=config.PRODUCT_DB_PATH,
        cache=cache,
    )
    _product_retriever_for_health = retriever

    profile_store = ProfileStore(
        db_path=config.PROFILE_DB_PATH,
        chroma_dir=config.PROFILE_CHROMA_DIR,
        embedding_model=config.EMBEDDING_MODEL,
    )

    llm = create_llm(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
        model=config.LLM_MODEL,
    )

    _agent = ShoppingGuideAgent(
        llm=llm,
        product_retriever=retriever,
        profile_store=profile_store,
        max_tool_rounds=config.MAX_TOOL_ROUNDS,
    )
    logger.info("Agent singleton initialized (model=%s)", config.LLM_MODEL)


def get_agent():
    """Return the global Agent instance, initializing on first call."""
    global _agent
    if _agent is None:
        _init_agent()
    return _agent


def get_conv_store():
    """Return the global ConversationStore instance."""
    global _conv_store
    if _conv_store is None:
        from src.db.conversation_store import ConversationStore
        db_path = str(BASE_DIR / "data" / "conversations.db")
        _conv_store = ConversationStore(db_path)
        logger.info("ConversationStore initialized at %s", db_path)
    return _conv_store


def clear_conversation_runtime(conv_id: str) -> None:
    """Clear graph checkpoints and profile data without forcing Agent startup."""
    if _agent is not None:
        try:
            _agent.clear_conversation_state(conv_id)
        except Exception as exc:
            logger.error("Failed to clear runtime state for %s: %s", conv_id, exc)
        return

    from src.config import config

    # Structured profile data can be removed directly without loading the
    # embedding model. Semantic memory is best-effort because Chroma is optional.
    import os
    import sqlite3
    if os.path.exists(config.PROFILE_DB_PATH):
        conn = sqlite3.connect(config.PROFILE_DB_PATH)
        try:
            for table in ("user_profiles", "memory_candidates"):
                try:
                    conn.execute(f"DELETE FROM {table} WHERE conv_id = ?", (conv_id,))
                except sqlite3.OperationalError:
                    pass
            conn.commit()
        finally:
            conn.close()

    if os.path.exists(config.AGENT_CHECKPOINT_DB_PATH):
        from langgraph.checkpoint.sqlite import SqliteSaver
        try:
            with SqliteSaver.from_conn_string(config.AGENT_CHECKPOINT_DB_PATH) as saver:
                saver.delete_thread(conv_id)
        except sqlite3.OperationalError:
            pass

    try:
        import chromadb
        client = chromadb.PersistentClient(path=config.PROFILE_CHROMA_DIR)
        client.get_collection("user_memory").delete(where={"conv_id": conv_id})
    except Exception:
        pass


def close_dependencies() -> None:
    """Release process-wide resources during application shutdown."""
    global _agent
    if _agent is not None:
        _agent.close()
        _agent = None


def get_component_status() -> dict:
    """Check health of all backend components."""
    status = {"llm": False, "chromadb": False, "redis": False, "database": False}

    # LLM
    try:
        agent = get_agent()
        status["llm"] = agent.llm is not None
    except Exception:
        pass

    # ChromaDB
    try:
        import chromadb
        from src.config import config
        client = chromadb.PersistentClient(path=config.PRODUCT_CHROMA_DIR)
        client.get_collection("product_descriptions")
        status["chromadb"] = True
    except Exception:
        pass

    # Redis
    try:
        if _product_retriever_for_health and _product_retriever_for_health._cache:
            if _product_retriever_for_health._cache._redis:
                _product_retriever_for_health._cache._redis.ping()
                status["redis"] = True
    except Exception:
        pass

    # SQLite
    try:
        import sqlite3
        from src.config import config
        conn = sqlite3.connect(config.PRODUCT_DB_PATH)
        conn.execute("SELECT 1 FROM products LIMIT 1")
        conn.close()
        status["database"] = True
    except Exception:
        pass

    return status
