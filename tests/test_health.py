"""Health checks must match the active index and configured components."""
import asyncio
from types import SimpleNamespace

from backend import dependencies
from backend.routers import health
from src.config import config


def test_health_ignores_disabled_optional_redis(monkeypatch):
    monkeypatch.setattr(config, "REDIS_ENABLED", False)
    monkeypatch.setattr(health, "get_component_status", lambda: {
        "llm": True, "chromadb": True, "redis": False, "database": True,
    })
    result = asyncio.run(health.health())
    assert result.status == "ok"
    assert result.components.redis is False
    monkeypatch.setattr(config, "REDIS_ENABLED", True)
    assert asyncio.run(health.health()).status == "degraded"


def test_component_status_checks_all_active_collections(monkeypatch):
    import chromadb
    from src.retrieval import index_manifest

    names = {"descriptions": "active-desc", "specs": "active-spec", "reviews": "active-review"}
    checked = []
    monkeypatch.setattr(index_manifest, "resolve_collection_names", lambda _: (names, "v-test"))
    monkeypatch.setattr(chromadb, "PersistentClient", lambda **_: SimpleNamespace(
        get_collection=lambda name: checked.append(name),
    ))
    monkeypatch.setattr(dependencies, "get_agent", lambda: SimpleNamespace(llm=object()))
    # Read only an isolated database, never the configured user's catalog.
    monkeypatch.setattr(config, "PRODUCT_DB_PATH", ":memory:")
    assert dependencies.get_component_status()["chromadb"] is True
    assert checked == list(names.values())

    def missing(name):
        if name == "active-spec":
            raise ValueError("missing active collection")

    monkeypatch.setattr(chromadb, "PersistentClient", lambda **_: SimpleNamespace(get_collection=missing))
    assert dependencies.get_component_status()["chromadb"] is False
