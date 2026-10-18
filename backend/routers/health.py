"""Health check endpoint."""

import logging

from fastapi import APIRouter

from backend.dependencies import get_component_status
from backend.schemas import ComponentStatus, HealthResponse
from src.config import config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health():
    """Return the health status of all backend components.

    Components checked: LLM, ChromaDB, Redis, SQLite database.
    """
    components = get_component_status()
    required = ["llm", "chromadb", "database"]
    if config.REDIS_ENABLED:
        required.append("redis")
    all_ok = all(components[name] for name in required)

    return HealthResponse(
        status="ok" if all_ok else "degraded",
        components=ComponentStatus(**components),
    )
