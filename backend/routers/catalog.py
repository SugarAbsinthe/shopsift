"""Expose only categories present in the configured, read-only catalog."""
from contextlib import closing
from pathlib import Path
import sqlite3

from fastapi import APIRouter
from src.config import config
from src.data.categories import CATEGORIES, EXAMPLES

router = APIRouter(prefix="/catalog", tags=["catalog"])


@router.get("/categories")
def categories():
    try:
        path = Path(config.PRODUCT_DB_PATH).resolve()
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
            counts = dict(conn.execute("SELECT category, count(*) FROM products GROUP BY category"))
    except (OSError, sqlite3.Error):
        return {"categories": [], "available": False}
    return {"available": True, "categories": [
        {"name": name, "count": counts[name], "example": EXAMPLES[name], "status": "trial"}
        for name in CATEGORIES if counts.get(name)
    ]}
