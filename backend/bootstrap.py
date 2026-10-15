"""Prepare an existing catalog, or import the bundled catalog on a fresh install.

No model API requests, replacement of user catalogs, or index deletion.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path
import sqlite3

from src.retrieval.index_manifest import (
    LEGACY_COLLECTIONS,
    MANIFEST_FILENAME,
    load_index_manifest,
)


class BootstrapError(RuntimeError):
    """A startup prerequisite is missing or inconsistent."""


_TABLE_FIELDS = {
    "products": {
        "product_id", "name", "brand", "category", "subcategory", "price",
        "rating", "sales_count", "release_date", "description", "specs",
    },
    "product_reviews": {"review_id", "product_id", "sentiment", "aspect", "content"},
}


def validate_database(path: Path, table: str) -> None:
    """Read-only checks; a missing SQLite file must never be created."""
    if not path.is_file():
        raise BootstrapError(f"Missing {path}; prepare your catalog as described in DEVELOPMENT.md.")
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            if conn.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise BootstrapError(f"{path}: SQLite integrity check failed.")
            fields = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
            missing = _TABLE_FIELDS[table] - fields
            if missing:
                raise BootstrapError(f"{path}: {table} is missing fields: {', '.join(sorted(missing))}.")
            if not conn.execute(f'SELECT 1 FROM "{table}" LIMIT 1').fetchone():
                raise BootstrapError(f"{path}: {table} is empty; prepare the catalog explicitly.")
    except sqlite3.Error as exc:
        raise BootstrapError(f"Cannot read {path} as a valid SQLite catalog.") from exc


def _load_model(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def _open_client(path: str):
    import chromadb
    return chromadb.PersistentClient(path=path)


def _build_index(*args, **kwargs):
    from src.embeddings.product_embedder import build_product_embeddings
    return build_product_embeddings(*args, **kwargs)


def prepare_runtime(settings, *, rebuild_index: bool = False) -> str:
    """Validate inputs, reuse a complete index or build the first version."""
    if not settings.OPENAI_API_KEY.strip() or settings.OPENAI_API_KEY.strip() in {
        "sk-placeholder", "your-api-key-here",
    }:
        raise BootstrapError("Set a real OPENAI_API_KEY in .env before starting the shopping assistant.")
    if not settings.OPENAI_BASE_URL.strip() or not settings.LLM_MODEL.strip():
        raise BootstrapError("Set OPENAI_BASE_URL and LLM_MODEL in .env.")

    products = Path(settings.PRODUCT_DB_PATH)
    reviews = products.parent / "product_reviews.db"
    from src.data.catalog import CatalogError, catalog_fingerprint
    from src.data.prepare_catalog import ensure_default_catalog
    try:
        ensure_default_catalog(products)
    except (CatalogError, OSError, sqlite3.Error) as exc:
        raise BootstrapError(f"Bundled catalog preparation failed: {exc}") from exc
    if (products.parent / f".{products.name}.catalog-import.json").exists():
        raise BootstrapError("Interrupted catalog import; rerun python -m src.data.prepare_catalog for this destination before indexing.")
    validate_database(products, "products")
    validate_database(reviews, "product_reviews")

    persist_dir = Path(settings.PRODUCT_CHROMA_DIR)
    manifest = load_index_manifest(persist_dir)
    if not rebuild_index and (persist_dir / MANIFEST_FILENAME).exists() and manifest is None:
        raise BootstrapError("Invalid retrieval_manifest.json; inspect it or explicitly rebuild the index.")
    if not rebuild_index and manifest and manifest.get("embedding_model") != settings.EMBEDDING_MODEL:
        raise BootstrapError("EMBEDDING_MODEL differs from the active index; align it or explicitly rebuild.")
    if not rebuild_index and manifest and manifest.get("catalog_fingerprint"):
        if catalog_fingerprint(products, reviews) != manifest["catalog_fingerprint"]:
            raise BootstrapError("Catalog content changed; explicitly rebuild the index with --rebuild-index.")

    # Warm the configured embedding model even when reusing an index. This also
    # prepares an initially empty named cache volume before the API starts.
    model = _load_model(settings.EMBEDDING_MODEL)
    client = _open_client(str(persist_dir))
    names = {getattr(collection, "name", collection) for collection in client.list_collections()}

    if not rebuild_index and (manifest is not None or names):
        active = manifest["collections"] if manifest else LEGACY_COLLECTIONS
        if not set(active.values()) <= names:
            raise BootstrapError("Active index collections are incomplete; explicitly rebuild the index.")
        counts = {key: client.get_collection(name).count() for key, name in active.items()}
        if counts["descriptions"] < 1:
            raise BootstrapError("Active product index is empty; explicitly rebuild the index.")
        if manifest and counts != manifest.get("counts"):
            raise BootstrapError("Active index counts differ from the manifest; explicitly rebuild the index.")
        version = manifest["version"] if manifest else "legacy (embedding model must match)"
        print(f"Reusing retrieval index {version}", flush=True)
        return "reused"

    manifest = _build_index(
        str(products), str(reviews), str(persist_dir), settings.EMBEDDING_MODEL,
        model=model, client=client,
    )
    print(f"Activated retrieval index {manifest['version']}", flush=True)
    return "built"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-index", action="store_true", help="Build and activate a new index version.")
    args = parser.parse_args()
    try:
        from src.config import config
        prepare_runtime(config, rebuild_index=args.rebuild_index)
    except BootstrapError as exc:
        print(f"Startup prerequisite: {exc}", flush=True)
        return 2
    except Exception as exc:
        # Do not echo arbitrary model/client errors that can contain credentials.
        print(f"Startup initialization failed ({type(exc).__name__}); check model access and catalog fields.", flush=True)
        return 1
    print("Startup data and embedding prerequisites are ready; model API access is checked during chat.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
