"""Build versioned product embeddings and activate them only after validation."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.retrieval.index_manifest import activate_index_manifest
from src.data.catalog import catalog_fingerprint, load_provenance


def _rows(db_path: str, query: str) -> list[dict]:
    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute(query)]
    finally:
        conn.close()


def _encode(model, documents: list[str]) -> list[list[float]]:
    if not documents:
        return []
    embeddings = model.encode(documents)
    return [embedding.tolist() for embedding in embeddings]


def build_product_embeddings(
    products_db: str,
    reviews_db: str,
    persist_dir: str,
    embedding_model_name: str,
    *,
    batch_size: int = 64,
    index_version: str | None = None,
    model=None,
    client=None,
) -> dict:
    """Build isolated collections; keep the old manifest active on failure."""
    if batch_size < 1 or batch_size > 1000:
        raise ValueError("batch_size must be between 1 and 1000")
    version = index_version or (
        datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]
    )
    safe_version = "".join(char for char in version if char.isalnum() or char in "-_")
    if not safe_version or safe_version != version or len(version) > 64:
        raise ValueError("index_version contains unsafe characters")

    content_fingerprint = catalog_fingerprint(products_db, reviews_db)
    embedding_model = model or SentenceTransformer(embedding_model_name)
    chroma_client = client or chromadb.PersistentClient(path=persist_dir)
    names = {
        "descriptions": f"product_descriptions_{version}",
        "specs": f"product_specs_{version}",
        "reviews": f"product_reviews_{version}",
    }
    created: list[str] = []
    try:
        products = _rows(products_db, "SELECT * FROM products ORDER BY product_id")
        reviews = _rows(reviews_db, "SELECT * FROM product_reviews ORDER BY review_id")
        review_sources = load_provenance(reviews_db, "review_provenance", [r["review_id"] for r in reviews])
        specs: list[dict] = []
        for product in products:
            for attribute, value in json.loads(product["specs"]).items():
                specs.append({
                    "id": f"spec_{product['product_id']}_{attribute}",
                    "document": f"{product['name']}（{product['category']}）的{attribute}: {value}",
                    "metadata": {
                        "product_id": product["product_id"],
                        "product_name": product["name"],
                        "brand": product["brand"],
                        "category": product["category"],
                        "subcategory": product["subcategory"],
                        "attribute": attribute,
                        "value": str(value),
                    },
                })

        datasets = {
            "descriptions": [
                {
                    "id": f"product_{product['product_id']}",
                    "document": product["description"],
                    "metadata": {
                        key: product[key]
                        for key in (
                            "product_id", "name", "brand", "category", "subcategory",
                            "price", "rating", "sales_count", "release_date",
                        )
                        if product[key] is not None and product[key] != ""
                    },
                }
                for product in products
            ],
            "specs": specs,
            "reviews": [
                {
                    "id": f"review_{review['review_id']}",
                    "document": f"[{review['sentiment']}] {review['aspect']}: {review['content']}",
                    "metadata": {
                        "product_id": review["product_id"],
                        "aspect": review["aspect"],
                        "sentiment": review["sentiment"],
                        "content": review["content"],
                        **{key: value for key, value in review_sources.get(review["review_id"], {}).items()
                           if isinstance(value, (str, int, float, bool))},
                    },
                }
                for review in reviews
            ],
        }

        collections = {}
        for key, name in names.items():
            # A partial final HNSW batch can remain memory-only even though
            # SQLite reports the full count. End on a batch/sync boundary so
            # metadata prefilters can resolve every ID after a process restart.
            count = len(datasets[key])
            threshold = next((size for size in range(min(batch_size, count), 2, -1)
                              if count % size == 0), max(3, count))
            collections[key] = chroma_client.create_collection(
                name, metadata={"hnsw:batch_size": threshold, "hnsw:sync_threshold": threshold}
            )
            created.append(name)

        for key, records in datasets.items():
            collection = collections[key]
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                documents = [record["document"] for record in batch]
                collection.add(
                    ids=[record["id"] for record in batch],
                    documents=documents,
                    embeddings=_encode(embedding_model, documents),
                    metadatas=[record["metadata"] for record in batch],
                )
            if collection.count() != len(records):
                raise RuntimeError(f"index count mismatch for {key}")

        manifest = {
            "schema_version": 1,
            "version": version,
            "embedding_model": embedding_model_name,
            "collections": names,
            "counts": {key: len(records) for key, records in datasets.items()},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "catalog_fingerprint": content_fingerprint,
        }
        if catalog_fingerprint(products_db, reviews_db) != content_fingerprint:
            raise RuntimeError("catalog changed during index construction")
        activate_index_manifest(persist_dir, manifest)
        return manifest
    except Exception:
        for name in created:
            try:
                chroma_client.delete_collection(name)
            except Exception:
                pass
        raise


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent.parent
    manifest = build_product_embeddings(
        str(base_dir / "data" / "products.db"),
        str(base_dir / "data" / "product_reviews.db"),
        str(base_dir / "data" / "product_chroma_db"),
        "BAAI/bge-small-zh-v1.5",
    )
    print(f"Activated retrieval index {manifest['version']}")


if __name__ == "__main__":
    main()
