"""Validate and import sourced JSONL into SQLite, without network/model calls."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager, closing
from pathlib import Path

from src.data.catalog import (
    CatalogError, DEFAULT_DATASET_DIR, PRODUCT_FIELDS, REVIEW_FIELDS,
    canonical_json, catalog_fingerprint, load_dataset, sha256_file,
)

DEFAULT_PRODUCT_DB = Path(__file__).resolve().parents[2] / "data" / "products.db"


@contextmanager
def _import_lock(path: Path):
    stream = path.open("a+b")
    try:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise CatalogError("Another catalog import is running") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        stream.close()


def _write_marker(path: Path, payload: dict) -> None:
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        stream.write(canonical_json(payload))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _safe_child(parent: Path, name: str) -> Path:
    child = parent / name
    if Path(name).name != name or child.resolve().parent != parent.resolve():
        raise CatalogError("Unsafe catalog recovery path")
    return child


def _recover(marker: Path, products: Path, reviews: Path) -> None:
    if not marker.exists():
        return
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        stages = [_safe_child(marker.parent, name) for name in payload["stages"]]
        if len(stages) != 2 or any(not p.name.startswith(".catalog-stage-") for p in stages):
            raise CatalogError("Invalid catalog recovery marker")
        expected = payload.get("outputs", {})
        if any(name not in {products.name, reviews.name} for name in expected):
            raise CatalogError("Unexpected catalog recovery output")
        finished = products.is_file() and reviews.is_file() and expected and all(
            sha256_file(path) == expected.get(path.name) for path in (products, reviews)
        )
        if not finished:
            owned = []
            for path in (products, reviews):
                if path.exists() and path.name in expected:
                    if sha256_file(path) != expected[path.name]:
                        raise CatalogError("Interrupted import output was changed; preserve it and repair explicitly")
                    owned.append(path)
            # Verify every output before removing any: a modified second file
            # must preserve the first file as well for explicit repair.
            for path in owned:
                path.unlink()
        for path in stages:
            path.unlink(missing_ok=True)
        marker.unlink()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CatalogError(f"Cannot recover interrupted catalog import: {exc}") from exc


def _write_database(path: Path, tables: dict, names: tuple[str, ...], metadata: dict) -> None:
    ddl = {
        "products": "product_id INTEGER PRIMARY KEY, name TEXT NOT NULL, brand TEXT NOT NULL, category TEXT NOT NULL, subcategory TEXT NOT NULL, price REAL NOT NULL CHECK(price > 0), rating REAL, sales_count INTEGER, release_date TEXT, description TEXT NOT NULL, specs TEXT NOT NULL",
        "product_reviews": "review_id INTEGER PRIMARY KEY, product_id INTEGER NOT NULL, sentiment TEXT NOT NULL, aspect TEXT NOT NULL, content TEXT NOT NULL",
        "product_provenance": "product_id INTEGER PRIMARY KEY REFERENCES products(product_id), sku_key TEXT UNIQUE NOT NULL, model_key TEXT NOT NULL, metadata TEXT NOT NULL",
        "review_provenance": "review_id INTEGER PRIMARY KEY REFERENCES product_reviews(review_id), summary_key TEXT NOT NULL, metadata TEXT NOT NULL",
        "catalog_sources": "source_id TEXT PRIMARY KEY, metadata TEXT NOT NULL",
    }
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("PRAGMA foreign_keys=ON")
        for table in names:
            conn.execute(f"CREATE TABLE {table} ({ddl[table]})")
            rows = tables[table]
            if rows:
                columns = list(rows[0])
                placeholders = ",".join("?" for _ in columns)
                conn.executemany(
                    f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
                    [tuple(row[column] for column in columns) for row in rows],
                )
        conn.execute("CREATE TABLE catalog_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.executemany("INSERT INTO catalog_metadata VALUES (?, ?)", [(key, str(value)) for key, value in metadata.items()])
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok" or conn.execute("PRAGMA foreign_key_check").fetchall():
            raise CatalogError("Imported database failed integrity validation")


def prepare_catalog(dataset_dir: Path | str, product_db: Path | str, *, require_complete: bool = False) -> dict:
    manifest, tables, report = load_dataset(dataset_dir, require_complete=require_complete)
    products = Path(product_db).resolve()
    reviews = products.parent / "product_reviews.db"
    if products == reviews:
        raise CatalogError("Product and review database paths must differ")
    products.parent.mkdir(parents=True, exist_ok=True)
    marker = products.parent / f".{products.name}.catalog-import.json"
    with _import_lock(products.parent / f".{products.name}.catalog-import.lock"):
        _recover(marker, products, reviews)
        if products.exists() or reviews.exists():
            if not products.is_file() or not reviews.is_file():
                raise CatalogError("Only one database exists; existing data is preserved")
            try:
                matches = catalog_fingerprint(products, reviews) == report["catalog_fingerprint"]
            except (sqlite3.Error, OSError, ValueError):
                matches = False
            if not matches:
                raise CatalogError("Existing catalog differs; choose an empty destination, existing data is preserved")
            return {**report, "import_status": "reused"}
        token = uuid.uuid4().hex
        stage_products = products.parent / f".catalog-stage-{token}-products.db"
        stage_reviews = products.parent / f".catalog-stage-{token}-reviews.db"
        payload = {"stages": [stage_products.name, stage_reviews.name], "outputs": {}}
        _write_marker(marker, payload)
        try:
            metadata = {"dataset_version": manifest["dataset_version"],
                        "catalog_fingerprint": report["catalog_fingerprint"],
                        "manifest_sha256": sha256_file(Path(dataset_dir) / "manifest.json")}
            _write_database(stage_products, tables, ("products", "product_provenance", "catalog_sources"), metadata)
            _write_database(stage_reviews, tables, ("product_reviews", "review_provenance"), metadata)
            if catalog_fingerprint(stage_products, stage_reviews) != report["catalog_fingerprint"]:
                raise CatalogError("Imported catalog fingerprint mismatch")
            payload["outputs"] = {products.name: sha256_file(stage_products), reviews.name: sha256_file(stage_reviews)}
            _write_marker(marker, payload)
            # Hard links publish a complete SQLite file without replacing any
            # concurrently created user file. Staging and targets share a volume.
            os.link(stage_products, products)
            os.link(stage_reviews, reviews)
            _recover(marker, products, reviews)
        except BaseException:
            _recover(marker, products, reviews)
            raise
    return {**report, "import_status": "imported"}


def ensure_default_catalog(product_db: Path | str) -> None:
    products = Path(product_db).resolve()
    default = DEFAULT_PRODUCT_DB
    marker = products.parent / f".{products.name}.catalog-import.json"
    if products == default.resolve() and (marker.exists() or (not products.exists() and not (products.parent / "product_reviews.db").exists())):
        prepare_catalog(DEFAULT_DATASET_DIR, products, require_complete=True)


def main() -> int:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--product-db", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.validate_only:
            _, _, report = load_dataset(args.dataset_dir)
        else:
            if args.product_db is None:
                from src.config import config
                args.product_db = Path(config.PRODUCT_DB_PATH)
            report = prepare_catalog(args.dataset_dir, args.product_db)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (CatalogError, OSError, sqlite3.Error) as exc:
        print(f"Catalog preparation failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
