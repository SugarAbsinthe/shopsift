"""Validated, atomic activation metadata for versioned retrieval indexes."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


MANIFEST_FILENAME = "retrieval_manifest.json"
LEGACY_COLLECTIONS = {
    "descriptions": "product_descriptions",
    "specs": "product_specs",
    "reviews": "product_reviews",
}
_SAFE_COLLECTION = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _validate_manifest(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != 1 or not isinstance(payload.get("version"), str):
        return None
    collections = payload.get("collections")
    if not isinstance(collections, dict):
        return None
    if set(collections) != set(LEGACY_COLLECTIONS):
        return None
    if not all(
        isinstance(name, str) and _SAFE_COLLECTION.fullmatch(name)
        for name in collections.values()
    ):
        return None
    fingerprint = payload.get("catalog_fingerprint")
    if fingerprint is not None and (not isinstance(fingerprint, str) or not re.fullmatch(r"[a-f0-9]{64}", fingerprint)):
        return None
    return payload


def load_index_manifest(persist_dir: str | Path) -> dict[str, Any] | None:
    path = Path(persist_dir) / MANIFEST_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _validate_manifest(payload)


def resolve_collection_names(
    persist_dir: str | Path,
) -> tuple[dict[str, str], str]:
    manifest = load_index_manifest(persist_dir)
    if manifest is None:
        return dict(LEGACY_COLLECTIONS), "legacy"
    return dict(manifest["collections"]), manifest["version"]


def activate_index_manifest(persist_dir: str | Path, payload: dict[str, Any]) -> None:
    """Atomically switch the active index after validating the full payload."""
    directory = Path(persist_dir)
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / MANIFEST_FILENAME
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=directory, prefix=".manifest-", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if _validate_manifest(json.loads(temp_path.read_text(encoding="utf-8"))) is None:
            raise ValueError("invalid retrieval manifest")
        os.replace(temp_path, probe)
    finally:
        if temp_path.exists():
            temp_path.unlink()
