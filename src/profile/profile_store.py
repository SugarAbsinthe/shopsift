"""Governed structured profiles plus an isolated semantic-memory layer."""

from __future__ import annotations

import math
import os
import sqlite3
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from backend.logging_config import hash_identifier, log
from src.profile.models import MEMORY_SCHEMA_VERSION, PROFILE_KEYS, MemoryCandidate


_VALID_MEMORY_TYPES = frozenset({"explicit", "inferred"})
_VALID_STATUSES = frozenset({"pending", "confirmed", "rejected", "expired", "deleted"})
_EXPLICIT_TTL_DAYS = 180
_INFERRED_TTL_DAYS = 30


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds")


def _now_iso() -> str:
    return _to_iso(_now())


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc)


def _days_since(iso_ts: str) -> float:
    try:
        return max(0, (_now() - _parse_iso(iso_ts)).total_seconds() / 86400.0)
    except (TypeError, ValueError):
        return 0


class ProfileStore:
    """Manage active constraints, governed candidates, and semantic references."""

    def __init__(
        self,
        db_path: str,
        chroma_dir: str,
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        decay_lambda: float = 0.05,
    ):
        self.db_path = db_path
        self.decay_lambda = decay_lambda
        self.model = SentenceTransformer(embedding_model)

        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_db()

        self.client = chromadb.PersistentClient(path=chroma_dir)
        try:
            self.memory_col = self.client.get_collection("user_memory")
        except Exception:
            self.memory_col = self.client.create_collection("user_memory")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            self._initialize_tables(conn)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize_tables(self, conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN IMMEDIATE")
        old_columns = {row["name"] for row in conn.execute("PRAGMA table_info(user_profiles)")}
        if old_columns and "category_scope" not in old_columns:
            conn.execute("ALTER TABLE user_profiles RENAME TO user_profiles_unscoped")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_id TEXT NOT NULL,
                profile_key TEXT NOT NULL,
                profile_value TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                last_updated TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'explicit',
                category_scope TEXT NOT NULL DEFAULT '',
                UNIQUE(conv_id, category_scope, profile_key)
            )
        """)
        profile_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(user_profiles)")
        }
        if "candidate_id" not in profile_columns:
            conn.execute("ALTER TABLE user_profiles ADD COLUMN candidate_id TEXT")
        if old_columns and "category_scope" not in old_columns:
            candidate_column = "p.candidate_id" if "candidate_id" in old_columns else "NULL"
            conn.execute(f"""
                INSERT INTO user_profiles
                    (id, conv_id, profile_key, profile_value, confidence, last_updated, source, candidate_id, category_scope)
                SELECT p.id, p.conv_id, p.profile_key, p.profile_value, p.confidence,
                       p.last_updated, p.source, {candidate_column},
                       CASE WHEN p.profile_key = 'product_category' THEN '' ELSE COALESCE(
                           (SELECT c.profile_value FROM user_profiles_unscoped c
                            WHERE c.conv_id = p.conv_id AND c.profile_key = 'product_category'
                            AND c.source = 'explicit'), '') END
                FROM user_profiles_unscoped p
            """)
            conn.execute("DROP TABLE user_profiles_unscoped")
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_profiles_conv
            ON user_profiles(conv_id)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_candidates (
                candidate_id TEXT PRIMARY KEY,
                conv_id TEXT NOT NULL,
                profile_key TEXT NOT NULL,
                profile_value TEXT NOT NULL,
                memory_type TEXT NOT NULL CHECK(memory_type IN ('explicit', 'inferred')),
                confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
                source_message_id TEXT NOT NULL,
                evidence_span TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('pending', 'confirmed', 'rejected', 'expired', 'deleted')),
                schema_version TEXT NOT NULL,
                category_scope TEXT NOT NULL DEFAULT ''
            )
        """)
        memory_columns = {row["name"] for row in conn.execute("PRAGMA table_info(memory_candidates)")}
        if "category_scope" not in memory_columns:
            conn.execute("ALTER TABLE memory_candidates ADD COLUMN category_scope TEXT NOT NULL DEFAULT ''")
            conn.execute("""
                UPDATE memory_candidates SET category_scope = COALESCE(
                    (SELECT p.profile_value FROM user_profiles p
                     WHERE p.conv_id = memory_candidates.conv_id AND p.profile_key = 'product_category'
                     AND p.source = 'explicit'), '')
                WHERE profile_key != 'product_category' AND status IN ('pending', 'confirmed')
            """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_candidates_conv_status
            ON memory_candidates(conv_id, status, created_at)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_candidates_conv_key
            ON memory_candidates(conv_id, profile_key, created_at)
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_candidates_scope_key "
                     "ON memory_candidates(conv_id, category_scope, profile_key, created_at)")
        self._migrate_legacy_profiles(conn)

    @staticmethod
    def _migrate_legacy_profiles(conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            "SELECT id, conv_id, profile_key, profile_value, confidence, "
            "last_updated, source, candidate_id, category_scope FROM user_profiles"
        ).fetchall()
        for row in rows:
            if row["candidate_id"] and conn.execute(
                "SELECT 1 FROM memory_candidates WHERE candidate_id = ?",
                (row["candidate_id"],),
            ).fetchone():
                continue
            candidate_id = f"legacy_{row['id']}"
            if conn.execute(
                "SELECT 1 FROM memory_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone():
                continue
            memory_type = "explicit" if row["source"] == "explicit" else "inferred"
            status = "confirmed" if memory_type == "explicit" else "pending"
            try:
                created_at = _parse_iso(row["last_updated"])
            except (TypeError, ValueError):
                created_at = _now()
            ttl = _EXPLICIT_TTL_DAYS if memory_type == "explicit" else _INFERRED_TTL_DAYS
            conn.execute("""
                INSERT INTO memory_candidates (
                    candidate_id, conv_id, profile_key, profile_value, memory_type,
                    confidence, source_message_id, evidence_span, created_at,
                    expires_at, status, schema_version, category_scope
                ) VALUES (?, ?, ?, ?, ?, ?, ?, '', ?, ?, ?, ?, ?)
            """, (
                candidate_id,
                row["conv_id"],
                row["profile_key"],
                row["profile_value"],
                memory_type,
                row["confidence"],
                f"legacy:{row['id']}",
                _to_iso(created_at),
                _to_iso(created_at + timedelta(days=ttl)),
                status,
                MEMORY_SCHEMA_VERSION,
                row["category_scope"],
            ))
            if status == "confirmed":
                conn.execute(
                    "UPDATE user_profiles SET candidate_id = ? WHERE id = ?",
                    (candidate_id, row["id"]),
                )

        # Preserve historical inference as a candidate, but stop using it as a
        # hard filter until the user confirms it.
        conn.execute("DELETE FROM user_profiles WHERE source != 'explicit'")

    @staticmethod
    def _validate_candidate_input(
        key: str,
        value: str,
        memory_type: str,
        confidence: float,
        source_id: str,
        evidence_span: str,
    ) -> None:
        if key not in PROFILE_KEYS:
            raise ValueError("unsupported profile key")
        if not isinstance(value, str) or not value.strip() or len(value) > 256:
            raise ValueError("profile value must contain 1-256 characters")
        if memory_type not in _VALID_MEMORY_TYPES:
            raise ValueError("invalid memory type")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("confidence must be numeric")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if not source_id or len(source_id) > 128:
            raise ValueError("invalid source message id")
        if len(evidence_span) > 160:
            raise ValueError("evidence span is too long")

    @staticmethod
    def _row_to_candidate(row: sqlite3.Row) -> MemoryCandidate:
        return MemoryCandidate(
            candidate_id=row["candidate_id"],
            conv_id=row["conv_id"],
            key=row["profile_key"],
            value=row["profile_value"],
            type=row["memory_type"],
            confidence=float(row["confidence"]),
            source_message_id=row["source_message_id"],
            evidence_span=row["evidence_span"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            status=row["status"],
            schema_version=row["schema_version"],
            category_scope=row["category_scope"],
        )

    @staticmethod
    def _candidate_by_id(
        conn: sqlite3.Connection, conv_id: str, candidate_id: str
    ) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM memory_candidates WHERE conv_id = ? AND candidate_id = ?",
            (conv_id, candidate_id),
        ).fetchone()

    def _expire_due(self, conn: sqlite3.Connection, conv_id: str) -> list[sqlite3.Row]:
        rows = conn.execute(
            "SELECT * FROM memory_candidates WHERE conv_id = ? "
            "AND status IN ('pending', 'confirmed') AND expires_at <= ?",
            (conv_id, _now_iso()),
        ).fetchall()
        for row in rows:
            conn.execute(
                "UPDATE memory_candidates SET status = 'expired' WHERE candidate_id = ?",
                (row["candidate_id"],),
            )
            if row["status"] == "confirmed":
                conn.execute(
                    "DELETE FROM user_profiles WHERE conv_id = ? AND candidate_id = ?",
                    (conv_id, row["candidate_id"]),
                )
        return rows

    @staticmethod
    def _conflicting_keys(key: str) -> tuple[str, ...]:
        if key == "preferred_brand":
            return ("exclude_brand",)
        if key == "exclude_brand":
            return ("preferred_brand",)
        return ()

    def _activate_candidate(
        self, conn: sqlite3.Connection, row: sqlite3.Row
    ) -> list[sqlite3.Row]:
        conflicts = conn.execute(
            "SELECT * FROM memory_candidates WHERE conv_id = ? AND status = 'confirmed' "
            "AND candidate_id != ? AND profile_key = ? AND category_scope = ?",
            (row["conv_id"], row["candidate_id"], row["profile_key"], row["category_scope"]),
        ).fetchall()
        for conflict_key in self._conflicting_keys(row["profile_key"]):
            opposite = conn.execute(
                "SELECT * FROM memory_candidates WHERE conv_id = ? AND status = 'confirmed' "
                "AND profile_key = ? AND category_scope = ?",
                (row["conv_id"], conflict_key, row["category_scope"]),
            ).fetchall()
            conflicts.extend(
                item for item in opposite
                if item["profile_value"].casefold() == row["profile_value"].casefold()
            )

        unique_conflicts = {item["candidate_id"]: item for item in conflicts}
        for conflict in unique_conflicts.values():
            conn.execute(
                "UPDATE memory_candidates SET status = 'expired' WHERE candidate_id = ?",
                (conflict["candidate_id"],),
            )
            conn.execute(
                "DELETE FROM user_profiles WHERE conv_id = ? AND candidate_id = ?",
                (row["conv_id"], conflict["candidate_id"]),
            )

        conn.execute(
            "UPDATE memory_candidates SET status = 'confirmed', memory_type = 'explicit' "
            "WHERE candidate_id = ?",
            (row["candidate_id"],),
        )
        conn.execute("""
            INSERT INTO user_profiles (
                conv_id, profile_key, profile_value, confidence, last_updated,
                source, candidate_id, category_scope
            ) VALUES (?, ?, ?, ?, ?, 'explicit', ?, ?)
            ON CONFLICT(conv_id, category_scope, profile_key) DO UPDATE SET
                profile_value = excluded.profile_value,
                confidence = excluded.confidence,
                last_updated = excluded.last_updated,
                source = excluded.source,
                candidate_id = excluded.candidate_id
        """, (
            row["conv_id"],
            row["profile_key"],
            row["profile_value"],
            row["confidence"],
            row["created_at"],
            row["candidate_id"],
            row["category_scope"],
        ))
        if row["profile_key"] == "product_category":
            self._bind_unscoped(conn, row["conv_id"], row["profile_value"])
        return list(unique_conflicts.values())

    @staticmethod
    def _active_category(conn: sqlite3.Connection, conv_id: str) -> str:
        row = conn.execute("SELECT profile_value FROM user_profiles WHERE conv_id = ? "
                           "AND profile_key = 'product_category' AND category_scope = ''", (conv_id,)).fetchone()
        return row[0] if row else ""

    @staticmethod
    def _bind_unscoped(conn: sqlite3.Connection, conv_id: str, category: str) -> None:
        """Bind requirements given before the user has chosen a category."""
        rows = conn.execute("SELECT * FROM memory_candidates WHERE conv_id = ? AND category_scope = '' "
                            "AND profile_key != 'product_category' AND status = 'confirmed'", (conv_id,)).fetchall()
        for row in rows:
            conn.execute("UPDATE memory_candidates SET status = 'expired' WHERE conv_id = ? AND category_scope = ? "
                         "AND profile_key = ? AND status = 'confirmed'", (conv_id, category, row["profile_key"]))
            conn.execute("DELETE FROM user_profiles WHERE conv_id = ? AND category_scope = ? AND profile_key = ?",
                         (conv_id, category, row["profile_key"]))
        conn.execute("UPDATE user_profiles SET category_scope = ? WHERE conv_id = ? AND category_scope = '' "
                     "AND profile_key != 'product_category'", (category, conv_id))
        conn.execute("UPDATE memory_candidates SET category_scope = ? WHERE conv_id = ? AND category_scope = '' "
                     "AND profile_key != 'product_category' AND status IN ('pending', 'confirmed')", (category, conv_id))

    def for_category(self, conv_id: str, category: str):
        """Return a turn-local view; never mutate a shared store's active context."""
        return _CategoryProfileView(self, conv_id, category)

    def serialize_effective(self, conv_id: str, profile: dict) -> str:
        return self.serialize_profile(conv_id, effective_profile=profile)

    def add_candidate(
        self,
        conv_id: str,
        key: str,
        value: str,
        *,
        memory_type: str,
        confidence: float,
        source_message_id: str,
        evidence_span: str,
        expires_at: str | None = None,
        category_scope: str | None = None,
    ) -> MemoryCandidate:
        """Create a candidate; explicit values become active immediately."""
        value = value.strip()
        evidence_span = evidence_span.strip()[:160]
        self._validate_candidate_input(
            key, value, memory_type, confidence, source_message_id, evidence_span
        )
        created = _now()
        if expires_at is None:
            ttl = _EXPLICIT_TTL_DAYS if memory_type == "explicit" else _INFERRED_TTL_DAYS
            expires_at = _to_iso(created + timedelta(days=ttl))
        else:
            try:
                parsed_expiry = _parse_iso(expires_at)
                if parsed_expiry <= created:
                    raise ValueError("expires_at must be in the future")
                expires_at = _to_iso(parsed_expiry)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid expires_at") from exc

        conn = self._connect()
        conn.execute("BEGIN IMMEDIATE")
        self._expire_due(conn, conv_id)
        scope = "" if key == "product_category" else (
            self._active_category(conn, conv_id) if category_scope is None else category_scope
        )
        existing = conn.execute(
            "SELECT * FROM memory_candidates WHERE conv_id = ? AND profile_key = ? "
            "AND profile_value = ? AND status IN ('pending', 'confirmed') "
            "AND category_scope = ? ORDER BY created_at DESC LIMIT 1",
            (conv_id, key, value, scope),
        ).fetchone()
        if existing:
            if memory_type == "explicit":
                conn.execute("""
                    UPDATE memory_candidates SET
                        memory_type = 'explicit', confidence = ?, source_message_id = ?,
                        evidence_span = ?, created_at = ?, expires_at = ?
                    WHERE candidate_id = ?
                """, (
                    float(confidence),
                    source_message_id,
                    evidence_span,
                    _to_iso(created),
                    expires_at,
                    existing["candidate_id"],
                ))
                promoted = self._candidate_by_id(conn, conv_id, existing["candidate_id"])
                conflicts = self._activate_candidate(conn, promoted)
                conn.commit()
                result = self._candidate_by_id(conn, conv_id, existing["candidate_id"])
                conn.close()
                log(
                    "profile_memory",
                    action=(
                        "promote_explicit"
                        if existing["status"] == "pending"
                        else "refresh_explicit"
                    ),
                    candidate_id=existing["candidate_id"],
                    key=key,
                    conflict_count=len(conflicts),
                    conversation_hash=hash_identifier(conv_id),
                )
                return self._row_to_candidate(result)
            conn.commit()
            conn.close()
            return self._row_to_candidate(existing)

        candidate_id = f"mem_{uuid.uuid4().hex[:20]}"
        conn.execute("""
            INSERT INTO memory_candidates (
                candidate_id, conv_id, profile_key, profile_value, memory_type,
                confidence, source_message_id, evidence_span, created_at,
                expires_at, status, schema_version, category_scope
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
        """, (
            candidate_id,
            conv_id,
            key,
            value,
            memory_type,
            float(confidence),
            source_message_id,
            evidence_span,
            _to_iso(created),
            expires_at,
            MEMORY_SCHEMA_VERSION,
            scope,
        ))
        row = self._candidate_by_id(conn, conv_id, candidate_id)
        conflicts: list[sqlite3.Row] = []
        if memory_type == "explicit":
            conflicts = self._activate_candidate(conn, row)
        conn.commit()
        result = self._candidate_by_id(conn, conv_id, candidate_id)
        conn.close()

        log(
            "profile_memory",
            action="create",
            candidate_id=candidate_id,
            key=key,
            memory_type=memory_type,
            status=result["status"],
            conversation_hash=hash_identifier(conv_id),
        )
        for conflict in conflicts:
            log(
                "profile_memory",
                action="conflict_resolved",
                candidate_id=candidate_id,
                conflict_candidate_id=conflict["candidate_id"],
                key=key,
                conflict_key=conflict["profile_key"],
                conversation_hash=hash_identifier(conv_id),
            )
        return self._row_to_candidate(result)

    def update(
        self,
        conv_id: str,
        key: str,
        value: str,
        confidence: float = 1.0,
        source: str = "explicit",
        source_message_id: str = "compatibility_api",
        evidence_span: str = "",
    ) -> MemoryCandidate:
        """Backward-compatible writes still pass through candidate governance."""
        memory_type = "explicit" if source == "explicit" else "inferred"
        return self.add_candidate(
            conv_id,
            key,
            value,
            memory_type=memory_type,
            confidence=confidence,
            source_message_id=source_message_id,
            evidence_span=evidence_span,
        )

    def confirm_candidate(self, conv_id: str, candidate_id: str) -> MemoryCandidate:
        conn = self._connect()
        conn.execute("BEGIN IMMEDIATE")
        self._expire_due(conn, conv_id)
        row = self._candidate_by_id(conn, conv_id, candidate_id)
        if row is None:
            conn.rollback()
            conn.close()
            raise KeyError("memory candidate not found")
        if row["status"] != "pending":
            conn.rollback()
            conn.close()
            raise ValueError("memory candidate is not pending")
        conn.execute(
            "UPDATE memory_candidates SET expires_at = ? WHERE candidate_id = ?",
            (_to_iso(_now() + timedelta(days=_EXPLICIT_TTL_DAYS)), candidate_id),
        )
        row = self._candidate_by_id(conn, conv_id, candidate_id)
        conflicts = self._activate_candidate(conn, row)
        conn.commit()
        result = self._candidate_by_id(conn, conv_id, candidate_id)
        conn.close()
        log(
            "profile_memory",
            action="confirm",
            candidate_id=candidate_id,
            key=row["profile_key"],
            conflict_count=len(conflicts),
            conversation_hash=hash_identifier(conv_id),
        )
        return self._row_to_candidate(result)

    def reject_candidate(self, conv_id: str, candidate_id: str) -> MemoryCandidate:
        conn = self._connect()
        conn.execute("BEGIN IMMEDIATE")
        self._expire_due(conn, conv_id)
        row = self._candidate_by_id(conn, conv_id, candidate_id)
        if row is None:
            conn.rollback()
            conn.close()
            raise KeyError("memory candidate not found")
        if row["status"] != "pending":
            conn.rollback()
            conn.close()
            raise ValueError("memory candidate is not pending")
        conn.execute(
            "UPDATE memory_candidates SET status = 'rejected' WHERE candidate_id = ?",
            (candidate_id,),
        )
        conn.commit()
        result = self._candidate_by_id(conn, conv_id, candidate_id)
        conn.close()
        log(
            "profile_memory",
            action="reject",
            candidate_id=candidate_id,
            key=row["profile_key"],
            conversation_hash=hash_identifier(conv_id),
        )
        return self._row_to_candidate(result)

    def delete_profile_key(self, conv_id: str, key: str, *, category_scope: str | None = None) -> int:
        if key not in PROFILE_KEYS:
            raise ValueError("unsupported profile key")
        conn = self._connect()
        conn.execute("BEGIN IMMEDIATE")
        scope = "" if key == "product_category" else (self._active_category(conn, conv_id) if category_scope is None else category_scope)
        cursor = conn.execute(
            "UPDATE memory_candidates SET status = 'deleted' "
            "WHERE conv_id = ? AND profile_key = ? AND category_scope = ? AND status IN ('pending', 'confirmed')",
            (conv_id, key, scope),
        )
        conn.execute(
            "DELETE FROM user_profiles WHERE conv_id = ? AND profile_key = ? AND category_scope = ?",
            (conv_id, key, scope),
        )
        count = cursor.rowcount
        conn.commit()
        conn.close()
        log(
            "profile_memory",
            action="delete",
            key=key,
            count=count,
            conversation_hash=hash_identifier(conv_id),
        )
        return count

    def delete_candidate(self, conv_id: str, candidate_id: str) -> MemoryCandidate:
        conn = self._connect()
        conn.execute("BEGIN IMMEDIATE")
        row = self._candidate_by_id(conn, conv_id, candidate_id)
        if row is None:
            conn.rollback()
            conn.close()
            raise KeyError("memory candidate not found")
        if row["status"] not in {"pending", "confirmed"}:
            conn.rollback()
            conn.close()
            raise ValueError("memory candidate is not active")
        conn.execute(
            "UPDATE memory_candidates SET status = 'deleted' WHERE candidate_id = ?",
            (candidate_id,),
        )
        if row["status"] == "confirmed":
            conn.execute(
                "DELETE FROM user_profiles WHERE conv_id = ? AND candidate_id = ?",
                (conv_id, candidate_id),
            )
        conn.commit()
        result = self._candidate_by_id(conn, conv_id, candidate_id)
        conn.close()
        log(
            "profile_memory",
            action="delete_candidate",
            candidate_id=candidate_id,
            key=row["profile_key"],
            conversation_hash=hash_identifier(conv_id),
        )
        return self._row_to_candidate(result)

    def list_candidates(
        self,
        conv_id: str,
        statuses: tuple[str, ...] | None = None,
    ) -> list[MemoryCandidate]:
        if statuses is not None and (
            not statuses or any(status not in _VALID_STATUSES for status in statuses)
        ):
            raise ValueError("invalid memory status filter")
        conn = self._connect()
        expired = self._expire_due(conn, conv_id)
        if statuses is None:
            rows = conn.execute(
                "SELECT * FROM memory_candidates WHERE conv_id = ? ORDER BY created_at DESC",
                (conv_id,),
            ).fetchall()
        else:
            placeholders = ",".join("?" for _ in statuses)
            rows = conn.execute(
                f"SELECT * FROM memory_candidates WHERE conv_id = ? "
                f"AND status IN ({placeholders}) ORDER BY created_at DESC",
                (conv_id, *statuses),
            ).fetchall()
        conn.commit()
        conn.close()
        for item in expired:
            log(
                "profile_memory",
                action="expire",
                candidate_id=item["candidate_id"],
                key=item["profile_key"],
                conversation_hash=hash_identifier(conv_id),
            )
        return [self._row_to_candidate(row) for row in rows]

    def get_structured(self, conv_id: str, *, category_scope: str | None = None) -> dict:
        """Return only confirmed, unexpired values eligible for hard filters."""
        candidates = self.list_candidates(conv_id, statuses=("confirmed",))
        category = next((c.value for c in candidates if c.key == "product_category"), "") if category_scope is None else category_scope
        profile = {}
        for candidate in candidates:
            if candidate.key != "product_category" and candidate.category_scope != category:
                continue
            if candidate.key in profile:
                continue
            effective = candidate.confidence * math.exp(
                -self.decay_lambda * _days_since(candidate.created_at)
            )
            if effective >= 0.15:
                profile[candidate.key] = {
                    "value": candidate.value,
                    "confidence": round(effective, 3),
                    "source": candidate.type,
                    "candidate_id": candidate.candidate_id,
                    "status": candidate.status,
                    "expires_at": candidate.expires_at,
                }
        if category_scope is not None:
            profile["product_category"] = {"value": category_scope, "confidence": 1.0, "source": "session"}
        return profile

    def clear_conv(self, conv_id: str) -> None:
        """Physically remove state when the containing conversation is deleted."""
        conn = self._connect()
        conn.execute("DELETE FROM user_profiles WHERE conv_id = ?", (conv_id,))
        conn.execute("DELETE FROM memory_candidates WHERE conv_id = ?", (conv_id,))
        conn.commit()
        conn.close()
        try:
            self.memory_col.delete(where={"conv_id": conv_id})
        except Exception:
            pass

    # Semantic memory stays reference-only and never feeds hard filters.
    def add_memory(
        self,
        conv_id: str,
        utterance: str,
        topic: str = "",
        metadata: Optional[dict] = None,
    ) -> None:
        embedding = self.model.encode(utterance).tolist()
        ts = _now_iso()
        meta = {"conv_id": conv_id, "topic": topic, "timestamp": ts}
        if metadata:
            meta.update(metadata)
        mem_id = f"mem_{conv_id}_{int(time.time() * 1000)}"
        self.memory_col.add(
            ids=[mem_id], documents=[utterance], embeddings=[embedding], metadatas=[meta]
        )

    def search_semantic(self, conv_id: str, query: str, top_k: int = 5) -> list[str]:
        embedding = self.model.encode(query).tolist()
        try:
            results = self.memory_col.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where={"conv_id": conv_id},
                include=["documents", "distances"],
            )
        except Exception:
            return []
        if not results["ids"] or not results["ids"][0]:
            return []
        memories = []
        for index, document in enumerate(results["documents"][0]):
            distance = results["distances"][0][index] if results["distances"] else 0
            memories.append(f"[dist={distance:.3f}] {document}")
        return memories

    def serialize_profile(self, conv_id: str, *, category_scope: str | None = None, effective_profile: dict | None = None) -> str:
        """Show confirmed constraints and pending candidates separately."""
        profile = self.get_structured(conv_id, category_scope=category_scope) if effective_profile is None else effective_profile
        pending = self.list_candidates(conv_id, statuses=("pending",))
        category = profile.get("product_category", {}).get("value", "")
        pending = [c for c in pending if c.key == "product_category" or c.category_scope == category]
        lines = []
        if profile:
            lines.append("## 本轮有效需求" if effective_profile is not None else "## 已确认画像")
            for key, info in profile.items():
                conf_pct = int(info["confidence"] * 100)
                lines.append(f"- {key}: {info['value']} (置信度 {conf_pct}%)")
        else:
            lines.append("(暂无已确认画像)")
        if pending:
            lines.append("## 待确认画像候选（不得作为硬约束）")
            for candidate in pending:
                evidence = f"，依据：{candidate.evidence_span}" if candidate.evidence_span else ""
                lines.append(
                    f"- [{candidate.candidate_id}] {candidate.key}: {candidate.value}{evidence}"
                )
        return "\n".join(lines)

    def serialize_memories(self, conv_id: str, query: str) -> str:
        memories = self.search_semantic(conv_id, query, top_k=3)
        if not memories:
            return ""
        return "## 用户历史偏好记忆（仅供参考）\n" + "\n".join(
            f"- {memory}" for memory in memories
        )


class _CategoryProfileView:
    """A per-turn category binding, safe across concurrent conversations."""

    def __init__(self, store: ProfileStore, conv_id: str, category: str):
        self.store, self.conv_id, self.category = store, conv_id, category

    def _check(self, conv_id: str) -> None:
        if conv_id != self.conv_id:
            raise ValueError("category view belongs to another conversation")

    def get_structured(self, conv_id: str) -> dict:
        self._check(conv_id)
        return self.store.get_structured(conv_id, category_scope=self.category)

    def serialize_profile(self, conv_id: str) -> str:
        self._check(conv_id)
        return self.store.serialize_profile(conv_id, category_scope=self.category)

    def serialize_effective(self, conv_id: str, profile: dict) -> str:
        self._check(conv_id)
        return self.store.serialize_profile(conv_id, category_scope=self.category, effective_profile=profile)

    def add_candidate(self, conv_id: str, key: str, value: str, **kwargs):
        self._check(conv_id)
        return self.store.add_candidate(conv_id, key, value, category_scope=self.category, **kwargs)

    def delete_profile_key(self, conv_id: str, key: str) -> int:
        self._check(conv_id)
        return self.store.delete_profile_key(conv_id, key, category_scope=self.category)

    def confirm_candidate(self, conv_id: str, candidate_id: str):
        self._check(conv_id)
        return self.store.confirm_candidate(conv_id, candidate_id)

    def reject_candidate(self, conv_id: str, candidate_id: str):
        self._check(conv_id)
        return self.store.reject_candidate(conv_id, candidate_id)

    def delete_candidate(self, conv_id: str, candidate_id: str):
        self._check(conv_id)
        return self.store.delete_candidate(conv_id, candidate_id)
