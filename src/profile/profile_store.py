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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_id TEXT NOT NULL,
                profile_key TEXT NOT NULL,
                profile_value TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                last_updated TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'explicit',
                UNIQUE(conv_id, profile_key)
            )
        """)
        profile_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(user_profiles)")
        }
        if "candidate_id" not in profile_columns:
            conn.execute("ALTER TABLE user_profiles ADD COLUMN candidate_id TEXT")
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
                schema_version TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_candidates_conv_status
            ON memory_candidates(conv_id, status, created_at)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_candidates_conv_key
            ON memory_candidates(conv_id, profile_key, created_at)
        """)
        self._migrate_legacy_profiles(conn)
        conn.commit()
        conn.close()

    @staticmethod
    def _migrate_legacy_profiles(conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            "SELECT id, conv_id, profile_key, profile_value, confidence, "
            "last_updated, source, candidate_id FROM user_profiles"
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
                    expires_at, status, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, '', ?, ?, ?, ?)
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
                    "DELETE FROM user_profiles WHERE conv_id = ? AND profile_key = ?",
                    (conv_id, row["profile_key"]),
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
            "AND candidate_id != ? AND profile_key = ?",
            (row["conv_id"], row["candidate_id"], row["profile_key"]),
        ).fetchall()
        for conflict_key in self._conflicting_keys(row["profile_key"]):
            opposite = conn.execute(
                "SELECT * FROM memory_candidates WHERE conv_id = ? AND status = 'confirmed' "
                "AND profile_key = ?",
                (row["conv_id"], conflict_key),
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
                "DELETE FROM user_profiles WHERE conv_id = ? AND profile_key = ?",
                (row["conv_id"], conflict["profile_key"]),
            )

        conn.execute(
            "UPDATE memory_candidates SET status = 'confirmed', memory_type = 'explicit' "
            "WHERE candidate_id = ?",
            (row["candidate_id"],),
        )
        conn.execute("""
            INSERT INTO user_profiles (
                conv_id, profile_key, profile_value, confidence, last_updated,
                source, candidate_id
            ) VALUES (?, ?, ?, ?, ?, 'explicit', ?)
            ON CONFLICT(conv_id, profile_key) DO UPDATE SET
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
        ))
        return list(unique_conflicts.values())

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
        existing = conn.execute(
            "SELECT * FROM memory_candidates WHERE conv_id = ? AND profile_key = ? "
            "AND profile_value = ? AND status IN ('pending', 'confirmed') "
            "ORDER BY created_at DESC LIMIT 1",
            (conv_id, key, value),
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
                expires_at, status, schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
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

    def delete_profile_key(self, conv_id: str, key: str) -> int:
        if key not in PROFILE_KEYS:
            raise ValueError("unsupported profile key")
        conn = self._connect()
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "UPDATE memory_candidates SET status = 'deleted' "
            "WHERE conv_id = ? AND profile_key = ? AND status IN ('pending', 'confirmed')",
            (conv_id, key),
        )
        conn.execute(
            "DELETE FROM user_profiles WHERE conv_id = ? AND profile_key = ?",
            (conv_id, key),
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
                "DELETE FROM user_profiles WHERE conv_id = ? AND profile_key = ?",
                (conv_id, row["profile_key"]),
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

    def get_structured(self, conv_id: str) -> dict:
        """Return only confirmed, unexpired values eligible for hard filters."""
        candidates = self.list_candidates(conv_id, statuses=("confirmed",))
        profile = {}
        for candidate in candidates:
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

    def serialize_profile(self, conv_id: str) -> str:
        """Show confirmed constraints and pending candidates separately."""
        profile = self.get_structured(conv_id)
        pending = self.list_candidates(conv_id, statuses=("pending",))
        lines = []
        if profile:
            lines.append("## 已确认画像")
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
