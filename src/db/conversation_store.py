"""Persistent conversation storage using SQLite.

Stores conversations and messages so that history survives
application restarts. Supports multi-conversation switching.
"""
import json
import uuid
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional


DDL = [
    """CREATE TABLE IF NOT EXISTS conversations (
        id          TEXT PRIMARY KEY,
        title       TEXT NOT NULL DEFAULT '新对话',
        model       TEXT NOT NULL DEFAULT '',
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS messages (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        conv_id     TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        role        TEXT NOT NULL,
        content     TEXT NOT NULL DEFAULT '',
        details     TEXT DEFAULT NULL,
        created_at  TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conv_id, id)",
]


class ConversationStore:
    """Manages conversation persistence."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self):
        conn = self._get_conn()
        try:
            for ddl in DDL:
                conn.execute(ddl)
            conn.commit()
        finally:
            conn.close()

    # ---- Conversations ----

    def create_conversation(self, title: str = "新对话", model: str = "") -> str:
        conv_id = uuid.uuid4().hex[:16]
        now = datetime.now().isoformat()
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO conversations (id, title, model, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (conv_id, title, model, now, now),
            )
            conn.commit()
            return conv_id
        finally:
            conn.close()

    def list_conversations(self, limit: int = 50) -> list[dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT id, title, model, created_at, updated_at FROM conversations "
                "ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_conversation(self, conv_id: str) -> Optional[dict]:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT id, title, model, created_at, updated_at FROM conversations WHERE id = ?",
                (conv_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def update_conversation(self, conv_id: str, title: str = None, model: str = None) -> None:
        now = datetime.now().isoformat()
        conn = self._get_conn()
        try:
            if title:
                conn.execute(
                    "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                    (title, now, conv_id),
                )
            if model:
                conn.execute(
                    "UPDATE conversations SET model = ?, updated_at = ? WHERE id = ?",
                    (model, now, conv_id),
                )
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))
            conn.commit()
        finally:
            conn.close()

    def delete_conversation(self, conv_id: str) -> None:
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM messages WHERE conv_id = ?", (conv_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
            conn.commit()
        finally:
            conn.close()

    # ---- Messages ----

    def add_message(self, conv_id: str, role: str, content: str, details: dict = None) -> int:
        now = datetime.now().isoformat()
        conn = self._get_conn()
        try:
            cur = conn.execute(
                "INSERT INTO messages (conv_id, role, content, details, created_at) VALUES (?, ?, ?, ?, ?)",
                (conv_id, role, content, json.dumps(details, ensure_ascii=False) if details else None, now),
            )
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))
            # Auto-title: use first user message (trimmed)
            if role == "user":
                existing = conn.execute("SELECT COUNT(*) FROM messages WHERE conv_id = ?", (conv_id,)).fetchone()
                if existing[0] <= 1:
                    title = content[:40] + ("..." if len(content) > 40 else "")
                    conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id))
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def get_messages(self, conv_id: str) -> list[dict]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT id, conv_id, role, content, details, created_at FROM messages "
                "WHERE conv_id = ? ORDER BY id ASC",
                (conv_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def clear_messages(self, conv_id: str) -> None:
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM messages WHERE conv_id = ?", (conv_id,))
            conn.commit()
        finally:
            conn.close()
