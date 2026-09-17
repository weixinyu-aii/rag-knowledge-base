"""SQLite-backed conversation history."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path


class SQLiteSessionStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session "
                "ON messages(session_id, id)"
            )

    def get(self, session_id: str, limit: int = 10) -> list[dict[str, str]]:
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                """
                SELECT role, content FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]

    def append_many(self, session_id: str, messages: list[tuple[str, str]]) -> None:
        if not messages:
            return
        with self._lock, self._connection() as connection:
            connection.executemany(
                "INSERT INTO messages(session_id, role, content) VALUES (?, ?, ?)",
                [(session_id, role, content) for role, content in messages],
            )

    def clear(self, session_id: str) -> None:
        with self._lock, self._connection() as connection:
            connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
