"""Local SQLite translation history persistence for SignBridge AI."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Any

from signbridge.config import HISTORY_DB_PATH


class TranslationHistoryStore:
    """Thread-safe local SQLite store for Sign-to-Speech, Speech-to-Sign, and Text-to-Sign history."""

    def __init__(self, db_path: str | Path = HISTORY_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        try:
            with closing(self._connect()) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        input_text TEXT NOT NULL,
                        output_text TEXT NOT NULL,
                        confidence REAL,
                        success INTEGER NOT NULL
                    )
                    """
                )
                conn.commit()
        except sqlite3.DatabaseError:
            # Handle corrupted DB file safely by backing up and recreating
            if self.db_path.exists():
                backup = self.db_path.with_suffix(".corrupt.bak")
                if backup.exists():
                    backup.unlink()
                self.db_path.rename(backup)
            with closing(self._connect()) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        input_text TEXT NOT NULL,
                        output_text TEXT NOT NULL,
                        confidence REAL,
                        success INTEGER NOT NULL
                    )
                    """
                )
                conn.commit()

    def add_entry(
        self,
        direction: str,
        input_text: str,
        output_text: str,
        confidence: float | None = None,
        success: bool = True,
    ) -> int:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with closing(self._connect()) as conn:
            cur = conn.execute(
                """
                INSERT INTO history (timestamp, direction, input_text, output_text, confidence, success)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ts, direction, input_text, output_text, confidence, int(bool(success))),
            )
            conn.commit()
            return int(cur.lastrowid)

    def get_recent(self, limit: int = 100) -> list[dict[str, Any]]:
        try:
            with closing(self._connect()) as conn:
                rows = conn.execute(
                    """
                    SELECT id, timestamp, direction, input_text, output_text, confidence, success
                    FROM history
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (int(limit),),
                ).fetchall()
                return [
                    {
                        "id": int(r["id"]),
                        "timestamp": str(r["timestamp"]),
                        "direction": str(r["direction"]),
                        "input_text": str(r["input_text"]),
                        "output_text": str(r["output_text"]),
                        "confidence": float(r["confidence"]) if r["confidence"] is not None else None,
                        "success": bool(r["success"]),
                    }
                    for r in rows
                ]
        except sqlite3.DatabaseError:
            return []

    def clear_history(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute("DELETE FROM history")
            conn.commit()

    def get_summary_stats(self) -> dict[str, Any]:
        entries = self.get_recent(limit=10000)
        total = len(entries)
        by_dir: dict[str, int] = {}
        confs: list[float] = []
        success_count = 0
        for e in entries:
            by_dir[e["direction"]] = by_dir.get(e["direction"], 0) + 1
            if e["success"]:
                success_count += 1
            if e["confidence"] is not None:
                confs.append(e["confidence"])
        return {
            "total_translations": total,
            "successful_translations": success_count,
            "by_direction": by_dir,
            "mean_logged_confidence": (sum(confs) / len(confs)) if confs else None,
        }
