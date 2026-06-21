from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class RunDatabase:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inference_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    input_path TEXT,
                    model_version TEXT,
                    num_detections INTEGER,
                    avg_confidence REAL,
                    latency_ms REAL,
                    status TEXT,
                    error_message TEXT
                )
            """)
            conn.commit()

    def log_run(
        self,
        input_path: str,
        model_version: str,
        num_detections: int,
        avg_confidence: float,
        latency_ms: float,
        status: str = "ok",
        error_message: str | None = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO inference_runs
                   (timestamp, input_path, model_version, num_detections, avg_confidence, latency_ms, status, error_message)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    input_path, model_version, num_detections,
                    avg_confidence, latency_ms, status, error_message,
                ),
            )
            conn.commit()
            return cur.lastrowid

    def get_recent(self, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM inference_runs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) as total_runs,
                    AVG(latency_ms) as avg_latency_ms,
                    SUM(CASE WHEN status != 'ok' THEN 1 ELSE 0 END) as errors,
                    AVG(num_detections) as avg_detections
                FROM inference_runs
            """).fetchone()
        return dict(row) if row else {}