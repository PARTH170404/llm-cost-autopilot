"""
app/db/database.py
SQLite database manager — schema creation and request logging.
"""

import sqlite3
import logging
import uuid
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from app.utils.config import get_setting

logger = logging.getLogger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS requests (
    id              TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    prompt          TEXT NOT NULL,
    prompt_len      INTEGER,
    complexity_tier TEXT,
    confidence      REAL,
    model_used      TEXT,
    forced_model    TEXT,
    content         TEXT,
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    total_tokens    INTEGER,
    latency_ms      REAL,
    cost            REAL,
    estimated_savings REAL,
    quality_score   REAL,
    escalated       INTEGER DEFAULT 0,
    error           TEXT
);

CREATE TABLE IF NOT EXISTS escalations (
    id              TEXT PRIMARY KEY,
    request_id      TEXT NOT NULL,
    from_model      TEXT,
    to_model        TEXT,
    reason          TEXT,
    timestamp       TEXT NOT NULL,
    FOREIGN KEY (request_id) REFERENCES requests(id)
);
"""


class Database:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or get_setting("database.path", "data/autopilot.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript(DDL)
        logger.info(f"Database ready at {self.db_path}")

    # ── Write ─────────────────────────────────────────────────────────────────

    def log_request(
        self,
        prompt: str,
        complexity_tier: str,
        confidence: float,
        model_used: str,
        content: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        latency_ms: float,
        cost: float,
        estimated_savings: float,
        quality_score: float | None = None,
        escalated: bool = False,
        forced_model: str | None = None,
        error: str | None = None,
    ) -> str:
        req_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO requests VALUES (
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )
                """,
                (
                    req_id,
                    datetime.utcnow().isoformat(),
                    prompt,
                    len(prompt),
                    complexity_tier,
                    confidence,
                    model_used,
                    forced_model,
                    content,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    latency_ms,
                    cost,
                    estimated_savings,
                    quality_score,
                    int(escalated),
                    error,
                ),
            )
        return req_id

    def log_escalation(self, request_id: str, from_model: str, to_model: str, reason: str):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO escalations VALUES (?,?,?,?,?,?)",
                (
                    str(uuid.uuid4()),
                    request_id,
                    from_model,
                    to_model,
                    reason,
                    datetime.utcnow().isoformat(),
                ),
            )

    def update_quality_score(self, request_id: str, score: float):
        with self._conn() as conn:
            conn.execute(
                "UPDATE requests SET quality_score=? WHERE id=?",
                (score, request_id),
            )

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*)                        AS total_requests,
                    COALESCE(SUM(total_tokens), 0)  AS total_tokens,
                    COALESCE(SUM(cost), 0)          AS total_cost,
                    COALESCE(SUM(estimated_savings), 0) AS total_savings,
                    COALESCE(AVG(latency_ms), 0)    AS avg_latency_ms,
                    COALESCE(AVG(escalated), 0)     AS escalation_rate
                FROM requests WHERE error IS NULL
                """
            ).fetchone()

            dist = conn.execute(
                """
                SELECT model_used, COUNT(*) as cnt
                FROM requests WHERE error IS NULL
                GROUP BY model_used
                """
            ).fetchall()

        return {
            "total_requests": row["total_requests"],
            "total_tokens": row["total_tokens"],
            "total_cost": row["total_cost"],
            "total_savings": row["total_savings"],
            "avg_latency_ms": row["avg_latency_ms"],
            "escalation_rate": row["escalation_rate"],
            "model_distribution": {r["model_used"]: r["cnt"] for r in dist},
        }

    def get_recent_requests(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM requests ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_latency_trend(self, limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, model_used, latency_ms, complexity_tier
                FROM requests WHERE error IS NULL
                ORDER BY timestamp DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
