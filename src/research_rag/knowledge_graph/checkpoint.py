from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research_rag.knowledge_graph.schema import (
    EntityCandidate,
    NodeType,
    candidate_aliases,
    entity_id,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 -- Python 3.10 support


class CheckpointStore:
    """Small local ledger for resumability, usage accounting, and entity aliases."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chunk_state (
                    chunk_id TEXT PRIMARY KEY,
                    signature TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    cost_usd REAL NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS chunk_state_document_idx
                    ON chunk_state(document_id);
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL,
                    model TEXT NOT NULL,
                    workers INTEGER NOT NULL,
                    resume INTEGER NOT NULL,
                    replace_mode INTEGER NOT NULL,
                    documents INTEGER NOT NULL DEFAULT 0,
                    completed_chunks INTEGER NOT NULL DEFAULT 0,
                    skipped_chunks INTEGER NOT NULL DEFAULT 0,
                    failed_chunks INTEGER NOT NULL DEFAULT 0,
                    retries INTEGER NOT NULL DEFAULT 0,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    cost_usd REAL NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS entity_alias (
                    node_type TEXT NOT NULL,
                    alias TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (node_type, alias)
                );
                CREATE INDEX IF NOT EXISTS entity_alias_id_idx
                    ON entity_alias(node_type, entity_id);
                CREATE TABLE IF NOT EXISTS usage_events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    model TEXT NOT NULL,
                    attempts INTEGER NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    cached_input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    reasoning_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS usage_events_run_idx ON usage_events(run_id);
                CREATE TABLE IF NOT EXISTS query_usage (
                    query_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    model TEXT NOT NULL,
                    attempts INTEGER NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    cached_input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    reasoning_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS operation_usage (
                    operation_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    question TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    cached_input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    reasoning_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def start_run(self, model: str, workers: int, resume: bool, replace: bool) -> str:
        run_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO runs "
                "(run_id, started_at, status, model, workers, resume, replace_mode) "
                "VALUES (?, ?, 'running', ?, ?, ?, ?)",
                (run_id, _now(), model, workers, int(resume), int(replace)),
            )
        return run_id

    def finish_run(
        self, run_id: str, totals: dict[str, Any], status_override: str | None = None
    ) -> None:
        status = status_override or (
            "completed_with_errors" if totals.get("failed_chunks", 0) else "completed"
        )
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE runs SET finished_at = ?, status = ?, documents = ?,
                    completed_chunks = ?, skipped_chunks = ?, failed_chunks = ?, retries = ?,
                    input_tokens = ?, cached_input_tokens = ?, output_tokens = ?,
                    reasoning_tokens = ?, total_tokens = ?, cost_usd = ?
                WHERE run_id = ?
                """,
                (
                    _now(), status, totals["documents"], totals["chunks"],
                    totals["skipped_chunks"], totals["failed_chunks"], totals["retries"],
                    totals["input_tokens"], totals["cached_input_tokens"],
                    totals["output_tokens"], totals["reasoning_tokens"],
                    totals["total_tokens"], totals["cost_usd"], run_id,
                ),
            )

    def record_usage(
        self,
        run_id: str,
        chunk_id: str,
        document_id: str,
        source: str,
        model: str,
        attempts: int,
        usage: Any,
        cost_usd: float,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO usage_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex, run_id, chunk_id, document_id, source, model, attempts,
                    usage.input_tokens, usage.cached_input_tokens, usage.output_tokens,
                    usage.reasoning_tokens, usage.total_tokens, cost_usd, _now(),
                ),
            )

    def run_usage(self, run_id: str) -> dict[str, int | float]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT coalesce(sum(input_tokens), 0) AS input_tokens,
                    coalesce(sum(cached_input_tokens), 0) AS cached_input_tokens,
                    coalesce(sum(output_tokens), 0) AS output_tokens,
                    coalesce(sum(reasoning_tokens), 0) AS reasoning_tokens,
                    coalesce(sum(total_tokens), 0) AS total_tokens,
                    coalesce(sum(cost_usd), 0) AS cost_usd
                FROM usage_events WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        return dict(row)

    def record_query_usage(
        self,
        query_id: str,
        question: str,
        model: str,
        attempts: int,
        usage: Any,
        cost_usd: float,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO query_usage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query_id, question, model, attempts, usage.input_tokens,
                    usage.cached_input_tokens, usage.output_tokens,
                    usage.reasoning_tokens, usage.total_tokens, cost_usd, _now(),
                ),
            )

    def record_operation_usage(
        self,
        request_id: str,
        operation: str,
        question: str,
        model: str,
        usage: Any,
        cost_usd: float,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO operation_usage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex, request_id, operation, question, model,
                    usage.input_tokens, usage.cached_input_tokens, usage.output_tokens,
                    usage.reasoning_tokens, usage.total_tokens, cost_usd, _now(),
                ),
            )

    def chunk_completed(self, chunk_id: str, signature: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status, signature FROM chunk_state WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
        return bool(row and row["status"] == "completed" and row["signature"] == signature)

    def mark_started(
        self, chunk_id: str, signature: str, document_id: str, source: str, model: str
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO chunk_state
                    (chunk_id, signature, document_id, source, model, status, updated_at)
                VALUES (?, ?, ?, ?, ?, 'running', ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    signature = excluded.signature, document_id = excluded.document_id,
                    source = excluded.source, model = excluded.model, status = 'running',
                    error = NULL, updated_at = excluded.updated_at, completed_at = NULL
                """,
                (chunk_id, signature, document_id, source, model, _now()),
            )

    def mark_completed(
        self,
        chunk_id: str,
        attempts: int,
        usage: Any,
        cost_usd: float,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE chunk_state SET status = 'completed', attempts = ?, error = NULL,
                    input_tokens = ?, cached_input_tokens = ?, output_tokens = ?,
                    reasoning_tokens = ?, total_tokens = ?, cost_usd = ?,
                    updated_at = ?, completed_at = ? WHERE chunk_id = ?
                """,
                (
                    attempts, usage.input_tokens, usage.cached_input_tokens,
                    usage.output_tokens, usage.reasoning_tokens, usage.total_tokens,
                    cost_usd, _now(), _now(), chunk_id,
                ),
            )

    def mark_failed(self, chunk_id: str, attempts: int, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE chunk_state SET status = 'failed', attempts = ?, error = ?, "
                "updated_at = ? WHERE chunk_id = ?",
                (attempts, error[:4000], _now(), chunk_id),
            )

    def clear_document(self, document_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM chunk_state WHERE document_id = ?", (document_id,))

    def resolve_entity(self, candidate: EntityCandidate, document_id: str) -> str:
        if candidate.type is NodeType.CLAIM:
            return entity_id(candidate, document_id)
        aliases = candidate_aliases(candidate)
        fallback = entity_id(candidate, document_id)
        if not aliases:
            return fallback

        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            placeholders = ",".join("?" for _ in aliases)
            rows = connection.execute(
                f"SELECT entity_id FROM entity_alias WHERE node_type = ? "
                f"AND alias IN ({placeholders}) ORDER BY entity_id",
                (candidate.type.value, *aliases),
            ).fetchall()
            resolved_id = rows[0]["entity_id"] if rows else fallback
            connection.executemany(
                """
                INSERT INTO entity_alias(node_type, alias, entity_id, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(node_type, alias) DO UPDATE SET
                    entity_id = excluded.entity_id, updated_at = excluded.updated_at
                """,
                [(candidate.type.value, alias, resolved_id, _now()) for alias in aliases],
            )
            connection.commit()
        return resolved_id

    def usage_summary(self) -> dict[str, Any]:
        with self._connect() as connection:
            usage = connection.execute(
                """
                SELECT count(*) AS api_responses,
                    coalesce(sum(input_tokens), 0) AS input_tokens,
                    coalesce(sum(cached_input_tokens), 0) AS cached_input_tokens,
                    coalesce(sum(output_tokens), 0) AS output_tokens,
                    coalesce(sum(reasoning_tokens), 0) AS reasoning_tokens,
                    coalesce(sum(total_tokens), 0) AS total_tokens,
                    coalesce(sum(cost_usd), 0) AS cost_usd
                FROM (
                    SELECT input_tokens, cached_input_tokens, output_tokens,
                        reasoning_tokens, total_tokens, cost_usd FROM usage_events
                    UNION ALL
                    SELECT input_tokens, cached_input_tokens, output_tokens,
                        reasoning_tokens, total_tokens, cost_usd FROM query_usage
                    UNION ALL
                    SELECT input_tokens, cached_input_tokens, output_tokens,
                        reasoning_tokens, total_tokens, cost_usd FROM operation_usage
                )
                """
            ).fetchone()
            statuses = connection.execute(
                "SELECT status, count(*) AS count FROM chunk_state GROUP BY status"
            ).fetchall()
            recent_runs = connection.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT 10"
            ).fetchall()
            recent_queries = connection.execute(
                "SELECT * FROM query_usage ORDER BY created_at DESC LIMIT 10"
            ).fetchall()
            recent_operations = connection.execute(
                "SELECT * FROM operation_usage ORDER BY created_at DESC LIMIT 20"
            ).fetchall()
        result = dict(usage)
        result["cost_usd"] = round(result["cost_usd"], 6)
        result["chunk_statuses"] = {row["status"]: row["count"] for row in statuses}
        result["recent_runs"] = [dict(row) for row in recent_runs]
        result["recent_queries"] = [dict(row) for row in recent_queries]
        result["recent_operations"] = [dict(row) for row in recent_operations]
        return result
