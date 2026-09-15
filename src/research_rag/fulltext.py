from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

from research_rag.models import Chunk

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "what", "when", "where", "which", "who", "why", "with",
}


class FullTextStore:
    """Persistent SQLite FTS5 index over the vector-RAG chunks."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    id UNINDEXED, source UNINDEXED, document_id UNINDEXED,
                    page UNINDEXED, chunk_index UNINDEXED, text,
                    tokenize='porter unicode61'
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def replace_source(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        rows = [
            (
                chunk.id, chunk.source, chunk.document_id, chunk.page,
                chunk.chunk_index, chunk.text,
            )
            for chunk in chunks
        ]
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM chunks_fts WHERE source = ?", (chunks[0].source,))
            connection.executemany(
                "INSERT INTO chunks_fts(id, source, document_id, page, chunk_index, text) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )

    def replace_all(self, rows: list[dict[str, Any]]) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM chunks_fts")
            connection.executemany(
                "INSERT INTO chunks_fts(id, source, document_id, page, chunk_index, text) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        row["id"], row["metadata"]["source"],
                        row["metadata"].get("document_id", ""), row["metadata"].get("page"),
                        row["metadata"]["chunk_index"], row["text"],
                    )
                    for row in rows
                ],
            )

    def search(self, question: str, top_k: int) -> list[dict[str, Any]]:
        terms = [
            term.casefold()
            for term in re.findall(r"[^\W_]+", question, flags=re.UNICODE)
            if len(term) >= 2 and term.casefold() not in STOP_WORDS
        ]
        if not terms:
            return []
        match_query = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, source, document_id, page, chunk_index, text,
                    bm25(chunks_fts) AS rank
                FROM chunks_fts WHERE chunks_fts MATCH ?
                ORDER BY rank LIMIT ?
                """,
                (match_query, top_k),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "text": row["text"],
                "metadata": {
                    "source": row["source"],
                    "document_id": row["document_id"],
                    "page": row["page"],
                    "chunk_index": row["chunk_index"],
                },
                "rank": float(row["rank"]),
            }
            for row in rows
        ]

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT count(*) FROM chunks_fts").fetchone()[0])
