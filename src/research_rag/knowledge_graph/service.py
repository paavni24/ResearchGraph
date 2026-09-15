from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

from research_rag.config import Settings
from research_rag.documents import chunk_document, discover_documents, load_document
from research_rag.knowledge_graph.checkpoint import CheckpointStore
from research_rag.knowledge_graph.extractor import EXTRACTION_VERSION, ExtractionResult
from research_rag.knowledge_graph.schema import resolve_extraction
from research_rag.models import Chunk, LoadedDocument

if TYPE_CHECKING:
    from research_rag.knowledge_graph.extractor import GraphExtractor
    from research_rag.knowledge_graph.store import Neo4jGraphStore


class KnowledgeGraphService:
    def __init__(
        self,
        settings: Settings,
        extractor: GraphExtractor | None = None,
        store: Neo4jGraphStore | None = None,
        checkpoints: CheckpointStore | None = None,
    ) -> None:
        settings.require_api_key()
        self.settings = settings
        if extractor is None:
            from research_rag.knowledge_graph.extractor import GraphExtractor

            extractor = GraphExtractor(
                settings.openai_api_key,
                settings.kg_extraction_model,
                settings.kg_max_retries,
                settings.kg_retry_base_seconds,
            )
        if store is None:
            from research_rag.knowledge_graph.store import Neo4jGraphStore

            store = Neo4jGraphStore(
                settings.neo4j_uri,
                settings.neo4j_username,
                settings.neo4j_password,
                settings.neo4j_database,
            )
        self.extractor = extractor
        self.store = store
        self.checkpoints = checkpoints or CheckpointStore(settings.kg_state_path)

    def ingest(
        self,
        paths: list[Path],
        max_documents: int | None = None,
        workers: int | None = None,
        resume: bool = True,
        replace: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        workers = workers or self.settings.kg_workers
        if workers < 1 or workers > 16:
            raise ValueError("workers must be between 1 and 16")
        if replace and resume:
            resume = False

        self.store.verify_connectivity()
        self.store.ensure_schema()
        discovered = discover_documents(paths)
        if max_documents is not None:
            discovered = discovered[:max_documents]

        totals: dict[str, Any] = {
            "documents": 0,
            "chunks": 0,
            "skipped_chunks": 0,
            "failed_chunks": 0,
            "retries": 0,
            "entities": 0,
            "relationships": 0,
            "rejected": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "errors": [],
        }
        run_id = self.checkpoints.start_run(
            self.settings.kg_extraction_model, workers, resume, replace
        )
        run_completed = False
        try:
            for path in discovered:
                self._ingest_document(path, workers, resume, replace, run_id, totals, progress)
            run_completed = True
        finally:
            totals.update(self.checkpoints.run_usage(run_id))
            totals["cost_usd"] = round(float(totals["cost_usd"]), 6)
            self.checkpoints.finish_run(
                run_id, totals, status_override=None if run_completed else "interrupted"
            )
        totals["run_id"] = run_id
        return totals

    def _ingest_document(
        self,
        path: Path,
        workers: int,
        resume: bool,
        replace: bool,
        run_id: str,
        totals: dict[str, Any],
        progress: Callable[[str], None] | None,
    ) -> None:
        document = load_document(path)
        chunks = chunk_document(
            document,
            self.settings.kg_chunk_size,
            self.settings.kg_chunk_overlap,
            self.settings.kg_extraction_model,
        )
        if replace:
            self.store.replace_document(document.document_id)
            self.checkpoints.clear_document(document.document_id)
        self.store.upsert_document(document)
        existing = self.store.existing_chunk_ids(document.document_id) if resume else set()
        pending = [
            chunk
            for chunk in chunks
            if not (
                chunk.id in existing
                and self.checkpoints.chunk_completed(chunk.id, self._signature(chunk))
            )
        ]
        skipped = len(chunks) - len(pending)
        totals["documents"] += 1
        totals["skipped_chunks"] += skipped
        if progress:
            progress(
                f"{path.name}: {len(pending)} pending, {skipped} resumed, "
                f"{len(chunks)} total; workers={workers}"
            )

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="kg-extract") as executor:
            futures = {
                executor.submit(self._process_chunk, document, chunk, run_id): chunk
                for chunk in pending
            }
            for completed, future in enumerate(as_completed(futures), 1):
                chunk = futures[future]
                try:
                    result = future.result()
                except Exception as error:  # noqa: BLE001 -- isolate failure to one chunk
                    attempts = int(getattr(error, "attempts", 1))
                    self.checkpoints.mark_failed(chunk.id, attempts, str(error))
                    totals["failed_chunks"] += 1
                    totals["retries"] += max(0, attempts - 1)
                    totals["errors"].append(
                        {"source": path.name, "chunk_index": chunk.chunk_index, "error": str(error)}
                    )
                else:
                    extraction_result, resolved, _cost = result
                    totals["chunks"] += 1
                    totals["retries"] += max(0, extraction_result.attempts - 1)
                    totals["entities"] += len(resolved.nodes)
                    totals["relationships"] += len(resolved.relationships) + 1
                    totals["rejected"] += resolved.rejected_relationships
                if progress and (completed == len(pending) or completed % 10 == 0):
                    progress(f"{path.name}: {completed}/{len(pending)} pending chunks handled")

    def _process_chunk(
        self, document: LoadedDocument, chunk: Chunk, run_id: str
    ) -> tuple[ExtractionResult, Any, float]:
        signature = self._signature(chunk)
        self.checkpoints.mark_started(
            chunk.id,
            signature,
            document.document_id,
            document.path.name,
            self.settings.kg_extraction_model,
        )
        result = self.extractor.extract(chunk, document.path.name)
        cost = self._cost(result)
        self.checkpoints.record_usage(
            run_id,
            chunk.id,
            document.document_id,
            document.path.name,
            self.settings.kg_extraction_model,
            result.attempts,
            result.usage,
            cost,
        )
        resolved = resolve_extraction(
            result.extraction,
            document.document_id,
            chunk.id,
            id_resolver=self.checkpoints.resolve_entity,
        )
        self.store.write_chunk(document, chunk, resolved, result.extraction.paper_title)
        self.checkpoints.mark_completed(chunk.id, result.attempts, result.usage, cost)
        return result, resolved, cost

    def _signature(self, chunk: Chunk) -> str:
        payload = {
            "version": EXTRACTION_VERSION,
            "model": self.settings.kg_extraction_model,
            "chunk_id": chunk.id,
            "text": chunk.text,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _cost(self, result: ExtractionResult) -> float:
        usage = result.usage
        cached = min(usage.cached_input_tokens, usage.input_tokens)
        uncached = usage.input_tokens - cached
        return (
            uncached * self.settings.kg_input_cost_per_million
            + cached * self.settings.kg_cached_input_cost_per_million
            + usage.output_tokens * self.settings.kg_output_cost_per_million
        ) / 1_000_000

    def stats(self) -> dict[str, Any]:
        self.store.verify_connectivity()
        return self.store.stats()

    def usage(self) -> dict[str, Any]:
        return self.checkpoints.usage_summary()

    def close(self) -> None:
        self.store.close()
