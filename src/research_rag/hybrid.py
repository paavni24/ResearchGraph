from __future__ import annotations

import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from research_rag.config import Settings
from research_rag.fulltext import FullTextStore
from research_rag.knowledge_graph.checkpoint import CheckpointStore
from research_rag.knowledge_graph.retrieval import GraphRetrievalService
from research_rag.models import HybridAnswer, HybridSource, TokenUsage

if TYPE_CHECKING:
    from research_rag.openai_client import OpenAIClient
    from research_rag.store import VectorStore


@dataclass
class _Candidate:
    id: str
    text: str
    metadata: dict[str, Any]
    methods: set[str] = field(default_factory=set)
    fused_score: float = 0.0
    rerank_score: float = 0.0
    graph_path: list[str] = field(default_factory=list)


class HybridRAGService:
    def __init__(
        self,
        settings: Settings,
        ai_client: OpenAIClient | None = None,
        vector_store: VectorStore | None = None,
        fulltext_store: FullTextStore | None = None,
        graph_service: GraphRetrievalService | None = None,
        checkpoints: CheckpointStore | None = None,
    ) -> None:
        settings.require_api_key()
        self.settings = settings
        if ai_client is None:
            from research_rag.openai_client import OpenAIClient

            ai_client = OpenAIClient(
                settings.openai_api_key, settings.embedding_model, settings.hybrid_answer_model
            )
        if vector_store is None:
            from research_rag.store import VectorStore

            vector_store = VectorStore(
                settings.chroma_path, settings.collection, settings.embedding_model
            )
        self.ai = ai_client
        self.vector = vector_store
        self.fulltext = fulltext_store or FullTextStore(settings.fulltext_path)
        self.checkpoints = checkpoints or CheckpointStore(settings.kg_state_path)
        self.graph = graph_service or GraphRetrievalService(
            settings, checkpoints=self.checkpoints
        )

    def ask(self, question: str, top_k: int | None = None) -> HybridAnswer:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        top_k = top_k or self.settings.hybrid_top_k
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        request_id = uuid.uuid4().hex
        self._sync_fulltext()

        errors: dict[str, str] = {}
        usage_by_operation: dict[str, TokenUsage] = {}
        costs_by_operation: dict[str, float] = {}
        per_retriever = self.settings.hybrid_per_retriever
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="hybrid-retrieve") as executor:
            vector_future = executor.submit(self._vector_retrieve, question, per_retriever)
            fulltext_future = executor.submit(self.fulltext.search, question, per_retriever)
            graph_future = executor.submit(self.graph.retrieve, question, per_retriever, None)
            try:
                vector_hits, embedding_usage = vector_future.result()
                usage_by_operation["embedding"] = embedding_usage
                costs_by_operation["embedding"] = self._embedding_cost(embedding_usage)
                self._record_usage(
                    request_id,
                    "embedding",
                    question,
                    self.settings.embedding_model,
                    embedding_usage,
                    costs_by_operation["embedding"],
                )
            except Exception as error:  # noqa: BLE001 -- one retriever may fail independently
                vector_hits = []
                errors["vector"] = str(error)
            try:
                fulltext_hits = fulltext_future.result()
            except Exception as error:  # noqa: BLE001 -- one retriever may fail independently
                fulltext_hits = []
                errors["fulltext"] = str(error)
            try:
                graph_result = graph_future.result()
                graph_hits = [chunk.model_dump() for chunk in graph_result.chunks]
                usage_by_operation["entity_extraction"] = TokenUsage(
                    input_tokens=int(graph_result.usage["input_tokens"]),
                    cached_input_tokens=int(graph_result.usage["cached_input_tokens"]),
                    output_tokens=int(graph_result.usage["output_tokens"]),
                    reasoning_tokens=int(graph_result.usage["reasoning_tokens"]),
                    total_tokens=int(graph_result.usage["total_tokens"]),
                )
                costs_by_operation["entity_extraction"] = float(
                    graph_result.usage["cost_usd"]
                )
            except Exception as error:  # noqa: BLE001 -- vector/FTS can still answer
                graph_hits = []
                errors["graph"] = str(error)

        candidates = self._fuse(vector_hits, fulltext_hits, graph_hits)
        shortlist = sorted(
            candidates.values(), key=lambda item: item.fused_score, reverse=True
        )[: self.settings.hybrid_candidate_limit]
        retriever_counts = {
            "vector": len(vector_hits),
            "fulltext": len(fulltext_hits),
            "graph": len(graph_hits),
            "fused_unique": len(candidates),
        }
        if not shortlist:
            return HybridAnswer(
                answer="No relevant passages were found in the indexed papers.",
                sources=[],
                retriever_counts=retriever_counts,
                usage=self._usage_dict(usage_by_operation, costs_by_operation),
                errors=errors,
            )

        candidate_ids = {f"c{index}": candidate for index, candidate in enumerate(shortlist)}
        try:
            rerank_result, rerank_usage = self.ai.rerank(
                question,
                [
                    (
                        candidate_id,
                        f"Source: {candidate.metadata.get('source', 'unknown')}, "
                        + f"page {candidate.metadata.get('page', 'unknown')}\n"
                        + candidate.text[: self.settings.hybrid_rerank_chars],
                    )
                    for candidate_id, candidate in candidate_ids.items()
                ],
                self.settings.hybrid_rerank_model,
            )
            usage_by_operation["rerank"] = rerank_usage
            costs_by_operation["rerank"] = self._text_cost(rerank_usage)
            self._record_usage(
                request_id,
                "rerank",
                question,
                self.settings.hybrid_rerank_model,
                rerank_usage,
                costs_by_operation["rerank"],
            )
            for score in rerank_result.scores:
                candidate = candidate_ids.get(score.candidate_id)
                if candidate is not None:
                    candidate.rerank_score = score.relevance_score
        except Exception as error:  # noqa: BLE001 -- fused ranking remains a safe fallback
            errors["rerank"] = str(error)
            for candidate in shortlist:
                candidate.rerank_score = candidate.fused_score * 1000

        ranked = sorted(
            shortlist,
            key=lambda item: (item.rerank_score, item.fused_score),
            reverse=True,
        )[:top_k]
        context_parts: list[str] = []
        sources: list[HybridSource] = []
        for index, candidate in enumerate(ranked, 1):
            metadata = candidate.metadata
            page = metadata.get("page")
            location = str(metadata.get("source", "unknown"))
            if page is not None:
                location += f", page {page}"
            context_parts.append(f"[{index}] {location}\n{candidate.text}")
            sources.append(
                HybridSource(
                    index=index,
                    id=candidate.id,
                    source=str(metadata.get("source", "unknown")),
                    page=int(page) if page is not None else None,
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    retrieval_methods=sorted(candidate.methods),
                    fused_score=round(candidate.fused_score, 6),
                    rerank_score=round(candidate.rerank_score, 3),
                    excerpt=candidate.text[:400],
                    graph_path=candidate.graph_path,
                )
            )
        answer, answer_usage = self.ai.answer_with_usage(
            question, "\n\n".join(context_parts), self.settings.hybrid_answer_model
        )
        usage_by_operation["answer"] = answer_usage
        costs_by_operation["answer"] = self._text_cost(answer_usage)
        self._record_usage(
            request_id,
            "answer",
            question,
            self.settings.hybrid_answer_model,
            answer_usage,
            costs_by_operation["answer"],
        )
        return HybridAnswer(
            answer=answer,
            sources=sources,
            retriever_counts=retriever_counts,
            usage=self._usage_dict(usage_by_operation, costs_by_operation),
            errors=errors,
        )

    def _vector_retrieve(
        self, question: str, top_k: int
    ) -> tuple[list[dict[str, Any]], TokenUsage]:
        embeddings, usage = self.ai.embed_with_usage([question])
        return self.vector.search(embeddings[0], top_k), usage

    def _sync_fulltext(self) -> None:
        if self.fulltext.count() != self.vector.count():
            self.fulltext.replace_all(self.vector.all_chunks())

    @staticmethod
    def _fuse(
        vector_hits: list[dict[str, Any]],
        fulltext_hits: list[dict[str, Any]],
        graph_hits: list[dict[str, Any]],
    ) -> dict[str, _Candidate]:
        candidates: dict[str, _Candidate] = {}
        text_keys: dict[str, str] = {}

        def add(
            method: str,
            rows: list[dict[str, Any]],
            weight: float,
        ) -> None:
            for rank, row in enumerate(rows, 1):
                text = row["text"]
                text_key = hashlib.sha256(" ".join(text.split()).encode()).hexdigest()
                candidate_id = text_keys.get(text_key, row["id"])
                metadata = row.get("metadata") or {
                    "source": row.get("source", ""),
                    "page": row.get("page"),
                    "chunk_index": row.get("chunk_index", 0),
                }
                candidate = candidates.get(candidate_id)
                if candidate is None:
                    candidate = _Candidate(candidate_id, text, metadata)
                    candidates[candidate_id] = candidate
                    text_keys[text_key] = candidate_id
                candidate.methods.add(method)
                candidate.fused_score += weight / (60 + rank)
                if method == "graph" and not candidate.graph_path:
                    candidate.graph_path = row.get("path_nodes", [])

        add("vector", vector_hits, 1.0)
        add("fulltext", fulltext_hits, 1.0)
        add("graph", graph_hits, 1.2)
        return candidates

    def _record_usage(
        self,
        request_id: str,
        operation: str,
        question: str,
        model: str,
        usage: TokenUsage,
        cost: float,
    ) -> None:
        self.checkpoints.record_operation_usage(
            request_id, operation, question, model, usage, cost
        )

    def _embedding_cost(self, usage: TokenUsage) -> float:
        return usage.input_tokens * self.settings.hybrid_embedding_cost_per_million / 1_000_000

    def _text_cost(self, usage: TokenUsage) -> float:
        cached = min(usage.cached_input_tokens, usage.input_tokens)
        return (
            (usage.input_tokens - cached) * self.settings.hybrid_input_cost_per_million
            + cached * self.settings.hybrid_cached_input_cost_per_million
            + usage.output_tokens * self.settings.hybrid_output_cost_per_million
        ) / 1_000_000

    def _usage_dict(
        self,
        usage_by_operation: dict[str, TokenUsage],
        costs_by_operation: dict[str, float],
    ) -> dict[str, dict[str, int | float]]:
        result: dict[str, dict[str, int | float]] = {}
        total = TokenUsage()
        total_cost = 0.0
        for operation, usage in usage_by_operation.items():
            cost = costs_by_operation.get(operation, 0.0)
            result[operation] = {
                "input_tokens": usage.input_tokens,
                "cached_input_tokens": usage.cached_input_tokens,
                "output_tokens": usage.output_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
                "total_tokens": usage.total_tokens,
                "cost_usd": round(cost, 6),
            }
            total = TokenUsage(
                input_tokens=total.input_tokens + usage.input_tokens,
                cached_input_tokens=total.cached_input_tokens + usage.cached_input_tokens,
                output_tokens=total.output_tokens + usage.output_tokens,
                reasoning_tokens=total.reasoning_tokens + usage.reasoning_tokens,
                total_tokens=total.total_tokens + usage.total_tokens,
            )
            total_cost += cost
        result["total"] = {
            "input_tokens": total.input_tokens,
            "cached_input_tokens": total.cached_input_tokens,
            "output_tokens": total.output_tokens,
            "reasoning_tokens": total.reasoning_tokens,
            "total_tokens": total.total_tokens,
            "cost_usd": round(total_cost, 6),
        }
        return result

    def close(self) -> None:
        self.graph.close()
