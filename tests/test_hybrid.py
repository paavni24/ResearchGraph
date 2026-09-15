from pathlib import Path

from research_rag.config import Settings
from research_rag.hybrid import HybridRAGService
from research_rag.knowledge_graph.retrieval import (
    GraphRetrievalResult,
    RetrievedGraphChunk,
)
from research_rag.models import RerankResult, RerankScore, TokenUsage


class FakeHybridAI:
    def embed_with_usage(self, texts):  # type: ignore[no-untyped-def]
        return [[1.0, 0.0]], TokenUsage(input_tokens=5, total_tokens=5)

    def rerank(self, question, candidates, model):  # type: ignore[no-untyped-def]
        return (
            RerankResult(
                scores=[
                    RerankScore(candidate_id="c0", relevance_score=70),
                    RerankScore(candidate_id="c1", relevance_score=95),
                ]
            ),
            TokenUsage(input_tokens=50, output_tokens=10, total_tokens=60),
        )

    def answer_with_usage(self, question, context, model):  # type: ignore[no-untyped-def]
        return "The Transformer uses attention [1].", TokenUsage(
            input_tokens=40, output_tokens=10, total_tokens=50
        )


class FakeVectorStore:
    def count(self) -> int:
        return 1

    def all_chunks(self):  # type: ignore[no-untyped-def]
        raise AssertionError("Already synchronized")

    def search(self, embedding, top_k):  # type: ignore[no-untyped-def]
        return [
            {
                "id": "shared",
                "text": "Vector and lexical candidate.",
                "metadata": {"source": "paper.pdf", "page": 1, "chunk_index": 0},
                "distance": 0.1,
            }
        ]


class FakeFullTextStore:
    def count(self) -> int:
        return 1

    def replace_all(self, rows):  # type: ignore[no-untyped-def]
        raise AssertionError("Already synchronized")

    def search(self, question, top_k):  # type: ignore[no-untyped-def]
        return [
            {
                "id": "shared",
                "text": "Vector and lexical candidate.",
                "metadata": {"source": "paper.pdf", "page": 1, "chunk_index": 0},
                "rank": -1.0,
            }
        ]


class FakeGraphService:
    def retrieve(self, question, top_k, depth):  # type: ignore[no-untyped-def]
        return GraphRetrievalResult(
            question=question,
            extracted_entities=[],
            matched_entities=[],
            chunks=[
                RetrievedGraphChunk(
                    id="graph",
                    source="paper.pdf",
                    page=2,
                    chunk_index=1,
                    text="The Transformer uses multi-head attention.",
                    score=2.0,
                    reason="relationship_evidence",
                    evidence="uses multi-head attention",
                    seed_entity="Transformer",
                    path_nodes=["Transformer", "Attention Is All You Need"],
                    path_relationships=["PROPOSES"],
                )
            ],
            usage={
                "input_tokens": 10,
                "cached_input_tokens": 0,
                "output_tokens": 5,
                "reasoning_tokens": 0,
                "total_tokens": 15,
                "cost_usd": 0.000013,
            },
        )

    def close(self) -> None:
        pass


def test_hybrid_fuses_reranks_and_answers(tmp_path: Path) -> None:
    settings = Settings(
        OPENAI_API_KEY="test",
        KG_STATE_PATH=tmp_path / "usage.sqlite3",
        HYBRID_CANDIDATE_LIMIT=10,
    )
    service = HybridRAGService(
        settings,
        ai_client=FakeHybridAI(),  # type: ignore[arg-type]
        vector_store=FakeVectorStore(),  # type: ignore[arg-type]
        fulltext_store=FakeFullTextStore(),  # type: ignore[arg-type]
        graph_service=FakeGraphService(),  # type: ignore[arg-type]
    )

    result = service.ask("How does the Transformer work?", top_k=2)

    assert result.answer.endswith("[1].")
    assert result.retriever_counts["fused_unique"] == 2
    assert result.sources[0].id == "graph"
    assert result.sources[1].retrieval_methods == ["fulltext", "vector"]
    assert result.usage["total"]["total_tokens"] == 130
