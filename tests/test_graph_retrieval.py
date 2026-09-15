from pathlib import Path

from research_rag.config import Settings
from research_rag.knowledge_graph.extractor import TokenUsage
from research_rag.knowledge_graph.retrieval import (
    GraphRetrievalService,
    QueryExtractionResult,
    QuestionEntity,
    QuestionEntityPlan,
)


class FakeQuestionExtractor:
    def extract(self, question: str) -> QueryExtractionResult:
        return QueryExtractionResult(
            plan=QuestionEntityPlan(
                entities=[QuestionEntity(name="Transformer", type="Method")]
            ),
            usage=TokenUsage(input_tokens=25, output_tokens=10, total_tokens=35),
            attempts=1,
        )


class FakeRetrievalStore:
    def __init__(self) -> None:
        self.mentions: list[dict] = []
        self.seeds: list[dict] = []

    def verify_connectivity(self) -> None:
        pass

    def lookup_entities(self, mentions: list[dict], per_mention: int) -> list[dict]:
        self.mentions = mentions
        return [
            {
                "id": "method-1",
                "type": "Method",
                "name": "Transformer",
                "description": "Attention-based architecture",
                "query_entity": "Transformer",
                "match_score": 1.0,
            }
        ]

    def retrieve_graph_chunks(
        self, seeds: list[dict], depth: int, top_k: int
    ) -> list[dict]:
        self.seeds = seeds
        return [
            {
                "id": "chunk-1",
                "source": "attention.pdf",
                "page": 2,
                "chunk_index": 1,
                "text": "We propose the Transformer.",
                "score": 2.05,
                "reason": "relationship_evidence",
                "evidence": "We propose the Transformer.",
                "seed_entity": "Transformer",
                "path_nodes": ["Transformer", "Attention Is All You Need"],
                "path_relationships": ["PROPOSES"],
            }
        ]

    def close(self) -> None:
        pass


def test_graph_retrieval_extracts_looks_up_and_traverses(tmp_path: Path) -> None:
    settings = Settings(
        OPENAI_API_KEY="test",
        KG_STATE_PATH=tmp_path / "state.sqlite3",
    )
    store = FakeRetrievalStore()
    service = GraphRetrievalService(
        settings,
        extractor=FakeQuestionExtractor(),  # type: ignore[arg-type]
        store=store,  # type: ignore[arg-type]
    )

    result = service.retrieve("What does the Transformer propose?", top_k=4, depth=2)

    assert store.mentions[0]["keys"] == ["transformer"]
    assert store.seeds[0]["id"] == "method-1"
    assert result.matched_entities[0].type == "Method"
    assert result.chunks[0].reason == "relationship_evidence"
    assert result.chunks[0].path_relationships == ["PROPOSES"]
    assert result.usage["total_tokens"] == 35
    assert service.checkpoints.usage_summary()["recent_queries"][0]["question"].startswith(
        "What does"
    )
