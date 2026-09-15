from pathlib import Path

from research_rag.config import Settings
from research_rag.service import RAGService


class FakeAI:
    def __init__(self) -> None:
        self.context = ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def answer(self, question: str, context: str) -> str:
        self.context = context
        return "The method improves retrieval [1]."


class FakeStore:
    def search(self, query_embedding: list[float], top_k: int) -> list[dict]:
        return [
            {
                "text": "The proposed method improves retrieval accuracy.",
                "metadata": {"source": "paper.pdf", "page": 4, "chunk_index": 2},
                "distance": 0.12,
            }
        ]


class FakeFullTextStore:
    def replace_source(self, chunks):  # type: ignore[no-untyped-def]
        pass


def test_ask_builds_numbered_context_and_sources() -> None:
    ai = FakeAI()
    service = RAGService(
        Settings(OPENAI_API_KEY="test", RAG_CHROMA_PATH=Path("unused")),
        ai_client=ai,  # type: ignore[arg-type]
        store=FakeStore(),  # type: ignore[arg-type]
        fulltext_store=FakeFullTextStore(),  # type: ignore[arg-type]
    )

    result = service.ask("What improves?")

    assert result.answer.endswith("[1].")
    assert result.sources[0].page == 4
    assert "[1] paper.pdf, page 4" in ai.context
