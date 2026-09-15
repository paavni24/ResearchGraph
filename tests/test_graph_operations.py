from pathlib import Path

from research_rag.config import Settings
from research_rag.knowledge_graph.checkpoint import CheckpointStore
from research_rag.knowledge_graph.extractor import ExtractionResult, TokenUsage
from research_rag.knowledge_graph.schema import EntityCandidate, GraphExtraction
from research_rag.knowledge_graph.service import KnowledgeGraphService


class FakeExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def extract(self, chunk, source):  # type: ignore[no-untyped-def]
        self.calls += 1
        return ExtractionResult(
            GraphExtraction(paper_title="Test", entities=[], relationships=[]),
            TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120),
            attempts=2,
        )


class FakeGraphStore:
    def __init__(self) -> None:
        self.chunk_ids: set[str] = set()
        self.replacements = 0

    def verify_connectivity(self) -> None:
        pass

    def ensure_schema(self) -> None:
        pass

    def upsert_document(self, document, title=None):  # type: ignore[no-untyped-def]
        pass

    def existing_chunk_ids(self, document_id: str) -> set[str]:
        return set(self.chunk_ids)

    def replace_document(self, document_id: str) -> None:
        self.replacements += 1
        self.chunk_ids.clear()

    def write_chunk(self, document, chunk, extraction, paper_title):  # type: ignore[no-untyped-def]
        self.chunk_ids.add(chunk.id)

    def close(self) -> None:
        pass


def test_resume_skips_committed_chunk_and_replace_rebuilds(tmp_path: Path) -> None:
    paper = tmp_path / "paper.txt"
    paper.write_text("A short research paper.")
    settings = Settings(
        OPENAI_API_KEY="test",
        KG_STATE_PATH=tmp_path / "state.sqlite3",
        KG_WORKERS=1,
    )
    extractor = FakeExtractor()
    store = FakeGraphStore()
    service = KnowledgeGraphService(
        settings,
        extractor=extractor,  # type: ignore[arg-type]
        store=store,  # type: ignore[arg-type]
    )

    first = service.ingest([paper])
    resumed = service.ingest([paper])
    replaced = service.ingest([paper], replace=True)

    assert first["chunks"] == 1
    assert first["retries"] == 1
    assert resumed["chunks"] == 0
    assert resumed["skipped_chunks"] == 1
    assert replaced["chunks"] == 1
    assert extractor.calls == 2
    assert store.replacements == 1
    assert service.usage()["api_responses"] == 2


def test_alias_registry_deduplicates_expanded_name(tmp_path: Path) -> None:
    checkpoints = CheckpointStore(tmp_path / "state.sqlite3")
    short = EntityCandidate(
        local_id="one", type="Model", name="BERT", aliases=["Bidirectional Encoder Representations from Transformers"]
    )
    expanded = EntityCandidate(
        local_id="two",
        type="Model",
        name="Bidirectional Encoder Representations from Transformers",
    )

    assert checkpoints.resolve_entity(short, "paper-one") == checkpoints.resolve_entity(
        expanded, "paper-two"
    )
