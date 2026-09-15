from pathlib import Path

import pytest
import tiktoken

from research_rag.config import Settings
from research_rag.documents import chunk_document
from research_rag.models import LoadedDocument, PageText


def test_chunking_preserves_overlap_and_page() -> None:
    document = LoadedDocument(
        path=Path("paper.pdf"),
        document_id="abc",
        pages=[PageText("one two three four five six seven", 3)],
    )

    chunks = chunk_document(document, size=5, overlap=2)

    assert [chunk.text.strip() for chunk in chunks] == [
        "one two three four five",
        "four five six seven",
    ]
    assert all(chunk.page == 3 for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="RAG_CHUNK_OVERLAP"):
        Settings(
            OPENAI_API_KEY="test",
            RAG_CHUNK_SIZE=100,
            RAG_CHUNK_OVERLAP=100,
        )


def test_chunking_splits_text_without_whitespace_by_tokens() -> None:
    document = LoadedDocument(
        path=Path("pathological.pdf"),
        document_id="large",
        pages=[PageText("研究" * 20_000, 1)],
    )

    chunks = chunk_document(document, size=350, overlap=60)
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")

    assert len(chunks) > 1
    assert all(len(encoding.encode(chunk.text)) <= 350 for chunk in chunks)
