from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class PageText:
    text: str
    page: int | None


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    document_id: str
    chunk_index: int
    page: int | None = None

    @property
    def metadata(self) -> dict[str, str | int]:
        data: dict[str, str | int] = {
            "source": self.source,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
        }
        if self.page is not None:
            data["page"] = self.page
        return data


@dataclass(frozen=True)
class LoadedDocument:
    path: Path
    document_id: str
    pages: list[PageText]


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


class Source(BaseModel):
    index: int
    source: str
    page: int | None = None
    chunk_index: int
    distance: float | None = None
    excerpt: str


class Answer(BaseModel):
    answer: str
    sources: list[Source]


class RerankScore(BaseModel):
    candidate_id: str
    relevance_score: float = Field(ge=0, le=100)


class RerankResult(BaseModel):
    scores: list[RerankScore]


class HybridSource(BaseModel):
    index: int
    id: str
    source: str
    page: int | None = None
    chunk_index: int
    retrieval_methods: list[str]
    fused_score: float
    rerank_score: float
    excerpt: str
    graph_path: list[str] = Field(default_factory=list)


class HybridAnswer(BaseModel):
    answer: str
    sources: list[HybridSource]
    retriever_counts: dict[str, int]
    usage: dict[str, dict[str, int | float]]
    errors: dict[str, str] = Field(default_factory=dict)


class HybridAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class IngestedFile(BaseModel):
    source: str
    document_id: str
    chunks: int


class IngestResponse(BaseModel):
    files: list[IngestedFile]
    total_chunks: int
