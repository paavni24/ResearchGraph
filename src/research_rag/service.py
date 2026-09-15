from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from research_rag.config import Settings
from research_rag.documents import chunk_document, discover_documents, load_document
from research_rag.models import Answer, IngestedFile, IngestResponse, Source

if TYPE_CHECKING:
    from research_rag.fulltext import FullTextStore
    from research_rag.openai_client import OpenAIClient
    from research_rag.store import VectorStore


class RAGService:
    def __init__(
        self,
        settings: Settings,
        ai_client: OpenAIClient | None = None,
        store: VectorStore | None = None,
        fulltext_store: FullTextStore | None = None,
    ) -> None:
        settings.require_api_key()
        self.settings = settings
        if ai_client is None:
            from research_rag.openai_client import OpenAIClient

            ai_client = OpenAIClient(
                settings.openai_api_key, settings.embedding_model, settings.chat_model
            )
        if store is None:
            from research_rag.store import VectorStore

            store = VectorStore(settings.chroma_path, settings.collection, settings.embedding_model)
        if fulltext_store is None:
            from research_rag.fulltext import FullTextStore

            fulltext_store = FullTextStore(settings.fulltext_path)
        self.ai = ai_client
        self.store = store
        self.fulltext = fulltext_store

    def ingest(self, paths: list[Path]) -> IngestResponse:
        files: list[IngestedFile] = []
        for path in discover_documents(paths):
            document = load_document(path)
            chunks = chunk_document(
                document,
                self.settings.chunk_size,
                self.settings.chunk_overlap,
                self.settings.embedding_model,
            )
            embeddings = self.ai.embed([chunk.text for chunk in chunks])
            self.store.replace_source(chunks, embeddings)
            self.fulltext.replace_source(chunks)
            files.append(
                IngestedFile(
                    source=path.name, document_id=document.document_id, chunks=len(chunks)
                )
            )
        return IngestResponse(files=files, total_chunks=sum(item.chunks for item in files))

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        query_embedding = self.ai.embed([question])[0]
        hits = self.store.search(query_embedding, top_k or self.settings.top_k)
        if not hits:
            return Answer(
                answer="No papers have been indexed yet. Ingest at least one document first.",
                sources=[],
            )

        sources: list[Source] = []
        context_parts: list[str] = []
        for index, hit in enumerate(hits, 1):
            metadata = hit["metadata"]
            page = metadata.get("page")
            label = f"{metadata['source']}" + (f", page {page}" if page else "")
            context_parts.append(f"[{index}] {label}\n{hit['text']}")
            sources.append(
                Source(
                    index=index,
                    source=str(metadata["source"]),
                    page=int(page) if page is not None else None,
                    chunk_index=int(metadata["chunk_index"]),
                    distance=float(hit["distance"]) if hit["distance"] is not None else None,
                    excerpt=hit["text"][:400],
                )
            )
        answer = self.ai.answer(question, "\n\n".join(context_parts))
        return Answer(answer=answer, sources=sources)
