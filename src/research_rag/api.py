import shutil
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from research_rag.config import get_settings
from research_rag.documents import SUPPORTED_SUFFIXES
from research_rag.hybrid import HybridRAGService
from research_rag.knowledge_graph.retrieval import (
    GraphRetrievalResult,
    GraphRetrievalService,
    GraphRetrieveRequest,
)
from research_rag.models import Answer, AskRequest, HybridAnswer, HybridAskRequest, IngestResponse
from research_rag.service import RAGService

app = FastAPI(title="Research Paper RAG", version="0.1.0")
WEB_DIRECTORY = Path(__file__).parent / "web"
app.mount("/static", StaticFiles(directory=WEB_DIRECTORY), name="static")


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(WEB_DIRECTORY / "index.html")


@lru_cache
def get_service() -> RAGService:
    return RAGService(get_settings())


@lru_cache
def get_graph_retrieval_service() -> GraphRetrievalService:
    return GraphRetrievalService(get_settings())


@lru_cache
def get_hybrid_service() -> HybridRAGService:
    return HybridRAGService(get_settings())


@app.get("/health")
def health(service: Annotated[RAGService, Depends(get_service)]) -> dict[str, int | str]:
    return {"status": "ok", "indexed_chunks": service.store.count()}


@app.post("/ingest", response_model=IngestResponse)
def ingest(
    files: Annotated[list[UploadFile], File(description="PDF, Markdown, or text papers")],
    service: Annotated[RAGService, Depends(get_service)],
) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")
    with TemporaryDirectory(prefix="research-rag-") as directory:
        paths: list[Path] = []
        for upload in files:
            filename = Path(upload.filename or "upload").name
            if Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
                raise HTTPException(status_code=415, detail=f"Unsupported file: {filename}")
            destination = Path(directory) / filename
            with destination.open("wb") as output:
                shutil.copyfileobj(upload.file, output)
            paths.append(destination)
        try:
            return service.ingest(paths)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/ask", response_model=Answer)
def ask(
    request: AskRequest, service: Annotated[RAGService, Depends(get_service)]
) -> Answer:
    try:
        return service.ask(request.question, request.top_k)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/graph/retrieve", response_model=GraphRetrievalResult)
def graph_retrieve(
    request: GraphRetrieveRequest,
    service: Annotated[GraphRetrievalService, Depends(get_graph_retrieval_service)],
) -> GraphRetrievalResult:
    try:
        return service.retrieve(request.question, request.top_k, request.depth)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/hybrid/ask", response_model=HybridAnswer)
def hybrid_ask(
    request: HybridAskRequest,
    service: Annotated[HybridRAGService, Depends(get_hybrid_service)],
) -> HybridAnswer:
    try:
        return service.ask(request.question, request.top_k)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
