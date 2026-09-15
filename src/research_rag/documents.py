import hashlib
import re
from pathlib import Path

import tiktoken
from pypdf import PdfReader

from research_rag.models import Chunk, LoadedDocument, PageText

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def discover_documents(paths: list[Path]) -> list[Path]:
    discovered: list[Path] = []
    for path in paths:
        if path.is_dir():
            discovered.extend(
                child
                for child in sorted(path.rglob("*"))
                if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            discovered.append(path)
        else:
            raise ValueError(f"Unsupported or missing document: {path}")
    return list(dict.fromkeys(path.resolve() for path in discovered))


def load_document(path: Path) -> LoadedDocument:
    raw = path.read_bytes()
    document_id = hashlib.sha256(raw).hexdigest()
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(path)
        pages = [
            PageText(_clean(page.extract_text() or ""), number)
            for number, page in enumerate(reader.pages, 1)
        ]
    elif suffix in {".txt", ".md"}:
        pages = [PageText(_clean(raw.decode("utf-8")), None)]
    else:
        raise ValueError(f"Unsupported document type: {suffix}")

    pages = [page for page in pages if page.text]
    if not pages:
        raise ValueError(f"No extractable text found in {path.name}. Scanned PDFs need OCR first.")
    return LoadedDocument(path=path, document_id=document_id, pages=pages)


def chunk_document(
    document: LoadedDocument,
    size: int,
    overlap: int,
    tokenizer_model: str = "text-embedding-3-small",
) -> list[Chunk]:
    """Split each page into overlapping token windows, preserving page references."""
    chunks: list[Chunk] = []
    step = size - overlap
    chunk_index = 0
    try:
        encoding = tiktoken.encoding_for_model(tokenizer_model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    for page in document.pages:
        tokens = encoding.encode(page.text, disallowed_special=())
        for start in range(0, len(tokens), step):
            window = tokens[start : start + size]
            if not window:
                break
            text = encoding.decode(window)
            while len(encoding.encode(text, disallowed_special=())) > size:
                text = text[:-1]
            if not text.strip():
                continue
            chunk_id = hashlib.sha256(
                f"{document.document_id}:{page.page}:{chunk_index}:{text}".encode()
            ).hexdigest()
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=text,
                    source=document.path.name,
                    document_id=document.document_id,
                    chunk_index=chunk_index,
                    page=page.page,
                )
            )
            chunk_index += 1
            if start + size >= len(tokens):
                break
    return chunks


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
