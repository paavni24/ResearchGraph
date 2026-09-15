# Research Paper RAG

A deliberately small, inspectable retrieval-augmented generation baseline. It:

1. extracts text from PDF, Markdown, and plain-text papers;
2. makes overlapping, page-aware chunks and embeds them;
3. persists vectors and metadata in a local Chroma database;
4. retrieves similar passages and asks an LLM for a grounded, cited answer.

## Quick start

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Put your OpenAI API key in .env
```

Ingest one file, several files, or a directory recursively:

```bash
research-rag ingest ./papers
research-rag ask "What limitations do the authors identify?"
```

Or start the HTTP API:

```bash
uvicorn research_rag.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API. Example requests:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -F "files=@papers/example.pdf"

curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the paper’s main contribution?","top_k":5}'
```

The application frontend is served at `http://127.0.0.1:8000/`. It provides hybrid literature
chat, expandable evidence cards, graph-path and retrieval-score inspection, per-query usage/cost,
and PDF/Markdown/text upload into the vector and full-text indexes. Neo4j extraction
remains a separate cost-controlled CLI operation.

## How it works

Each PDF page is cleaned and split into token windows (350 tokens with 60 tokens of overlap by default). Token-aware splitting prevents malformed or whitespace-free PDF text from exceeding embedding input limits. Chunks are embedded in batches and upserted into a cosine-similarity Chroma collection. Re-ingesting a filename replaces its prior chunks. At query time, the question is embedded with the same model, the closest chunks are numbered, and the LLM is instructed to answer only from those chunks and cite them as `[1]`, `[2]`, and so on.

The `/ask` response includes the answer plus source filename, page, chunk number, distance, and an excerpt. Lower cosine distance is a closer match.

Configuration lives in `.env`; see [.env.example](.env.example). If you change the embedding model, use a new `RAG_COLLECTION`, because embeddings from different models must not be mixed. Image-only/scanned PDFs need OCR before ingestion.

## Layout

```text
src/research_rag/
  documents.py      extraction and chunking
  fulltext.py       persistent SQLite FTS5/BM25 retrieval
  hybrid.py         three-way fusion, reranking, and answering
  openai_client.py  embeddings and LLM generation
  store.py          persistent Chroma retrieval
  service.py        ingestion and RAG orchestration
  api.py            FastAPI endpoints
  cli.py            command-line interface
```

## Verify

```bash
pytest
ruff check .
```

This is a baseline, not a production system. Natural next steps are semantic/section-aware chunking,
OCR, retrieval evaluation datasets, answer-faithfulness checks, access controls, tracing, and
background ingestion for large corpora.

## Knowledge graph (separate pipeline)

The repository also contains an entity/relation extraction pipeline that does not use embeddings or Chroma. It creates `Paper`, `Author`, `Method`, `Model`, `Dataset`, `Task`, `Metric`, `Concept`, `Organization`, `Claim`, and `Chunk` nodes in Neo4j and restricts edges to the schema defined in `knowledge_graph/schema.py`.

Run Neo4j using Neo4j Desktop, Neo4j Aura, or the included Compose service, then refresh the editable installation:

```bash
docker compose up -d neo4j
pip install -e ".[dev]"
```

Build a small trial graph first because extraction makes one LLM request per graph chunk:

```bash
research-graph ingest ./papers/attention_is_all_you_need.pdf
research-graph stats
research-graph usage
```

Then process more documents when the output and API cost are acceptable:

```bash
research-graph ingest ./papers --max-documents 5
research-graph ingest ./papers/attention_is_all_you_need.pdf --replace
```

Neo4j Browser is available at `http://localhost:7474`. The username defaults to `neo4j`; set a
strong `NEO4J_PASSWORD` in your private `.env`. `Organization` is extracted as a node but remains
disconnected because the requested core schema does not define an organization relationship.

Graph ingestion includes exact API usage/cost accounting, resumable SQLite checkpoints,
exponential retries, bounded concurrency, alias-aware entity deduplication, and document-level
replace semantics.

Retrieve chunks using the knowledge graph without invoking Chroma:

```bash
research-graph retrieve "What architecture does Attention Is All You Need propose?"
```

This performs structured question-entity extraction, entity lookup, bounded graph traversal, and
provenance-aware chunk ranking. It is the graph-only retrieval stage intended to feed a later hybrid
vector-plus-graph retriever.

## Hybrid retrieval and answering

The complete path combines all three retrieval systems:

```text
Question
├── Chroma semantic vector retrieval
├── SQLite FTS5/BM25 lexical retrieval
└── Neo4j entity lookup and graph traversal
        ↓
weighted reciprocal-rank fusion
        ↓
structured LLM reranking
        ↓
grounded LLM answer with [1], [2] citations
```

```bash
research-rag hybrid-ask "How does the Transformer use multi-head attention?"
research-rag hybrid-ask "Which datasets are used for machine translation?" --top-k 6
```

The API equivalent is `POST /hybrid/ask`. Its response includes the answer, cited passages,
contributing retrieval methods, fusion and reranking scores, graph paths, retriever counts, usage,
cost, and isolated retriever errors. If Neo4j is unavailable, vector and full-text retrieval can
still answer.

Vector ingestion writes the same chunks to Chroma and SQLite FTS5. Existing Chroma data is copied
into FTS5 automatically when their counts differ, without regenerating embeddings or calling OpenAI.
