import argparse
import json
import sys
from pathlib import Path

from research_rag.config import get_settings
from research_rag.knowledge_graph.checkpoint import CheckpointStore
from research_rag.knowledge_graph.retrieval import GraphRetrievalService
from research_rag.knowledge_graph.service import KnowledgeGraphService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and inspect the research knowledge graph")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="Extract entities/relations into Neo4j")
    ingest.add_argument("paths", nargs="+", type=Path, help="Files or directories")
    ingest.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Only process the first N documents (useful for cost-controlled trials)",
    )
    ingest.add_argument(
        "--workers", type=int, default=None, help="Concurrent extraction requests (default: KG_WORKERS)"
    )
    ingest.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip chunks committed to both SQLite and Neo4j (default: enabled)",
    )
    ingest.add_argument(
        "--replace",
        action="store_true",
        help="Delete each document's previous evidence-backed graph and rebuild it",
    )
    commands.add_parser("stats", help="Show node and relationship counts")
    commands.add_parser("usage", help="Show persisted token usage, cost, and recent runs")
    retrieve = commands.add_parser(
        "retrieve", help="Extract question entities and retrieve chunks through Neo4j"
    )
    retrieve.add_argument("question")
    retrieve.add_argument("--top-k", type=int, default=None)
    retrieve.add_argument("--depth", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    if args.command == "usage":
        print(json.dumps(CheckpointStore(settings.kg_state_path).usage_summary(), indent=2))
        return
    if args.command == "retrieve":
        retrieval = GraphRetrievalService(settings)
        try:
            result = retrieval.retrieve(args.question, args.top_k, args.depth)
            print(json.dumps(result.model_dump(), indent=2))
        finally:
            retrieval.close()
        return

    service = KnowledgeGraphService(settings)
    try:
        if args.command == "ingest":
            result = service.ingest(
                args.paths,
                max_documents=args.max_documents,
                workers=args.workers,
                resume=args.resume,
                replace=args.replace,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
        elif args.command == "stats":
            result = service.stats()
        print(json.dumps(result, indent=2))
        if args.command == "ingest" and result["failed_chunks"]:
            raise SystemExit(1)
    finally:
        service.close()


if __name__ == "__main__":
    main()
