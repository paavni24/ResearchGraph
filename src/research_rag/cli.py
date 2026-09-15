import argparse
import json
from pathlib import Path

from research_rag.config import get_settings
from research_rag.hybrid import HybridRAGService
from research_rag.service import RAGService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest and query research papers")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="Chunk, embed, and index documents")
    ingest.add_argument("paths", nargs="+", type=Path, help="Files or directories")

    ask = commands.add_parser("ask", help="Ask a grounded question")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=None)
    hybrid = commands.add_parser(
        "hybrid-ask", help="Combine vector, full-text, and graph retrieval before answering"
    )
    hybrid.add_argument("question")
    hybrid.add_argument("--top-k", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "hybrid-ask":
        service = HybridRAGService(get_settings())
        try:
            result = service.ask(args.question, args.top_k)
            print(json.dumps(result.model_dump(), indent=2))
        finally:
            service.close()
        return
    service = RAGService(get_settings())
    if args.command == "ingest":
        result = service.ingest(args.paths)
    else:
        result = service.ask(args.question, args.top_k)
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
