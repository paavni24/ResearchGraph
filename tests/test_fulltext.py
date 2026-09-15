from pathlib import Path

from research_rag.fulltext import FullTextStore
from research_rag.models import Chunk


def test_fulltext_search_uses_bm25_and_replaces_source(tmp_path: Path) -> None:
    store = FullTextStore(tmp_path / "fulltext.sqlite3")
    transformer = Chunk("one", "Transformer attention architecture", "paper.pdf", "doc", 0, 1)
    unrelated = Chunk("two", "Convolutional image classifier", "paper.pdf", "doc", 1, 2)
    store.replace_source([transformer, unrelated])

    hits = store.search("How does Transformer attention work?", 5)

    assert hits[0]["id"] == "one"
    assert hits[0]["metadata"]["page"] == 1

    replacement = Chunk("three", "A replacement passage", "paper.pdf", "doc", 0, 3)
    store.replace_source([replacement])
    assert store.count() == 1

