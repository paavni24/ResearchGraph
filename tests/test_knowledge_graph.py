from research_rag.knowledge_graph.schema import (
    GraphExtraction,
    NodeType,
    RelationshipType,
    resolve_extraction,
)


def test_resolves_valid_schema_edges_and_rejects_invalid_directions() -> None:
    extraction = GraphExtraction.model_validate(
        {
            "paper_title": "A Paper",
            "entities": [
                {
                    "local_id": "author",
                    "type": "Author",
                    "name": "Ada Lovelace",
                    "description": "",
                    "external_id": "",
                },
                {
                    "local_id": "method",
                    "type": "Method",
                    "name": "A Method",
                    "description": "",
                    "external_id": "",
                },
            ],
            "relationships": [
                {
                    "source_id": "author",
                    "type": "AUTHORED",
                    "target_id": "CURRENT_PAPER",
                    "evidence": "Ada authored the paper.",
                    "confidence": 0.99,
                },
                {
                    "source_id": "method",
                    "type": "PROPOSES",
                    "target_id": "CURRENT_PAPER",
                    "evidence": "Wrong direction.",
                    "confidence": 0.5,
                },
            ],
        }
    )

    resolved = resolve_extraction(extraction, "paper-id", "chunk-id")

    assert len(resolved.relationships) == 1
    assert resolved.relationships[0].type is RelationshipType.AUTHORED
    assert resolved.relationships[0].source_type is NodeType.AUTHOR
    assert resolved.rejected_relationships == 1


def test_claim_identity_is_scoped_to_its_paper() -> None:
    extraction = GraphExtraction.model_validate(
        {
            "paper_title": None,
            "entities": [
                {
                    "local_id": "claim",
                    "type": "Claim",
                    "name": "Accuracy improves.",
                    "description": "",
                    "external_id": "",
                }
            ],
            "relationships": [],
        }
    )

    first = resolve_extraction(extraction, "paper-one", "chunk").nodes[0].id
    second = resolve_extraction(extraction, "paper-two", "chunk").nodes[0].id

    assert first != second
