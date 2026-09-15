from collections.abc import Iterable
from typing import Any

from neo4j import GraphDatabase, ManagedTransaction

from research_rag.knowledge_graph.schema import (
    NodeType,
    ResolvedExtraction,
    ResolvedNode,
    ResolvedRelationship,
)
from research_rag.models import Chunk, LoadedDocument


class Neo4jGraphStore:
    def __init__(self, uri: str, username: str, password: str, database: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database

    def close(self) -> None:
        self.driver.close()

    def verify_connectivity(self) -> None:
        self.driver.verify_connectivity()

    def ensure_schema(self) -> None:
        with self.driver.session(database=self.database) as session:
            for node_type in NodeType:
                constraint = f"{node_type.value.lower()}_id_unique"
                session.run(
                    f"CREATE CONSTRAINT {constraint} IF NOT EXISTS "
                    f"FOR (n:{node_type.value}) REQUIRE n.id IS UNIQUE"
                ).consume()

    def upsert_document(self, document: LoadedDocument, title: str | None = None) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                "MERGE (p:Paper {id: $id}) "
                "SET p.name = coalesce($title, p.name, $source), p.source = $source, "
                "p.document_ids = CASE WHEN $id IN coalesce(p.document_ids, []) "
                "THEN p.document_ids ELSE coalesce(p.document_ids, []) + $id END",
                id=document.document_id,
                title=title,
                source=document.path.name,
            ).consume()

    def existing_chunk_ids(self, document_id: str) -> set[str]:
        with self.driver.session(database=self.database) as session:
            rows = session.run(
                "MATCH (c:Chunk {document_id: $document_id}) RETURN c.id AS id",
                document_id=document_id,
            ).data()
        return {row["id"] for row in rows}

    def replace_document(self, document_id: str) -> None:
        """Remove facts supported by this document while preserving shared graph entities."""
        with self.driver.session(database=self.database) as session:
            session.execute_write(self._replace_document_transaction, document_id)

    @staticmethod
    def _replace_document_transaction(tx: ManagedTransaction, document_id: str) -> None:
        chunk_ids = [
            row["id"]
            for row in tx.run(
                "MATCH (c:Chunk {document_id: $document_id}) RETURN c.id AS id",
                document_id=document_id,
            )
        ]
        if chunk_ids:
            tx.run(
                "MATCH ()-[r]->() WHERE r.evidence_chunk_id IN $chunk_ids DELETE r",
                chunk_ids=chunk_ids,
            ).consume()
        tx.run(
            "MATCH (c:Chunk {document_id: $document_id}) DETACH DELETE c",
            document_id=document_id,
        ).consume()
        tx.run(
            "MATCH (c:Claim {document_id: $document_id}) DETACH DELETE c",
            document_id=document_id,
        ).consume()
        tx.run(
            "MATCH (n) WHERE $document_id IN coalesce(n.document_ids, []) "
            "SET n.document_ids = [value IN n.document_ids WHERE value <> $document_id]",
            document_id=document_id,
        ).consume()
        tx.run(
            "MATCH (n) WHERE NOT n:Paper AND NOT n:Chunk AND NOT (n)--() "
            "DELETE n"
        ).consume()
        tx.run(
            "MATCH (p:Paper) WHERE p.source IS NULL AND NOT (p)--() DELETE p"
        ).consume()

    def write_chunk(
        self,
        document: LoadedDocument,
        chunk: Chunk,
        extraction: ResolvedExtraction,
        paper_title: str | None,
    ) -> None:
        with self.driver.session(database=self.database) as session:
            session.execute_write(
                self._write_chunk_transaction,
                document,
                chunk,
                extraction,
                paper_title,
            )

    @staticmethod
    def _write_chunk_transaction(
        tx: ManagedTransaction,
        document: LoadedDocument,
        chunk: Chunk,
        extraction: ResolvedExtraction,
        paper_title: str | None,
    ) -> None:
        tx.run(
            "MERGE (p:Paper {id: $paper_id}) "
            "SET p.name = coalesce($title, p.name, $source), p.source = $source, "
            "p.document_ids = CASE WHEN $paper_id IN coalesce(p.document_ids, []) "
            "THEN p.document_ids ELSE coalesce(p.document_ids, []) + $paper_id END "
            "MERGE (c:Chunk {id: $chunk_id}) "
            "SET c.text = $text, c.page = $page, c.chunk_index = $chunk_index, "
            "c.document_id = $paper_id, c.document_ids = [$paper_id], c.source = $source "
            "MERGE (c)-[:PART_OF]->(p)",
            paper_id=document.document_id,
            title=paper_title,
            source=document.path.name,
            chunk_id=chunk.id,
            text=chunk.text,
            page=chunk.page,
            chunk_index=chunk.chunk_index,
        ).consume()
        Neo4jGraphStore._upsert_nodes(tx, extraction.nodes)
        Neo4jGraphStore._upsert_relationships(tx, extraction.relationships, chunk.id)

    @staticmethod
    def _upsert_nodes(tx: ManagedTransaction, nodes: Iterable[ResolvedNode]) -> None:
        for node in nodes:
            tx.run(
                f"MERGE (n:{node.type.value} {{id: $id}}) "
                "SET n.name = $name, "
                "n.description = CASE WHEN $description = '' THEN n.description "
                "ELSE $description END, "
                "n.external_id = CASE WHEN $external_id = '' THEN n.external_id "
                "ELSE $external_id END, "
                "n.aliases = reduce(values = coalesce(n.aliases, []), alias IN $aliases | "
                "CASE WHEN alias IN values THEN values ELSE values + alias END), "
                "n.document_ids = CASE WHEN $document_id IN coalesce(n.document_ids, []) "
                "THEN n.document_ids ELSE coalesce(n.document_ids, []) + $document_id END, "
                "n.document_id = CASE WHEN $node_type = 'Claim' THEN $document_id "
                "ELSE n.document_id END",
                id=node.id,
                name=node.name,
                description=node.description,
                external_id=node.external_id,
                aliases=list(node.aliases),
                document_id=node.document_id,
                node_type=node.type.value,
            ).consume()

    @staticmethod
    def _upsert_relationships(
        tx: ManagedTransaction,
        relationships: Iterable[ResolvedRelationship],
        evidence_chunk_id: str,
    ) -> None:
        for relationship in relationships:
            tx.run(
                f"MATCH (s:{relationship.source_type.value} {{id: $source_id}}) "
                f"MATCH (t:{relationship.target_type.value} {{id: $target_id}}) "
                f"MERGE (s)-[r:{relationship.type.value} "
                "{evidence_chunk_id: $evidence_chunk_id}]->(t) "
                "SET r.evidence = $evidence, r.confidence = $confidence",
                source_id=relationship.source_id,
                target_id=relationship.target_id,
                evidence_chunk_id=evidence_chunk_id,
                evidence=relationship.evidence,
                confidence=relationship.confidence,
            ).consume()

    def stats(self) -> dict[str, Any]:
        with self.driver.session(database=self.database) as session:
            node_rows = session.run(
                "MATCH (n) UNWIND labels(n) AS label "
                "RETURN label, count(*) AS count ORDER BY label"
            ).data()
            relationship_rows = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY type"
            ).data()
        return {
            "nodes": {row["label"]: row["count"] for row in node_rows},
            "relationships": {row["type"]: row["count"] for row in relationship_rows},
        }

    def lookup_entities(
        self, mentions: list[dict[str, Any]], per_mention: int
    ) -> list[dict[str, Any]]:
        if not mentions:
            return []
        query = """
        UNWIND $mentions AS mention
        CALL (mention) {
            MATCH (n)
            WHERE mention.type IN labels(n)
            WITH mention, n,
                CASE
                    WHEN any(key IN mention.keys WHERE key IN coalesce(n.aliases, []))
                        THEN 1.0
                    WHEN any(key IN mention.keys WHERE
                        toLower(trim(coalesce(n.name, ''))) = key OR
                        toLower(trim(coalesce(n.external_id, ''))) = key)
                        THEN 1.0
                    WHEN any(key IN mention.keys WHERE size(key) >= 3 AND
                        (toLower(coalesce(n.name, '')) CONTAINS key OR
                        key CONTAINS toLower(coalesce(n.name, ''))))
                        THEN 0.65
                    ELSE 0.0
                END AS score
            WHERE score > 0
            RETURN n, score
            ORDER BY score DESC, n.name
            LIMIT $per_mention
        }
        RETURN n.id AS id, mention.type AS type, coalesce(n.name, n.id) AS name,
            coalesce(n.description, '') AS description, mention.name AS query_entity,
            score AS match_score
        """
        with self.driver.session(database=self.database) as session:
            return session.run(
                query, mentions=mentions, per_mention=per_mention
            ).data()

    def retrieve_graph_chunks(
        self, seeds: list[dict[str, Any]], depth: int, top_k: int
    ) -> list[dict[str, Any]]:
        if not seeds:
            return []
        if not 1 <= depth <= 5:
            raise ValueError("depth must be between 1 and 5")
        candidate_limit = max(50, top_k * 5)
        evidence_query = f"""
        UNWIND $seeds AS seed_data
        MATCH (seed {{id: seed_data.id}})
        CALL (seed, seed_data) {{
            MATCH p = (seed)-[*1..{depth}]-(context)
            UNWIND relationships(p) AS evidence_rel
            WITH seed_data, p, evidence_rel
            WHERE evidence_rel.evidence_chunk_id IS NOT NULL
            MATCH (chunk:Chunk {{id: evidence_rel.evidence_chunk_id}})
            RETURN chunk, p, evidence_rel,
                seed_data.score + 1.0 +
                coalesce(evidence_rel.confidence, 0.0) * 0.1 - length(p) * 0.05 AS score
            ORDER BY score DESC
            LIMIT $candidate_limit
        }}
        RETURN chunk.id AS id, coalesce(chunk.source, '') AS source, chunk.page AS page,
            chunk.chunk_index AS chunk_index, chunk.text AS text, score,
            'relationship_evidence' AS reason, evidence_rel.evidence AS evidence,
            seed_data.name AS seed_entity,
            [node IN nodes(p) | coalesce(node.name, node.source, node.id)] AS path_nodes,
            [relationship IN relationships(p) | type(relationship)] AS path_relationships
        """
        structural_query = f"""
        UNWIND $seeds AS seed_data
        MATCH (seed {{id: seed_data.id}})
        CALL (seed, seed_data) {{
            MATCH p = (seed)-[*1..{depth}]-(chunk:Chunk)
            RETURN chunk, p, seed_data.score + 0.4 / length(p) AS score
            ORDER BY length(p), chunk.chunk_index
            LIMIT $candidate_limit
        }}
        RETURN chunk.id AS id, coalesce(chunk.source, '') AS source, chunk.page AS page,
            chunk.chunk_index AS chunk_index, chunk.text AS text, score,
            'graph_traversal' AS reason, null AS evidence, seed_data.name AS seed_entity,
            [node IN nodes(p) | coalesce(node.name, node.source, node.id)] AS path_nodes,
            [relationship IN relationships(p) | type(relationship)] AS path_relationships
        """
        with self.driver.session(database=self.database) as session:
            evidence_rows = session.run(
                evidence_query, seeds=seeds, candidate_limit=candidate_limit
            ).data()
            structural_rows = session.run(
                structural_query, seeds=seeds, candidate_limit=candidate_limit
            ).data()

        chunks_by_id: dict[str, dict[str, Any]] = {}
        for row in [*evidence_rows, *structural_rows]:
            row["score"] = round(float(row["score"]), 6)
            current = chunks_by_id.get(row["id"])
            if current is None or row["score"] > current["score"]:
                chunks_by_id[row["id"]] = row
        return sorted(
            chunks_by_id.values(),
            key=lambda item: (-item["score"], item["source"], item["chunk_index"]),
        )[:top_k]
