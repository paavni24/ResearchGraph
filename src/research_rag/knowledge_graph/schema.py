import hashlib
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    PAPER = "Paper"
    AUTHOR = "Author"
    METHOD = "Method"
    MODEL = "Model"
    DATASET = "Dataset"
    TASK = "Task"
    METRIC = "Metric"
    CONCEPT = "Concept"
    ORGANIZATION = "Organization"
    CLAIM = "Claim"
    CHUNK = "Chunk"


class RelationshipType(str, Enum):
    AUTHORED = "AUTHORED"
    CITES = "CITES"
    PROPOSES = "PROPOSES"
    USES = "USES"
    EVALUATES_ON = "EVALUATES_ON"
    USES_MODEL = "USES_MODEL"
    REPORTS = "REPORTS"
    IMPROVES_ON = "IMPROVES_ON"
    RELATED_TO = "RELATED_TO"
    MAKES_CLAIM = "MAKES_CLAIM"
    SUPPORTED_BY = "SUPPORTED_BY"
    PART_OF = "PART_OF"


ALLOWED_ENDPOINTS: dict[RelationshipType, tuple[NodeType, NodeType]] = {
    RelationshipType.AUTHORED: (NodeType.AUTHOR, NodeType.PAPER),
    RelationshipType.CITES: (NodeType.PAPER, NodeType.PAPER),
    RelationshipType.PROPOSES: (NodeType.PAPER, NodeType.METHOD),
    RelationshipType.USES: (NodeType.PAPER, NodeType.DATASET),
    RelationshipType.EVALUATES_ON: (NodeType.PAPER, NodeType.TASK),
    RelationshipType.USES_MODEL: (NodeType.PAPER, NodeType.MODEL),
    RelationshipType.REPORTS: (NodeType.PAPER, NodeType.METRIC),
    RelationshipType.IMPROVES_ON: (NodeType.METHOD, NodeType.METHOD),
    RelationshipType.RELATED_TO: (NodeType.METHOD, NodeType.CONCEPT),
    RelationshipType.MAKES_CLAIM: (NodeType.PAPER, NodeType.CLAIM),
    RelationshipType.SUPPORTED_BY: (NodeType.CLAIM, NodeType.CHUNK),
    RelationshipType.PART_OF: (NodeType.CHUNK, NodeType.PAPER),
}

CURRENT_PAPER = "CURRENT_PAPER"
CURRENT_CHUNK = "CURRENT_CHUNK"


class EntityCandidate(BaseModel):
    local_id: str = Field(min_length=1, max_length=80)
    type: NodeType
    name: str = Field(min_length=1, max_length=1000)
    description: str = Field(default="", max_length=2000)
    external_id: str = Field(default="", max_length=500)
    aliases: list[str] = Field(default_factory=list)


class RelationshipCandidate(BaseModel):
    source_id: str = Field(min_length=1, max_length=80)
    type: RelationshipType
    target_id: str = Field(min_length=1, max_length=80)
    evidence: str = Field(min_length=1, max_length=2000)
    confidence: float = Field(ge=0, le=1)


class GraphExtraction(BaseModel):
    paper_title: str | None = Field(default=None, max_length=1000)
    entities: list[EntityCandidate]
    relationships: list[RelationshipCandidate]


@dataclass(frozen=True)
class ResolvedNode:
    id: str
    type: NodeType
    name: str
    description: str
    external_id: str
    aliases: tuple[str, ...]
    document_id: str


@dataclass(frozen=True)
class ResolvedRelationship:
    source_id: str
    source_type: NodeType
    type: RelationshipType
    target_id: str
    target_type: NodeType
    evidence: str
    confidence: float


@dataclass(frozen=True)
class ResolvedExtraction:
    nodes: list[ResolvedNode]
    relationships: list[ResolvedRelationship]
    rejected_relationships: int


def canonical_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def candidate_aliases(entity: EntityCandidate) -> tuple[str, ...]:
    values = [entity.name, entity.external_id, *entity.aliases]
    return tuple(sorted({canonical_name(value) for value in values if value.strip()}))


def entity_id(entity: EntityCandidate, document_id: str) -> str:
    identity = canonical_name(entity.external_id or entity.name)
    if entity.type is NodeType.CLAIM:
        identity = f"{document_id}:{identity}"
    digest = hashlib.sha256(f"{entity.type.value}:{identity}".encode()).hexdigest()
    return digest


def resolve_extraction(
    extraction: GraphExtraction,
    document_id: str,
    chunk_id: str,
    id_resolver: Callable[[EntityCandidate, str], str] | None = None,
) -> ResolvedExtraction:
    references: dict[str, tuple[NodeType, str]] = {
        CURRENT_PAPER: (NodeType.PAPER, document_id),
        CURRENT_CHUNK: (NodeType.CHUNK, chunk_id),
    }
    nodes: list[ResolvedNode] = []
    for candidate in extraction.entities:
        if candidate.local_id in references:
            continue
        resolved_id = (
            id_resolver(candidate, document_id)
            if id_resolver is not None
            else entity_id(candidate, document_id)
        )
        references.setdefault(candidate.local_id, (candidate.type, resolved_id))
        nodes.append(
            ResolvedNode(
                id=resolved_id,
                type=candidate.type,
                name=candidate.name.strip(),
                description=candidate.description.strip(),
                external_id=candidate.external_id.strip(),
                aliases=candidate_aliases(candidate),
                document_id=document_id,
            )
        )

    relationships: list[ResolvedRelationship] = []
    rejected = 0
    for candidate in extraction.relationships:
        source = references.get(candidate.source_id)
        target = references.get(candidate.target_id)
        expected = ALLOWED_ENDPOINTS[candidate.type]
        if (
            source is None
            or target is None
            or (source[0], target[0]) != expected
            or candidate.type is RelationshipType.PART_OF
        ):
            rejected += 1
            continue
        relationships.append(
            ResolvedRelationship(
                source_id=source[1],
                source_type=source[0],
                type=candidate.type,
                target_id=target[1],
                target_type=target[0],
                evidence=candidate.evidence.strip(),
                confidence=candidate.confidence,
            )
        )
    return ResolvedExtraction(nodes, relationships, rejected)
