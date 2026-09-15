from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from pydantic import BaseModel, Field

from research_rag.config import Settings
from research_rag.knowledge_graph.checkpoint import CheckpointStore
from research_rag.knowledge_graph.schema import NodeType, canonical_name
from research_rag.models import TokenUsage

if TYPE_CHECKING:
    from research_rag.knowledge_graph.store import Neo4jGraphStore


QUERY_INSTRUCTIONS = """Extract named entities from a research question for knowledge-graph lookup.
Allowed types are Paper, Author, Method, Model, Dataset, Task, Metric, Concept, and Organization.
Use the most specific canonical name present or clearly referenced in the question. Include explicit
abbreviations, expanded names, DOI/arXiv IDs, or spelling variants as aliases. Do not emit generic
category words such as "paper", "method", "dataset", or "metric" by themselves. Do not answer the
question. Return an empty entity list if the question contains no searchable named entity."""

QUERYABLE_TYPES = {
    NodeType.PAPER,
    NodeType.AUTHOR,
    NodeType.METHOD,
    NodeType.MODEL,
    NodeType.DATASET,
    NodeType.TASK,
    NodeType.METRIC,
    NodeType.CONCEPT,
    NodeType.ORGANIZATION,
}


class QuestionEntity(BaseModel):
    name: str = Field(min_length=1, max_length=1000)
    type: NodeType
    aliases: list[str] = Field(default_factory=list)


class QuestionEntityPlan(BaseModel):
    entities: list[QuestionEntity]


@dataclass(frozen=True)
class QueryExtractionResult:
    plan: QuestionEntityPlan
    usage: TokenUsage
    attempts: int


class MatchedGraphEntity(BaseModel):
    id: str
    type: str
    name: str
    description: str = ""
    query_entity: str
    match_score: float


class RetrievedGraphChunk(BaseModel):
    id: str
    source: str
    page: int | None = None
    chunk_index: int
    text: str
    score: float
    reason: str
    evidence: str | None = None
    seed_entity: str
    path_nodes: list[str]
    path_relationships: list[str]


class GraphRetrievalResult(BaseModel):
    question: str
    extracted_entities: list[QuestionEntity]
    matched_entities: list[MatchedGraphEntity]
    chunks: list[RetrievedGraphChunk]
    usage: dict[str, int | float]


class GraphRetrieveRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    depth: int | None = Field(default=None, ge=1, le=5)


class QuestionEntityExtractor:
    def __init__(
        self, api_key: str, model: str, max_retries: int = 4, retry_base_seconds: float = 1.0
    ) -> None:
        self.client = OpenAI(api_key=api_key, max_retries=0)
        self.model = model
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds

    def extract(self, question: str) -> QueryExtractionResult:
        attempts = 0
        while True:
            attempts += 1
            try:
                response = self.client.responses.parse(
                    model=self.model,
                    instructions=QUERY_INSTRUCTIONS,
                    input=question,
                    text_format=QuestionEntityPlan,
                    store=False,
                )
                if response.output_parsed is None:
                    raise RuntimeError("Question entity extraction returned no parsed output")
                usage = response.usage
                input_details = getattr(usage, "input_tokens_details", None)
                output_details = getattr(usage, "output_tokens_details", None)
                plan = QuestionEntityPlan(
                    entities=[
                        entity
                        for entity in response.output_parsed.entities
                        if entity.type in QUERYABLE_TYPES
                    ]
                )
                return QueryExtractionResult(
                    plan=plan,
                    usage=TokenUsage(
                        input_tokens=getattr(usage, "input_tokens", 0) or 0,
                        cached_input_tokens=getattr(input_details, "cached_tokens", 0) or 0,
                        output_tokens=getattr(usage, "output_tokens", 0) or 0,
                        reasoning_tokens=getattr(output_details, "reasoning_tokens", 0) or 0,
                        total_tokens=getattr(usage, "total_tokens", 0) or 0,
                    ),
                    attempts=attempts,
                )
            except (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            ) as error:
                if attempts > self.max_retries:
                    error.attempts = attempts
                    raise
                delay = self.retry_base_seconds * (2 ** (attempts - 1))
                time.sleep(delay + random.uniform(0, delay * 0.25))


class GraphRetrievalService:
    def __init__(
        self,
        settings: Settings,
        extractor: QuestionEntityExtractor | None = None,
        store: Neo4jGraphStore | None = None,
        checkpoints: CheckpointStore | None = None,
    ) -> None:
        settings.require_api_key()
        self.settings = settings
        self.extractor = extractor or QuestionEntityExtractor(
            settings.openai_api_key,
            settings.kg_query_model,
            settings.kg_max_retries,
            settings.kg_retry_base_seconds,
        )
        if store is None:
            from research_rag.knowledge_graph.store import Neo4jGraphStore

            store = Neo4jGraphStore(
                settings.neo4j_uri,
                settings.neo4j_username,
                settings.neo4j_password,
                settings.neo4j_database,
            )
        self.store = store
        self.checkpoints = checkpoints or CheckpointStore(settings.kg_state_path)

    def retrieve(
        self, question: str, top_k: int | None = None, depth: int | None = None
    ) -> GraphRetrievalResult:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        top_k = top_k or self.settings.kg_retrieval_top_k
        depth = depth or self.settings.kg_retrieval_depth
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        if not 1 <= depth <= 5:
            raise ValueError("depth must be between 1 and 5")

        self.store.verify_connectivity()
        extraction = self.extractor.extract(question)
        cost = self._cost(extraction.usage)
        query_id = uuid.uuid4().hex
        self.checkpoints.record_query_usage(
            query_id,
            question,
            self.settings.kg_query_model,
            extraction.attempts,
            extraction.usage,
            cost,
        )

        mentions = [
            {
                "mention_id": position,
                "name": entity.name,
                "type": entity.type.value,
                "keys": sorted(
                    {
                        canonical_name(value)
                        for value in [entity.name, *entity.aliases]
                        if value.strip()
                    }
                ),
            }
            for position, entity in enumerate(extraction.plan.entities)
        ]
        raw_matches = self.store.lookup_entities(mentions, self.settings.kg_entity_matches)
        matches_by_id: dict[str, dict[str, Any]] = {}
        for match in raw_matches:
            current = matches_by_id.get(match["id"])
            if current is None or match["match_score"] > current["match_score"]:
                matches_by_id[match["id"]] = match
        matches = sorted(
            matches_by_id.values(), key=lambda item: (-item["match_score"], item["name"])
        )
        seeds = [
            {"id": match["id"], "name": match["name"], "score": match["match_score"]}
            for match in matches
        ]
        chunks = self.store.retrieve_graph_chunks(seeds, depth, top_k) if seeds else []
        usage = extraction.usage
        return GraphRetrievalResult(
            question=question,
            extracted_entities=extraction.plan.entities,
            matched_entities=[MatchedGraphEntity.model_validate(match) for match in matches],
            chunks=[RetrievedGraphChunk.model_validate(chunk) for chunk in chunks],
            usage={
                "input_tokens": usage.input_tokens,
                "cached_input_tokens": usage.cached_input_tokens,
                "output_tokens": usage.output_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
                "total_tokens": usage.total_tokens,
                "cost_usd": round(cost, 6),
            },
        )

    def _cost(self, usage: TokenUsage) -> float:
        cached = min(usage.cached_input_tokens, usage.input_tokens)
        uncached = usage.input_tokens - cached
        return (
            uncached * self.settings.kg_input_cost_per_million
            + cached * self.settings.kg_cached_input_cost_per_million
            + usage.output_tokens * self.settings.kg_output_cost_per_million
        ) / 1_000_000

    def close(self) -> None:
        self.store.close()
