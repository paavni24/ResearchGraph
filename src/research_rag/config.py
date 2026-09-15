from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    chat_model: str = Field(default="gpt-5-mini", alias="RAG_CHAT_MODEL")
    embedding_model: str = Field(
        default="text-embedding-3-small", alias="RAG_EMBEDDING_MODEL"
    )
    chroma_path: Path = Field(default=Path("data/chroma"), alias="RAG_CHROMA_PATH")
    collection: str = Field(default="research_papers", alias="RAG_COLLECTION")
    chunk_size: int = Field(default=350, ge=50, alias="RAG_CHUNK_SIZE")
    chunk_overlap: int = Field(default=60, ge=0, alias="RAG_CHUNK_OVERLAP")
    top_k: int = Field(default=5, ge=1, le=20, alias="RAG_TOP_K")
    fulltext_path: Path = Field(
        default=Path("data/fulltext.sqlite3"), alias="RAG_FULLTEXT_PATH"
    )

    hybrid_rerank_model: str = Field(default="gpt-5-mini", alias="HYBRID_RERANK_MODEL")
    hybrid_answer_model: str = Field(default="gpt-5-mini", alias="HYBRID_ANSWER_MODEL")
    hybrid_per_retriever: int = Field(default=8, ge=1, le=30, alias="HYBRID_PER_RETRIEVER")
    hybrid_candidate_limit: int = Field(default=20, ge=3, le=60, alias="HYBRID_CANDIDATE_LIMIT")
    hybrid_top_k: int = Field(default=6, ge=1, le=20, alias="HYBRID_TOP_K")
    hybrid_rerank_chars: int = Field(default=2000, ge=200, le=8000, alias="HYBRID_RERANK_CHARS")
    hybrid_embedding_cost_per_million: float = Field(
        default=0.02, ge=0, alias="HYBRID_EMBEDDING_COST_PER_MILLION"
    )
    hybrid_input_cost_per_million: float = Field(
        default=0.25, ge=0, alias="HYBRID_INPUT_COST_PER_MILLION"
    )
    hybrid_cached_input_cost_per_million: float = Field(
        default=0.025, ge=0, alias="HYBRID_CACHED_INPUT_COST_PER_MILLION"
    )
    hybrid_output_cost_per_million: float = Field(
        default=2.0, ge=0, alias="HYBRID_OUTPUT_COST_PER_MILLION"
    )

    kg_extraction_model: str = Field(default="gpt-5-mini", alias="KG_EXTRACTION_MODEL")
    kg_chunk_size: int = Field(default=1200, ge=100, le=8000, alias="KG_CHUNK_SIZE")
    kg_chunk_overlap: int = Field(default=100, ge=0, alias="KG_CHUNK_OVERLAP")
    kg_workers: int = Field(default=3, ge=1, le=16, alias="KG_WORKERS")
    kg_max_retries: int = Field(default=4, ge=0, le=10, alias="KG_MAX_RETRIES")
    kg_retry_base_seconds: float = Field(default=1.0, ge=0.1, le=30, alias="KG_RETRY_BASE_SECONDS")
    kg_state_path: Path = Field(default=Path("data/kg_state.sqlite3"), alias="KG_STATE_PATH")
    kg_input_cost_per_million: float = Field(default=0.25, ge=0, alias="KG_INPUT_COST_PER_MILLION")
    kg_cached_input_cost_per_million: float = Field(
        default=0.025, ge=0, alias="KG_CACHED_INPUT_COST_PER_MILLION"
    )
    kg_output_cost_per_million: float = Field(
        default=2.0, ge=0, alias="KG_OUTPUT_COST_PER_MILLION"
    )
    kg_query_model: str = Field(default="gpt-5-mini", alias="KG_QUERY_MODEL")
    kg_retrieval_depth: int = Field(default=3, ge=1, le=5, alias="KG_RETRIEVAL_DEPTH")
    kg_retrieval_top_k: int = Field(default=8, ge=1, le=50, alias="KG_RETRIEVAL_TOP_K")
    kg_entity_matches: int = Field(default=3, ge=1, le=10, alias="KG_ENTITY_MATCHES")
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", alias="NEO4J_USERNAME")
    neo4j_password: str = Field(default="", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")

    @model_validator(mode="after")
    def validate_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
        if self.kg_chunk_overlap >= self.kg_chunk_size:
            raise ValueError("KG_CHUNK_OVERLAP must be smaller than KG_CHUNK_SIZE")
        return self

    def require_api_key(self) -> None:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and add it.")


@lru_cache
def get_settings() -> Settings:
    return Settings()
