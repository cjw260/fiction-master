from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", REPO_ROOT / "backend" / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "小说大师"
    app_env: str = "development"
    api_prefix: str = "/api/v1"
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )

    fiction_dir: Path = REPO_ROOT / "fiction"
    data_dir: Path = REPO_ROOT / "data"
    max_fiction_file_mb: int = 100
    auto_sync_on_startup: bool = True

    dashscope_api_key: str | None = None
    chat_api_key: str | None = None
    chat_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    chat_model: str = "qwen-plus"
    embedding_api_key: str | None = None
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "text-embedding-v4"
    embedding_dimension: int = 1024
    embedding_batch_size: int = 10
    embedding_concurrency: int = 4
    rerank_api_key: str | None = None
    rerank_url: str = (
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    )
    rerank_model: str = "qwen3-rerank"

    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "fiction_chunks"

    chunk_target_chars: int = 900
    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 150
    retrieval_candidate_limit: int = 30
    retrieval_evidence_limit: int = 8
    citation_excerpt_chars: int = 300
    max_question_chars: int = 4000

    # LightRAG is an optional sidecar. It supplies graph leads only; final
    # evidence and citations still come from the primary Qdrant index.
    lightrag_enabled: bool = False
    lightrag_base_url: str = "http://127.0.0.1:9621"
    lightrag_api_key: str | None = None
    lightrag_query_timeout_seconds: float = 10.0
    lightrag_index_timeout_seconds: float = 180.0
    lightrag_index_poll_interval_seconds: float = 2.0
    lightrag_index_max_wait_seconds: int = 3600
    lightrag_index_batch_size: int = 20
    lightrag_request_retries: int = 2
    lightrag_index_book_titles: Annotated[list[str], NoDecode] = Field(default_factory=list)
    lightrag_mode: Literal["local", "global", "hybrid", "mix"] = "mix"
    lightrag_top_k: int = 10
    lightrag_chunk_top_k: int = 12
    lightrag_max_entity_tokens: int = 2500
    lightrag_max_relation_tokens: int = 3500
    lightrag_max_total_tokens: int = 8000
    lightrag_enable_rerank: bool = False
    lightrag_expanded_query_limit: int = 3
    lightrag_rrf_weight: float = 0.7

    @field_validator("cors_origins", "lightrag_index_book_titles", mode="before")
    @classmethod
    def parse_comma_separated_list(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def resolve_paths(self) -> Settings:
        if not self.fiction_dir.is_absolute():
            self.fiction_dir = (REPO_ROOT / self.fiction_dir).resolve()
        if not self.data_dir.is_absolute():
            self.data_dir = (REPO_ROOT / self.data_dir).resolve()
        if self.lightrag_max_entity_tokens + self.lightrag_max_relation_tokens >= (
            self.lightrag_max_total_tokens
        ):
            raise ValueError(
                "LIGHTRAG_MAX_TOTAL_TOKENS must exceed entity and relation token budgets"
            )
        if self.lightrag_rrf_weight > 1:
            raise ValueError("LIGHTRAG_RRF_WEIGHT must be at most 1")
        if self.lightrag_enabled and not self.lightrag_api_key:
            raise ValueError("LIGHTRAG_API_KEY is required when LIGHTRAG_ENABLED=true")
        return self

    @field_validator("lightrag_request_retries")
    @classmethod
    def nonnegative_lightrag_retries(cls, value: int) -> int:
        if value < 0:
            raise ValueError("LIGHTRAG_REQUEST_RETRIES must be non-negative")
        return value

    @field_validator(
        "lightrag_query_timeout_seconds",
        "lightrag_index_timeout_seconds",
        "lightrag_index_poll_interval_seconds",
        "lightrag_rrf_weight",
    )
    @classmethod
    def positive_lightrag_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("LightRAG timeout and weight settings must be positive")
        return value

    @field_validator(
        "lightrag_index_max_wait_seconds",
        "lightrag_index_batch_size",
        "lightrag_top_k",
        "lightrag_chunk_top_k",
        "lightrag_max_entity_tokens",
        "lightrag_max_relation_tokens",
        "lightrag_max_total_tokens",
        "lightrag_expanded_query_limit",
    )
    @classmethod
    def positive_lightrag_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("LightRAG limits must be positive")
        return value

    @property
    def database_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.data_dir / 'fiction_master.db'}"

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def resolved_chat_key(self) -> str | None:
        return self.chat_api_key or self.dashscope_api_key

    @property
    def resolved_embedding_key(self) -> str | None:
        return self.embedding_api_key or self.dashscope_api_key

    @property
    def resolved_rerank_key(self) -> str | None:
        return self.rerank_api_key or self.dashscope_api_key

    def ensure_directories(self) -> None:
        self.fiction_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
