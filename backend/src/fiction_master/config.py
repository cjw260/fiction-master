from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

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

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def resolve_paths(self) -> Settings:
        if not self.fiction_dir.is_absolute():
            self.fiction_dir = (REPO_ROOT / self.fiction_dir).resolve()
        if not self.data_dir.is_absolute():
            self.data_dir = (REPO_ROOT / self.data_dir).resolve()
        return self

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
