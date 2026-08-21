from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ApiModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class HealthResponse(ApiModel):
    status: Literal["ok", "degraded"]
    database: bool
    vector_store: bool
    models_configured: bool
    version: str


class ModelStatus(ApiModel):
    chat_model: str
    embedding_model: str
    rerank_model: str
    chat_configured: bool
    embedding_configured: bool
    rerank_configured: bool


class BookResponse(ApiModel):
    id: str
    relative_path: str
    file_format: str
    title: str
    author: str | None
    file_size: int
    word_count: int
    chapter_count: int
    chunk_count: int
    status: str
    error: str | None
    duplicate_of: str | None
    updated_at: datetime


class JobResponse(ApiModel):
    id: str
    job_type: str
    target_book_id: str | None
    status: str
    total_files: int
    completed_files: int
    current_file: str | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class JobAccepted(ApiModel):
    job_id: str
    status: Literal["queued"] = "queued"


class ScopeRequest(ApiModel):
    mode: Literal["auto", "books"] = "auto"
    book_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_scope(self) -> ScopeRequest:
        self.book_ids = list(
            dict.fromkeys(book_id.strip() for book_id in self.book_ids if book_id.strip())
        )
        if self.mode == "books" and not self.book_ids:
            raise ValueError("book_ids is required when scope mode is books")
        if self.mode == "auto":
            self.book_ids = []
        return self


class ConversationCreate(ApiModel):
    title: str | None = Field(default=None, max_length=256)
    scope: ScopeRequest = Field(default_factory=ScopeRequest)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


class ConversationUpdate(ApiModel):
    title: str | None = Field(default=None, min_length=1, max_length=256)
    scope: ScopeRequest | None = None

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("title must not be blank")
        return normalized


class CitationResponse(ApiModel):
    id: str
    ordinal: int
    book_id: str | None
    book_title: str
    chapter_title: str
    chunk_id: str
    excerpt: str
    start_offset: int
    end_offset: int


class MessageResponse(ApiModel):
    id: str
    conversation_id: str
    role: str
    status: str
    content: str
    model: str | None
    usage: dict[str, object]
    latency_ms: int | None
    error: str | None
    created_at: datetime
    citations: list[CitationResponse] = Field(default_factory=list)


class ConversationResponse(ApiModel):
    id: str
    title: str
    scope_mode: str
    book_ids: list[str]
    created_at: datetime
    updated_at: datetime


class ConversationDetail(ConversationResponse):
    messages: list[MessageResponse] = Field(default_factory=list)


class ChatRequest(ApiModel):
    content: str = Field(min_length=1, max_length=4000)
    scope: ScopeRequest | None = None

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("content must not be blank")
        return normalized


class ErrorBody(ApiModel):
    code: str
    message: str
    retryable: bool = False
