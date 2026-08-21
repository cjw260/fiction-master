from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from fiction_master.db import Base


def new_id() -> str:
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(UTC)


class Book(Base):
    __tablename__ = "books"
    __table_args__ = (Index("ix_books_content_hash", "content_hash"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    relative_path: Mapped[str] = mapped_column(String(1024), unique=True, index=True)
    file_format: Mapped[str] = mapped_column(String(16))
    title: Mapped[str] = mapped_column(String(512))
    author: Mapped[str | None] = mapped_column(String(256), nullable=True)
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    chapter_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    active_index_version: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="discovered", index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    duplicate_of: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("books.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    chapters: Mapped[list[Chapter]] = relationship(
        back_populates="book", cascade="all, delete-orphan", order_by="Chapter.ordinal"
    )


class Chapter(Base):
    __tablename__ = "chapters"
    __table_args__ = (Index("ix_chapters_book_ordinal", "book_id", "ordinal", unique=True),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    book_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("books.id", ondelete="CASCADE"), index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    volume_title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    title: Mapped[str] = mapped_column(String(512))
    start_offset: Mapped[int] = mapped_column(Integer)
    end_offset: Mapped[int] = mapped_column(Integer)

    book: Mapped[Book] = relationship(back_populates="chapters")


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    job_type: Mapped[str] = mapped_column(String(32), default="sync")
    target_book_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    total_files: Mapped[int] = mapped_column(Integer, default=0)
    completed_files: Mapped[int] = mapped_column(Integer, default=0)
    current_file: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    title: Mapped[str] = mapped_column(String(256), default="新对话")
    scope_mode: Mapped[str] = mapped_column(String(16), default="auto")
    book_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, index=True
    )

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(24), default="completed")
    content: Mapped[str] = mapped_column(Text, default="")
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    usage: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
    citations: Mapped[list[Citation]] = relationship(
        back_populates="message", cascade="all, delete-orphan", order_by="Citation.ordinal"
    )


class Citation(Base):
    __tablename__ = "citations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    message_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("messages.id", ondelete="CASCADE"), index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    book_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    book_title: Mapped[str] = mapped_column(String(512))
    chapter_title: Mapped[str] = mapped_column(String(512))
    chunk_id: Mapped[str] = mapped_column(String(64))
    excerpt: Mapped[str] = mapped_column(Text)
    start_offset: Mapped[int] = mapped_column(Integer)
    end_offset: Mapped[int] = mapped_column(Integer)

    message: Mapped[Message] = relationship(back_populates="citations")
