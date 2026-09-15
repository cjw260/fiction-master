from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from fiction_master.config import Settings
from fiction_master.errors import NotFoundError
from fiction_master.ingestion.chunker import chunk_book
from fiction_master.ingestion.graph_indexer import GraphIndexingService
from fiction_master.ingestion.parsers import SUPPORTED_SUFFIXES, ParsedBook, parse_file
from fiction_master.models import Book, Chapter, GraphIndex, IngestionJob, utc_now
from fiction_master.rag.lightrag import GraphChapter
from fiction_master.rag.providers import EmbeddingProvider
from fiction_master.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class IngestionService:
    def __init__(
        self,
        *,
        settings: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        graph_indexer: GraphIndexingService | None = None,
    ) -> None:
        self.settings = settings
        self.session_factory = session_factory
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.graph_indexer = graph_indexer
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def recover_interrupted_jobs(self) -> None:
        """Make in-memory jobs retryable after an unclean process restart."""
        message = "Indexing interrupted by service restart"
        async with self.session_factory() as session:
            jobs = list(
                (
                    await session.scalars(
                        select(IngestionJob).where(IngestionJob.status.in_(["queued", "running"]))
                    )
                ).all()
            )
            for job in jobs:
                job.status = "failed"
                job.error = message
                job.finished_at = utc_now()

            books = list(
                (await session.scalars(select(Book).where(Book.status == "indexing"))).all()
            )
            for book in books:
                book.status = "ready" if book.active_index_version else "error"
                book.error = message
            await session.commit()

    async def create_sync_job(self) -> IngestionJob:
        async with self.session_factory() as session:
            running = await session.scalar(
                select(IngestionJob)
                .where(
                    IngestionJob.job_type == "sync",
                    IngestionJob.status.in_(["queued", "running"]),
                )
                .order_by(IngestionJob.created_at.desc())
            )
            if running:
                return running
            job = IngestionJob(job_type="sync", status="queued")
            session.add(job)
            await session.commit()
            await session.refresh(job)
        self._schedule(job.id)
        return job

    async def create_reindex_job(self, book_id: str) -> IngestionJob:
        async with self.session_factory() as session:
            book = await session.get(Book, book_id)
            if book is None:
                raise NotFoundError("book")
            running = await session.scalar(
                select(IngestionJob)
                .where(
                    IngestionJob.target_book_id == book_id,
                    IngestionJob.status.in_(["queued", "running"]),
                )
                .order_by(IngestionJob.created_at.desc())
            )
            if running:
                return running
            job = IngestionJob(
                job_type="reindex", target_book_id=book_id, status="queued", total_files=1
            )
            session.add(job)
            await session.commit()
            await session.refresh(job)
        self._schedule(job.id)
        return job

    def _schedule(self, job_id: str) -> None:
        task = asyncio.create_task(self._run_job(job_id), name=f"ingestion-{job_id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def discover_files(self) -> list[Path]:
        root = self.settings.fiction_dir.resolve()
        files: list[Path] = []
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            try:
                relative = path.resolve().relative_to(root)
            except ValueError:
                logger.warning("Skipping fiction path outside root: %s", path)
                continue
            if relative.name.casefold() == "readme.md" or any(
                part.startswith(".") for part in relative.parts
            ):
                continue
            files.append(path)
        return sorted(files, key=lambda item: item.relative_to(root).as_posix())

    async def _run_job(self, job_id: str) -> None:
        async with self._lock:
            async with self.session_factory() as session:
                job = await session.get(IngestionJob, job_id)
                if job is None:
                    return
                job.status = "running"
                job.started_at = utc_now()
                await session.commit()

            errors: list[str] = []
            try:
                target_book_id: str | None
                async with self.session_factory() as session:
                    job = await session.get(IngestionJob, job_id)
                    target_book_id = job.target_book_id if job else None

                if target_book_id:
                    async with self.session_factory() as session:
                        book = await session.get(Book, target_book_id)
                        if book is None:
                            raise NotFoundError("book")
                        paths = [self.settings.fiction_dir / book.relative_path]
                else:
                    paths = self.discover_files()

                await self._update_job(job_id, total_files=len(paths))
                seen_paths: set[str] = set()
                for index, path in enumerate(paths, start=1):
                    relative = path.relative_to(self.settings.fiction_dir).as_posix()
                    seen_paths.add(relative)
                    await self._update_job(job_id, current_file=relative)
                    try:
                        await self._process_file(path, force=target_book_id is not None)
                    except Exception as exc:
                        logger.exception("Failed to index %s", relative)
                        errors.append(f"{relative}: {exc}")
                        await self._mark_book_error(relative, exc)
                    await self._update_job(job_id, completed_files=index)

                if target_book_id is None:
                    await self._mark_missing_books(seen_paths)
                await self._finish_job(job_id, errors)
            except asyncio.CancelledError:
                await asyncio.shield(self._fail_job(job_id, "Index job cancelled"))
                raise
            except Exception as exc:
                logger.exception("Ingestion job %s failed", job_id)
                await self._fail_job(job_id, str(exc))

    async def _process_file(self, path: Path, *, force: bool) -> None:
        if not path.exists():
            relative = path.relative_to(self.settings.fiction_dir).as_posix()
            await self._mark_book_missing(relative)
            return
        max_bytes = self.settings.max_fiction_file_mb * 1024 * 1024
        file_size = path.stat().st_size
        if file_size > max_bytes:
            raise ValueError(
                f"File exceeds {self.settings.max_fiction_file_mb} MB limit: {path.name}"
            )
        relative = path.relative_to(self.settings.fiction_dir).as_posix()
        fingerprint = await asyncio.to_thread(sha256_file, path)

        duplicate_version: str | None = None
        duplicate_found = False
        graph_only = False
        book_id: str | None = None
        old_version: str | None = None
        async with self.session_factory() as session:
            book = await session.scalar(select(Book).where(Book.relative_path == relative))
            if book is None:
                book = Book(
                    relative_path=relative,
                    file_format=path.suffix.lower().lstrip("."),
                    title=path.stem,
                    file_size=file_size,
                    status="discovered",
                )
                session.add(book)
                await session.commit()
                await session.refresh(book)
            if not force and book.content_hash == fingerprint and book.status == "ready":
                if self.graph_indexer is None or not book.active_index_version:
                    return
                graph_row = await session.scalar(
                    select(GraphIndex).where(
                        GraphIndex.book_id == book.id,
                        GraphIndex.index_version == book.active_index_version,
                        GraphIndex.status.in_(["queued", "indexing", "ready", "paused"]),
                    )
                )
                if graph_row is not None:
                    return
                graph_only = True
                book_id = book.id
                old_version = book.active_index_version
            duplicate = (
                None
                if graph_only
                else await session.scalar(
                    select(Book).where(
                        Book.content_hash == fingerprint,
                        Book.id != book.id,
                        Book.status == "ready",
                    )
                )
            )
            if duplicate is not None:
                book_id = book.id
                duplicate_version = book.active_index_version
                book.status = "duplicate"
                book.duplicate_of = duplicate.id
                book.content_hash = fingerprint
                book.file_size = file_size
                book.active_index_version = None
                book.error = f"Duplicate of {duplicate.relative_path}"
                await session.commit()
                duplicate_found = True
            elif not graph_only:
                book.status = "indexing"
                book.error = None
                book.duplicate_of = None
                await session.commit()
                book_id = book.id
                old_version = book.active_index_version

        if duplicate_found:
            if self.graph_indexer and book_id:
                await self.graph_indexer.remove_book(book_id)
            if duplicate_version:
                try:
                    await self.vector_store.delete_index_version(duplicate_version)
                except Exception:
                    logger.exception("Failed to clean duplicate's old index %s", duplicate_version)
            return
        if book_id is None:
            raise RuntimeError("Book identity was not initialized")

        parsed = await asyncio.to_thread(parse_file, path)
        if graph_only:
            if old_version is None:
                raise RuntimeError("Graph-only indexing requires an active primary index")
            await self._schedule_graph_index(
                book_id=book_id,
                index_version=old_version,
                content_hash=fingerprint,
                parsed=parsed,
            )
            return
        chunks = chunk_book(
            parsed,
            target_chars=self.settings.chunk_target_chars,
            max_chars=self.settings.chunk_max_chars,
            overlap_chars=self.settings.chunk_overlap_chars,
        )
        if not chunks:
            raise ValueError("No indexable text chunks found")
        async with self.session_factory() as session:
            book = await session.get(Book, book_id)
            if book is None:
                raise NotFoundError("book")
            book.title = parsed.title
            book.author = parsed.author
            book.file_size = file_size
            book.word_count = parsed.word_count
            book.chapter_count = len(parsed.chapters)
            book.chunk_count = len(chunks)
            await session.commit()
        embedding_inputs = [
            f"书名：{parsed.title}\n章节：{chunk.chapter_title}\n正文：{chunk.content}"
            for chunk in chunks
        ]
        dense_vectors = await self.embedding_provider.embed(embedding_inputs)
        index_version = str(uuid4())
        try:
            await self.vector_store.upsert_book(
                book_id=book_id,
                book_title=parsed.title,
                author=parsed.author,
                relative_path=relative,
                index_version=index_version,
                chunks=chunks,
                dense_vectors=dense_vectors,
            )
        except asyncio.CancelledError:
            await asyncio.shield(self._discard_index(index_version))
            raise
        except Exception:
            await self._discard_index(index_version)
            raise

        try:
            async with self.session_factory() as session:
                book = await session.get(Book, book_id)
                if book is None:
                    raise NotFoundError("book")
                await session.execute(delete(Chapter).where(Chapter.book_id == book_id))
                session.add_all(
                    [
                        Chapter(
                            book_id=book_id,
                            ordinal=chapter.ordinal,
                            volume_title=chapter.volume_title,
                            title=chapter.title,
                            start_offset=chapter.start_offset,
                            end_offset=chapter.end_offset,
                        )
                        for chapter in parsed.chapters
                    ]
                )
                book.title = parsed.title
                book.author = parsed.author
                book.file_format = path.suffix.lower().lstrip(".")
                book.file_size = file_size
                book.word_count = parsed.word_count
                book.chapter_count = len(parsed.chapters)
                book.chunk_count = len(chunks)
                book.content_hash = fingerprint
                book.active_index_version = index_version
                book.status = "ready"
                book.error = None
                book.duplicate_of = None
                await session.commit()
        except Exception:
            await self.vector_store.delete_index_version(index_version)
            raise
        if old_version and old_version != index_version:
            try:
                await self.vector_store.delete_index_version(old_version)
            except Exception:
                logger.exception("Failed to clean old index version %s", old_version)
        await self._schedule_graph_index(
            book_id=book_id,
            index_version=index_version,
            content_hash=fingerprint,
            parsed=parsed,
        )

    async def _schedule_graph_index(
        self,
        *,
        book_id: str,
        index_version: str,
        content_hash: str,
        parsed: ParsedBook,
    ) -> None:
        if self.graph_indexer is None:
            return
        allowed_titles = self.settings.lightrag_index_book_titles
        if allowed_titles and parsed.title not in allowed_titles:
            logger.info(
                "Skipping LightRAG graph indexing for %s because it is not in "
                "LIGHTRAG_INDEX_BOOK_TITLES",
                parsed.title,
            )
            return
        chapters = [
            GraphChapter(
                ordinal=chapter.ordinal,
                title=chapter.title,
                content=chapter.content,
                volume_title=chapter.volume_title,
            )
            for chapter in parsed.chapters
        ]
        try:
            await self.graph_indexer.schedule(
                book_id=book_id,
                index_version=index_version,
                content_hash=content_hash,
                book_title=parsed.title,
                author=parsed.author,
                chapters=chapters,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            # The primary Qdrant index is already committed and must remain
            # usable even if the optional graph scheduler itself fails.
            logger.exception("Failed to schedule optional graph index for %s", book_id)

    async def _discard_index(self, index_version: str) -> None:
        try:
            await self.vector_store.delete_index_version(index_version)
        except Exception:
            logger.exception("Failed to discard incomplete index version %s", index_version)

    async def _mark_book_error(self, relative: str, exc: Exception) -> None:
        async with self.session_factory() as session:
            book = await session.scalar(select(Book).where(Book.relative_path == relative))
            if book is None:
                book = Book(
                    relative_path=relative,
                    file_format=Path(relative).suffix.lower().lstrip("."),
                    title=Path(relative).stem,
                    status="error",
                )
                session.add(book)
            book.status = "ready" if book.active_index_version else "error"
            book.error = str(exc)[:4000]
            await session.commit()

    async def _mark_book_missing(self, relative: str) -> None:
        await self._mark_missing_books(set(), only_relative=relative)

    async def _mark_missing_books(
        self, seen_paths: set[str], *, only_relative: str | None = None
    ) -> None:
        versions: list[str] = []
        graph_book_ids: list[str] = []
        async with self.session_factory() as session:
            query = select(Book)
            if only_relative is not None:
                query = query.where(Book.relative_path == only_relative)
            books = list((await session.scalars(query)).all())
            for book in books:
                if only_relative is None and book.relative_path in seen_paths:
                    continue
                if book.active_index_version:
                    versions.append(book.active_index_version)
                graph_book_ids.append(book.id)
                book.active_index_version = None
                book.status = "missing"
                book.error = "Source file is missing"
            await session.commit()
        for version in versions:
            try:
                await self.vector_store.delete_index_version(version)
            except Exception:
                logger.exception("Failed to remove missing book index %s", version)
        if self.graph_indexer:
            for book_id in graph_book_ids:
                await self.graph_indexer.remove_book(book_id)

    async def _update_job(self, job_id: str, **values: object) -> None:
        async with self.session_factory() as session:
            job = await session.get(IngestionJob, job_id)
            if job is None:
                return
            for key, value in values.items():
                setattr(job, key, value)
            await session.commit()

    async def _finish_job(self, job_id: str, errors: list[str]) -> None:
        async with self.session_factory() as session:
            job = await session.get(IngestionJob, job_id)
            if job is None:
                return
            job.status = "succeeded_with_errors" if errors else "succeeded"
            job.error = "\n".join(errors)[:8000] if errors else None
            job.current_file = None
            job.finished_at = utc_now()
            await session.commit()

    async def _fail_job(self, job_id: str, error: str) -> None:
        async with self.session_factory() as session:
            job = await session.get(IngestionJob, job_id)
            if job is None:
                return
            job.status = "failed"
            job.error = error[:8000]
            job.finished_at = utc_now()
            await session.commit()
