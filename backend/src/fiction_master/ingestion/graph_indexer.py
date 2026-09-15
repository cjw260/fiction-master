from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from fiction_master.config import Settings
from fiction_master.models import Book, GraphIndex, utc_now
from fiction_master.rag.lightrag import (
    GraphChapter,
    LightRagClient,
    LightRagError,
    graph_file_source,
    project_graph_chapter,
)

logger = logging.getLogger(__name__)


class GraphIndexingService:
    """Runs optional LightRAG indexing without blocking the primary index."""

    def __init__(
        self,
        *,
        settings: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        client: LightRagClient,
    ) -> None:
        self.settings = settings
        self.session_factory = session_factory
        self.client = client
        self._schedule_lock = asyncio.Lock()
        self._insert_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()
        self._row_tasks: dict[str, asyncio.Task[None]] = {}

    async def recover_interrupted(self) -> None:
        async with self.session_factory() as session:
            rows = list(
                (
                    await session.scalars(
                        select(GraphIndex).where(
                            GraphIndex.status.in_(["queued", "indexing", "removing"])
                        )
                    )
                ).all()
            )
            for row in rows:
                row.status = "error"
                row.error = "Graph indexing interrupted by service restart"
                row.finished_at = utc_now()
            await session.commit()

    async def close(self) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def wait_for_idle(self) -> None:
        tasks = list(self._tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def needs_index(self, *, book_id: str, index_version: str) -> bool:
        async with self.session_factory() as session:
            row = await session.scalar(
                select(GraphIndex).where(
                    GraphIndex.book_id == book_id,
                    GraphIndex.index_version == index_version,
                    GraphIndex.status.in_(["queued", "indexing", "ready", "paused"]),
                )
            )
            return row is None

    async def schedule(
        self,
        *,
        book_id: str,
        index_version: str,
        content_hash: str,
        book_title: str,
        author: str | None,
        chapters: Sequence[GraphChapter],
    ) -> GraphIndex:
        if not chapters:
            raise ValueError("Cannot build a graph index without chapters")
        async with self._schedule_lock:
            async with self.session_factory() as session:
                row = await session.scalar(
                    select(GraphIndex).where(
                        GraphIndex.book_id == book_id,
                        GraphIndex.index_version == index_version,
                    )
                )
                if row is not None and row.status in {"queued", "indexing", "ready", "paused"}:
                    return row
                file_sources = [
                    graph_file_source(book_id, index_version, chapter.ordinal)
                    for chapter in chapters
                ]
                if row is None:
                    row = GraphIndex(
                        book_id=book_id,
                        index_version=index_version,
                        content_hash=content_hash,
                        status="queued",
                        file_sources=file_sources,
                    )
                    session.add(row)
                else:
                    content_changed = row.content_hash != content_hash
                    row.content_hash = content_hash
                    row.status = "queued"
                    if content_changed:
                        row.track_ids = []
                        row.document_ids = []
                    row.file_sources = file_sources
                    row.error = None
                    row.finished_at = None
                await session.commit()
                await session.refresh(row)
                row_id = row.id

            task = asyncio.create_task(
                self._run(
                    row_id=row_id,
                    book_title=book_title,
                    author=author,
                    chapters=list(chapters),
                ),
                name=f"graph-index-{row_id}",
            )
            self._remember_task(task, row_id=row_id)
            return row

    async def remove_book(self, book_id: str) -> None:
        async with self._schedule_lock:
            async with self.session_factory() as session:
                row_ids = list(
                    (
                        await session.scalars(
                            select(GraphIndex.id).where(GraphIndex.book_id == book_id)
                        )
                    ).all()
                )
                if not row_ids:
                    return

            active_tasks = [
                task
                for row_id in row_ids
                if (task := self._row_tasks.get(row_id)) is not None and not task.done()
            ]
            for task in active_tasks:
                task.cancel()
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)

            async with self.session_factory() as session:
                rows = list(
                    (
                        await session.scalars(
                            select(GraphIndex).where(GraphIndex.book_id == book_id)
                        )
                    ).all()
                )
                document_ids = list(
                    dict.fromkeys(doc_id for row in rows for doc_id in row.document_ids)
                )
                for row in rows:
                    row.status = "removing"
                    row.error = None
                await session.commit()
            task = asyncio.create_task(
                self._remove_rows(row_ids=row_ids, document_ids=document_ids),
                name=f"graph-remove-{book_id}",
            )
            self._remember_task(task)

    async def _run(
        self,
        *,
        row_id: str,
        book_title: str,
        author: str | None,
        chapters: Sequence[GraphChapter],
    ) -> None:
        try:
            await self._update_row(row_id, status="indexing", error=None)
            async with self.session_factory() as session:
                row = await session.get(GraphIndex, row_id)
                if row is None:
                    return
                file_sources = list(row.file_sources)
                track_ids = list(row.track_ids)
                document_ids = list(row.document_ids)
            texts = [
                project_graph_chapter(book_title=book_title, author=author, chapter=chapter)
                for chapter in chapters
            ]
            batch_size = self.settings.lightrag_index_batch_size
            async with asyncio.timeout(self.settings.lightrag_index_max_wait_seconds):
                document_ids = await self._poll_tracks(row_id, track_ids, document_ids)
                if len(document_ids) > len(texts):
                    raise LightRagError(
                        "LightRAG returned more documents than the graph index contains"
                    )
                for start in range(len(document_ids), len(texts), batch_size):
                    async with self._insert_lock:
                        track_id = await self.client.insert_chapters(
                            texts=texts[start : start + batch_size],
                            file_sources=file_sources[start : start + batch_size],
                        )
                    track_ids.append(track_id)
                    await self._update_row(row_id, track_ids=list(track_ids))
                    document_ids = await self._poll_tracks(row_id, [track_id], document_ids)
            if len(document_ids) != len(texts):
                raise LightRagError(
                    "LightRAG completed indexing without returning every document ID"
                )

            activated = await self._activate_if_current(row_id, document_ids)
            if not activated:
                await self._delete_documents_safely(
                    document_ids,
                    "Failed to clean an obsolete LightRAG graph index",
                )
                return
            await self._supersede_older_rows(row_id)
        except asyncio.CancelledError:
            await asyncio.shield(
                self._update_row(
                    row_id,
                    status="error",
                    error="Graph indexing cancelled",
                    finished_at=utc_now(),
                )
            )
            raise
        except Exception as exc:
            logger.exception("LightRAG graph indexing failed for %s", row_id)
            await self._update_row(
                row_id,
                status="error",
                error=str(exc)[:4000],
                finished_at=utc_now(),
            )

    async def _poll_tracks(
        self,
        row_id: str,
        track_ids: Sequence[str],
        document_ids: Sequence[str],
    ) -> list[str]:
        collected = list(document_ids)
        for track_id in track_ids:
            while True:
                status = await self.client.track_status(track_id)
                collected = list(dict.fromkeys([*collected, *status.document_ids]))
                await self._update_row(row_id, document_ids=list(collected))
                if status.complete:
                    if status.failed:
                        raise LightRagError(status.error or "LightRAG document indexing failed")
                    break
                await asyncio.sleep(self.settings.lightrag_index_poll_interval_seconds)
        return collected

    async def _supersede_older_rows(self, current_row_id: str) -> None:
        async with self.session_factory() as session:
            current = await session.get(GraphIndex, current_row_id)
            if current is None:
                return
            older = list(
                (
                    await session.scalars(
                        select(GraphIndex).where(
                            GraphIndex.book_id == current.book_id,
                            GraphIndex.id != current.id,
                            GraphIndex.status != "removing",
                        )
                    )
                ).all()
            )
            document_ids = list(
                dict.fromkeys(doc_id for row in older for doc_id in row.document_ids)
            )
            for row in older:
                row.status = "superseded"
                row.finished_at = row.finished_at or utc_now()
            await session.commit()
        if document_ids:
            # Old sources cannot become final evidence because retrieval is
            # scoped to the new source prefix and Qdrant active version.
            await self._delete_documents_safely(
                document_ids,
                "Failed to clean superseded LightRAG documents",
            )

    async def _activate_if_current(self, row_id: str, document_ids: Sequence[str]) -> bool:
        """Atomically activate a graph only while its primary index is current."""

        async with self.session_factory() as session:
            row = await session.get(GraphIndex, row_id)
            if row is None:
                return False
            book = await session.get(Book, row.book_id)
            if (
                book is None
                or book.status != "ready"
                or book.active_index_version != row.index_version
                or row.status in {"removing", "superseded"}
            ):
                row.status = "superseded"
                row.document_ids = list(document_ids)
                row.error = None
                row.finished_at = utc_now()
                await session.commit()
                return False
            row.status = "ready"
            row.document_ids = list(document_ids)
            row.error = None
            row.finished_at = utc_now()
            await session.commit()
            return True

    async def _remove_rows(self, *, row_ids: Sequence[str], document_ids: Sequence[str]) -> None:
        try:
            await self.client.delete_documents(document_ids)
        except asyncio.CancelledError:
            await asyncio.shield(self._mark_rows_error(row_ids, "Graph removal cancelled"))
            raise
        except Exception as exc:
            logger.exception("Failed to remove LightRAG documents")
            await self._mark_rows_error(row_ids, str(exc))
            return
        async with self.session_factory() as session:
            for row_id in row_ids:
                row = await session.get(GraphIndex, row_id)
                if row is not None:
                    await session.delete(row)
            await session.commit()

    async def _mark_rows_error(self, row_ids: Sequence[str], error: str) -> None:
        async with self.session_factory() as session:
            for row_id in row_ids:
                row = await session.get(GraphIndex, row_id)
                if row is not None:
                    row.status = "error"
                    row.error = error[:4000]
                    row.finished_at = utc_now()
            await session.commit()

    async def _update_row(self, row_id: str, **values: object) -> None:
        async with self.session_factory() as session:
            row = await session.get(GraphIndex, row_id)
            if row is None:
                return
            for key, value in values.items():
                setattr(row, key, value)
            await session.commit()

    async def _delete_documents_safely(self, document_ids: Sequence[str], log_message: str) -> None:
        if not document_ids:
            return
        try:
            await self.client.delete_documents(document_ids)
        except Exception:
            logger.exception(log_message)

    def _remember_task(self, task: asyncio.Task[None], *, row_id: str | None = None) -> None:
        self._tasks.add(task)
        if row_id is not None:
            self._row_tasks[row_id] = task

        def forget(completed: asyncio.Task[None]) -> None:
            self._tasks.discard(completed)
            if row_id is not None and self._row_tasks.get(row_id) is completed:
                self._row_tasks.pop(row_id, None)

        task.add_done_callback(forget)
