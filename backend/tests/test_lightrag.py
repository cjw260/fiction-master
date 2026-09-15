from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
import respx
from httpx import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from fiction_master.config import Settings
from fiction_master.db import Database
from fiction_master.ingestion.graph_indexer import GraphIndexingService
from fiction_master.models import Book, GraphIndex
from fiction_master.rag.agent import AgentState, RagAgent
from fiction_master.rag.lightrag import (
    GraphChapter,
    GraphTrackStatus,
    LightRagClient,
    LightRagError,
    extract_graph_queries,
    graph_file_source,
    graph_source_prefix,
    plan_graph_retrieval,
)
from fiction_master.rag.metrics import AnswerRunMetrics
from fiction_master.rag.providers import ChatProvider, RerankResult
from fiction_master.rag.vector_store import SearchHit, VectorStore


def test_graph_plan_and_query_extraction_are_source_scoped() -> None:
    assert plan_graph_retrieval("两个人的关系如何变化？", available=True).enabled
    assert plan_graph_retrieval("全书的主要势力有哪些？", available=True).mode == "global"
    assert not plan_graph_retrieval("主角拿到了什么？", available=True).enabled
    assert not plan_graph_retrieval("人物关系是什么？", available=False).enabled

    payload = {
        "data": {
            "relationships": [
                {
                    "src_id": "甲",
                    "tgt_id": "乙",
                    "description": "从盟友逐渐变成对手",
                    "file_path": graph_file_source("book-1", "version-1", 2),
                },
                {
                    "src_id": "外书人物",
                    "tgt_id": "另一个人",
                    "description": "不应进入查询",
                    "file_path": graph_file_source("book-2", "version-9", 1),
                },
            ],
            "entities": [],
            "chunks": [],
        }
    }
    queries = extract_graph_queries(
        payload,
        allowed_source_prefixes=[graph_source_prefix("book-1", "version-1")],
        limit=3,
    )

    assert queries == ["甲 乙 从盟友逐渐变成对手"]
    assert "/" not in graph_file_source("book-1", "version-1", 1)


def test_lightrag_requires_an_api_key_when_enabled(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="LIGHTRAG_API_KEY"):
        Settings(
            fiction_dir=tmp_path / "fiction",
            data_dir=tmp_path / "data",
            lightrag_enabled=True,
            lightrag_api_key=None,
        )


@pytest.mark.asyncio
@respx.mock
async def test_lightrag_rest_contract_and_authentication(tmp_path: Path) -> None:
    settings = Settings(
        fiction_dir=tmp_path / "fiction",
        data_dir=tmp_path / "data",
        auto_sync_on_startup=False,
        lightrag_enabled=True,
        lightrag_base_url="http://lightrag.test:9621",
        lightrag_api_key="test-secret",
        lightrag_request_retries=0,
    )
    insert_route = respx.post("http://lightrag.test:9621/documents/texts").mock(
        return_value=Response(200, json={"status": "success", "track_id": "track-1"})
    )
    respx.get("http://lightrag.test:9621/documents/track_status/track-1").mock(
        return_value=Response(
            200,
            json={
                "status": "success",
                "documents": [{"id": "doc-1", "status": "processed"}],
            },
        )
    )
    query_route = respx.post("http://lightrag.test:9621/query/data").mock(
        return_value=Response(200, json={"status": "success", "data": {"chunks": []}})
    )
    delete_route = respx.delete("http://lightrag.test:9621/documents/delete_document").mock(
        return_value=Response(200, json={"status": "deletion_started"})
    )
    client = LightRagClient(settings)
    try:
        track_id = await client.insert_chapters(
            texts=["章节正文"],
            file_sources=[graph_file_source("book", "version", 1)],
        )
        status = await client.track_status(track_id)
        payload = await client.query_data("人物关系是什么？", mode="mix")
        await client.delete_documents(status.document_ids)
    finally:
        await client.close()

    assert track_id == "track-1"
    assert status.complete and not status.failed
    assert status.document_ids == ["doc-1"]
    assert payload["status"] == "success"
    assert insert_route.calls[0].request.headers["X-API-Key"] == "test-secret"
    insert_body = json.loads(insert_route.calls[0].request.content)
    assert insert_body == {
        "texts": ["章节正文"],
        "file_sources": [graph_file_source("book", "version", 1)],
    }
    query_body = json.loads(query_route.calls[0].request.content)
    assert query_body["mode"] == "mix"
    assert query_body["enable_rerank"] is False
    delete_body = json.loads(delete_route.calls[0].request.content)
    assert delete_body["doc_ids"] == ["doc-1"]


@pytest.mark.asyncio
@respx.mock
async def test_lightrag_insert_retries_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(
        fiction_dir=tmp_path / "fiction",
        data_dir=tmp_path / "data",
        auto_sync_on_startup=False,
        lightrag_enabled=True,
        lightrag_base_url="http://lightrag.test:9621",
        lightrag_api_key="test-secret",
        lightrag_request_retries=2,
    )
    attempts = 0

    def respond(_request: object) -> Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return Response(409, json={"detail": "pipeline reservation conflict"})
        return Response(200, json={"status": "success", "track_id": "track-retried"})

    respx.post("http://lightrag.test:9621/documents/texts").mock(side_effect=respond)
    sleep_delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", record_sleep)
    client = LightRagClient(settings)
    try:
        track_id = await client.insert_chapters(
            texts=["章节正文"],
            file_sources=[graph_file_source("book", "version", 1)],
        )
    finally:
        await client.close()

    assert track_id == "track-retried"
    assert attempts == 3
    assert sleep_delays == [1.0, 2.0]


class FakeGraphClient:
    def __init__(self) -> None:
        self.inserted: list[tuple[list[str], list[str]]] = []
        self.deleted: list[list[str]] = []
        self.queries: list[tuple[str, str]] = []

    async def insert_chapters(self, *, texts: Sequence[str], file_sources: Sequence[str]) -> str:
        self.inserted.append((list(texts), list(file_sources)))
        return f"track-{len(self.inserted)}"

    async def track_status(self, track_id: str) -> GraphTrackStatus:
        batch_index = int(track_id.rsplit("-", 1)[1]) - 1
        document_count = len(self.inserted[batch_index][0])
        return GraphTrackStatus(
            complete=True,
            failed=False,
            document_ids=[f"doc-{track_id}-{item}" for item in range(1, document_count + 1)],
        )

    async def delete_documents(self, document_ids: Sequence[str]) -> None:
        self.deleted.append(list(document_ids))

    async def query_data(self, query: str, *, mode: str) -> dict[str, object]:
        self.queries.append((query, mode))
        return {
            "status": "success",
            "data": {
                "relationships": [
                    {
                        "src_id": "甲",
                        "tgt_id": "乙",
                        "description": "甲帮助乙后因误会反目",
                        "file_path": graph_file_source("book-1", "version-1", 1),
                    }
                ],
                "entities": [],
                "chunks": [],
            },
        }


class ConcurrentInsertGraphClient(FakeGraphClient):
    def __init__(self) -> None:
        super().__init__()
        self.active_inserts = 0
        self.max_active_inserts = 0

    async def insert_chapters(self, *, texts: Sequence[str], file_sources: Sequence[str]) -> str:
        self.active_inserts += 1
        self.max_active_inserts = max(self.max_active_inserts, self.active_inserts)
        try:
            await asyncio.sleep(0.01)
            return await super().insert_chapters(texts=texts, file_sources=file_sources)
        finally:
            self.active_inserts -= 1


@pytest.mark.asyncio
async def test_graph_indexer_serializes_batch_submissions(tmp_path: Path) -> None:
    settings = Settings(
        fiction_dir=tmp_path / "fiction",
        data_dir=tmp_path / "data",
        auto_sync_on_startup=False,
        lightrag_enabled=True,
        lightrag_api_key="test-secret",
        lightrag_index_poll_interval_seconds=0.001,
    )
    settings.ensure_directories()
    database = Database(settings)
    await database.create_schema()
    fake_client = ConcurrentInsertGraphClient()
    indexer = GraphIndexingService(
        settings=settings,
        session_factory=database.session_factory,
        client=cast(LightRagClient, fake_client),
    )
    try:
        async with database.session_factory() as session:
            books = [
                Book(
                    relative_path=f"book-{index}.txt",
                    file_format="txt",
                    title=f"测试小说 {index}",
                    content_hash=str(index) * 64,
                    status="ready",
                    active_index_version=f"version-{index}",
                )
                for index in (1, 2)
            ]
            session.add_all(books)
            await session.commit()
            book_ids = [book.id for book in books]

        for index, book_id in enumerate(book_ids, start=1):
            await indexer.schedule(
                book_id=book_id,
                index_version=f"version-{index}",
                content_hash=str(index) * 64,
                book_title=f"测试小说 {index}",
                author=None,
                chapters=[GraphChapter(ordinal=1, title="第一章", content="正文")],
            )
        await indexer.wait_for_idle()

        assert len(fake_client.inserted) == 2
        assert fake_client.max_active_inserts == 1
    finally:
        await indexer.close()
        await database.close()


@pytest.mark.asyncio
async def test_graph_indexer_batches_tracks_and_supersedes_old_version(tmp_path: Path) -> None:
    settings = Settings(
        fiction_dir=tmp_path / "fiction",
        data_dir=tmp_path / "data",
        auto_sync_on_startup=False,
        lightrag_enabled=True,
        lightrag_api_key="test-secret",
        lightrag_index_batch_size=2,
        lightrag_index_poll_interval_seconds=0.001,
    )
    settings.ensure_directories()
    database = Database(settings)
    await database.create_schema()
    fake_client = FakeGraphClient()
    indexer = GraphIndexingService(
        settings=settings,
        session_factory=database.session_factory,
        client=cast(LightRagClient, fake_client),
    )
    try:
        async with database.session_factory() as session:
            book = Book(
                relative_path="book.txt",
                file_format="txt",
                title="测试小说",
                content_hash="a" * 64,
                status="ready",
                active_index_version="version-1",
            )
            session.add(book)
            await session.commit()
            book_id = book.id

        chapters = [
            GraphChapter(ordinal=index, title=f"第{index}章", content=f"正文 {index}")
            for index in range(1, 4)
        ]
        first = await indexer.schedule(
            book_id=book_id,
            index_version="version-1",
            content_hash="a" * 64,
            book_title="测试小说",
            author=None,
            chapters=chapters,
        )
        await indexer.wait_for_idle()

        async with database.session_factory() as session:
            first_row = await session.get(GraphIndex, first.id)
            assert first_row is not None
            assert first_row.status == "ready"
            assert first_row.track_ids == ["track-1", "track-2"]
            assert first_row.document_ids == [
                "doc-track-1-1",
                "doc-track-1-2",
                "doc-track-2-1",
            ]
            assert all(
                source.startswith(graph_source_prefix(book_id, "version-1"))
                for source in first_row.file_sources
            )

        async with database.session_factory() as session:
            book = await session.get(Book, book_id)
            assert book is not None
            book.active_index_version = "version-2"
            book.content_hash = "b" * 64
            await session.commit()

        second = await indexer.schedule(
            book_id=book_id,
            index_version="version-2",
            content_hash="b" * 64,
            book_title="测试小说",
            author=None,
            chapters=chapters[:1],
        )
        await indexer.wait_for_idle()

        async with database.session_factory() as session:
            rows = list(
                (
                    await session.scalars(select(GraphIndex).where(GraphIndex.book_id == book_id))
                ).all()
            )
            status_by_id = {row.id: row.status for row in rows}
            assert status_by_id[first.id] == "superseded"
            assert status_by_id[second.id] == "ready"
        assert fake_client.deleted == [["doc-track-1-1", "doc-track-1-2", "doc-track-2-1"]]
        assert len(fake_client.inserted) == 3
    finally:
        await indexer.close()
        await database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_status", ["error", "paused"])
async def test_graph_indexer_resumes_saved_tracks_without_reinserting(
    tmp_path: Path, initial_status: str
) -> None:
    settings = Settings(
        fiction_dir=tmp_path / "fiction",
        data_dir=tmp_path / "data",
        auto_sync_on_startup=False,
        lightrag_enabled=True,
        lightrag_api_key="test-secret",
        lightrag_index_poll_interval_seconds=0.001,
    )
    settings.ensure_directories()
    database = Database(settings)
    await database.create_schema()
    fake_client = FakeGraphClient()
    indexer = GraphIndexingService(
        settings=settings,
        session_factory=database.session_factory,
        client=cast(LightRagClient, fake_client),
    )
    try:
        async with database.session_factory() as session:
            book = Book(
                relative_path="resume.txt",
                file_format="txt",
                title="恢复测试",
                content_hash="a" * 64,
                status="ready",
                active_index_version="version-1",
            )
            session.add(book)
            await session.flush()
            source = graph_file_source(book.id, "version-1", 1)
            row = GraphIndex(
                book_id=book.id,
                index_version="version-1",
                content_hash="a" * 64,
                status=initial_status,
                track_ids=["track-1"],
                file_sources=[source],
                error="Graph indexing interrupted by service restart",
            )
            session.add(row)
            await session.commit()
            book_id = book.id
            row_id = row.id

        fake_client.inserted.append((["先前已经提交的正文"], [source]))
        await indexer.schedule(
            book_id=book_id,
            index_version="version-1",
            content_hash="a" * 64,
            book_title="恢复测试",
            author=None,
            chapters=[GraphChapter(ordinal=1, title="第一章", content="正文")],
        )
        await indexer.wait_for_idle()

        async with database.session_factory() as session:
            recovered = await session.get(GraphIndex, row_id)
            assert recovered is not None
            assert recovered.status == ("paused" if initial_status == "paused" else "ready")
            assert recovered.track_ids == ["track-1"]
            assert recovered.document_ids == (
                [] if initial_status == "paused" else ["doc-track-1-1"]
            )
        assert not await indexer.needs_index(book_id=book_id, index_version="version-1")
        assert len(fake_client.inserted) == 1
    finally:
        await indexer.close()
        await database.close()


class BlockingGraphClient(FakeGraphClient):
    def __init__(self) -> None:
        super().__init__()
        self.poll_started = asyncio.Event()

    async def track_status(self, track_id: str) -> GraphTrackStatus:
        self.poll_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_remove_book_cancels_an_inflight_graph_index(tmp_path: Path) -> None:
    settings = Settings(
        fiction_dir=tmp_path / "fiction",
        data_dir=tmp_path / "data",
        auto_sync_on_startup=False,
        lightrag_enabled=True,
        lightrag_api_key="test-secret",
        lightrag_index_poll_interval_seconds=0.001,
    )
    settings.ensure_directories()
    database = Database(settings)
    await database.create_schema()
    fake_client = BlockingGraphClient()
    indexer = GraphIndexingService(
        settings=settings,
        session_factory=database.session_factory,
        client=cast(LightRagClient, fake_client),
    )
    try:
        async with database.session_factory() as session:
            book = Book(
                relative_path="remove.txt",
                file_format="txt",
                title="删除测试",
                content_hash="a" * 64,
                status="ready",
                active_index_version="version-1",
            )
            session.add(book)
            await session.commit()
            book_id = book.id

        await indexer.schedule(
            book_id=book_id,
            index_version="version-1",
            content_hash="a" * 64,
            book_title="删除测试",
            author=None,
            chapters=[GraphChapter(ordinal=1, title="第一章", content="正文")],
        )
        await asyncio.wait_for(fake_client.poll_started.wait(), timeout=1)
        await indexer.remove_book(book_id)
        await indexer.wait_for_idle()

        async with database.session_factory() as session:
            assert (
                await session.scalar(select(GraphIndex).where(GraphIndex.book_id == book_id))
                is None
            )
    finally:
        await indexer.close()
        await database.close()


class FakeEmbedding:
    model = "fake-embedding"
    dimension = 4

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _text in texts]


class FakeRerank:
    model = "fake-rerank"

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int
    ) -> list[RerankResult]:
        return [RerankResult(index=index, score=1.0 - index * 0.1) for index in range(top_n)]


class GraphAwareVectorStore:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(
        self,
        *,
        query: str,
        dense_vector: Sequence[float] | None,
        active_versions: Sequence[str],
        limit: int,
    ) -> list[SearchHit]:
        self.queries.append(query)
        content = (
            "甲与乙从盟友变成对手的原文。"
            if query != "甲和乙的关系如何变化？"
            else "甲最初帮助乙。"
        )
        return [
            SearchHit(
                chunk_id=f"chunk-{len(self.queries)}",
                score=0.9,
                book_id="book-1",
                book_title="测试小说",
                author=None,
                chapter_title="第一章",
                chapter_ordinal=1,
                content=content,
                start_offset=0,
                end_offset=len(content),
            )
        ]


@pytest.mark.asyncio
async def test_agent_uses_graph_only_as_scoped_qdrant_expansion(tmp_path: Path) -> None:
    settings = Settings(
        fiction_dir=tmp_path / "fiction",
        data_dir=tmp_path / "data",
        auto_sync_on_startup=False,
        embedding_dimension=4,
        lightrag_enabled=True,
        lightrag_api_key="test-secret",
        retrieval_candidate_limit=5,
        retrieval_evidence_limit=3,
    )
    fake_graph = FakeGraphClient()
    vector_store = GraphAwareVectorStore()
    agent = RagAgent(
        settings=settings,
        session_factory=cast(async_sessionmaker[AsyncSession], None),
        chat_provider=cast(ChatProvider, None),  # not used by _retrieve
        embedding_provider=FakeEmbedding(),
        rerank_provider=FakeRerank(),
        vector_store=cast(VectorStore, vector_store),
        lightrag_client=cast(LightRagClient, fake_graph),
    )

    async def progress(_stage: str, _payload: dict[str, object]) -> None:
        return None

    metrics = AnswerRunMetrics()
    state = cast(
        AgentState,
        {
            "question": "甲和乙的关系如何变化？",
            "history": [],
            "scope_mode": "books",
            "requested_book_ids": ["book-1"],
            "attempt": 0,
            "progress": progress,
            "standalone_query": "甲和乙的关系如何变化？",
            "active_versions": ["version-1"],
            "graph_source_prefixes": [graph_source_prefix("book-1", "version-1")],
            "use_graph": True,
            "graph_mode": "mix",
            "metrics": metrics,
        },
    )

    result = await agent._retrieve(state)

    assert fake_graph.queries == [("甲和乙的关系如何变化？", "mix")]
    assert vector_store.queries == [
        "甲和乙的关系如何变化？",
        "甲 乙 甲帮助乙后因误会反目",
    ]
    assert metrics.graph_calls == 1
    assert metrics.graph_used
    assert metrics.graph_queries == 1
    assert not metrics.graph_fallback
    assert len(result["evidence"]) == 2  # type: ignore[arg-type]


class FailingGraphClient(FakeGraphClient):
    async def query_data(self, query: str, *, mode: str) -> dict[str, object]:
        self.queries.append((query, mode))
        raise LightRagError("sidecar unavailable")


@pytest.mark.asyncio
async def test_agent_falls_back_when_lightrag_is_unavailable(tmp_path: Path) -> None:
    settings = Settings(
        fiction_dir=tmp_path / "fiction",
        data_dir=tmp_path / "data",
        auto_sync_on_startup=False,
        embedding_dimension=4,
        lightrag_enabled=True,
        lightrag_api_key="test-secret",
    )
    fake_graph = FailingGraphClient()
    vector_store = GraphAwareVectorStore()
    agent = RagAgent(
        settings=settings,
        session_factory=cast(async_sessionmaker[AsyncSession], None),
        chat_provider=cast(ChatProvider, None),
        embedding_provider=FakeEmbedding(),
        rerank_provider=FakeRerank(),
        vector_store=cast(VectorStore, vector_store),
        lightrag_client=cast(LightRagClient, fake_graph),
    )

    async def progress(_stage: str, _payload: dict[str, object]) -> None:
        return None

    metrics = AnswerRunMetrics()
    state = cast(
        AgentState,
        {
            "question": "甲和乙为什么反目？",
            "history": [],
            "scope_mode": "books",
            "requested_book_ids": ["book-1"],
            "attempt": 0,
            "progress": progress,
            "standalone_query": "甲和乙为什么反目？",
            "active_versions": ["version-1"],
            "graph_source_prefixes": [graph_source_prefix("book-1", "version-1")],
            "use_graph": True,
            "graph_mode": "mix",
            "metrics": metrics,
        },
    )

    result = await agent._retrieve(state)

    assert vector_store.queries == ["甲和乙为什么反目？"]
    assert result["evidence"]
    assert metrics.graph_calls == 1
    assert not metrics.graph_used
    assert metrics.graph_fallback


@pytest.mark.asyncio
@pytest.mark.parametrize("primary_outcome", ["success", "error", "cancel"])
async def test_graph_query_is_bounded_and_cleaned_up(tmp_path: Path, primary_outcome: str) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class SlowGraph(FakeGraphClient):
        async def query_data(self, query: str, *, mode: str) -> dict[str, object]:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return {}

    class PrimarySearch(GraphAwareVectorStore):
        async def search(self, **kwargs) -> list[SearchHit]:
            await started.wait()
            if primary_outcome == "error":
                raise RuntimeError("primary search failed")
            if primary_outcome == "cancel":
                raise asyncio.CancelledError
            return await super().search(**kwargs)

    settings = Settings(
        fiction_dir=tmp_path / "fiction",
        data_dir=tmp_path / "data",
        lightrag_enabled=True,
        lightrag_api_key="test-secret",
        lightrag_query_timeout_seconds=0.05,
    )
    agent = RagAgent(
        settings=settings,
        session_factory=cast(async_sessionmaker[AsyncSession], None),
        chat_provider=cast(ChatProvider, None),
        embedding_provider=FakeEmbedding(),
        rerank_provider=FakeRerank(),
        vector_store=cast(VectorStore, PrimarySearch()),
        lightrag_client=cast(LightRagClient, SlowGraph()),
    )

    async def progress(_stage: str, _payload: dict[str, object]) -> None:
        pass

    metrics = AnswerRunMetrics()
    state = cast(
        AgentState,
        {
            "question": "甲和乙为什么反目？",
            "attempt": 0,
            "progress": progress,
            "active_versions": ["version-1"],
            "graph_source_prefixes": [graph_source_prefix("book-1", "version-1")],
            "use_graph": True,
            "metrics": metrics,
        },
    )
    if primary_outcome == "error":
        with pytest.raises(RuntimeError, match="primary search failed"):
            await agent._retrieve(state)
    elif primary_outcome == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await agent._retrieve(state)
    else:
        result = await asyncio.wait_for(agent._retrieve(state), timeout=1)
        assert result["evidence"]
        assert metrics.graph_fallback
        assert not metrics.graph_used
    assert cancelled.is_set()
