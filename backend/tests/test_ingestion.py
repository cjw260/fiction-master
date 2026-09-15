import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
from sqlalchemy import select

from fiction_master.config import Settings
from fiction_master.models import Book, GraphIndex, IngestionJob
from fiction_master.rag.lightrag import GraphTrackStatus, LightRagClient
from fiction_master.services import AppServices


class FakeEmbedding:
    model = "fake-embedding"
    dimension = 4

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, 0.0, 0.0, 0.0] for _text in texts]


class FakeGraphClient:
    def __init__(self) -> None:
        self.insert_calls = 0
        self.deleted: list[list[str]] = []

    async def insert_chapters(self, *, texts: Sequence[str], file_sources: Sequence[str]) -> str:
        assert len(texts) == len(file_sources)
        self.insert_calls += 1
        return f"track-{self.insert_calls}"

    async def track_status(self, track_id: str) -> GraphTrackStatus:
        return GraphTrackStatus(
            complete=True,
            failed=False,
            document_ids=[f"doc-{track_id}"],
        )

    async def delete_documents(self, document_ids: Sequence[str]) -> None:
        self.deleted.append(list(document_ids))


async def wait_for_job(services: AppServices, job_id: str) -> IngestionJob:
    for _ in range(200):
        async with services.database.session_factory() as session:
            job = await session.get(IngestionJob, job_id)
            if job and job.status not in {"queued", "running"}:
                return job
        await asyncio.sleep(0.02)
    raise AssertionError("ingestion job did not finish")


@pytest.mark.asyncio
async def test_incremental_sync_and_missing_file(tmp_path: Path) -> None:
    fiction_dir = tmp_path / "fiction"
    fiction_dir.mkdir()
    source = fiction_dir / "sample.txt"
    source.write_text(
        "《测试小说》\n作者：作者\n\n第一章 开始\n\n主角在雨中找到一把钥匙。",
        encoding="utf-8",
    )
    (fiction_dir / "README.md").write_text("# 素材说明\n这不是一本小说。", encoding="utf-8")
    settings = Settings(
        fiction_dir=fiction_dir,
        data_dir=tmp_path / "data",
        embedding_dimension=4,
        auto_sync_on_startup=False,
    )
    services = await AppServices.create(settings)
    fake = FakeEmbedding()
    services.ingestion.embedding_provider = fake
    try:
        first = await services.ingestion.create_sync_job()
        result = await wait_for_job(services, first.id)
        assert result.status == "succeeded"
        async with services.database.session_factory() as session:
            book = await session.scalar(select(Book).where(Book.relative_path == "sample.txt"))
            assert book is not None
            assert book.status == "ready"
            assert book.chapter_count == 1
            assert book.chunk_count == 1
            all_books = list((await session.scalars(select(Book))).all())
            assert len(all_books) == 1
        assert fake.calls == 1

        second = await services.ingestion.create_sync_job()
        await wait_for_job(services, second.id)
        assert fake.calls == 1

        source.unlink()
        third = await services.ingestion.create_sync_job()
        await wait_for_job(services, third.id)
        async with services.database.session_factory() as session:
            book = await session.scalar(select(Book).where(Book.relative_path == "sample.txt"))
            assert book is not None
            assert book.status == "missing"
            assert book.active_index_version is None
    finally:
        await services.close()


@pytest.mark.asyncio
async def test_recover_interrupted_job_and_book(tmp_path: Path) -> None:
    settings = Settings(
        fiction_dir=tmp_path / "fiction",
        data_dir=tmp_path / "data",
        embedding_dimension=4,
        auto_sync_on_startup=False,
    )
    services = await AppServices.create(settings)
    try:
        async with services.database.session_factory() as session:
            book = Book(
                relative_path="interrupted.txt",
                file_format="txt",
                title="中断测试",
                status="indexing",
                active_index_version="old-version",
            )
            job = IngestionJob(job_type="sync", status="running")
            session.add_all([book, job])
            await session.commit()
            book_id = book.id
            job_id = job.id

        await services.ingestion.recover_interrupted_jobs()

        async with services.database.session_factory() as session:
            recovered_book = await session.get(Book, book_id)
            recovered_job = await session.get(IngestionJob, job_id)
            assert recovered_book is not None
            assert recovered_book.status == "ready"
            assert recovered_book.error == "Indexing interrupted by service restart"
            assert recovered_job is not None
            assert recovered_job.status == "failed"
            assert recovered_job.finished_at is not None
    finally:
        await services.close()


@pytest.mark.asyncio
async def test_primary_sync_schedules_optional_graph_index(tmp_path: Path) -> None:
    fiction_dir = tmp_path / "fiction"
    fiction_dir.mkdir()
    (fiction_dir / "sample.txt").write_text(
        "《图谱测试》\n作者：作者\n\n第一章 相遇\n\n甲在雨中帮助了乙。",
        encoding="utf-8",
    )
    settings = Settings(
        fiction_dir=fiction_dir,
        data_dir=tmp_path / "data",
        embedding_dimension=4,
        auto_sync_on_startup=False,
        lightrag_enabled=True,
        lightrag_api_key="test-secret",
        lightrag_index_poll_interval_seconds=0.001,
        lightrag_index_book_titles=[],
    )
    services = await AppServices.create(settings)
    fake_embedding = FakeEmbedding()
    fake_graph = FakeGraphClient()
    services.ingestion.embedding_provider = fake_embedding
    assert services.graph_indexer is not None
    services.graph_indexer.client = cast(LightRagClient, fake_graph)
    try:
        job = await services.ingestion.create_sync_job()
        result = await wait_for_job(services, job.id)
        assert result.status == "succeeded"
        await services.graph_indexer.wait_for_idle()

        async with services.database.session_factory() as session:
            book = await session.scalar(select(Book).where(Book.relative_path == "sample.txt"))
            assert book is not None
            graph_index = await session.scalar(
                select(GraphIndex).where(
                    GraphIndex.book_id == book.id,
                    GraphIndex.index_version == book.active_index_version,
                )
            )
            assert graph_index is not None
            assert graph_index.status == "ready"
            assert graph_index.document_ids == ["doc-track-1"]
        assert fake_embedding.calls == 1
        assert fake_graph.insert_calls == 1

        unchanged = await services.ingestion.create_sync_job()
        await wait_for_job(services, unchanged.id)
        await services.graph_indexer.wait_for_idle()
        assert fake_embedding.calls == 1
        assert fake_graph.insert_calls == 1
    finally:
        await services.close()
