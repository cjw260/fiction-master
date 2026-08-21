import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest
from sqlalchemy import select

from fiction_master.config import Settings
from fiction_master.models import Book, IngestionJob
from fiction_master.services import AppServices


class FakeEmbedding:
    model = "fake-embedding"
    dimension = 4

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, 0.0, 0.0, 0.0] for _text in texts]


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
