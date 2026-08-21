from pathlib import Path

import pytest

from fiction_master.config import Settings
from fiction_master.ingestion.chunker import TextChunk
from fiction_master.rag.vector_store import VectorStore


@pytest.mark.asyncio
async def test_local_hybrid_vector_store(tmp_path: Path) -> None:
    settings = Settings(
        fiction_dir=tmp_path / "fiction",
        data_dir=tmp_path / "data",
        embedding_dimension=4,
        auto_sync_on_startup=False,
    )
    store = VectorStore(settings)
    await store.initialize()
    try:
        chunks = [
            TextChunk(1, 1, "第一章", None, "林舟得到一枚星纹铜片。", 0, 12),
            TextChunk(2, 2, "第二章", None, "港口停着一艘白色帆船。", 20, 33),
        ]
        await store.upsert_book(
            book_id="book-1",
            book_title="星海纪事",
            author="测试作者",
            relative_path="sample.txt",
            index_version="version-1",
            chunks=chunks,
            dense_vectors=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        )
        hits = await store.search(
            query="星纹铜片",
            dense_vector=[1.0, 0.0, 0.0, 0.0],
            active_versions=["version-1"],
            limit=5,
        )
        assert hits
        assert hits[0].chapter_title == "第一章"
        assert "铜片" in hits[0].content
    finally:
        await store.close()
