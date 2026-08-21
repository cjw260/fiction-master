from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

import jieba
from qdrant_client import QdrantClient, models

from fiction_master.config import Settings
from fiction_master.ingestion.chunker import TextChunk


@dataclass(slots=True)
class SearchHit:
    chunk_id: str
    score: float
    book_id: str
    book_title: str
    author: str | None
    chapter_title: str
    chapter_ordinal: int
    content: str
    start_offset: int
    end_offset: int


TOKEN_RE = re.compile(r"[\w\u3400-\u9fff]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [
        token.casefold()
        for token in jieba.lcut(text)
        if token.strip() and TOKEN_RE.fullmatch(token.strip())
    ]


def _token_index(token: str) -> int:
    return int.from_bytes(hashlib.blake2s(token.encode("utf-8"), digest_size=4).digest(), "big")


def _bm25_sparse(text: str, *, query: bool) -> models.SparseVector:
    tokens = _tokens(text)
    if not tokens:
        return models.SparseVector(indices=[], values=[])
    frequencies = Counter(tokens)
    weights: dict[int, float] = {}
    if query:
        for token, count in frequencies.items():
            weights[_token_index(token)] = 1.0 + math.log(count)
    else:
        k1, b, average_length = 1.2, 0.75, 500.0
        length_norm = k1 * (1.0 - b + b * len(tokens) / average_length)
        for token, count in frequencies.items():
            weights[_token_index(token)] = count / (count + length_norm)
    ordered = sorted(weights.items())
    return models.SparseVector(
        indices=[index for index, _weight in ordered],
        values=[weight for _index, weight in ordered],
    )


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client_lock = asyncio.Lock()
        if settings.qdrant_url:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                timeout=60,
            )
        else:
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            # Local Qdrant is called from asyncio worker threads. Disabling SQLite's
            # thread affinity and serializing client calls below keeps that access safe.
            self.client = QdrantClient(
                path=str(settings.qdrant_path),
                force_disable_check_same_thread=True,
            )
        self.collection = settings.qdrant_collection

    async def initialize(self) -> None:
        def initialize_sync() -> None:
            if self.client.collection_exists(self.collection):
                return
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.settings.embedding_dimension,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )

        async with self._client_lock:
            await asyncio.to_thread(initialize_sync)

    async def health(self) -> bool:
        try:
            async with self._client_lock:
                await asyncio.to_thread(self.client.get_collections)
            return True
        except Exception:
            return False

    async def close(self) -> None:
        async with self._client_lock:
            await asyncio.to_thread(self.client.close)

    async def _sparse_documents(self, texts: Sequence[str]) -> list[models.SparseVector]:
        def encode() -> list[models.SparseVector]:
            return [_bm25_sparse(text, query=False) for text in texts]

        return await asyncio.to_thread(encode)

    async def _sparse_query(self, text: str) -> models.SparseVector:
        def encode() -> models.SparseVector:
            return _bm25_sparse(text, query=True)

        return await asyncio.to_thread(encode)

    async def upsert_book(
        self,
        *,
        book_id: str,
        book_title: str,
        author: str | None,
        relative_path: str,
        index_version: str,
        chunks: Sequence[TextChunk],
        dense_vectors: Sequence[Sequence[float]],
    ) -> list[str]:
        if len(chunks) != len(dense_vectors):
            raise ValueError("Chunk and vector counts do not match")
        sparse_vectors = await self._sparse_documents([chunk.content for chunk in chunks])
        point_ids = [
            str(uuid5(NAMESPACE_URL, f"fiction-master:{book_id}:{index_version}:{chunk.ordinal}"))
            for chunk in chunks
        ]
        points = [
            models.PointStruct(
                id=point_id,
                vector={"dense": list(dense), "bm25": sparse},
                payload={
                    "book_id": book_id,
                    "book_title": book_title,
                    "author": author,
                    "relative_path": relative_path,
                    "index_version": index_version,
                    "chunk_ordinal": chunk.ordinal,
                    "chapter_ordinal": chunk.chapter_ordinal,
                    "chapter_title": chunk.chapter_title,
                    "volume_title": chunk.volume_title,
                    "content": chunk.content,
                    "start_offset": chunk.start_offset,
                    "end_offset": chunk.end_offset,
                },
            )
            for point_id, chunk, dense, sparse in zip(
                point_ids, chunks, dense_vectors, sparse_vectors, strict=True
            )
        ]

        def upsert() -> None:
            for start in range(0, len(points), 256):
                self.client.upsert(
                    collection_name=self.collection,
                    points=points[start : start + 256],
                    wait=True,
                )

        async with self._client_lock:
            await asyncio.to_thread(upsert)
        return point_ids

    async def delete_index_version(self, index_version: str) -> None:
        selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="index_version", match=models.MatchValue(value=index_version)
                    )
                ]
            )
        )
        async with self._client_lock:
            await asyncio.to_thread(
                self.client.delete,
                collection_name=self.collection,
                points_selector=selector,
                wait=True,
            )

    async def search(
        self,
        *,
        query: str,
        dense_vector: Sequence[float] | None,
        active_versions: Sequence[str],
        limit: int,
    ) -> list[SearchHit]:
        if not active_versions:
            return []
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="index_version",
                    match=models.MatchAny(any=list(active_versions)),
                )
            ]
        )
        prefetch: list[models.Prefetch] = []
        if dense_vector is not None:
            prefetch.append(
                models.Prefetch(
                    query=list(dense_vector), using="dense", filter=query_filter, limit=limit
                )
            )
        try:
            sparse_query = await self._sparse_query(query)
            prefetch.append(
                models.Prefetch(query=sparse_query, using="bm25", filter=query_filter, limit=limit)
            )
        except Exception:
            if not prefetch:
                raise

        def query_sync() -> object:
            if len(prefetch) == 1:
                item = prefetch[0]
                return self.client.query_points(
                    collection_name=self.collection,
                    query=item.query,
                    using=item.using,
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                )
            return self.client.query_points(
                collection_name=self.collection,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )

        async with self._client_lock:
            response = await asyncio.to_thread(query_sync)
        points = getattr(response, "points", [])
        hits: list[SearchHit] = []
        for point in points:
            payload = point.payload or {}
            hits.append(
                SearchHit(
                    chunk_id=str(point.id),
                    score=float(point.score),
                    book_id=str(payload.get("book_id", "")),
                    book_title=str(payload.get("book_title", "")),
                    author=payload.get("author"),
                    chapter_title=str(payload.get("chapter_title", "")),
                    chapter_ordinal=int(payload.get("chapter_ordinal", 0)),
                    content=str(payload.get("content", "")),
                    start_offset=int(payload.get("start_offset", 0)),
                    end_offset=int(payload.get("end_offset", 0)),
                )
            )
        return hits
