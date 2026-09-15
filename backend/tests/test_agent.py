from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import cast

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from fiction_master.config import Settings
from fiction_master.rag.agent import (
    AgentState,
    RagAgent,
    SourceCitationStreamNormalizer,
    normalize_source_citations,
)
from fiction_master.rag.metrics import AnswerRunMetrics
from fiction_master.rag.providers import CompletionResult, RerankResult, StreamChunk
from fiction_master.rag.vector_store import SearchHit, VectorStore


def test_normalize_source_citations_accepts_malformed_model_output() -> None:
    answer = (
        '如<source id="1">、<source id="3>、<source id=4>、'
        "<source id='6' book='测试'>所示，结论成立。</source>"
    )

    assert normalize_source_citations(answer) == "如[1]、[3]、[4]、[6]所示，结论成立。"


def test_stream_normalizer_holds_source_tags_split_across_chunks() -> None:
    normalizer = SourceCitationStreamNormalizer()
    chunks = ["依据 <sou", 'rce id="5', '">原文</source', ">。"]

    answer = "".join(normalizer.feed(chunk) for chunk in chunks) + normalizer.flush()

    assert answer == "依据 [5]原文。"


def _hit(chunk_id: str, content: str, score: float = 0.9) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        score=score,
        book_id="book-1",
        book_title="测试小说",
        author=None,
        chapter_title="第二百七十六章",
        chapter_ordinal=276,
        content=content,
        start_offset=0,
        end_offset=len(content),
    )


class MultiQueryChat:
    model = "fake-chat"

    async def complete(
        self, messages: Sequence[dict[str, str]], *, temperature: float = 0.1
    ) -> CompletionResult:
        return CompletionResult(
            content=(
                '{"sufficient":false,"revised_queries":['
                '"月关 死亡 结局","月关 最后一次出场","唐三 月关 最后一战"]}'
            ),
            usage={"prompt_tokens": 10, "completion_tokens": 6},
        )

    async def stream(
        self, messages: Sequence[dict[str, str]], *, temperature: float = 0.2
    ) -> AsyncIterator[StreamChunk]:
        if False:
            yield StreamChunk()


class RecordingEmbedding:
    model = "fake-embedding"
    dimension = 4

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[1.0, 0.0, 0.0, 0.0] for _text in texts]


class RecordingRerank:
    model = "fake-rerank"

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int
    ) -> list[RerankResult]:
        self.queries.append(query)
        ordered = sorted(
            range(len(documents)),
            key=lambda index: "连尸体也没有留下" not in documents[index],
        )
        return [
            RerankResult(index=index, score=1.0 - rank * 0.1)
            for rank, index in enumerate(ordered[:top_n])
        ]


class RecordingVectorStore:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.escape = _hit("escape", "菊斗罗月关试图逃遁，随后向唐三求饶。")
        self.death = _hit("death", "唐三感觉到月关连尸体也没有留下。")

    async def search(
        self,
        *,
        query: str,
        dense_vector: Sequence[float] | None,
        active_versions: Sequence[str],
        limit: int,
    ) -> list[SearchHit]:
        self.queries.append(query)
        if query == "菊斗罗死了吗":
            return [self.escape]
        return [self.death, self.escape]


@pytest.mark.asyncio
async def test_insufficient_first_round_triggers_batched_multi_query(tmp_path: Path) -> None:
    embedding = RecordingEmbedding()
    rerank = RecordingRerank()
    vector_store = RecordingVectorStore()
    settings = Settings(
        fiction_dir=tmp_path / "fiction",
        data_dir=tmp_path / "data",
        auto_sync_on_startup=False,
        embedding_dimension=4,
        retrieval_candidate_limit=5,
        retrieval_evidence_limit=3,
    )
    agent = RagAgent(
        settings=settings,
        session_factory=cast(async_sessionmaker[AsyncSession], None),
        chat_provider=MultiQueryChat(),
        embedding_provider=embedding,
        rerank_provider=rerank,
        vector_store=cast(VectorStore, vector_store),
    )
    progress_events: list[tuple[str, dict[str, object]]] = []

    async def progress(stage: str, payload: dict[str, object]) -> None:
        progress_events.append((stage, payload))

    metrics = AnswerRunMetrics()
    first_state = cast(
        AgentState,
        {
            "question": "菊斗罗死了吗",
            "history": [],
            "scope_mode": "books",
            "requested_book_ids": ["book-1"],
            "attempt": 0,
            "progress": progress,
            "standalone_query": "菊斗罗死了吗",
            "active_versions": ["version-1"],
            "metrics": metrics,
        },
    )

    first_retrieval = await agent._retrieve(first_state)
    assert vector_store.queries == ["菊斗罗死了吗"]
    assert embedding.batches == [["菊斗罗死了吗"]]

    graded_state = cast(AgentState, {**first_state, **first_retrieval})
    grade = await agent._grade(graded_state)
    retry_state = cast(AgentState, {**graded_state, **grade})
    assert agent._after_grade(retry_state) == "retry"
    assert retry_state.get("revised_queries") == [
        "月关 死亡 结局",
        "月关 最后一次出场",
        "唐三 月关 最后一战",
    ]

    second_retrieval = await agent._retrieve(retry_state)

    assert embedding.batches[1] == [
        "菊斗罗死了吗",
        "月关 死亡 结局",
        "月关 最后一次出场",
        "唐三 月关 最后一战",
    ]
    assert vector_store.queries[1:] == embedding.batches[1]
    assert second_retrieval["evidence"][0].chunk_id == "death"  # type: ignore[index,union-attr]
    assert len({hit.chunk_id for hit in second_retrieval["hits"]}) == 2  # type: ignore[union-attr]
    assert metrics.retrieval_rounds == 2
    assert metrics.embedding_calls == 2
    assert metrics.rerank_calls == 2
    assert metrics.chat_calls == 1
    assert any(payload.get("query_count") == 4 for _stage, payload in progress_events)
