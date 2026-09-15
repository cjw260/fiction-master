import json
import re
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import httpx
import pytest
from asgi_lifespan import LifespanManager
from sqlalchemy import func, select

from fiction_master.config import Settings
from fiction_master.ingestion.chunker import TextChunk
from fiction_master.main import create_app
from fiction_master.models import Book, Citation, Message
from fiction_master.rag.providers import CompletionResult, RerankResult, StreamChunk


class FakeChat:
    model = "fake-chat"

    async def complete(
        self, messages: Sequence[dict[str, str]], *, temperature: float = 0.1
    ) -> CompletionResult:
        return CompletionResult(
            content='{"sufficient":true}',
            usage={"prompt_tokens": 5, "completion_tokens": 2},
        )

    async def stream(
        self, messages: Sequence[dict[str, str]], *, temperature: float = 0.2
    ) -> AsyncIterator[StreamChunk]:
        answer = (
            '主角在雨中找到了一把钥匙。<source id="1>'
            if "<source" in messages[-1]["content"]
            else "你好，我是小说大师。"
        )
        yield StreamChunk(text=answer)
        yield StreamChunk(usage={"prompt_tokens": 7, "completion_tokens": 4})


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


@pytest.mark.asyncio
async def test_health_conversation_and_smalltalk_stream(tmp_path: Path) -> None:
    settings = Settings(
        fiction_dir=tmp_path / "fiction",
        data_dir=tmp_path / "data",
        embedding_dimension=4,
        auto_sync_on_startup=False,
        dashscope_api_key=None,
        chat_api_key=None,
        embedding_api_key=None,
        rerank_api_key=None,
    )
    app = create_app(settings)
    async with LifespanManager(app):
        app.state.services.agent.chat_provider = FakeChat()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get("/api/v1/health")
            assert health.status_code == 200
            assert health.json()["status"] == "degraded"

            created = await client.post("/api/v1/conversations", json={"scope": {"mode": "auto"}})
            assert created.status_code == 201
            conversation_id = created.json()["id"]

            response = await client.post(
                f"/api/v1/conversations/{conversation_id}/messages/stream",
                json={"content": "你好"},
            )
            assert response.status_code == 200
            assert "event: delta" in response.text
            assert "小说大师" in response.text
            assert "event: done" in response.text
            done_match = re.search(r"event: done\ndata: (.+)\n", response.text)
            assert done_match is not None
            done_message = json.loads(done_match.group(1))["message"]
            assert done_message["metrics"]["sources"] == {
                "books": 0,
                "chapters": 0,
                "evidence": 0,
            }
            assert done_message["metrics"]["retrieval"]["rounds"] == 0
            assert done_message["metrics"]["calls"] == {
                "chat": 1,
                "embedding": 0,
                "rerank": 0,
                "graph": 0,
            }
            assert done_message["metrics"]["tokens"] == {"input": 7, "output": 4}
            assert done_message["metrics"]["timing"]["first_token_ms"] is not None

            detail = await client.get(f"/api/v1/conversations/{conversation_id}")
            persisted_assistant = next(
                message for message in detail.json()["messages"] if message["role"] == "assistant"
            )
            assert persisted_assistant["metrics"] == done_message["metrics"]

            async with app.state.services.database.session_factory() as session:
                book = Book(
                    relative_path="sample.txt",
                    file_format="txt",
                    title="测试小说",
                    status="ready",
                    active_index_version="version-1",
                )
                session.add(book)
                await session.commit()
                book_id = book.id
            await app.state.services.vector_store.upsert_book(
                book_id=book_id,
                book_title="测试小说",
                author=None,
                relative_path="sample.txt",
                index_version="version-1",
                chunks=[TextChunk(1, 1, "第一章", None, "主角在雨中找到了一把钥匙。", 0, 14)],
                dense_vectors=[[1.0, 0.0, 0.0, 0.0]],
            )
            app.state.services.agent.embedding_provider = FakeEmbedding()
            app.state.services.agent.rerank_provider = FakeRerank()

            rag_conversation = await client.post(
                "/api/v1/conversations",
                json={"scope": {"mode": "books", "book_ids": [book_id]}},
            )
            rag_response = await client.post(
                f"/api/v1/conversations/{rag_conversation.json()['id']}/messages/stream",
                json={"content": "主角找到了什么？"},
            )
            rag_done_match = re.search(r"event: done\ndata: (.+)\n", rag_response.text)
            assert rag_done_match is not None
            rag_message = json.loads(rag_done_match.group(1))["message"]
            assert "<source" not in rag_response.text
            assert rag_message["content"] == "主角在雨中找到了一把钥匙。[1]"
            assert rag_message["metrics"]["sources"] == {
                "books": 1,
                "chapters": 1,
                "evidence": 1,
            }
            assert rag_message["metrics"]["retrieval"] == {
                "rounds": 1,
                "dense": True,
                "bm25": True,
                "rerank": True,
                "graph": False,
                "graph_mode": None,
                "graph_queries": 0,
                "graph_fallback": False,
            }
            assert rag_message["metrics"]["calls"] == {
                "chat": 2,
                "embedding": 1,
                "rerank": 1,
                "graph": 0,
            }
            assert rag_message["metrics"]["tokens"] == {"input": 12, "output": 6}
            assert len(rag_message["citations"]) == 1

            blank = await client.post(
                f"/api/v1/conversations/{conversation_id}/messages/stream",
                json={"content": "   "},
            )
            assert blank.status_code == 422

            async with app.state.services.database.session_factory() as session:
                assistant = await session.scalar(
                    select(Message).where(
                        Message.conversation_id == conversation_id,
                        Message.role == "assistant",
                    )
                )
                assert assistant is not None
                first_assistant_id = assistant.id
                session.add(
                    Citation(
                        message_id=assistant.id,
                        ordinal=1,
                        book_title="测试小说",
                        chapter_title="第一章",
                        chunk_id="chunk-1",
                        excerpt="测试原文",
                        start_offset=0,
                        end_offset=4,
                    )
                )
                await session.commit()

            deleted = await client.delete(f"/api/v1/conversations/{conversation_id}")
            assert deleted.status_code == 204
            async with app.state.services.database.session_factory() as session:
                assert (
                    await session.scalar(
                        select(func.count())
                        .select_from(Message)
                        .where(Message.conversation_id == conversation_id)
                    )
                    == 0
                )
                assert (
                    await session.scalar(
                        select(func.count())
                        .select_from(Citation)
                        .where(Citation.message_id == first_assistant_id)
                    )
                    == 0
                )
