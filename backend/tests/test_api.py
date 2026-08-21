from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import httpx
import pytest
from asgi_lifespan import LifespanManager
from sqlalchemy import func, select

from fiction_master.config import Settings
from fiction_master.main import create_app
from fiction_master.models import Citation, Message
from fiction_master.rag.providers import CompletionResult


class FakeChat:
    model = "fake-chat"

    async def complete(
        self, messages: Sequence[dict[str, str]], *, temperature: float = 0.1
    ) -> CompletionResult:
        return CompletionResult(content='{"sufficient":true}', usage={})

    async def stream(
        self, messages: Sequence[dict[str, str]], *, temperature: float = 0.2
    ) -> AsyncIterator[str]:
        yield "你好，我是小说大师。"


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
                assert await session.scalar(select(func.count()).select_from(Message)) == 0
                assert await session.scalar(select(func.count()).select_from(Citation)) == 0
