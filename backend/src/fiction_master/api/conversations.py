from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, select
from sqlalchemy.orm import selectinload

from fiction_master.api.dependencies import get_services
from fiction_master.errors import FictionMasterError, NotFoundError
from fiction_master.models import Citation, Conversation, Message, utc_now
from fiction_master.rag.agent import PreparedAnswer, cited_evidence
from fiction_master.schemas import (
    ChatRequest,
    CitationResponse,
    ConversationCreate,
    ConversationDetail,
    ConversationResponse,
    ConversationUpdate,
    MessageResponse,
)
from fiction_master.services import AppServices

router = APIRouter(tags=["conversations"])


def sse(event: str, data: dict[str, object]) -> bytes:
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {encoded}\n\n".encode()


@router.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations(
    services: AppServices = Depends(get_services),
) -> list[ConversationResponse]:
    async with services.database.session_factory() as session:
        conversations = list(
            (
                await session.scalars(select(Conversation).order_by(Conversation.updated_at.desc()))
            ).all()
        )
    return [ConversationResponse.model_validate(item) for item in conversations]


@router.post(
    "/conversations", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED
)
async def create_conversation(
    body: ConversationCreate, services: AppServices = Depends(get_services)
) -> ConversationResponse:
    conversation = Conversation(
        title=body.title or "新对话",
        scope_mode=body.scope.mode,
        book_ids=body.scope.book_ids,
    )
    async with services.database.session_factory() as session:
        session.add(conversation)
        await session.commit()
        await session.refresh(conversation)
    return ConversationResponse.model_validate(conversation)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: str, services: AppServices = Depends(get_services)
) -> ConversationDetail:
    async with services.database.session_factory() as session:
        conversation = await session.scalar(
            select(Conversation)
            .where(Conversation.id == conversation_id)
            .options(selectinload(Conversation.messages).selectinload(Message.citations))
        )
        if conversation is None:
            raise NotFoundError("conversation")
        return ConversationDetail(
            **ConversationResponse.model_validate(conversation).model_dump(),
            messages=[MessageResponse.model_validate(message) for message in conversation.messages],
        )


@router.patch("/conversations/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: str,
    body: ConversationUpdate,
    services: AppServices = Depends(get_services),
) -> ConversationResponse:
    async with services.database.session_factory() as session:
        conversation = await session.get(Conversation, conversation_id)
        if conversation is None:
            raise NotFoundError("conversation")
        if body.title is not None:
            conversation.title = body.title.strip()
        if body.scope is not None:
            conversation.scope_mode = body.scope.mode
            conversation.book_ids = body.scope.book_ids
        conversation.updated_at = utc_now()
        await session.commit()
        await session.refresh(conversation)
    return ConversationResponse.model_validate(conversation)


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str, services: AppServices = Depends(get_services)
) -> None:
    async with services.database.session_factory() as session:
        result = await session.execute(
            delete(Conversation).where(Conversation.id == conversation_id)
        )
        if result.rowcount == 0:  # type: ignore[attr-defined]
            raise NotFoundError("conversation")
        await session.commit()


async def _save_failed_message(
    services: AppServices, message_id: str, error: str, status_value: str = "failed"
) -> None:
    async with services.database.session_factory() as session:
        message = await session.get(Message, message_id)
        if message is None:
            return
        message.status = status_value
        message.error = error[:4000]
        await session.commit()


@router.post("/conversations/{conversation_id}/messages/stream")
async def stream_message(
    conversation_id: str,
    body: ChatRequest,
    request: Request,
    services: AppServices = Depends(get_services),
) -> StreamingResponse:
    async with services.database.session_factory() as session:
        conversation = await session.get(Conversation, conversation_id)
        if conversation is None:
            raise NotFoundError("conversation")
        scope_mode = body.scope.mode if body.scope else conversation.scope_mode
        book_ids = body.scope.book_ids if body.scope else list(conversation.book_ids)
        history_rows = list(
            (
                await session.scalars(
                    select(Message)
                    .where(
                        Message.conversation_id == conversation_id,
                        Message.status == "completed",
                    )
                    .order_by(Message.created_at.desc())
                    .limit(8)
                )
            ).all()
        )
        history = [
            {"role": message.role, "content": message.content}
            for message in reversed(history_rows)
            if message.role in {"user", "assistant"}
        ]
        user_message = Message(
            conversation_id=conversation_id,
            role="user",
            status="completed",
            content=body.content.strip(),
        )
        assistant_message = Message(
            conversation_id=conversation_id,
            role="assistant",
            status="streaming",
            content="",
            model=services.settings.chat_model,
        )
        session.add_all([user_message, assistant_message])
        if conversation.title == "新对话":
            conversation.title = body.content.strip()[:24]
        conversation.updated_at = utc_now()
        await session.commit()
        await session.refresh(assistant_message)
        assistant_id = assistant_message.id

    async def generate() -> AsyncIterator[bytes]:
        started = time.perf_counter()
        answer_parts: list[str] = []
        prepare_task: asyncio.Task[PreparedAnswer] | None = None
        yield sse("status", {"run_id": assistant_id, "stage": "accepted", "detail": "已接收问题"})
        try:
            progress_queue: asyncio.Queue[tuple[str, dict[str, object]]] = asyncio.Queue()

            async def progress(stage: str, payload: dict[str, object]) -> None:
                await progress_queue.put((stage, payload))

            prepare_task = asyncio.create_task(
                services.agent.prepare(
                    question=body.content.strip(),
                    history=history,
                    scope_mode=scope_mode,
                    requested_book_ids=book_ids,
                    progress=progress,
                )
            )
            while not prepare_task.done():
                try:
                    stage_name, payload = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                    yield sse("status", {"run_id": assistant_id, "stage": stage_name, **payload})
                except TimeoutError:
                    if await request.is_disconnected():
                        prepare_task.cancel()
                        raise asyncio.CancelledError from None
            while not progress_queue.empty():
                stage_name, payload = progress_queue.get_nowait()
                yield sse("status", {"run_id": assistant_id, "stage": stage_name, **payload})
            prepared: PreparedAnswer = await prepare_task

            yield sse(
                "status",
                {"run_id": assistant_id, "stage": "writing", "detail": "正在组织回答"},
            )
            if not prepared.no_retrieval and not prepared.evidence:
                fallback = "我没有在当前已索引的小说原文中找到足够依据，暂时无法可靠回答这个问题。"
                answer_parts.append(fallback)
                yield sse("delta", {"run_id": assistant_id, "text": fallback})
            else:
                async for text_part in services.agent.stream_answer(prepared):
                    if await request.is_disconnected():
                        raise asyncio.CancelledError
                    answer_parts.append(text_part)
                    yield sse("delta", {"run_id": assistant_id, "text": text_part})

            answer = "".join(answer_parts).strip()
            selected_citations = cited_evidence(answer, prepared.evidence)
            citation_models: list[Citation] = []
            async with services.database.session_factory() as session:
                assistant = await session.get(Message, assistant_id)
                if assistant is None:
                    raise NotFoundError("message")
                assistant.status = "completed"
                assistant.content = answer
                assistant.latency_ms = int((time.perf_counter() - started) * 1000)
                for ordinal, hit in selected_citations:
                    citation = Citation(
                        message_id=assistant_id,
                        ordinal=ordinal,
                        book_id=hit.book_id,
                        book_title=hit.book_title,
                        chapter_title=hit.chapter_title,
                        chunk_id=hit.chunk_id,
                        excerpt=hit.content[: services.settings.citation_excerpt_chars],
                        start_offset=hit.start_offset,
                        end_offset=hit.end_offset,
                    )
                    session.add(citation)
                    citation_models.append(citation)
                await session.commit()
                for citation in citation_models:
                    await session.refresh(citation)
                await session.refresh(assistant)
                message_payload = MessageResponse(
                    id=assistant.id,
                    conversation_id=assistant.conversation_id,
                    role=assistant.role,
                    status=assistant.status,
                    content=assistant.content,
                    model=assistant.model,
                    usage=assistant.usage,
                    latency_ms=assistant.latency_ms,
                    error=assistant.error,
                    created_at=assistant.created_at,
                    citations=[CitationResponse.model_validate(item) for item in citation_models],
                ).model_dump(mode="json")
            for citation in citation_models:
                yield sse(
                    "citation",
                    CitationResponse.model_validate(citation).model_dump(mode="json"),
                )
            yield sse("done", {"run_id": assistant_id, "message": message_payload})
        except asyncio.CancelledError:
            if prepare_task is not None and not prepare_task.done():
                prepare_task.cancel()
                await asyncio.gather(prepare_task, return_exceptions=True)
            await asyncio.shield(
                _save_failed_message(services, assistant_id, "Client disconnected", "cancelled")
            )
            raise
        except FictionMasterError as exc:
            await _save_failed_message(services, assistant_id, exc.message)
            yield sse(
                "error",
                {
                    "run_id": assistant_id,
                    "code": exc.code,
                    "message": exc.message,
                    "retryable": exc.retryable,
                },
            )
        except Exception as exc:
            await _save_failed_message(services, assistant_id, str(exc))
            yield sse(
                "error",
                {
                    "run_id": assistant_id,
                    "code": "INTERNAL_ERROR",
                    "message": "回答生成失败，请稍后重试",
                    "retryable": True,
                },
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
