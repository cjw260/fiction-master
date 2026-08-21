from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text

from fiction_master import __version__
from fiction_master.api.dependencies import get_services
from fiction_master.schemas import HealthResponse, ModelStatus
from fiction_master.services import AppServices

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(services: AppServices = Depends(get_services)) -> HealthResponse:
    database_ok = True
    try:
        async with services.database.session_factory() as session:
            await session.execute(text("SELECT 1"))
    except Exception:
        database_ok = False
    vector_ok = await services.vector_store.health()
    models_configured = bool(
        services.settings.resolved_chat_key and services.settings.resolved_embedding_key
    )
    return HealthResponse(
        status="ok" if database_ok and vector_ok and models_configured else "degraded",
        database=database_ok,
        vector_store=vector_ok,
        models_configured=models_configured,
        version=__version__,
    )


@router.get("/system/models", response_model=ModelStatus)
async def models(services: AppServices = Depends(get_services)) -> ModelStatus:
    settings = services.settings
    return ModelStatus(
        chat_model=settings.chat_model,
        embedding_model=settings.embedding_model,
        rerank_model=settings.rerank_model,
        chat_configured=bool(settings.resolved_chat_key),
        embedding_configured=bool(settings.resolved_embedding_key),
        rerank_configured=bool(settings.resolved_rerank_key),
    )
