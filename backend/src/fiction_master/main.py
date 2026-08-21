from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from fiction_master.api import conversations, library, system
from fiction_master.config import Settings, get_settings
from fiction_master.errors import FictionMasterError
from fiction_master.schemas import ErrorBody
from fiction_master.services import AppServices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        services = await AppServices.create(app_settings)
        app.state.services = services
        if app_settings.auto_sync_on_startup:
            await services.ingestion.create_sync_job()
        try:
            yield
        finally:
            await services.close()

    app = FastAPI(
        title="小说大师 API",
        version="0.1.0",
        description="从本地 fiction 目录构建索引并提供带原文引用的小说 RAG 问答。",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(FictionMasterError)
    async def fiction_error_handler(_request: Request, exc: FictionMasterError) -> JSONResponse:
        status_code = 404 if exc.code == "NOT_FOUND" else 400
        return JSONResponse(
            status_code=status_code,
            content=ErrorBody(
                code=exc.code, message=exc.message, retryable=exc.retryable
            ).model_dump(),
        )

    app.include_router(system.router, prefix=app_settings.api_prefix)
    app.include_router(library.router, prefix=app_settings.api_prefix)
    app.include_router(conversations.router, prefix=app_settings.api_prefix)

    front_dist = Path(__file__).resolve().parents[3] / "front" / "dist"
    if front_dist.joinpath("index.html").exists():
        app.mount("/", StaticFiles(directory=front_dist, html=True), name="frontend")
    return app


app = create_app()
