from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy import select

from fiction_master.api.dependencies import get_services
from fiction_master.errors import NotFoundError
from fiction_master.models import Book, GraphIndex, IngestionJob
from fiction_master.schemas import BookResponse, JobAccepted, JobResponse
from fiction_master.services import AppServices

router = APIRouter(tags=["library"])


@router.get("/books", response_model=list[BookResponse])
async def list_books(services: AppServices = Depends(get_services)) -> list[BookResponse]:
    async with services.database.session_factory() as session:
        books = list((await session.scalars(select(Book).order_by(Book.title))).all())
        graph_rows = (
            list((await session.scalars(select(GraphIndex))).all())
            if services.settings.lightrag_enabled
            else []
        )
    graph_by_version = {
        (row.book_id, row.index_version): (row.status, row.error)
        for row in sorted(graph_rows, key=lambda item: item.updated_at)
    }
    return [
        BookResponse.model_validate(book).model_copy(
            update={
                "graph_status": (
                    graph_by_version.get((book.id, book.active_index_version), ("pending", None))[0]
                    if services.settings.lightrag_enabled and book.active_index_version
                    else "disabled"
                ),
                "graph_error": (
                    graph_by_version.get((book.id, book.active_index_version), ("pending", None))[1]
                    if services.settings.lightrag_enabled and book.active_index_version
                    else None
                ),
            }
        )
        for book in books
    ]


@router.post("/library/sync", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def sync_library(services: AppServices = Depends(get_services)) -> JobAccepted:
    job = await services.ingestion.create_sync_job()
    return JobAccepted(job_id=job.id)


@router.post(
    "/books/{book_id}/reindex", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED
)
async def reindex_book(book_id: str, services: AppServices = Depends(get_services)) -> JobAccepted:
    job = await services.ingestion.create_reindex_job(book_id)
    return JobAccepted(job_id=job.id)


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, services: AppServices = Depends(get_services)) -> JobResponse:
    async with services.database.session_factory() as session:
        job = await session.get(IngestionJob, job_id)
        if job is None:
            raise NotFoundError("job")
    return JobResponse.model_validate(job)
