from __future__ import annotations

from dataclasses import dataclass

from fiction_master.config import Settings
from fiction_master.db import Database
from fiction_master.ingestion.service import IngestionService
from fiction_master.rag.agent import RagAgent
from fiction_master.rag.providers import (
    DashScopeRerankProvider,
    OpenAICompatibleChatProvider,
    OpenAICompatibleEmbeddingProvider,
)
from fiction_master.rag.vector_store import VectorStore


@dataclass(slots=True)
class AppServices:
    settings: Settings
    database: Database
    vector_store: VectorStore
    chat_provider: OpenAICompatibleChatProvider
    embedding_provider: OpenAICompatibleEmbeddingProvider
    rerank_provider: DashScopeRerankProvider
    ingestion: IngestionService
    agent: RagAgent

    @classmethod
    async def create(cls, settings: Settings) -> AppServices:
        settings.ensure_directories()
        database = Database(settings)
        await database.create_schema()
        vector_store = VectorStore(settings)
        await vector_store.initialize()
        chat_provider = OpenAICompatibleChatProvider(settings)
        embedding_provider = OpenAICompatibleEmbeddingProvider(settings)
        rerank_provider = DashScopeRerankProvider(settings)
        ingestion = IngestionService(
            settings=settings,
            session_factory=database.session_factory,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
        )
        await ingestion.recover_interrupted_jobs()
        agent = RagAgent(
            settings=settings,
            session_factory=database.session_factory,
            chat_provider=chat_provider,
            embedding_provider=embedding_provider,
            rerank_provider=rerank_provider,
            vector_store=vector_store,
        )
        return cls(
            settings=settings,
            database=database,
            vector_store=vector_store,
            chat_provider=chat_provider,
            embedding_provider=embedding_provider,
            rerank_provider=rerank_provider,
            ingestion=ingestion,
            agent=agent,
        )

    async def close(self) -> None:
        await self.ingestion.close()
        await self.rerank_provider.close()
        await self.vector_store.close()
        await self.database.close()
