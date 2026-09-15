from __future__ import annotations

from dataclasses import dataclass

from fiction_master.config import Settings
from fiction_master.db import Database
from fiction_master.ingestion.graph_indexer import GraphIndexingService
from fiction_master.ingestion.service import IngestionService
from fiction_master.rag.agent import RagAgent
from fiction_master.rag.lightrag import LightRagClient
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
    lightrag_client: LightRagClient | None
    graph_indexer: GraphIndexingService | None
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
        lightrag_client = LightRagClient(settings) if settings.lightrag_enabled else None
        graph_indexer = (
            GraphIndexingService(
                settings=settings,
                session_factory=database.session_factory,
                client=lightrag_client,
            )
            if lightrag_client is not None
            else None
        )
        if graph_indexer is not None:
            await graph_indexer.recover_interrupted()
        ingestion = IngestionService(
            settings=settings,
            session_factory=database.session_factory,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            graph_indexer=graph_indexer,
        )
        await ingestion.recover_interrupted_jobs()
        agent = RagAgent(
            settings=settings,
            session_factory=database.session_factory,
            chat_provider=chat_provider,
            embedding_provider=embedding_provider,
            rerank_provider=rerank_provider,
            vector_store=vector_store,
            lightrag_client=lightrag_client,
        )
        return cls(
            settings=settings,
            database=database,
            vector_store=vector_store,
            chat_provider=chat_provider,
            embedding_provider=embedding_provider,
            rerank_provider=rerank_provider,
            lightrag_client=lightrag_client,
            graph_indexer=graph_indexer,
            ingestion=ingestion,
            agent=agent,
        )

    async def close(self) -> None:
        await self.ingestion.close()
        if self.graph_indexer is not None:
            await self.graph_indexer.close()
        if self.lightrag_client is not None:
            await self.lightrag_client.close()
        await self.rerank_provider.close()
        await self.vector_store.close()
        await self.database.close()
