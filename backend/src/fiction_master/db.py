from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from fiction_master.config import Settings


class Base(DeclarativeBase):
    pass


class Database:
    def __init__(self, settings: Settings) -> None:
        self.engine: AsyncEngine = create_async_engine(
            settings.database_url,
            connect_args={"check_same_thread": False},
        )

        @event.listens_for(self.engine.sync_engine, "connect")
        def enable_sqlite_foreign_keys(dbapi_connection: Any, _connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def create_schema(self) -> None:
        from fiction_master import models  # noqa: F401

        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            # Versions before 0.1 did not enable SQLite foreign keys, so clean up
            # records that could have survived a bulk conversation deletion.
            await connection.execute(
                text(
                    "DELETE FROM citations WHERE NOT EXISTS "
                    "(SELECT 1 FROM messages WHERE messages.id = citations.message_id)"
                )
            )
            await connection.execute(
                text(
                    "DELETE FROM messages WHERE NOT EXISTS "
                    "(SELECT 1 FROM conversations "
                    "WHERE conversations.id = messages.conversation_id)"
                )
            )

    async def close(self) -> None:
        await self.engine.dispose()

    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as session:
            yield session
