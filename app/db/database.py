from __future__ import annotations

import contextlib
from typing import AsyncIterator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# База данных для заявок и сообщений
TICKETS_DATABASE_URL = "sqlite+aiosqlite:///./db/tickets.db"
# База данных для знаний
KNOWLEDGE_DATABASE_URL = "sqlite+aiosqlite:///./db/knowledge.db"

# Движки баз данных
tickets_engine = create_async_engine(TICKETS_DATABASE_URL, echo=False, future=True)
knowledge_engine = create_async_engine(KNOWLEDGE_DATABASE_URL, echo=False, future=True)

# Сессии
TicketsSessionLocal = async_sessionmaker(tickets_engine, expire_on_commit=False)
KnowledgeSessionLocal = async_sessionmaker(knowledge_engine, expire_on_commit=False)


async def init_db() -> None:
    """Создание таблиц баз данных при запуске приложения."""
    from app.db.models import Base, KnowledgeBase

    # Создаем таблицы для заявок
    async with tickets_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Создаем таблицы для базы знаний
    async with knowledge_engine.begin() as conn:
        await conn.run_sync(KnowledgeBase.metadata.create_all)


@contextlib.asynccontextmanager
async def lifespan_tickets_session() -> AsyncIterator[AsyncSession]:
    """Сессия для работы с заявками в рамках жизненного цикла FastAPI."""
    async with TicketsSessionLocal() as session:
        yield session


@contextlib.asynccontextmanager
async def lifespan_knowledge_session() -> AsyncIterator[AsyncSession]:
    """Сессия для работы с базой знаний в рамках жизненного цикла FastAPI."""
    async with KnowledgeSessionLocal() as session:
        yield session


async def get_tickets_session() -> AsyncIterator[AsyncSession]:
    """Получить сессию для работы с заявками."""
    async with TicketsSessionLocal() as session:
        yield session


async def get_knowledge_session() -> AsyncIterator[AsyncSession]:
    """Получить сессию для работы с базой знаний."""
    async with KnowledgeSessionLocal() as session:
        yield session
