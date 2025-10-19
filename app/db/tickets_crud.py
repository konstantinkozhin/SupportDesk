from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

from sqlalchemy import delete, select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db import models


# ========== TICKETS ==========


async def get_open_ticket_by_chat_id(
    session: AsyncSession,
    telegram_chat_id: str,
) -> Optional[models.Ticket]:
    """Получить открытую заявку для данного чата."""
    result = await session.execute(
        select(models.Ticket).where(
            and_(
                models.Ticket.telegram_chat_id == telegram_chat_id,
                models.Ticket.status.in_(
                    [models.TicketStatus.OPEN, models.TicketStatus.IN_PROGRESS]
                ),
            )
        )
    )
    return result.scalar_one_or_none()


async def create_ticket(
    session: AsyncSession,
    telegram_chat_id: str,
    title: Optional[str] = None,
) -> models.Ticket:
    """Создать новую заявку."""
    ticket = models.Ticket(
        telegram_chat_id=telegram_chat_id,
        title="Временная заявка",  # Временный заголовок
        status=models.TicketStatus.OPEN,
    )
    session.add(ticket)
    await session.commit()
    await session.refresh(ticket)

    # Обновляем заголовок с номером заявки
    if title:
        ticket.title = f"Заявка #{ticket.id} - {title}"
    else:
        ticket.title = f"Заявка #{ticket.id} - Пользователь {telegram_chat_id}"
    await session.commit()

    return ticket


async def list_tickets(
    session: AsyncSession,
    *,
    status: Optional[models.TicketStatus] = None,
    archived: bool = False,
) -> list[models.Ticket]:
    """Получить список заявок."""
    stmt = (
        select(models.Ticket)
        .options(selectinload(models.Ticket.messages))
        .order_by(models.Ticket.updated_at.desc())
    )

    if status:
        stmt = stmt.where(models.Ticket.status == status)
    elif archived:
        # Архив - закрытые и архивированные заявки
        stmt = stmt.where(
            models.Ticket.status.in_(
                [models.TicketStatus.CLOSED, models.TicketStatus.ARCHIVED]
            )
        )
    else:
        # Активные заявки - только открытые и в работе
        stmt = stmt.where(
            models.Ticket.status.in_(
                [models.TicketStatus.OPEN, models.TicketStatus.IN_PROGRESS]
            )
        )

    result = await session.execute(stmt)
    return result.scalars().all()


async def get_ticket_by_id(
    session: AsyncSession,
    ticket_id: int,
) -> Optional[models.Ticket]:
    """Получить заявку по ID."""
    return await session.get(models.Ticket, ticket_id)


async def get_ticket_with_messages(
    session: AsyncSession,
    ticket_id: int,
) -> Optional[tuple[models.Ticket, list[models.Message]]]:
    """Получить заявку с сообщениями по ID."""
    # Получаем тикет
    result = await session.execute(
        select(models.Ticket).where(models.Ticket.id == ticket_id)
    )
    ticket = result.scalar_one_or_none()

    if ticket is None:
        return None

    # Получаем сообщения
    messages_result = await session.execute(
        select(models.Message)
        .where(models.Message.ticket_id == ticket_id)
        .order_by(models.Message.created_at)
    )
    messages = messages_result.scalars().all()

    return ticket, list(messages)


async def update_ticket_status(
    session: AsyncSession,
    ticket_id: int,
    status: models.TicketStatus,
) -> Optional[models.Ticket]:
    """Обновить статус заявки."""
    ticket = await session.get(models.Ticket, ticket_id)
    if ticket is None:
        return None

    ticket.status = status
    ticket.updated_at = datetime.utcnow()

    if status == models.TicketStatus.CLOSED:
        ticket.closed_at = datetime.utcnow()
    elif status == models.TicketStatus.ARCHIVED:
        if not ticket.closed_at:
            ticket.closed_at = datetime.utcnow()

    await session.commit()
    await session.refresh(ticket)
    return ticket


async def update_ticket_summary(
    session: AsyncSession,
    ticket_id: int,
    summary: str,
) -> Optional[models.Ticket]:
    """Обновить краткое описание (summary) заявки."""
    ticket = await session.get(models.Ticket, ticket_id)
    if ticket is None:
        return None

    ticket.summary = summary
    ticket.updated_at = datetime.utcnow()

    await session.commit()
    await session.refresh(ticket)
    return ticket


async def update_ticket_classification(
    session: AsyncSession,
    ticket_id: int,
    classification: str,
) -> Optional[models.Ticket]:
    """Обновить классификацию заявки."""
    ticket = await session.get(models.Ticket, ticket_id)
    if ticket is None:
        return None

    ticket.classification = classification
    ticket.updated_at = datetime.utcnow()

    await session.commit()
    await session.refresh(ticket)
    return ticket


async def mark_operator_requested(
    session: AsyncSession,
    ticket_id: int,
) -> Optional[models.Ticket]:
    """Отметить что для тикета запрошен оператор."""
    ticket = await session.get(models.Ticket, ticket_id)
    if ticket is None:
        return None

    ticket.operator_requested = True
    ticket.updated_at = datetime.utcnow()

    await session.commit()
    await session.refresh(ticket)
    return ticket


async def set_first_response_time(
    session: AsyncSession,
    ticket_id: int,
) -> Optional[models.Ticket]:
    """Записать время первого ответа оператора (если еще не записано)."""
    ticket = await session.get(models.Ticket, ticket_id)
    if ticket is None or ticket.first_response_at is not None:
        return None

    ticket.first_response_at = datetime.utcnow()
    ticket.updated_at = datetime.utcnow()

    await session.commit()
    await session.refresh(ticket)
    return ticket


# ========== MESSAGES ==========


async def add_message(
    session: AsyncSession,
    ticket_id: int,
    sender: str,
    text: str,
    telegram_message_id: Optional[int] = None,
    vk_message_id: Optional[int] = None,
    is_system: bool = False,
    created_at: Optional[datetime] = None,
    is_read: bool = False,
) -> models.Message:
    """Добавить сообщение в заявку."""
    message = models.Message(
        ticket_id=ticket_id,
        sender=sender,
        text=text,
        telegram_message_id=telegram_message_id,
        vk_message_id=vk_message_id,
        is_system=is_system,
        is_read=is_read,
    )
    if created_at:
        message.created_at = created_at
    session.add(message)

    # Обновляем время последнего обновления заявки
    ticket = await session.get(models.Ticket, ticket_id)
    if ticket:
        ticket.updated_at = datetime.utcnow()

    await session.commit()
    await session.refresh(message)
    return message


async def list_messages_for_ticket(
    session: AsyncSession,
    ticket_id: int,
    include_system: bool = True,
) -> list[models.Message]:
    """Получить все сообщения заявки."""
    stmt = select(models.Message).where(models.Message.ticket_id == ticket_id)

    if not include_system:
        stmt = stmt.where(models.Message.is_system == False)

    stmt = stmt.order_by(models.Message.created_at)
    result = await session.execute(stmt)
    return result.scalars().all()


async def mark_ticket_messages_as_read(
    session: AsyncSession,
    ticket_id: int,
) -> int:
    """Отметить все сообщения от user/bot в заявке как прочитанные."""
    from sqlalchemy import update

    stmt = (
        update(models.Message)
        .where(
            and_(
                models.Message.ticket_id == ticket_id,
                models.Message.sender.in_(["user", "bot"]),
                models.Message.is_read == False,
            )
        )
        .values(is_read=True)
    )

    result = await session.execute(stmt)
    await session.commit()
    return result.rowcount


# ========== KNOWLEDGE (DocumentChunks) ==========


async def load_all_chunks(session: AsyncSession) -> list[models.DocumentChunk]:
    """Загрузить все чанки документов из базы знаний."""
    result = await session.execute(
        select(models.DocumentChunk).order_by(models.DocumentChunk.id)
    )
    return result.scalars().all()


async def get_chunk_by_id(
    session: AsyncSession, chunk_id: int
) -> Optional[models.DocumentChunk]:
    """Получить чанк по ID."""
    result = await session.execute(
        select(models.DocumentChunk).where(models.DocumentChunk.id == chunk_id)
    )
    return result.scalar_one_or_none()


async def count_document_chunks(session: AsyncSession) -> int:
    """Подсчитать количество чанков в базе знаний."""
    result = await session.execute(
        select(func.count()).select_from(models.DocumentChunk)
    )
    count = result.scalar_one()
    return int(count or 0)


async def delete_chunks_by_source(session: AsyncSession, source_file: str) -> int:
    """Удалить все чанки из конкретного файла."""
    result = await session.execute(
        delete(models.DocumentChunk).where(
            models.DocumentChunk.source_file == source_file
        )
    )
    await session.commit()
    return result.rowcount


async def add_document_chunk(
    session: AsyncSession,
    content: str,
    source_file: str,
    chunk_index: int,
    start_char: int = 0,
    end_char: int = 0,
    embedding: bytes | None = None,
    chunk_metadata: str | None = None,
) -> models.DocumentChunk:
    """Добавить чанк документа в базу знаний."""
    chunk = models.DocumentChunk(
        content=content,
        source_file=source_file,
        chunk_index=chunk_index,
        start_char=start_char,
        end_char=end_char,
        embedding=embedding,
        chunk_metadata=chunk_metadata,
    )
    session.add(chunk)
    await session.commit()
    await session.refresh(chunk)
    return chunk


async def add_document_chunks(
    session: AsyncSession, chunks_data: list[tuple]
) -> list[models.DocumentChunk]:
    """Добавить несколько чанков документов в базу знаний.

    Args:
        chunks_data: список кортежей (content, source_file, chunk_index, start_char, end_char, embedding)
    """
    chunks = []
    for chunk_tuple in chunks_data:
        # Распаковываем кортеж
        content, source_file, chunk_index, start_char, end_char, embedding = chunk_tuple

        chunk = models.DocumentChunk(
            content=content,
            source_file=source_file,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            embedding=embedding,
            chunk_metadata=None,
        )
        session.add(chunk)
        chunks.append(chunk)

    await session.commit()

    # Обновляем все чанки после commit
    for chunk in chunks:
        await session.refresh(chunk)

    return chunks


async def delete_all_chunks(session: AsyncSession) -> int:
    """Удалить все чанки из базы знаний."""
    result = await session.execute(delete(models.DocumentChunk))
    await session.commit()
    return result.rowcount


# ========== DASHBOARD STATISTICS ==========


async def get_tickets_stats(session: AsyncSession) -> dict:
    """Получить статистику по тикетам для дашборда."""
    # Подсчет тикетов по статусам
    status_stats = {}
    for status in [
        models.TicketStatus.OPEN,
        models.TicketStatus.IN_PROGRESS,
        models.TicketStatus.CLOSED,
        models.TicketStatus.ARCHIVED,
    ]:
        result = await session.execute(
            select(func.count())
            .select_from(models.Ticket)
            .where(models.Ticket.status == status)
        )
        status_stats[status.value] = result.scalar_one() or 0

    # Общее количество тикетов
    total_result = await session.execute(
        select(func.count()).select_from(models.Ticket)
    )
    total_tickets = total_result.scalar_one() or 0

    # Тикеты за последние 7 дней
    from datetime import datetime, timedelta

    week_ago = datetime.utcnow() - timedelta(days=7)
    week_result = await session.execute(
        select(func.count())
        .select_from(models.Ticket)
        .where(models.Ticket.created_at >= week_ago)
    )
    tickets_this_week = week_result.scalar_one() or 0

    # Тикеты за сегодня
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_result = await session.execute(
        select(func.count())
        .select_from(models.Ticket)
        .where(models.Ticket.created_at >= today)
    )
    tickets_today = today_result.scalar_one() or 0

    return {
        "total_tickets": total_tickets,
        "tickets_today": tickets_today,
        "tickets_this_week": tickets_this_week,
        "status_distribution": status_stats,
    }


async def get_response_time_stats(session: AsyncSession) -> dict:
    """Получить статистику времени ответа."""
    # Получаем закрытые тикеты с их сообщениями
    query = select(models.Ticket).where(
        models.Ticket.status.in_(
            [models.TicketStatus.CLOSED, models.TicketStatus.ARCHIVED]
        )
    )

    result = await session.execute(query)
    tickets = result.scalars().all()

    response_times = []
    for ticket in tickets:
        # Получаем сообщения для каждого тикета отдельно
        messages_result = await session.execute(
            select(models.Message)
            .where(models.Message.ticket_id == ticket.id)
            .order_by(models.Message.created_at)
        )
        messages = messages_result.scalars().all()

        user_message = None
        bot_response = None

        for message in messages:
            if message.sender == "user" and user_message is None:
                user_message = message
            elif (
                message.sender in ["bot", "operator"]
                and user_message
                and bot_response is None
            ):
                bot_response = message
                break

        if user_message and bot_response:
            response_time = (
                bot_response.created_at - user_message.created_at
            ).total_seconds() / 60  # в минутах
            response_times.append(response_time)

    if not response_times:
        return {"avg_response_time": 0, "min_response_time": 0, "max_response_time": 0}

    return {
        "avg_response_time": sum(response_times) / len(response_times),
        "min_response_time": min(response_times),
        "max_response_time": max(response_times),
    }


async def get_daily_tickets_stats(session: AsyncSession, days: int = 30) -> list[dict]:
    """Получить статистику тикетов по дням за последние N дней."""
    from datetime import datetime, timedelta

    end_date = datetime.utcnow().replace(
        hour=23, minute=59, second=59, microsecond=999999
    )
    start_date = end_date - timedelta(days=days - 1)
    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

    daily_stats = []
    current_date = start_date

    while current_date <= end_date:
        next_date = current_date + timedelta(days=1)

        result = await session.execute(
            select(func.count())
            .select_from(models.Ticket)
            .where(
                and_(
                    models.Ticket.created_at >= current_date,
                    models.Ticket.created_at < next_date,
                )
            )
        )
        count = result.scalar_one() or 0

        daily_stats.append(
            {"date": current_date.strftime("%Y-%m-%d"), "tickets_count": count}
        )

        current_date = next_date

    return daily_stats


async def get_daily_time_metrics(session: AsyncSession, days: int = 30) -> list[dict]:
    """Получить средние времена реакции и обработки по дням за последние N дней."""
    from datetime import datetime, timedelta

    end_date = datetime.utcnow().replace(
        hour=23, minute=59, second=59, microsecond=999999
    )
    start_date = end_date - timedelta(days=days - 1)
    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

    result = await session.execute(
        select(models.Ticket).where(
            and_(
                models.Ticket.created_at >= start_date,
                models.Ticket.created_at <= end_date,
            )
        )
    )
    tickets = result.scalars().all()

    buckets: dict[str, dict[str, list[float]]] = {}

    for ticket in tickets:
        day_key = ticket.created_at.date().isoformat()
        bucket = buckets.setdefault(day_key, {"response": [], "resolution": []})

        if ticket.first_response_at:
            response_delta = (
                ticket.first_response_at - ticket.created_at
            ).total_seconds() / 60
            if response_delta >= 0:
                bucket["response"].append(response_delta)

        if ticket.closed_at:
            resolution_delta = (
                ticket.closed_at - ticket.created_at
            ).total_seconds() / 60
            if resolution_delta >= 0:
                bucket["resolution"].append(resolution_delta)

    output: list[dict[str, object]] = []
    current_date = start_date

    from math import isfinite

    while current_date <= end_date:
        day_key = current_date.date().isoformat()
        bucket = buckets.get(day_key, {"response": [], "resolution": []})

        def average(values: list[float]) -> float | None:
            if not values:
                return None
            total = sum(values)
            count = len(values)
            mean = total / count if count else None
            return round(mean, 1) if mean is not None and isfinite(mean) else None

        output.append(
            {
                "date": day_key,
                "avg_response_minutes": average(bucket["response"]),
                "avg_resolution_minutes": average(bucket["resolution"]),
            }
        )

        current_date += timedelta(days=1)

    return output


async def get_average_response_time(session: AsyncSession) -> Optional[float]:
    """
    Получить среднее время отклика (в минутах) - от создания заявки до первого ответа оператора.
    Возвращает None если нет данных.
    """
    result = await session.execute(
        select(
            func.avg(
                func.julianday(models.Ticket.first_response_at)
                - func.julianday(models.Ticket.created_at)
            )
            * 24
            * 60  # Конвертируем дни в минуты
        ).where(models.Ticket.first_response_at.isnot(None))
    )

    avg_minutes = result.scalar_one()
    return round(avg_minutes, 1) if avg_minutes else None


async def get_average_resolution_time(session: AsyncSession) -> Optional[float]:
    """
    Получить среднее время обработки заявки (в минутах) - от создания до закрытия.
    Возвращает None если нет данных.
    """
    result = await session.execute(
        select(
            func.avg(
                func.julianday(models.Ticket.closed_at)
                - func.julianday(models.Ticket.created_at)
            )
            * 24
            * 60  # Конвертируем дни в минуты
        ).where(models.Ticket.closed_at.isnot(None))
    )

    avg_minutes = result.scalar_one()
    return round(avg_minutes, 1) if avg_minutes else None


# ========== DOCUMENT CHUNKS - Случайный выбор ==========


async def get_random_chunk(
    session: AsyncSession,
) -> Optional[models.DocumentChunk]:
    """Получить случайный чанк документа из базы знаний."""
    result = await session.execute(
        select(models.DocumentChunk).order_by(func.random()).limit(1)
    )
    return result.scalar_one_or_none()
