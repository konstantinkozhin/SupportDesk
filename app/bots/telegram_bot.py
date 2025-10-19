from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from collections import defaultdict

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
import html

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db import tickets_crud as crud, models
from app.config import load_telegram_responses
from app.rag import RAGResult, RAGService
from app.services import ConnectionManager
from app.db import TicketRead, MessageRead

logger = logging.getLogger(__name__)

USER_SENDER = "user"
BOT_SENDER = "bot"
OPERATOR_REQUEST_CALLBACK = "request_operator"

# Словарь блокировок для каждого пользователя (предотвращение спама)
user_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

# Глобальная ссылка на бот для отправки сообщений из agent_tools
_bot_instance: Bot | None = None
_session_maker: async_sessionmaker[AsyncSession] | None = None


def set_bot_instance(bot: Bot, session_maker: async_sessionmaker[AsyncSession]) -> None:
    """Установить глобальную ссылку на бот для использования в agent_tools"""
    global _bot_instance, _session_maker
    _bot_instance = bot
    _session_maker = session_maker
    logger.info("Bot instance set for agent tools")


REQUEST_OPERATOR_KEYBOARD = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(
                text="Позвать оператора", callback_data=OPERATOR_REQUEST_CALLBACK
            )
        ]
    ]
)


def _extract_title(message: Message = None, user_obj=None) -> str:
    """Извлекает имя пользователя из сообщения или объекта пользователя"""
    user = user_obj if user_obj else (message.from_user if message else None)
    if user:
        if user.full_name:
            return user.full_name
        if user.username:
            return f"@{user.username}"
        if user.first_name:
            return user.first_name

    # Fallback к chat_id если нет информации о пользователе
    if message:
        return f"Пользователь {message.chat.id}"
    return "Неизвестный пользователь"


async def send_agent_action_to_telegram(chat_id: int, action_text: str) -> None:
    """Отправляет действие агента в Telegram и сохраняет в БД

    Используется из agent_tools для отправки сообщений о действиях агента.

    Args:
        chat_id: ID чата Telegram
        action_text: Текст действия (например, "🔍 Поиск в базе знаний...")
    """
    global _bot_instance, _session_maker

    if not _bot_instance or not _session_maker:
        logger.warning("Bot instance not set, cannot send agent action")
        return

    try:
        # Отправляем сообщение в Telegram (без таймаута чтобы избежать проблем с context manager)
        try:
            await _bot_instance.send_message(
                chat_id=chat_id,
                text=f"<i>{action_text}</i>",  # Курсивом для отличия от обычных сообщений
                parse_mode="HTML",
                request_timeout=5,  # Явный таймаут вместо context manager
            )
        except Exception as send_error:
            logger.warning(f"Failed to send message to Telegram: {send_error}")
            # Продолжаем сохранять в БД даже если отправка не удалась

        # Сохраняем в БД (автоматически попадёт в веб через broadcast)
        async with _session_maker() as session:
            from app.main import connection_manager

            ticket = await crud.get_open_ticket_by_chat_id(session, chat_id)
            if ticket:
                # Проверяем, есть ли активные соединения для этого тикета
                should_mark_as_read = connection_manager.has_active_chat_connections(
                    ticket.id
                )

                # Добавляем сообщение как от бота
                db_message = await crud.add_message(
                    session,
                    ticket_id=ticket.id,
                    sender=BOT_SENDER,
                    text=action_text,
                    is_read=should_mark_as_read,
                )

                await session.refresh(db_message)
                await session.refresh(ticket, ["messages"])

                # Broadcast в веб-интерфейс
                from app.db import MessageRead, TicketRead

                # Отправляем сообщение
                await connection_manager.broadcast_message(
                    ticket.id, MessageRead.from_orm(db_message).model_dump(mode="json")
                )

                # Обновляем список тикетов
                tickets = await crud.list_tickets(session, archived=False)
                tickets_payload = []
                for t in tickets:
                    ticket_data = TicketRead.from_orm(t).model_dump(mode="json")
                    unread_count = sum(
                        1
                        for msg in t.messages
                        if msg.sender in ["user", "bot"] and not msg.is_read
                    )
                    ticket_data["unread_count"] = min(unread_count, 99)
                    tickets_payload.append(ticket_data)

                await connection_manager.broadcast_conversations(tickets_payload)

        logger.info(
            f"Sent agent action to Telegram chat {chat_id}: {action_text[:50]}..."
        )
    except Exception as e:
        logger.error(f"Failed to send agent action to Telegram: {e}")


def create_dispatcher(
    session_maker: async_sessionmaker[AsyncSession],
    connection_manager: ConnectionManager,
    rag_service: RAGService,
    knowledge_base=None,
) -> Dispatcher:
    router = Router()

    async def _serialize_tickets(session: AsyncSession) -> list[dict]:
        tickets = await crud.list_tickets(session, archived=False)
        result = []
        for ticket in tickets:
            ticket_data = TicketRead.from_orm(ticket).model_dump(mode="json")
            # Подсчитываем непрочитанные сообщения от user и bot
            unread_count = sum(
                1
                for msg in ticket.messages
                if msg.sender in ["user", "bot"] and not msg.is_read
            )
            # Ограничиваем максимум 99
            ticket_data["unread_count"] = min(unread_count, 99)
            result.append(ticket_data)
        return result

    async def _broadcast_tickets() -> None:
        async with session_maker() as session:
            tickets_payload = await _serialize_tickets(session)
        await connection_manager.broadcast_conversations(tickets_payload)

    async def _broadcast_message(
        conversation_id: int, message_schema: MessageRead
    ) -> None:
        await connection_manager.broadcast_message(
            conversation_id, message_schema.model_dump(mode="json")
        )

    async def _persist_message(
        message: Message, text: str, sender: str = USER_SENDER
    ) -> tuple[int | None, bool]:
        """Сохраняет сообщение в существующую заявку (если есть)"""
        chat_id = message.chat.id
        async with session_maker() as session:
            ticket = await crud.get_open_ticket_by_chat_id(session, chat_id)
            if ticket is None:
                return None, False

            should_mark_as_read = connection_manager.has_active_chat_connections(
                ticket.id
            )

            db_message = await crud.add_message(
                session,
                ticket_id=ticket.id,
                sender=sender,
                text=text,
                telegram_message_id=(
                    message.message_id if sender == USER_SENDER else None
                ),
                is_read=should_mark_as_read,
            )

            await session.refresh(db_message)
            await session.refresh(ticket, ["messages"])

            # Автоматическое обновление ТОЛЬКО классификации (только для сообщений пользователя)
            # Приоритет устанавливается ТОЛЬКО через MCP tool агентом
            if sender == USER_SENDER:
                try:
                    from app.rag.agent_tools import auto_update_classification

                    # Подсчитываем количество сообщений от пользователя
                    user_message_count = sum(
                        1 for msg in ticket.messages if msg.sender == USER_SENDER
                    )

                    # Формируем историю диалога
                    dialogue_parts = []
                    for msg in ticket.messages:
                        role = (
                            "Пользователь"
                            if msg.sender == USER_SENDER
                            else "Бот" if msg.sender == BOT_SENDER else "Оператор"
                        )
                        dialogue_parts.append(f"{role}: {msg.text}")
                    dialogue_history = "\n".join(dialogue_parts)

                    # Вызываем автоматическое обновление ТОЛЬКО классификации
                    update_result = auto_update_classification(
                        conversation_id=ticket.id,
                        dialogue_history=dialogue_history,
                        message_count=user_message_count,
                        current_classification=ticket.classification,
                    )

                    if update_result.get("updated"):
                        logger.info(
                            f"Auto-updated classification for ticket {ticket.id}: "
                            f"classification={update_result.get('classification')}"
                        )
                        # Обновляем ticket в текущей сессии
                        await session.refresh(ticket)

                except Exception as e:
                    logger.warning(f"Failed to auto-update classification: {e}")

            tickets_payload = await _serialize_tickets(session)

        await _broadcast_message(ticket.id, MessageRead.from_orm(db_message))
        await connection_manager.broadcast_conversations(tickets_payload)
        return ticket.id, True

    def _make_concise_labels(candidates: list[str], max_len: int = 30) -> list[str]:
        """Generate concise 2-3 word labels from candidate strings.

        - Normalize common tokens (e.g., wifi -> Wi‑Fi)
        - Prefer meaningful short phrases (2-3 words)
        - Ensure labels are unique (append short differentiator if needed)
        - Do not cut words in the middle; add ellipsis if truncated
        """

        def normalize(s: str) -> str:
            if not s:
                return ""
            s = s.strip()
            # Normalize wifi variants to Wi‑Fi
            s = (
                s.replace("wifi", "Wi‑Fi")
                .replace("wi‑fi", "Wi‑Fi")
                .replace("WiFi", "Wi‑Fi")
            )
            # collapse excessive whitespace
            s = " ".join(s.split())
            return s

        labels: list[str] = []
        seen: set[str] = set()

        for orig in candidates:
            s = normalize(orig)
            if not s:
                s = orig or ""
            # pick first 2-3 words if possible
            parts = s.split()
            if len(parts) >= 3:
                cand = " ".join(parts[:3])
            else:
                cand = " ".join(parts)

            # fallback to shorter if too long
            if len(cand) > max_len:
                # try first two words
                cand = " ".join(parts[:2]) if len(parts) >= 2 else parts[0][:max_len]

            # final truncate without cutting words
            if len(cand) > max_len:
                # truncate to max_len but avoid cutting last word
                truncated = cand[:max_len]
                if " " in truncated:
                    truncated = truncated.rsplit(" ", 1)[0]
                cand = truncated.rstrip() + "..."

            # Ensure uniqueness: if duplicate, append a short suffix number
            base = cand
            i = 1
            while cand in seen:
                i += 1
                suffix = f" ({i})"
                allowed = max_len - len(suffix)
                if len(base) > allowed:
                    trimmed = base[:allowed].rsplit(" ", 1)[0]
                    cand = f"{trimmed}{suffix}"
                else:
                    cand = f"{base}{suffix}"

            seen.add(cand)
            labels.append(cand)

        return labels

    async def _create_ticket_and_add_message(message: Message, text: str) -> int:
        """Создает новую заявку и добавляет первое сообщение"""
        chat_id = message.chat.id
        title = _extract_title(message)

        # Генерируем summary на основе истории чата ПЕРЕД созданием тикета
        try:
            summary = await rag_service.generate_ticket_summary_from_chat_history(
                chat_id
            )
            logger.info(
                f"Generated summary from chat history for user {chat_id}: {summary[:50]}..."
            )
        except Exception as e:
            logger.warning(
                f"Failed to generate summary from chat history for user {chat_id}: {e}"
            )
            summary = "Запрос помощи от пользователя"

        async with session_maker() as session:
            # Создаем новую заявку
            ticket = await crud.create_ticket(session, chat_id, title)

            # Проверяем, есть ли активные подключения к этому чату (маловероятно для новой заявки, но проверим)
            should_mark_as_read = connection_manager.has_active_chat_connections(
                ticket.id
            )

            # Получаем и добавляем историю чата из RAG сервиса
            try:
                chat_history = rag_service.get_chat_history_since_last_ticket(chat_id)
                logger.info(
                    f"Retrieved chat history for user {chat_id}: {len(chat_history)} messages"
                )

                # Добавляем все сообщения из истории чата
                for chat_msg in chat_history:
                    sender = USER_SENDER if chat_msg.is_user else BOT_SENDER
                    await crud.add_message(
                        session,
                        ticket_id=ticket.id,
                        sender=sender,
                        text=chat_msg.message,
                        is_system=False,
                        is_read=should_mark_as_read,
                    )

                rag_service.mark_ticket_created(chat_id)

                # Сохраняем summary в БД
                await crud.update_ticket_summary(session, ticket.id, summary)

                # Проверяем, есть ли текущее сообщение в истории
                last_user_messages = [
                    msg.message for msg in chat_history if msg.is_user
                ]
                if text in last_user_messages:
                    # Текущее сообщение уже в истории, берём последнее добавленное
                    messages = await crud.list_messages_for_ticket(session, ticket.id)
                    db_message = messages[-1] if messages else None
                    if not db_message:
                        raise Exception("No messages found after adding history")
                else:
                    # Добавляем текущее сообщение
                    db_message = await crud.add_message(
                        session,
                        ticket_id=ticket.id,
                        sender=USER_SENDER,
                        text=text,
                        telegram_message_id=message.message_id,
                        is_read=should_mark_as_read,
                    )

            except Exception as e:
                logger.warning(
                    f"Failed to process chat history for user {chat_id}: {e}"
                )
                db_message = await crud.add_message(
                    session,
                    ticket_id=ticket.id,
                    sender=USER_SENDER,
                    text=text,
                    telegram_message_id=message.message_id,
                    is_read=should_mark_as_read,
                )

        await _broadcast_message(ticket.id, MessageRead.from_orm(db_message))
        await _broadcast_tickets()
        return ticket.id

    async def _send_bot_message(ticket_id: int, text: str) -> None:
        """Отправляет сообщение от имени бота в конкретную заявку"""
        async with session_maker() as session:
            should_mark_as_read = connection_manager.has_active_chat_connections(
                ticket_id
            )

            db_message = await crud.add_message(
                session,
                ticket_id=ticket_id,
                sender=BOT_SENDER,
                text=text,
                is_read=should_mark_as_read,
            )

        await _broadcast_message(ticket_id, MessageRead.from_orm(db_message))
        await _broadcast_tickets()

    async def _send_bot_message_to_ticket(chat_id: int, text: str) -> None:
        """Сохраняет сообщение бота в тикет пользователя (если тикет существует)"""
        async with session_maker() as session:
            ticket = await crud.get_open_ticket_by_chat_id(session, chat_id)
            if ticket:
                await _send_bot_message(ticket.id, text)

    async def _get_average_response_time() -> str:
        """Вычисляет среднее время ответа оператора"""
        try:
            async with session_maker() as session:
                from sqlalchemy import select, func
                from datetime import datetime, timedelta

                # Получаем закрытые заявки за последние 30 дней
                thirty_days_ago = datetime.utcnow() - timedelta(days=30)
                stmt = select(models.Ticket).where(
                    models.Ticket.status == models.TicketStatus.CLOSED,
                    models.Ticket.created_at >= thirty_days_ago,
                )
                result = await session.execute(stmt)
                tickets = result.scalars().all()

                if not tickets:
                    return "обычно быстро"

                # Вычисляем среднее время между созданием и первым ответом оператора
                response_times = []
                for ticket in tickets:
                    messages = await crud.list_messages_for_ticket(session, ticket.id)
                    # Находим первое сообщение оператора
                    operator_message = next(
                        (m for m in messages if m.sender == "operator"), None
                    )
                    if operator_message and ticket.created_at:
                        delta = operator_message.created_at - ticket.created_at
                        response_times.append(delta.total_seconds() / 60)  # в минутах

                if not response_times:
                    return "обычно быстро"

                avg_minutes = sum(response_times) / len(response_times)

                if avg_minutes < 1:
                    return "менее минуты"
                elif avg_minutes < 5:
                    return f"{int(avg_minutes)} мин"
                elif avg_minutes < 60:
                    return f"{int(avg_minutes)} минут"
                else:
                    hours = int(avg_minutes / 60)
                    return f"около {hours} ч"

        except Exception as e:
            logger.warning(f"Failed to calculate average response time: {e}")
            return "обычно быстро"

    async def _answer_with_rag_only(
        message: Message, user_text: str, ticket_id: int = None
    ) -> None:
        """Отвечает пользователю через RAG и сохраняет ответ в тикет (если есть)

        Args:
            message: Сообщение пользователя
            user_text: Текст сообщения
            ticket_id: ID тикета для передачи в RAG (если None - используется chat_id)
        """
        await message.bot.send_chat_action(message.chat.id, "typing")

        # Если ticket_id не передан, пытаемся получить его
        conversation_id = ticket_id
        if conversation_id is None:
            async with session_maker() as session:
                ticket = await crud.get_open_ticket_by_chat_id(session, message.chat.id)
                if ticket:
                    conversation_id = ticket.id
                else:
                    # Fallback на chat_id если тикета нет
                    conversation_id = message.chat.id
                    logger.warning(
                        f"No ticket found for chat {message.chat.id}, using chat_id as conversation_id"
                    )

        try:
            # Вызываем RAG сервис с правильным conversation_id (ticket.id)
            try:
                rag_result = await rag_service.generate_reply(
                    conversation_id, user_text
                )
            except TypeError:
                rag_result = await asyncio.to_thread(
                    rag_service.generate_reply, conversation_id, user_text
                )
        except Exception as exc:
            logger.error(f"RAG generation failed: {exc}")
            await message.answer(
                "Не смогла обработать запрос. Попробуйте ещё раз или позовите оператора.",
                reply_markup=REQUEST_OPERATOR_KEYBOARD,
            )
            return

        answer_text = (
            rag_result.final_answer
            or "Я пока не нашла ответ. Попробуйте уточнить вопрос."
        )

        # Проверяем есть ли похожие предложения для показа кнопок
        if rag_result.similar_suggestions:
            suggestions = rag_result.similar_suggestions
            print(
                f"TELEGRAM BOT: Showing {len(suggestions)} similar suggestions as buttons"
            )

            # Показываем основной ответ
            await message.answer(answer_text, parse_mode="HTML")

            # Сохраняем ответ бота в тикет
            await _send_bot_message_to_ticket(message.chat.id, answer_text)

            # Создаём кнопки из предложений
            buttons = []
            for item in suggestions[:3]:  # Максимум 3 кнопки
                chunk_id = item.get("id")
                preview = item.get("preview", "")[:60]  # Ограничиваем длину
                buttons.append(
                    [
                        InlineKeyboardButton(
                            text=f"📄 {preview}",
                            callback_data=f"similar::{chunk_id}",
                        )
                    ]
                )

            if buttons:
                similar_kb = InlineKeyboardMarkup(inline_keyboard=buttons)
                await message.answer(
                    "🔍 Возможно, ваша проблема похожа на одну из этих:",
                    reply_markup=similar_kb,
                )

            return  # Завершаем обработку

        # Проверяем, есть ли открытый тикет и не был ли уже запрошен оператор
        async with session_maker() as session:
            existing_ticket = await crud.get_open_ticket_by_chat_id(
                session, message.chat.id
            )
            operator_already_requested = (
                existing_ticket.operator_requested if existing_ticket else False
            )

        # Проверяем нужен ли оператор (явный запрос или низкая уверенность)
        needs_operator = (
            rag_result.operator_requested or rag_result.confidence_score > 0.6
        ) and not operator_already_requested

        if needs_operator:
            avg_response_time = await _get_average_response_time()
            combined_text = (
                f"{answer_text}\n\n"
                f"⏱ Среднее время ответа: <b>{avg_response_time}</b>\n\n"
                f"Подключить оператора?"
            )
            confirm_keyboard = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="✅ Да, подключить оператора",
                            callback_data=OPERATOR_REQUEST_CALLBACK,
                        )
                    ]
                ]
            )
            await message.answer(
                combined_text, reply_markup=confirm_keyboard, parse_mode="HTML"
            )
            # Сохраняем ответ бота в тикет
            await _send_bot_message_to_ticket(message.chat.id, combined_text)
        else:
            # Уверенный ответ - показываем ответ и кнопки KB (если доступны)
            await message.answer(answer_text, parse_mode="HTML")

            # Сохраняем ответ бота в тикет
            await _send_bot_message_to_ticket(message.chat.id, answer_text)

            # Показываем быстрые кнопки из knowledge_base
            if knowledge_base is not None:
                try:
                    top = await knowledge_base.search_top_k(user_text, top_k=3)
                    if top:
                        candidates = [(entry.get("question") or "") for entry in top]
                        labels = _make_concise_labels(candidates, max_len=30)
                        buttons = []
                        for entry, label in zip(top, labels):
                            eid = entry.get("id")
                            buttons.append(
                                [
                                    InlineKeyboardButton(
                                        text=label, callback_data=f"kb::{eid}"
                                    )
                                ]
                            )
                        topic_kb = InlineKeyboardMarkup(inline_keyboard=buttons)
                        await message.answer(
                            "Возможно, пригодится один из этих быстрых ответов:",
                            reply_markup=topic_kb,
                        )
                except Exception as e:
                    logger.warning(f"Failed to show KB buttons: {e}")

    @router.callback_query(F.data.startswith("kb::"))
    async def on_kb_callback(query: CallbackQuery) -> None:
        """Обработчик нажатия на быстрые кнопки KB"""
        await query.answer()

        data = query.data or ""
        parts = data.split("::")
        if len(parts) < 2:
            return

        try:
            entry_id = int(parts[1])
        except ValueError:
            return

        if knowledge_base is None:
            return

        try:
            entry = await knowledge_base.get_by_id(entry_id)
            if not entry:
                await query.message.answer("Извините, не удалось найти ответ.")
                return

            # Показываем что выбрал пользователь
            question = entry.get("question", "")
            if query.message and question:
                await query.message.edit_text(f'Вы выбрали: "{html.escape(question)}"')

            # Отправляем ответ
            answer_text = entry.get("answer", "Ответ не найден.")
            safe_answer = html.escape(answer_text)

            if query.message:
                await query.message.answer(safe_answer, parse_mode="HTML")
            else:
                await query.bot.send_message(
                    query.from_user.id, safe_answer, parse_mode="HTML"
                )

        except Exception as e:
            logger.error(f"Error in KB callback: {e}")
            await query.message.answer("Произошла ошибка при получении ответа.")

    @router.callback_query(F.data.startswith("similar::"))
    async def on_similar_callback(query: CallbackQuery) -> None:
        """Обработчик нажатия на кнопки с похожими проблемами"""
        await query.answer()

        data = query.data or ""
        parts = data.split("::")
        if len(parts) < 2:
            return

        try:
            chunk_id = int(parts[1])
        except ValueError:
            return

        try:
            from app.db.database import KnowledgeSessionLocal
            from app.db import tickets_crud as knowledge_crud

            async with KnowledgeSessionLocal() as session:
                # Получаем чанк по ID
                chunk = await knowledge_crud.get_chunk_by_id(session, chunk_id)

                if not chunk:
                    await query.message.answer("Извините, решение не найдено.")
                    return

                # Показываем решение пользователю
                solution_text = chunk.content.strip()
                source_info = (
                    f"\n\n📚 Источник: {chunk.source_file}" if chunk.source_file else ""
                )

                full_response = f"<b>Решение:</b>\n\n{html.escape(solution_text)}{html.escape(source_info)}"

                # Удаляем кнопки из сообщения после выбора
                if query.message:
                    try:
                        await query.message.edit_reply_markup(reply_markup=None)
                    except Exception:
                        pass

                    await query.message.answer(full_response, parse_mode="HTML")
                else:
                    await query.bot.send_message(
                        query.from_user.id, full_response, parse_mode="HTML"
                    )

        except Exception as e:
            logger.error(f"Error in similar problems callback: {e}")
            await query.message.answer("Произошла ошибка при получении решения.")

    async def _send_bot_message(
        ticket_id: int, text: str, is_system: bool = False
    ) -> None:
        # Проверяем, есть ли активные подключения к этому чату
        should_mark_as_read = connection_manager.has_active_chat_connections(ticket_id)
        async with session_maker() as session:
            db_message = await crud.add_message(
                session,
                ticket_id,
                BOT_SENDER,
                text,
                is_system=is_system,
                is_read=should_mark_as_read,
            )
            # add_message уже делает commit, просто обновляем объект
            await session.refresh(db_message)
            tickets_payload = await _serialize_tickets(session)
        await _broadcast_message(ticket_id, MessageRead.from_orm(db_message))
        await connection_manager.broadcast_conversations(tickets_payload)

    @router.message(CommandStart())
    async def on_start(message: Message) -> None:
        user_name = (
            message.from_user.first_name if message.from_user else "пользователь"
        )

        # Получаем base_url с автоматическим определением (ngrok/env/config)
        from app.utils import get_base_url

        base_url = get_base_url()

        telegram_responses = load_telegram_responses()
        greeting = telegram_responses.get("start_greeting", "").format(
            user_name=user_name, base_url=base_url
        )

        if not greeting.strip():
            greeting = (
                f"👋 Привет, {user_name}! Я бот технической поддержки. Чем могу помочь?\n\n"
                f'💡 Посмотрите наш <a href="{base_url}/faq">FAQ с ответами на популярные вопросы</a>'
            )

        await message.answer(greeting, parse_mode="HTML")

    @router.message(F.voice)
    async def on_voice(message: Message) -> None:
        """Обработчик голосовых сообщений"""
        chat_id = message.chat.id
        lock = user_locks[chat_id]

        if lock.locked():
            await message.answer(
                "⏳ Пожалуйста, дождитесь ответа на предыдущее сообщение"
            )
            return

        async with lock:
            voice = message.voice
            if not voice:
                await message.answer("Не удалось получить голосовое сообщение.")
                return

            processing_msg = await message.answer(
                "🎤 Обрабатываю голосовое сообщение..."
            )

            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_file:
                await message.bot.download(voice.file_id, temp_file.name)
                temp_file_path = temp_file.name

            try:
                transcribed_text = await rag_service.speech_to_text.transcribe_audio(
                    temp_file_path
                )

                if not transcribed_text:
                    await processing_msg.edit_text(
                        "❌ Не удалось распознать речь. Попробуйте еще раз или напишите текстом."
                    )
                    return

                await processing_msg.delete()
                ticket_id, has_ticket = await _persist_message(
                    message, transcribed_text
                )

                if not has_ticket or not ticket_id:
                    await _answer_with_rag_only(message, transcribed_text)

            except Exception as e:
                logger.error(f"Error processing voice message: {e}")
                await message.answer(
                    "❌ Произошла ошибка при обработке голосового сообщения. Попробуйте написать текстом."
                )
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

    @router.message(F.text)
    async def on_text(message: Message) -> None:
        user_text = message.text or ""
        chat_id = message.chat.id
        lock = user_locks[chat_id]

        if lock.locked():
            await message.answer(
                "⏳ Пожалуйста, дождитесь ответа на предыдущее сообщение"
            )
            return

        async with lock:
            # Проверяем, есть ли открытая заявка
            async with session_maker() as session:
                ticket = await crud.get_open_ticket_by_chat_id(session, chat_id)

                if ticket is None:
                    # НЕТ ЗАЯВКИ - создаём новую при первом сообщении
                    logger.info(
                        f"Creating new ticket for chat {chat_id} on first message"
                    )
                    ticket_id = await _create_ticket_and_add_message(message, user_text)

                    # Отвечаем через RAG с передачей ticket_id
                    await _answer_with_rag_only(message, user_text, ticket_id)
                    return

                # ЕСТЬ ЗАЯВКА - проверяем, подключен ли оператор
                if ticket.operator_requested:
                    # Оператор подключен - только сохраняем сообщение, бот молчит
                    ticket_id, has_ticket = await _persist_message(message, user_text)
                    return
                else:
                    # Оператор НЕ подключен - сохраняем сообщение И отвечаем через RAG
                    ticket_id, has_ticket = await _persist_message(message, user_text)
                    await _answer_with_rag_only(message, user_text, ticket.id)

    @router.message(F.caption)
    async def on_caption(message: Message) -> None:
        caption_text = message.caption or ""
        ticket_id, has_ticket = await _persist_message(message, caption_text)

        warning = "Пока могу обрабатывать только текстовые сообщения."
        await message.answer(warning, reply_markup=REQUEST_OPERATOR_KEYBOARD)

        if has_ticket and ticket_id:
            await _send_bot_message(ticket_id, warning)
        elif caption_text:
            await _answer_with_rag_only(message, caption_text)

    @router.message()
    async def on_other(message: Message) -> None:
        ticket_id, has_ticket = await _persist_message(
            message, "[unsupported message type]"
        )

        warning = "Пока поддерживаются только текстовые сообщения. Пожалуйста, напишите ваш вопрос."
        await message.answer(warning, reply_markup=REQUEST_OPERATOR_KEYBOARD)

        if has_ticket and ticket_id:
            await _send_bot_message(ticket_id, warning)

    @router.callback_query(lambda query: query.data == OPERATOR_REQUEST_CALLBACK)
    async def on_request_operator(callback_query: CallbackQuery) -> None:
        chat = callback_query.message.chat if callback_query.message else None
        if chat is None:
            await callback_query.answer()
            return

        async with session_maker() as session:
            ticket = await crud.get_open_ticket_by_chat_id(session, chat.id)

            if ticket is None:
                # Создаём новую заявку с историей чата
                chat_history = rag_service.get_chat_history_since_last_ticket(chat.id)
                logger.info(
                    f"Creating ticket with chat history: {len(chat_history)} messages"
                )

                # Генерируем summary
                try:
                    summary = (
                        await rag_service.generate_ticket_summary_from_chat_history(
                            chat.id
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate summary: {e}")
                    summary = "Запрос помощи от пользователя"

                # Создаём заявку
                title = (
                    _extract_title(user_obj=callback_query.from_user)
                    if callback_query.from_user
                    else f"Заявка от {chat.id}"
                )
                ticket = await crud.create_ticket(session, chat.id, title)

                should_mark_as_read = connection_manager.has_active_chat_connections(
                    ticket.id
                )

                # Добавляем историю чата
                for chat_msg in chat_history:
                    sender = USER_SENDER if chat_msg.is_user else BOT_SENDER
                    await crud.add_message(
                        session,
                        ticket_id=ticket.id,
                        sender=sender,
                        text=chat_msg.message,
                        is_system=False,
                        is_read=should_mark_as_read,
                    )

                rag_service.mark_ticket_created(chat.id)
                await crud.update_ticket_summary(session, ticket.id, summary)

            # Переводим в статус "открыта" если была в работе
            if ticket.status != models.TicketStatus.OPEN:
                await crud.update_ticket_status(
                    session, ticket.id, models.TicketStatus.OPEN
                )

            # Устанавливаем флаг что оператор запрошен
            await crud.mark_operator_requested(session, ticket.id)

            # Генерируем summary и классификацию
            messages = await crud.list_messages_for_ticket(session, ticket.id)
            if messages:
                try:
                    summary = await rag_service.generate_ticket_summary(
                        messages, ticket_id=ticket.id
                    )
                    await crud.update_ticket_summary(session, ticket.id, summary)
                except Exception as e:
                    logger.warning(f"Failed to generate summary: {e}")

                # Автоматическая классификация
                try:
                    from app.rag.agent_tools import _classify_request_internal

                    dialogue_text = "\n".join(
                        [
                            f"{'Пользователь' if msg.sender == 'user' else 'Бот'}: {msg.text}"
                            for msg in messages[-10:]
                        ]
                    )

                    if dialogue_text.strip():
                        classification_result = _classify_request_internal(
                            dialogue_history=dialogue_text
                        )
                        if "Классификация проблемы:" in classification_result:
                            categories = classification_result.split(
                                "Классификация проблемы:"
                            )[1].strip()
                            await crud.update_ticket_classification(
                                session, ticket.id, categories
                            )
                            logger.info(
                                f"Generated classification for ticket {ticket.id}: {categories}"
                            )
                except Exception as e:
                    logger.warning(f"Failed to generate classification: {e}")

            tickets_payload = await _serialize_tickets(session)

        rag_service.reset_history(ticket.id)
        await callback_query.answer("✅ Заявка создана")

        # Убираем кнопку из сообщения
        if callback_query.message:
            try:
                await callback_query.message.edit_reply_markup(reply_markup=None)
            except Exception as e:
                logger.debug(f"Could not remove keyboard: {e}")

        await connection_manager.broadcast_conversations(tickets_payload)

        notice = "✅ Заявка создана. Ожидайте ответа оператора."
        await callback_query.message.answer(notice)
        await _send_bot_message(ticket.id, notice, is_system=True)

    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    return dispatcher


async def start_bot(bot: Bot, dispatcher: Dispatcher) -> None:
    try:
        await dispatcher.start_polling(bot)
    except asyncio.CancelledError:
        raise
