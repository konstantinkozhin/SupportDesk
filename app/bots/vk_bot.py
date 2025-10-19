from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections import defaultdict

import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType
import vk_api.exceptions
import requests
import json
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db import tickets_crud as crud, models
from app.config import load_vk_responses
from app.rag import RAGResult, RAGService
from app.services import ConnectionManager
from app.db import TicketRead, MessageRead

logger = logging.getLogger(__name__)

# Module-level VK API client (set when create_vk_bot initializes)
VK_API: any = None

# Module-level VK responses configuration
vk_responses: dict = {}

# transient per-user pending options when keyboard isn't supported by client
pending_options: dict[int, list[str]] = {}

USER_SENDER = "user"
BOT_SENDER = "bot"
OPERATOR_REQUEST_CALLBACK = "request_operator"

# Словарь блокировок для каждого пользователя (предотвращение спама)
user_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)


def _extract_title_vk(user_id: int, vk_session) -> str:
    """Извлекает имя пользователя из VK"""
    try:
        user_info = vk_session.method(
            "users.get", {"user_ids": user_id, "fields": "first_name,last_name"}
        )[0]
        first_name = user_info.get("first_name", "")
        last_name = user_info.get("last_name", "")
        if first_name or last_name:
            return f"{first_name} {last_name}".strip()
        return f"VK User {user_id}"
    except Exception as e:
        logger.warning(f"Failed to get user info for {user_id}: {e}")
        return f"VK User {user_id}"


def create_vk_bot(
    session_maker: async_sessionmaker[AsyncSession],
    connection_manager: ConnectionManager,
    rag_service: RAGService,
    vk_token: str,
):
    logger.info("VK: Starting VK bot creation...")
    logger.info(f"VK: Token provided: {'YES' if vk_token else 'NO'}")

    vk_session = vk_api.VkApi(token=vk_token)
    vk = vk_session.get_api()
    # expose module-level client for other modules to use when needed
    global VK_API, vk_responses
    VK_API = vk
    vk_responses = load_vk_responses()

    # Флаг типа longpoll
    is_bot_longpoll = False  # Начинаем с личных сообщений

    longpoll = None

    # Сначала пробуем VkLongPoll для личных сообщений (более надежный)
    try:
        logger.info("VK: Attempting VkLongPoll for personal messages...")
        longpoll = VkLongPoll(vk_session)
        logger.info("VK: VkLongPoll initialized successfully for personal messages")
    except Exception as e:
        logger.warning(
            f"VK: VkLongPoll failed: {e}, trying VkBotLongPoll for groups..."
        )
        try:
            # Fallback: пробуем VkBotLongPoll для сообществ
            logger.info("VK: Attempting to get group info...")
            group_info = vk.groups.getById()
            group_id = group_info[0]["id"]
            logger.info(f"VK: Using group ID {group_id}")
            longpoll = VkBotLongPoll(vk_session, group_id=group_id)
            is_bot_longpoll = True
            logger.info("VK: VkBotLongPoll initialized successfully")
        except vk_api.exceptions.ApiError as e:
            logger.error(f"VK: ApiError during group access: {e}")
            if "longpoll for this group is not enabled" in str(e):
                logger.warning(
                    "VK: LongPoll не включен для сообщества, trying polling mode..."
                )
                try:
                    # Второй fallback: используем polling через messages.getConversations
                    logger.info("VK: Using polling mode for VK messages")
                    is_bot_longpoll = False
                    longpoll = None  # Будем использовать polling
                except Exception as poll_e:
                    logger.error(f"VK: Polling setup also failed: {poll_e}")
                    logger.error("VK: VK bot disabled - all methods failed")
                    return None
            else:
                logger.error(f"VK: Failed to initialize VkBotLongPoll: {e}")
                raise
        except Exception as e:
            logger.error(f"VK: Failed to initialize VkBotLongPoll: {e}")
            raise

    # ... остальной код ...

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

    async def _persist_message_vk(
        user_id: int, text: str, sender: str = USER_SENDER, message_id: int = None
    ) -> tuple[int | None, bool]:
        """Сохраняет сообщение в существующую заявку (если есть)"""
        chat_id = f"vk_{user_id}"  # Префикс для VK
        async with session_maker() as session:
            # Ищем ОТКРЫТУЮ заявку для данного чата
            ticket = await crud.get_open_ticket_by_chat_id(session, chat_id)
            logger.info(
                f"VK: Looking for open ticket for chat_id={chat_id}, found: {ticket.id if ticket else None}"
            )
            if ticket is None:
                # Нет открытой заявки - это обычное общение с ботом
                return None, False

            # Проверяем, есть ли активные подключения к этому чату
            should_mark_as_read = connection_manager.has_active_chat_connections(
                ticket.id
            )
            logger.info(
                f"📨 VK Message from {sender} to ticket #{ticket.id}: has_active={should_mark_as_read}"
            )

            # Есть открытая заявка - добавляем сообщение
            db_message = await crud.add_message(
                session,
                ticket_id=ticket.id,
                sender=sender,
                text=text,
                vk_message_id=message_id if sender == USER_SENDER else None,
                is_read=should_mark_as_read,  # Сразу помечаем как прочитанное, если чат открыт
            )
            logger.info(
                f"✅ VK Message #{db_message.id} created with is_read={db_message.is_read}"
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
                            f"VK: Auto-updated classification for ticket {ticket.id}: "
                            f"classification={update_result.get('classification')}"
                        )
                        # Обновляем ticket в текущей сессии
                        await session.refresh(ticket)

                except Exception as e:
                    logger.warning(f"VK: Failed to auto-update classification: {e}")

            tickets_payload = await _serialize_tickets(session)

        await _broadcast_message(ticket.id, MessageRead.from_orm(db_message))
        await connection_manager.broadcast_conversations(tickets_payload)
        return ticket.id, True

    async def _create_ticket_and_add_message_vk(
        user_id: int, text: str, message_id: int
    ) -> int:
        """Создает новую заявку и добавляет первое сообщение"""
        chat_id = f"vk_{user_id}"
        logger.info(
            f"VK: Creating ticket for chat_id={chat_id}, initial text={text[:80]}..."
        )
        title = _extract_title_vk(user_id, vk_session)
        async with session_maker() as session:
            # Создаем новую заявку
            ticket = await crud.create_ticket(session, chat_id, title)
            logger.info(
                f"VK: New ticket created with id={ticket.id} for chat_id={chat_id}"
            )

            # Проверяем, есть ли активные подключения к этому чату (маловероятно для новой заявки, но проверим)
            should_mark_as_read = connection_manager.has_active_chat_connections(
                ticket.id
            )

            # Пытаемся получить и добавить историю чата из RAG сервиса
            try:
                print(
                    f"VK BOT DEBUG: Creating ticket for user {user_id}, current message: {text}"
                )

                # Получаем историю с момента последней заявки или с начала
                chat_history = rag_service.get_chat_history_since_last_ticket(chat_id)
                print(
                    f"VK BOT DEBUG: Retrieved segmented chat history for user {user_id}: {len(chat_history)} messages"
                )
                logger.info(
                    f"Retrieved segmented chat history for user {user_id}: {len(chat_history)} messages"
                )

                # Логируем содержимое истории для отладки
                for i, msg in enumerate(chat_history):
                    sender_type = "USER" if msg.is_user else "BOT"
                    print(
                        f"VK BOT DEBUG: History message {i+1}: [{sender_type}] {msg.message[:50]}..."
                    )
                    logger.debug(
                        f"History message {i+1}: [{sender_type}] {msg.message[:50]}..."
                    )

                # Добавляем все сообщения из релевантной истории чата
                for i, chat_msg in enumerate(chat_history):
                    try:
                        sender = USER_SENDER if chat_msg.is_user else BOT_SENDER
                        # Для новых сообщений не передавать created_at, чтобы использовалось текущее время
                        await crud.add_message(
                            session,
                            ticket_id=ticket.id,
                            sender=sender,
                            text=chat_msg.message,
                            vk_message_id=None,  # Для истории не указываем
                            is_system=False,
                            is_read=should_mark_as_read,
                        )
                        print(
                            f"VK BOT DEBUG: Added history message {i+1}/{len(chat_history)}: {sender} - {chat_msg.message[:30]}..."
                        )
                        logger.debug(
                            f"Added history message {i+1}/{len(chat_history)}: {sender}"
                        )
                    except Exception as e:
                        print(f"VK BOT DEBUG: Failed to add history message {i+1}: {e}")
                        logger.warning(f"Failed to add history message {i+1}: {e}")

                # Отмечаем создание заявки
                rag_service.mark_ticket_created(chat_id)

                # НЕ очищаем историю чата - оставляем для будущих заявок
                logger.info(f"Marked ticket creation for user {chat_id}")

                # Проверяем, есть ли уже текущее сообщение в истории
                last_user_messages = [
                    msg.message for msg in chat_history if msg.is_user
                ]
                if text in last_user_messages:
                    # Текущее сообщение уже есть в истории, не дублируем
                    # Берем последнее добавленное сообщение как db_message для ответа
                    messages = await crud.list_messages_for_ticket(session, ticket.id)
                    db_message = messages[-1] if messages else None
                    if not db_message:
                        # Если по какой-то причине сообщений нет, добавляем текущее
                        raise Exception("No messages found after adding history")
                else:
                    # Добавляем текущее сообщение, если его нет в истории
                    db_message = await crud.add_message(
                        session,
                        ticket_id=ticket.id,
                        sender=USER_SENDER,
                        text=text,
                        vk_message_id=message_id,
                        is_read=should_mark_as_read,
                    )

            except Exception as e:
                # Если что-то пошло не так с историей, просто добавляем текущее сообщение
                logger.warning(
                    f"Failed to process chat history for user {user_id}: {e}"
                )
                db_message = await crud.add_message(
                    session,
                    ticket_id=ticket.id,
                    sender=USER_SENDER,
                    text=text,
                    vk_message_id=message_id,
                    is_read=should_mark_as_read,
                )

            # Генерируем summary сразу после создания заявки
            try:
                messages = await crud.list_messages_for_ticket(session, ticket.id)
                summary = await rag_service.generate_ticket_summary(
                    messages, ticket_id=ticket.id
                )
                # Сохраняем summary в БД
                await crud.update_ticket_summary(session, ticket.id, summary)
                logger.info(
                    f"Generated and saved summary for VK ticket {ticket.id}: {summary[:50]}..."
                )

                # Автоматическая классификация при передаче оператору
                try:
                    # Формируем историю диалога для классификации
                    dialogue_text = ""
                    for msg in messages[-10:]:  # Берем последние 10 сообщений
                        sender_name = "Пользователь" if msg.sender == "user" else "Бот"
                        dialogue_text += f"{sender_name}: {msg.text}\n"

                    if dialogue_text.strip():
                        # Используем встроенную функцию классификации из agent_tools
                        from app.rag.agent_tools import _classify_request_internal

                        classification_result = _classify_request_internal(
                            dialogue_history=dialogue_text
                        )

                        # Извлекаем категории из результата (формат: "Классификация проблемы: Категория1, Категория2")
                        if "Классификация проблемы:" in classification_result:
                            categories = classification_result.split(
                                "Классификация проблемы:"
                            )[1].strip()
                            await crud.update_ticket_classification(
                                session, ticket.id, categories
                            )
                            logger.info(
                                f"Generated classification for VK ticket {ticket.id}: {categories}"
                            )
                except Exception as e:
                    logger.warning(
                        f"Failed to generate classification for VK ticket: {e}"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to generate summary for VK ticket {ticket.id}: {e}"
                )

            # Обновляем статус заявки на OPEN если нужно
            if ticket.status != models.TicketStatus.OPEN:
                await crud.update_ticket_status(
                    session, ticket.id, models.TicketStatus.OPEN
                )

            tickets_payload = await _serialize_tickets(session)

        rag_service.reset_history(ticket.id)
        await _broadcast_message(ticket.id, MessageRead.from_orm(db_message))
        await connection_manager.broadcast_conversations(tickets_payload)

        notice = "✅ Заявка создана. Ожидайте ответа оператора."
        await _send_vk_message(user_id, notice)
        await _send_bot_message(ticket.id, notice, is_system=True)

        return ticket.id

    async def _send_bot_message(ticket_id: int, text: str, is_system: bool = False):
        """Отправляет сообщение бота в тикет"""
        async with session_maker() as session:
            db_message = await crud.add_message(
                session,
                ticket_id=ticket_id,
                sender=BOT_SENDER,
                text=text,
                is_system=is_system,
                is_read=False,
            )
            await session.refresh(db_message)
            await _broadcast_message(ticket_id, MessageRead.from_orm(db_message))
            tickets_payload = await _serialize_tickets(session)
            await connection_manager.broadcast_conversations(tickets_payload)

    async def _send_vk_message(user_id: int, text: str, keyboard: dict | None = None):
        """Отправляет сообщение в VK"""
        try:
            logger.info(
                f"VK: Attempting to send message to user {user_id}: {text[:100]}..."
            )
            # Проверяем, что user_id положительный (для пользователей)
            if user_id > 0:
                logger.info(f"VK: User ID is positive: {user_id}, proceeding with send")
                # Use peer_id (works for users and chats) and a random_id generated by vk_api
                try:
                    random_id = vk_api.utils.get_random_id()
                except Exception:
                    random_id = 0
                params = {"peer_id": user_id, "message": text, "random_id": random_id}
                if keyboard:
                    try:
                        # Ensure keyboard explicitly contains inline:true
                        if isinstance(keyboard, dict) and "inline" not in keyboard:
                            keyboard["inline"] = True
                        params["keyboard"] = json.dumps(keyboard, ensure_ascii=False)
                        try:
                            # extract option labels and store in pending_options for numeric replies
                            rows = (
                                keyboard.get("buttons", [])
                                if isinstance(keyboard, dict)
                                else []
                            )
                            opts = []
                            for row in rows:
                                for btn in row:
                                    try:
                                        lbl = btn.get("action", {}).get("label")
                                    except Exception:
                                        lbl = None
                                    if lbl:
                                        opts.append(lbl)
                            if opts:
                                pending_options[user_id] = opts
                        except Exception:
                            pass
                    except Exception:
                        params["keyboard"] = json.dumps(keyboard)
                logger.debug(f"VK: Sending params: {params}")
                try:
                    result = vk.messages.send(**params)
                    logger.info(
                        f"VK: Message sent successfully to user {user_id}, result: {result}"
                    )
                except vk_api.exceptions.ApiError as api_e:
                    # Handle common VK errors (e.g., chat bot feature not enabled)
                    logger.error(
                        f"VK: ApiError while sending message to {user_id}: {api_e}"
                    )
                    # If chat bot feature not enabled (912), advise enabling or use community token
                    try:
                        if (
                            hasattr(api_e, "code")
                            and api_e.code == 912
                            or "912" in str(api_e)
                        ):
                            logger.error(
                                "VK: ApiError [912] detected - Chat bot feature may be disabled for this token. Enable Chat bot feature in group settings or use a community token."
                            )
                    except Exception:
                        pass
                    # Fallback: if keyboard was provided, send a numbered text fallback
                    if keyboard:
                        try:
                            # build numbered fallback from keyboard buttons
                            rows = (
                                keyboard.get("buttons", [])
                                if isinstance(keyboard, dict)
                                else []
                            )
                            options = []
                            idx = 1
                            for row in rows:
                                for btn in row:
                                    label = None
                                    try:
                                        label = btn.get("action", {}).get("label")
                                    except Exception:
                                        label = str(btn)
                                    if label:
                                        options.append(f"{idx}. {label}")
                                        idx += 1
                            if options:
                                fallback_text = (
                                    text
                                    + "\n\n"
                                    + "Пожалуйста, выберите опцию:\n"
                                    + "\n".join(options)
                                )
                                vk.messages.send(
                                    peer_id=user_id,
                                    message=fallback_text,
                                    random_id=vk_api.utils.get_random_id(),
                                )
                                try:
                                    # store pending numeric options mapping
                                    pending_options[user_id] = [
                                        opt.split(". ", 1)[1] if ". " in opt else opt
                                        for opt in options
                                    ]
                                except Exception:
                                    pass
                        except Exception as e:
                            logger.debug(
                                f"VK: Failed to send fallback numbered message: {e}"
                            )
                    return
            else:
                logger.warning(f"VK: Invalid user_id {user_id}, cannot send message")
        except Exception as e:
            logger.error(f"VK: Failed to send VK message to {user_id}: {e}")
            logger.error(f"VK: Error details: {type(e).__name__}: {e}")
            # Try to get more details about the error
            if hasattr(e, "error_code"):
                logger.error(f"VK: Error code: {e.error_code}")
            if hasattr(e, "error_msg"):
                logger.error(f"VK: Error message: {e.error_msg}")

    async def _send_vk_typing(user_id: int):
        """Отправляет статус 'печатает' в VK"""
        try:
            logger.debug(f"VK: Sending typing status to user {user_id}")
            # VK API: setActivity показывает статус "печатает" на 10 секунд
            await asyncio.to_thread(
                vk.messages.setActivity, user_id=user_id, type="typing"
            )
        except Exception as e:
            logger.warning(f"VK: Failed to send typing status to {user_id}: {e}")

    def _build_vk_keyboard(buttons: list[list[str]]) -> dict | None:
        """Builds VK inline keyboard-like payload as a simple JSON keyboard.

        buttons: list of rows, each row is a list of strings (labels).
        VK doesn't support Telegram-style inline buttons for personal messages the same way,
        but we can send a JSON keyboard used by some clients or simply include numbered options.
        We'll return a minimal structure compatible with VK messages API 'keyboard' param.
        """
        try:
            # Simple VK keyboard structure with inline and payloads
            kb = {"one_time": False, "buttons": [], "inline": True}
            for row in buttons:
                kb_row = []
                for label in row:
                    # action.payload must be a JSON-encoded string
                    try:
                        payload = json.dumps({"topic": label}, ensure_ascii=False)
                    except Exception:
                        payload = json.dumps({"topic": str(label)})
                    kb_row.append(
                        {
                            "action": {
                                "type": "text",
                                "payload": payload,
                                "label": label,
                            },
                            "color": "default",
                        }
                    )
                kb["buttons"].append(kb_row)
            # register pending options for users when this keyboard will be sent
            # Note: we cannot know user_id here; caller should set pending_options[user_id] after send
            return kb
        except Exception:
            return None

    def _build_confirm_operator_keyboard() -> dict | None:
        """Build a keyboard with Confirm/Cancel for operator request."""
        try:
            kb = {"one_time": True, "buttons": [], "inline": True}
            yes_payload = json.dumps({"confirm_operator": "yes"}, ensure_ascii=False)
            no_payload = json.dumps({"confirm_operator": "no"}, ensure_ascii=False)
            kb["buttons"].append(
                [
                    {
                        "action": {
                            "type": "text",
                            "payload": yes_payload,
                            "label": "Да",
                        },
                        "color": "primary",
                    },
                    {
                        "action": {
                            "type": "text",
                            "payload": no_payload,
                            "label": "Нет",
                        },
                        "color": "default",
                    },
                ]
            )
            return kb
        except Exception:
            return None

    async def _send_bot_message(ticket_id: int, text: str, is_system: bool = False):
        """Отправляет сообщение бота в тикет"""
        async with session_maker() as session:
            db_message = await crud.add_message(
                session,
                ticket_id=ticket_id,
                sender=BOT_SENDER,
                text=text,
                is_system=is_system,
                is_read=False,
            )
            await session.refresh(db_message)
            await _broadcast_message(ticket_id, MessageRead.from_orm(db_message))
            tickets_payload = await _serialize_tickets(session)
            await connection_manager.broadcast_conversations(tickets_payload)

    async def _get_average_response_time() -> str:
        """Вычисляет среднее время ответа оператора"""
        try:
            async with session_maker() as session:
                from sqlalchemy import select, func
                from datetime import datetime, timedelta

                thirty_days_ago = datetime.utcnow() - timedelta(days=30)
                stmt = select(models.Ticket).where(
                    models.Ticket.status == models.TicketStatus.CLOSED,
                    models.Ticket.created_at >= thirty_days_ago,
                )
                result = await session.execute(stmt)
                tickets = result.scalars().all()

                if not tickets:
                    return "обычно быстро"

                response_times = []
                for ticket in tickets:
                    messages = await crud.list_messages_for_ticket(session, ticket.id)
                    operator_message = next(
                        (m for m in messages if m.sender == "operator"), None
                    )
                    if operator_message and ticket.created_at:
                        delta = operator_message.created_at - ticket.created_at
                        response_times.append(delta.total_seconds() / 60)

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

    async def _send_bot_message_to_ticket_vk(user_id: int, text: str) -> None:
        """Сохраняет сообщение бота в тикет пользователя VK (если тикет существует)"""
        chat_id = f"vk_{user_id}"
        async with session_maker() as session:
            ticket = await crud.get_open_ticket_by_chat_id(session, chat_id)
            if ticket:
                should_mark_as_read = connection_manager.has_active_chat_connections(
                    ticket.id
                )

                db_message = await crud.add_message(
                    session,
                    ticket_id=ticket.id,
                    sender=BOT_SENDER,
                    text=text,
                    is_read=should_mark_as_read,
                )

                await _broadcast_message(ticket.id, MessageRead.from_orm(db_message))
                await _broadcast_tickets()

    async def _answer_with_rag_only_vk(user_id: int, user_text: str) -> None:
        """Отвечает пользователю через RAG и сохраняет ответ в тикет (если есть)"""
        logger.info(
            f"VK: Starting RAG processing for user {user_id}, text: {user_text[:50]}..."
        )

        # Отправляем статус "печатает"
        await _send_vk_typing(user_id)

        try:
            conversation_id = f"vk_{user_id}"
            logger.info(
                f"VK: Calling RAG generate_reply for conversation {conversation_id}"
            )
            # Если это HybridRAGService - метод асинхронный, просто await
            # Если это RAGService - метод синхронный, используем to_thread
            try:
                # Пробуем вызвать как async
                rag_result: RAGResult = await rag_service.generate_reply(
                    conversation_id,
                    user_text,
                )
            except TypeError:
                # Если метод синхронный, вызываем через to_thread
                rag_result: RAGResult = await asyncio.to_thread(
                    rag_service.generate_reply,
                    conversation_id,
                    user_text,
                )
            logger.info(
                f"VK: RAG result received: operator_requested={rag_result.operator_requested}, answer_length={len(rag_result.final_answer) if rag_result.final_answer else 0}"
            )
        except Exception as exc:
            logger.exception("VK RAG generation failed: %s", exc)
            fallback = "Не смогла обработать запрос. Попробуйте ещё раз или позовите оператора."
            await _send_vk_message(user_id, fallback)
            return

        if rag_result.operator_requested:
            logger.info(f"VK: RAG requested operator for user {user_id}")
            # RAG решил, что нужен оператор - показываем предложение
            avg_response_time = await _get_average_response_time()

            combined_text = (
                f"{rag_result.final_answer}\n\n"
                "Хотите связаться с живым оператором? После подтверждения мы создадим заявку и уведомим доступных операторов.\n\n"
                f"⏱ Среднее время ответа: {avg_response_time}\n\n"
                "Если хотите подключить оператора — отправьте 'да'."
            )
            # For VK we don't send a 'No' option; instruct user to reply 'да' to call operator.
            await _send_vk_message(user_id, combined_text)

            # Сохраняем ответ бота в тикет
            await _send_bot_message_to_ticket_vk(user_id, combined_text)

            # Сохраняем в историю RAG
            rag_service.add_to_history(conversation_id, user_text, False)
            rag_service.add_to_history(conversation_id, combined_text, True)
        else:
            logger.info(f"VK: RAG provided direct answer for user {user_id}")
            response_text = f"{rag_result.final_answer}"
            logger.info(
                f"VK: Sending RAG response to user {user_id}: {response_text[:100]}..."
            )
            await _send_vk_message(user_id, response_text)

            # Сохраняем ответ бота в тикет
            await _send_bot_message_to_ticket_vk(user_id, response_text)

            # Сохраняем в истории RAG
            rag_service.add_to_history(conversation_id, user_text, False)
            rag_service.add_to_history(conversation_id, rag_result.final_answer, True)

    async def _handle_vk_voice_message(user_id: int, attachment: dict, message_id: int):
        """Обрабатывает голосовое сообщение из VK"""
        try:
            logger.info(f"VK: Processing voice message from user {user_id}")

            # Normalize attachment: it may be a dict or a short string; if it's not a dict, try to fetch full message
            if not isinstance(attachment, dict):
                logger.debug(
                    "VK: attachment is not dict, attempting to fetch message by id to resolve attachments"
                )
                try:
                    msg_info = vk.messages.getById(message_ids=message_id)
                    items = msg_info.get("items", [])
                    if items:
                        msg_obj = items[0]
                        atts = msg_obj.get("attachments", [])
                        # find first audio_message attachment
                        found = None
                        for att in atts:
                            if (
                                isinstance(att, dict)
                                and att.get("type") == "audio_message"
                            ):
                                found = att
                                break
                        if found:
                            attachment = found
                        else:
                            logger.warning(
                                f"VK: No audio_message attachment found in fetched message {message_id}"
                            )
                            await _send_vk_message(
                                user_id,
                                vk_responses.get(
                                    "voice_processing_error",
                                    "❌ Не удалось получить голосовое сообщение.",
                                ),
                            )
                            return
                    else:
                        logger.warning(
                            f"VK: getById returned no items for message {message_id}"
                        )
                        await _send_vk_message(
                            user_id,
                            vk_responses.get(
                                "voice_processing_error",
                                "❌ Не удалось получить голосовое сообщение.",
                            ),
                        )
                        return
                except Exception as e:
                    logger.debug(
                        f"VK: Failed to fetch message by id for attachment: {e}"
                    )
                    await _send_vk_message(
                        user_id,
                        vk_responses.get(
                            "voice_processing_error",
                            "❌ Не удалось получить голосовое сообщение.",
                        ),
                    )
                    return

            # НЕ сохраняем placeholder — будем сохранять только расшифровку.
            # Это позволяет в UI показывать только понятную расшифровку, а не '[Голосовое сообщение]'.
            ticket_id_tmp = None
            was_ticket = False

            # Отправляем статус "печатает"
            await _send_vk_typing(user_id)

            # Уведомляем пользователя о начале обработки
            processing_msg = vk_responses.get(
                "processing_message", "⏳ Обрабатываю ваш запрос..."
            )
            await _send_vk_message(user_id, processing_msg)

            # Получаем URL аудио файл (поддерживаем разные варианты полей)
            audio_message = {}
            if isinstance(attachment, dict):
                audio_message = (
                    attachment.get("audio_message") or attachment.get("audio") or {}
                )
            # try multiple keys commonly used
            audio_url = None
            if isinstance(audio_message, dict):
                audio_url = (
                    audio_message.get("link_mp3")
                    or audio_message.get("link_ogg")
                    or audio_message.get("link")
                    or audio_message.get("url")
                )

            if not audio_url:
                # Попробуем получить через messages.getById (ещё один шанс)
                try:
                    msg_info = vk.messages.getById(message_ids=message_id)
                    items = msg_info.get("items", [])
                    if items:
                        msg_obj = items[0]
                        atts = msg_obj.get("attachments", [])
                        for att in atts:
                            if (
                                isinstance(att, dict)
                                and att.get("type") == "audio_message"
                            ):
                                am = att.get("audio_message") or att.get("audio") or {}
                                audio_url = (
                                    am.get("link_mp3")
                                    or am.get("link_ogg")
                                    or am.get("link")
                                    or am.get("url")
                                )
                                if audio_url:
                                    break
                except Exception as e:
                    logger.debug(f"VK: getById extra attempt failed: {e}")

            if not audio_url:
                await _send_vk_message(
                    user_id,
                    vk_responses.get(
                        "voice_processing_error",
                        "❌ Не удалось получить голосовое сообщение.",
                    ),
                )
                # Mark placeholder as system note if persisted earlier
                try:
                    if ticket_id_tmp:
                        await _send_bot_message(
                            ticket_id_tmp,
                            "[Голосовое сообщение получено, но не удалось скачать аудио]",
                            is_system=True,
                        )
                except Exception:
                    pass
                return

            # Скачиваем аудио файл
            import requests
            import tempfile
            import os

            # Скачиваем в отдельном потоке (синхронно)
            response = await asyncio.to_thread(requests.get, audio_url, timeout=20)
            if response.status_code != 200:
                await _send_vk_message(
                    user_id,
                    vk_responses.get(
                        "voice_download_error",
                        "❌ Не удалось скачать голосовое сообщение.",
                    ),
                )
                return

            # Сохраняем во временный файл
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name

            try:
                # Преобразуем голос в текст
                try:
                    transcribed_text = (
                        await rag_service.speech_to_text.transcribe_audio(
                            temp_file_path
                        )
                    )
                except Exception as e:
                    logger.exception(
                        f"VK: speech_to_text failed for user {user_id}: {e}"
                    )
                    transcribed_text = None

                if not transcribed_text:
                    # Если распознавание не удалось, уведомляем пользователя и НЕ создаём пустую запись
                    await _send_vk_message(
                        user_id,
                        vk_responses.get(
                            "speech_recognition_error",
                            "❌ Не удалось распознать речь. Попробуйте еще раз или напишите текстом.",
                        ),
                    )
                    return

                # Обрабатываем расшифровку как обычное текстовое сообщение — сохраняем транскрипт
                ticket_id, has_ticket = await _persist_message_vk(
                    user_id, transcribed_text, USER_SENDER, message_id
                )

                # Если есть заявка, сообщение уже сохранено — ничего не отправляем от бота
                # Если нет заявки — отвечаем через RAG
                if not has_ticket or not ticket_id:
                    await _answer_with_rag_only_vk(user_id, transcribed_text)

            finally:
                # Удаляем временный файл
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        except Exception as e:
            logger.exception(
                f"VK: Error processing voice message from user {user_id}: {e}"
            )
            await _send_vk_message(
                user_id,
                vk_responses.get(
                    "voice_general_error",
                    "❌ Произошла ошибка при обработке голосового сообщения. Попробуйте написать текстом.",
                ),
            )

    async def handle_vk_message(event):
        """Обрабатывает входящее сообщение из VK"""
        # Normalize different event shapes coming from VkLongPoll / VkBotLongPoll
        user_id = None
        text = ""
        message_id = 0
        attachments = []

        try:
            # Case: event.object -> VkBotLongPoll event wrapper
            if hasattr(event, "object") and isinstance(event.object, dict):
                msg = event.object.get("message") or event.object
                # payload may be present in bot events
                payload_raw = msg.get("payload") if isinstance(msg, dict) else None
                payload = None
                if payload_raw:
                    try:
                        payload = json.loads(payload_raw)
                    except Exception:
                        payload = None
                user_id = msg.get("from_id") or msg.get("peer_id") or msg.get("user_id")
                text = msg.get("text") or ""
                message_id = msg.get("id") or msg.get("message_id") or 0
                attachments = msg.get("attachments") or []
                # If payload contains a topic, override text with it for processing
                if payload and isinstance(payload, dict):
                    if payload.get("topic"):
                        text = payload.get("topic")
                    # operator confirm payload handling
                    if payload.get("confirm_operator"):
                        choice = str(payload.get("confirm_operator")).lower()
                        if choice == "yes" or choice == "y" or choice == "да":
                            # create ticket now
                            ticket_id = await _create_ticket_and_add_message_vk(
                                user_id,
                                "Пользователь подтвердил подключение оператора",
                                message_id,
                            )
                            if ticket_id:
                                await _broadcast_tickets()
                            # notify user
                            await _send_vk_message(
                                user_id,
                                vk_responses.get(
                                    "operator_connected",
                                    "✅ Оператор будет подключен. Ожидайте ответа.",
                                ),
                            )
                        else:
                            await _send_vk_message(
                                user_id,
                                vk_responses.get(
                                    "operator_cancelled",
                                    "Отмена подключения оператора.",
                                ),
                            )
                        # We handled the payload action — stop further processing
                        return
            # Case: event.message (some wrappers)
            elif hasattr(event, "message") and isinstance(event.message, dict):
                msg = event.message
                payload_raw = msg.get("payload") if isinstance(msg, dict) else None
                payload = None
                if payload_raw:
                    try:
                        payload = json.loads(payload_raw)
                    except Exception:
                        payload = None
                user_id = msg.get("from_id") or msg.get("peer_id")
                text = msg.get("text") or ""
                message_id = msg.get("id") or 0
                attachments = msg.get("attachments") or []
                if payload and isinstance(payload, dict) and payload.get("topic"):
                    text = payload.get("topic")
            # Case: simple Event with attrs user_id/text
            elif hasattr(event, "user_id"):
                user_id = getattr(event, "user_id", None)
                text = getattr(event, "text", "") or ""
                message_id = getattr(event, "message_id", getattr(event, "id", 0))
                attachments = getattr(event, "attachments", []) or []
            else:
                logger.error(f"VK: Event missing required attributes: {dir(event)}")
                return
        except Exception as e:
            logger.exception(f"VK: Failed to normalize event: {e}")
            return

        # attachments may be strings (short forms) or dicts; normalize
        def _find_audio_in_attachments(atts_list):
            if not atts_list:
                return None
            for att in atts_list:
                try:
                    if isinstance(att, dict):
                        typ = att.get("type")
                        if typ == "audio_message":
                            return att
                        # some shapes may contain audio under 'audio_message' or 'audio'
                        inner = att.get("audio_message") or att.get("audio")
                        if isinstance(inner, dict) and (
                            inner.get("link_mp3")
                            or inner.get("link_ogg")
                            or inner.get("url")
                            or inner.get("link")
                        ):
                            return {"type": "audio_message", "audio_message": inner}
                        # doc attachments may contain audio files
                        if typ == "doc":
                            doc = att.get("doc") or {}
                            if doc.get("mime_type", "").startswith("audio") or doc.get(
                                "ext"
                            ) in ["mp3", "ogg"]:
                                return {
                                    "type": "audio_message",
                                    "audio_message": {
                                        "link": doc.get("url")
                                        or doc.get("preview", {}).get("photo")
                                    },
                                }
                    elif isinstance(att, str):
                        # short form string like 'audio_message123_456' - ignore here, we'll fetch full message
                        continue
                except Exception:
                    continue
            return None

        # quick check in provided attachments
        audio_att = None
        if attachments and isinstance(attachments, list):
            audio_att = _find_audio_in_attachments(attachments)
            if audio_att:
                logger.info(
                    f"VK: Detected audio attachment in provided attachments for user {user_id}"
                )
                await _handle_vk_voice_message(user_id, audio_att, message_id)
                return

        # If there is no text and no audio found yet, try to fetch the message by id to inspect attachments
        if (not text or not text.strip()) and message_id:
            try:
                msg_info = vk.messages.getById(message_ids=message_id)
                items = msg_info.get("items", [])
                if items:
                    msg_obj = items[0]
                    atts = msg_obj.get("attachments", [])
                    audio_att = _find_audio_in_attachments(atts)
                    if audio_att:
                        logger.info(
                            f"VK: Detected audio attachment after getById for user {user_id}"
                        )
                        await _handle_vk_voice_message(user_id, audio_att, message_id)
                        return
            except Exception as e:
                logger.debug(f"VK: getById secondary check failed: {e}")

        logger.info(
            f"VK: Processing message from user {user_id}: '{text}' (id={message_id})"
        )

        # If user replied with a single number and we have pending options, map it
        if isinstance(text, str) and text.strip().isdigit():
            try:
                num = int(text.strip())
                opts = pending_options.get(user_id)
                if opts and 1 <= num <= len(opts):
                    mapped = opts[num - 1]
                    logger.info(
                        f"VK: Mapped numeric reply {num} -> '{mapped}' for user {user_id}"
                    )
                    text = mapped
                    # clear pending options after selection
                    try:
                        del pending_options[user_id]
                    except KeyError:
                        pass
            except Exception:
                pass

        # Проверяем блокировку
        async with user_locks[user_id]:
            # Проверяем, есть ли открытая заявка
            async with session_maker() as session:
                ticket = await crud.get_open_ticket_by_chat_id(session, f"vk_{user_id}")

                if ticket is None:
                    # НЕТ ЗАЯВКИ - создаём новую при первом сообщении
                    logger.info(
                        f"VK: Creating new ticket for user {user_id} on first message"
                    )
                    ticket_id = await _create_ticket_and_add_message_vk(
                        user_id, text, message_id
                    )

                    # Отвечаем через RAG
                    await _answer_with_rag_only_vk(user_id, text)
                    return

                # ЕСТЬ ЗАЯВКА - проверяем, подключен ли оператор
                if ticket.operator_requested:
                    # Оператор подключен - только сохраняем сообщение, бот молчит
                    await _persist_message_vk(user_id, text, USER_SENDER, message_id)
                    await _broadcast_tickets()
                    return
                else:
                    # Оператор НЕ подключен - сохраняем сообщение И отвечаем через RAG
                    await _persist_message_vk(user_id, text, USER_SENDER, message_id)
                    await _answer_with_rag_only_vk(user_id, text)
                    return

            # Нет открытой заявки - проверяем специальные команды
            low = text.lower().strip()
            agree_phrases = [
                "да",
                "yes",
                "подключить оператора",
                "да, подключить оператора",
            ]
            operator_phrases = [
                "позови оператора",
                "позовите оператора",
                "вызвать оператора",
                "хочу оператора",
                "подключите оператора",
                "позови",
                "позовите",
            ]

            # Если пользователь явно ответил 'да' на предложение подключить оператора
            if low in agree_phrases:
                logger.info(f"VK: User {user_id} agreed to connect operator")
                ticket_id = await _create_ticket_and_add_message_vk(
                    user_id, text, message_id
                )
                if ticket_id:
                    await _broadcast_tickets()
                return

            # Если пользователь просит оператора разными фразами или использует слово 'оператор' с глаголом
            def requests_operator(s: str) -> bool:
                if not s:
                    return False
                for p in operator_phrases:
                    if p in s:
                        return True
                # heuristic: 'оператор' + nearby verb-like words
                if "оператор" in s:
                    verbs = ["поз", "выз", "подкл", "хоч", "нужн"]
                    for v in verbs:
                        if v in s:
                            return True
                return False

            if requests_operator(low):
                logger.info(f"VK: User {user_id} requested operator via phrase: {text}")
                # Instead of creating ticket immediately, send instruction to reply 'да' to call operator
                avg_response_time = await _get_average_response_time()
                confirm_text = vk_responses.get(
                    "operator_request_confirmation", ""
                ).format(avg_response_time=avg_response_time)
                await _send_vk_message(user_id, confirm_text)
                # Save a system note in history that user asked to call operator (not creating ticket yet)
                try:
                    rag_service.add_to_history(f"vk_{user_id}", text, False)
                    rag_service.add_to_history(f"vk_{user_id}", confirm_text, True)
                except Exception:
                    pass
                return

            # Проверяем, выбрал ли пользователь тему
            topic_selected = await _check_topic_selection_vk(user_id, text)
            if topic_selected:
                logger.info(f"VK: User {user_id} selected topic: {topic_selected}")
                # Обрабатываем выбранную тему как новый запрос
                await _answer_with_rag_only_vk(user_id, topic_selected)
                return

            # Обычный ответ через RAG
            logger.info(f"VK: Processing через RAG для пользователя {user_id}")
            await _answer_with_rag_only_vk(user_id, text)

    async def _check_topic_selection_vk(user_id: int, text: str) -> str | None:
        """Проверяет, выбрал ли пользователь тему из предложенных"""
        try:
            # Извлекаем числовой user_id для истории
            user_id_int = user_id
            chat_history = rag_service.get_chat_history(user_id_int)

            # Ищем последнее сообщение бота с темами
            for msg in reversed(chat_history):
                if not msg.is_user and "Хотите узнать подробнее" in msg.message:
                    # Разбираем темы из сообщения
                    lines = msg.message.split("\n")
                    topics = []
                    for line in lines:
                        if line.strip().startswith("• "):
                            topic = line.strip()[2:]  # Убираем "• "
                            topics.append(topic)

                    if topics:
                        # Проверяем, является ли текст номером или названием темы
                        text_lower = text.lower().strip()

                        # Проверяем номер темы (1, 2, 3...)
                        try:
                            topic_index = int(text) - 1  # 1-based to 0-based
                            if 0 <= topic_index < len(topics):
                                return topics[topic_index]
                        except ValueError:
                            pass

                        # Проверяем точное совпадение названия темы
                        for topic in topics:
                            if text_lower == topic.lower():
                                return topic

                        # Проверяем частичное совпадение
                        for topic in topics:
                            if (
                                text_lower in topic.lower()
                                or topic.lower() in text_lower
                            ):
                                return topic

                    break  # Проверяем только последнее сообщение с темами

            return None
        except Exception as e:
            logger.exception(
                f"VK: Error checking topic selection for user {user_id}: {e}"
            )
            return None

    # Очередь для сообщений из потока
    message_queue = asyncio.Queue()

    # Основной цикл прослушивания
    def run_vk_bot_sync():
        """Синхронная функция для работы в отдельном потоке"""
        logger.info(
            f"VK: Starting VK longpoll listener (bot_mode={is_bot_longpoll})..."
        )
        try:
            for event in longpoll.listen():
                logger.info(
                    f"VK: Received event type={event.type}, user_id={getattr(event, 'user_id', 'N/A')}, text={getattr(event, 'text', 'N/A')}"
                )
                if is_bot_longpoll:
                    # VkBotLongPoll
                    if event.type == VkBotEventType.MESSAGE_NEW:
                        logger.info(
                            f"VK: Processing bot message from user {event.user_id}: {event.text}"
                        )
                        # Помещаем событие в очередь для обработки в asyncio
                        asyncio.run_coroutine_threadsafe(message_queue.put(event), loop)
                    else:
                        logger.debug(f"VK: Ignoring bot event type={event.type}")
                else:
                    # VkLongPoll fallback
                    if event.type == VkEventType.MESSAGE_NEW and event.to_me:
                        logger.info(
                            f"VK: Processing personal message to me from user {event.user_id}: {event.text}"
                        )
                        # Помещаем событие в очередь для обработки в asyncio
                        asyncio.run_coroutine_threadsafe(message_queue.put(event), loop)
                    else:
                        logger.debug(
                            f"VK: Ignoring personal event type={event.type}, to_me={getattr(event, 'to_me', 'N/A')}"
                        )
        except Exception as e:
            logger.error(f"VK: LongPoll listener crashed: {e}")
            logger.error(f"VK: Error type: {type(e).__name__}")

    def run_vk_polling():
        """Polling функция для получения сообщений"""
        import time

        logger.info("VK: Starting VK polling listener...")
        last_message_id = 0

        while True:
            try:
                # Получаем новые сообщения
                response = vk.messages.getConversations(count=20, filter="unread")
                conversations = response.get("items", [])

                for conv in conversations:
                    conversation = conv.get("conversation", {})
                    last_message = conv.get("last_message", {})

                    if last_message and last_message.get("id", 0) > last_message_id:
                        user_id = last_message.get("from_id", 0)
                        text = last_message.get("text", "")
                        message_id = last_message.get("id", 0)

                        # Проверяем, что это сообщение пользователю (не от нас)
                        if (
                            user_id > 0 and text
                        ):  # user_id > 0 означает пользователя, не сообщество
                            logger.info(
                                f"VK: Polled message from user {user_id}: {text}"
                            )

                            # Создаем объект события для совместимости
                            class MockEvent:
                                def __init__(self, user_id, text, message_id):
                                    self.user_id = user_id
                                    self.text = text
                                    self.message_id = message_id

                            event = MockEvent(user_id, text, message_id)
                            asyncio.run_coroutine_threadsafe(
                                message_queue.put(event), loop
                            )

                            # Помечаем как прочитанное
                            try:
                                vk.messages.markAsRead(peer_id=user_id)
                            except Exception as e:
                                logger.debug(f"VK: Failed to mark message as read: {e}")

                            last_message_id = max(last_message_id, message_id)

                time.sleep(2)  # Проверяем каждые 2 секунды

            except Exception as e:
                logger.error(f"VK: Polling error: {e}")
                time.sleep(5)  # Ждем 5 секунд при ошибке

    # Получаем текущий event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Запускаем longpoll в отдельном потоке или polling
    if longpoll is not None:
        # LongPoll режим
        thread = threading.Thread(target=run_vk_bot_sync, daemon=True)
        thread.start()
    else:
        # Polling режим
        logger.info("VK: Starting polling mode")
        thread = threading.Thread(target=run_vk_polling, daemon=True)
        thread.start()

    async def run_vk_bot():
        """Асинхронная функция обработки очереди сообщений"""
        logger.info("VK: Async message processor started")
        while True:
            logger.debug("VK: Waiting for message in queue...")
            # Ждем новое сообщение из очереди
            event = await message_queue.get()
            logger.info(
                f"VK: Processing message from queue: user {event.user_id}, text: {event.text}"
            )
            await handle_vk_message(event)
            message_queue.task_done()
            logger.debug("VK: Message processed")

    return run_vk_bot


async def start_vk_bot(run_vk_bot_func):
    try:
        logger.info("VK: Starting VK bot task...")
        await run_vk_bot_func()
    except asyncio.CancelledError:
        logger.info("VK: Bot task cancelled")
        raise
    except Exception as e:
        logger.error(f"VK: Bot task failed: {e}")
        logger.exception("VK bot error details:")
        raise
