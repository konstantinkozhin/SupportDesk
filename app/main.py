from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import AsyncIterator

from dotenv import load_dotenv

load_dotenv()


# Настройка логирования
def setup_logging():
    """Настройка системы логирования"""
    # Создаем директорию для логов если её нет
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    # Формат логов
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Очищаем существующие обработчики
    root_logger.handlers.clear()

    # Обработчик для файла (с ротацией)
    log_file = os.path.join(
        logs_dir, f"support_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"  # 10 MB
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))

    # Обработчик для консоли
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))

    # Добавляем обработчики
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Настраиваем уровни для сторонних библиотек
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("aiogram").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    logging.info("=" * 80)
    logging.info("Система логирования инициализирована")
    logging.info(f"Логи сохраняются в: {log_file}")
    logging.info("=" * 80)


# Инициализируем логирование при импорте модуля
setup_logging()

from aiogram import Bot
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    Body,
)
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from app import auth
from app.bots import (
    create_dispatcher,
    start_bot,
    create_vk_bot,
    start_vk_bot,
    set_bot_instance,
)
from app.config import load_rag_config, load_app_config
from app.db import (
    models,
    tickets_crud as crud,
    tickets_crud,  # Для работы с чанками knowledge
    get_tickets_session,
    get_knowledge_session,
    init_db,
    TicketsSessionLocal,
    KnowledgeSessionLocal,
    TicketRead,
    KnowledgeStats,
    MessageCreate,
    MessageRead,
)
from app.rag.hybrid_service import get_hybrid_rag_service, HybridRAGService
from app.services import ConnectionManager, SimulatorService
from app.auth import require_permission, require_admin, get_user_permissions

logger = logging.getLogger(__name__)

connection_manager = ConnectionManager()

# Глобальная ссылка на главный event loop для использования в agent_tools
_main_loop = None


templates = Jinja2Templates(directory="app/templates")


def _serialize_tickets(tickets: list[models.Ticket]) -> list[dict]:
    result = []
    for ticket in tickets:
        # Ensure telegram_chat_id is a string when it contains non-numeric VK ids like 'vk_123'
        try:
            # Try direct conversion first
            ticket_data = TicketRead.from_orm(ticket).model_dump(mode="json")
        except Exception:
            # Fallback: coerce telegram_chat_id to string and build dict manually
            try:
                raw = ticket.__dict__.copy()
                raw["telegram_chat_id"] = str(getattr(ticket, "telegram_chat_id", ""))
                # Build minimal payload matching TicketRead fields
                ticket_data = {
                    "id": raw.get("id"),
                    "telegram_chat_id": raw.get("telegram_chat_id"),
                    "title": raw.get("title"),
                    "summary": raw.get("summary"),
                    "status": raw.get("status"),
                    "priority": raw.get("priority"),
                    "operator_requested": raw.get("operator_requested", False),
                    "created_at": raw.get("created_at"),
                    "first_response_at": raw.get("first_response_at"),
                    "closed_at": raw.get("closed_at"),
                    "updated_at": raw.get("updated_at"),
                }
            except Exception:
                # As last resort, skip this ticket
                logger.exception(
                    "Failed to serialize ticket %s", getattr(ticket, "id", "<unknown>")
                )
                continue
        # Подсчитываем непрочитанные сообщения от user и bot
        unread_count = sum(
            1
            for msg in ticket.messages
            if msg.sender in ["user", "bot"] and not msg.is_read
        )
        # Ограничиваем значение 99
        ticket_data["unread_count"] = min(unread_count, 99)
        result.append(ticket_data)
    return result


def _serialize_message(message: models.Message) -> dict:
    return MessageRead.from_orm(message).model_dump(mode="json")


async def _broadcast_conversations_update(
    session: AsyncSession, manager: ConnectionManager
) -> None:
    """Обновить список заявок для всех подключенных клиентов"""
    tickets = await crud.list_tickets(session, archived=False)
    await manager.broadcast_conversations(_serialize_tickets(tickets))


async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _main_loop

    logger.info("🚀 Starting application lifespan...")

    # Сохраняем ссылку на главный event loop
    import asyncio

    _main_loop = asyncio.get_running_loop()
    logger.info(f"Main event loop stored: {_main_loop}")

    # Определяем и логируем базовый URL приложения
    from app.utils import get_base_url

    base_url = get_base_url()
    logger.info(f"🌐 Application BASE_URL: {base_url}")

    await init_db()
    app.state.connection_manager = connection_manager

    # Загружаем полный конфиг (RAG + App)
    from app.config import load_config

    full_config = load_config()

    rag_config = load_rag_config()
    embeddings_cfg = rag_config.get("embeddings", {})

    # Используем гибридный RAG сервис с агентом
    rag_service = get_hybrid_rag_service()
    await rag_service.prepare()
    app.state.rag = rag_service

    # Прогреваем SentenceTransformer для agent_tools (ленивая инициализация)
    logger.info("Warming up SentenceTransformer for agent tools...")
    from app.rag.agent_tools import get_sentence_transformer

    get_sentence_transformer()  # Инициализирует модель один раз
    logger.info("SentenceTransformer ready")

    # Инициализируем FAQ service
    logger.info("Initializing FAQ service...")
    from app.rag.faq_service import set_faq_config, set_faq_cache

    faq_config = rag_config.get("faq", {})
    faq_update_interval = faq_config.get("update_interval_minutes", 5)
    faq_top_chunks = faq_config.get("top_chunks_count", 10)
    set_faq_config(faq_update_interval, faq_top_chunks)

    # Устанавливаем начальный FAQ (placeholder до первого обновления)
    initial_faq = [
        {
            "question": "Добро пожаловать в FAQ!",
            "answer": "Часто задаваемые вопросы автоматически обновляются на основе популярных запросов. Первое обновление произойдёт через несколько минут.",
        }
    ]
    set_faq_cache(initial_faq)

    logger.info(
        f"FAQ service ready (update every {faq_update_interval} minutes, top {faq_top_chunks} chunks)"
    )

    # Автоматически загружаем QA.xlsx если база знаний пустая
    logger.info("Checking if knowledge base needs initial data...")
    try:
        from app.rag.faq_service import load_qa_xlsx_if_empty
        import os

        qa_file_path = os.path.join(os.getcwd(), "QA.xlsx")
        async with KnowledgeSessionLocal() as session:
            loaded_count = await load_qa_xlsx_if_empty(session, qa_file_path)
            if loaded_count > 0:
                logger.info(
                    f"✅ Knowledge base initialized with {loaded_count} QA pairs from QA.xlsx"
                )
    except Exception as e:
        logger.error(f"❌ Error loading initial QA data: {e}")

    # Инициализируем симулятор
    simulator_service = SimulatorService(rag_service)
    app.state.simulator = simulator_service

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    bot: Bot | None = None
    dispatcher = None
    bot_task: asyncio.Task | None = None
    popularity_task: asyncio.Task | None = None

    if token:
        bot = Bot(token=token)
        # Pass rag_service to dispatcher for Telegram bot
        dispatcher = create_dispatcher(
            TicketsSessionLocal, connection_manager, rag_service, None
        )
        # Установить bot instance для использования в agent_tools
        set_bot_instance(bot, TicketsSessionLocal)
        app.state.bot = bot
        app.state.dispatcher = dispatcher
        bot_task = asyncio.create_task(start_bot(bot, dispatcher))
    else:
        logger.warning(
            "TELEGRAM_BOT_TOKEN is not set. Telegram integration is disabled."
        )
        app.state.bot = None
        app.state.dispatcher = None

    # VK Bot
    vk_token = os.getenv("VK_ACCESS_TOKEN")
    vk_bot_task: asyncio.Task | None = None
    logger.info(f"VK: Token found: {'YES' if vk_token else 'NO'}")
    if vk_token:
        logger.info("VK: Attempting to create VK bot...")
        vk_run_bot = create_vk_bot(
            TicketsSessionLocal, connection_manager, rag_service, vk_token
        )
        if vk_run_bot is not None:
            logger.info("VK: Bot created successfully, starting task...")
            app.state.vk_bot = vk_run_bot
            vk_bot_task = asyncio.create_task(start_vk_bot(vk_run_bot))
            logger.info("VK: Bot task started")
        else:
            logger.warning("VK: Bot disabled due to configuration issues")
            app.state.vk_bot = None
    else:
        logger.warning("VK_ACCESS_TOKEN is not set. VK integration is disabled.")
        app.state.vk_bot = None

    try:
        logger.info("✅ Application startup complete")
        yield
    finally:
        logger.info("🛑 Shutting down application...")
        if bot_task:
            logger.info("Stopping Telegram bot...")
            bot_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await bot_task
        if vk_bot_task:
            logger.info("Stopping VK bot...")
            vk_bot_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await vk_bot_task
        if popularity_task:
            popularity_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await popularity_task
        if bot:
            logger.info("Closing bot session...")
            await bot.session.close()
        logger.info("Closing all connections...")
        await connection_manager.close_all()
        logger.info("✅ Application shutdown complete")


app = FastAPI(title="Support Desk", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/health")
async def health_check():
    """Health check endpoint для Docker"""
    return {"status": "ok", "service": "Support Desk"}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Главная страница - перенаправляет после аутентификации пользователей"""
    if not auth.is_authenticated_request(request):
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/tickets", response_class=HTMLResponse)
@require_permission("tickets")
async def tickets_page(request: Request):
    """Страница заявок"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/tickets/{ticket_id}", response_class=HTMLResponse)
@require_permission("tickets")
async def ticket_detail_page(request: Request, ticket_id: int):
    """Страница конкретной заявки - возвращает тот же HTML, JavaScript загрузит нужные данные"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if auth.is_authenticated_request(request):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def login(request: Request):
    data = await request.json()
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    user = auth.validate_credentials(username, password)
    if not user:
        raise HTTPException(status_code=401, detail="Неверный логин или пароль")

    response = JSONResponse({"success": True})
    auth.issue_session_cookie(response, user["id"])
    return response


@app.post("/logout")
async def logout(request: Request):
    accept = request.headers.get("accept", "") or ""
    if "application/json" in accept and "text/html" not in accept:
        response = JSONResponse({"success": True})
    else:
        response = RedirectResponse(url="/login", status_code=303)

    # Очищаем cookie сессии
    auth.clear_session_cookie(response)

    # Добавляем заголовки для очистки кеша
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response


@app.get("/api/me")
async def get_current_user(request: Request):
    """Получить информацию о текущем пользователе"""
    if not auth.is_authenticated_request(request):
        raise HTTPException(status_code=401, detail="Not authenticated")

    from app.auth import user_manager

    user_id = auth.get_user_id_from_request(request)
    user = user_manager.get_user_by_id(user_id)

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {"username": user["username"], "authenticated": True}


@app.get("/api/permissions")
async def get_user_permissions_api(request: Request):
    """Получить права текущего активного пользователя"""
    if not auth.is_authenticated_request(request):
        raise HTTPException(status_code=401, detail="Not authenticated")

    permissions = get_user_permissions(request)
    return {"available_pages": permissions}


@app.get("/admin/knowledge", response_class=HTMLResponse)
@require_permission("knowledge")
async def knowledge_admin(request: Request):
    async with KnowledgeSessionLocal() as session:
        total = await tickets_crud.count_document_chunks(session)
    return templates.TemplateResponse(
        "knowledge.html",
        {"request": request, "entry_count": total},
    )


@app.get("/dashboard", response_class=HTMLResponse)
@require_permission("dashboard")
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/simulator", response_class=HTMLResponse)
@require_permission("simulator")
async def simulator(request: Request):
    return templates.TemplateResponse("simulator.html", {"request": request})


# ==================== FAQ (PUBLIC) ====================


@app.get("/faq", response_class=HTMLResponse)
async def faq_page(request: Request):
    """Публичная страница FAQ для пользователей"""
    return templates.TemplateResponse("faq.html", {"request": request})


@app.get("/api/faq")
async def get_faq():
    """Получить список FAQ вопросов (обновляется автоматически на основе популярных запросов)"""
    try:
        from app.rag.faq_service import get_faq_cache, start_faq_update_background

        # Запускаем обновление в фоне если нужно (неблокирующе)
        rag_config = load_rag_config()
        update_started = start_faq_update_background(rag_config)
        if update_started:
            logger.info("[API] FAQ background update started")

        # Сразу возвращаем текущий кэшированный FAQ (не ждём обновления)
        faq_items = get_faq_cache()

        logger.info(f"[API] FAQ requested, returning {len(faq_items)} items")

        return {"items": faq_items, "count": len(faq_items)}
    except Exception as e:
        logger.error(f"[API] Error getting FAQ: {e}")
        # Возвращаем пустой список в случае ошибки
        return {"items": [], "count": 0}


@app.get("/api/faq/search")
async def search_faq(q: str = ""):
    """Поиск по FAQ"""
    try:
        from app.rag.faq_service import search_faq

        results = await search_faq(q)

        logger.info(f"[API] FAQ search '{q}' returned {len(results)} results")

        return {"items": results, "count": len(results), "query": q}
    except Exception as e:
        logger.error(f"[API] Error searching FAQ: {e}")
        return {"items": [], "count": 0, "query": q}


@app.get("/api/dashboard/stats")
async def get_dashboard_stats(
    request: Request,
    session: AsyncSession = Depends(get_tickets_session),
    _: None = Depends(auth.ensure_api_auth),
):
    if not auth.is_authenticated_request(request):
        raise HTTPException(status_code=401, detail="Authentication required")

    # Получаем статистику
    tickets_stats = await crud.get_tickets_stats(session)
    response_time_stats = await crud.get_response_time_stats(session)
    daily_stats = await crud.get_daily_tickets_stats(session, days=30)
    daily_time_metrics = await crud.get_daily_time_metrics(session, days=30)

    # Метрики времени
    avg_response_time = await crud.get_average_response_time(session)
    avg_resolution_time = await crud.get_average_resolution_time(session)

    # Статистика по базе знаний
    async with KnowledgeSessionLocal() as knowledge_session:
        knowledge_count = await tickets_crud.count_document_chunks(knowledge_session)

    return {
        "tickets": tickets_stats,
        "response_times": response_time_stats,
        "daily_tickets": daily_stats,
        "daily_time_metrics": daily_time_metrics,
        "knowledge_entries": knowledge_count,
        "avg_response_time_minutes": avg_response_time,
        "avg_resolution_time_minutes": avg_resolution_time,
    }


@app.get("/api/dashboard/active_sessions")
async def get_active_sessions_stats(
    request: Request,
    session: AsyncSession = Depends(get_tickets_session),
    _: None = Depends(auth.ensure_api_auth),
):
    """Получить статистику ВСЕХ активных диалогов (бот-пользователь И оператор-пользователь) по классам и приоритетам"""
    if not auth.is_authenticated_request(request):
        raise HTTPException(status_code=401, detail="Authentication required")

    # Получаем ВСЕ активные тикеты (статус open или in_progress)
    active_tickets = await crud.list_tickets(session, archived=False)

    logger.info(f"[DASHBOARD] Найдено активных тикетов: {len(active_tickets)}")

    # Разделяем тикеты на группы
    bot_only_tickets = []  # Только диалоги с ботом (operator_requested=False)
    operator_tickets = []  # Диалоги с оператором (operator_requested=True)

    # Статистика по приоритетам (только для диалогов с ботом)
    bot_priority_stats = {
        "low": 0,
        "medium": 0,
        "high": 0,
    }

    # Статистика по классам (только для диалогов с ботом)
    bot_classification_stats = {}

    # Подсчет высокоприоритетных диалогов с ботом
    high_priority_bot_count = 0

    for ticket in active_tickets:
        logger.info(
            f"[DASHBOARD] Тикет #{ticket.id}: priority={ticket.priority}, classification={ticket.classification}, operator_requested={ticket.operator_requested}"
        )

        # Разделяем по типу
        if ticket.operator_requested:
            operator_tickets.append(ticket)
        else:
            bot_only_tickets.append(ticket)

            # Подсчет по приоритетам (только для бота)
            priority = ticket.priority or "medium"
            if priority in bot_priority_stats:
                bot_priority_stats[priority] += 1

            # Подсчет высокоприоритетных
            if priority == "high":
                high_priority_bot_count += 1

            # Подсчет по классификациям (только для бота)
            if ticket.classification:
                # Классификация может содержать несколько категорий через запятую
                categories = [cat.strip() for cat in ticket.classification.split(",")]
                for category in categories:
                    if category:
                        bot_classification_stats[category] = (
                            bot_classification_stats.get(category, 0) + 1
                        )

    result = {
        "total_active": len(active_tickets),
        "bot_only": len(bot_only_tickets),
        "with_operator": len(operator_tickets),
        "high_priority_bot": high_priority_bot_count,
        "bot_by_priority": bot_priority_stats,
        "bot_by_classification": bot_classification_stats,
    }

    logger.info(f"[DASHBOARD] Результат: {result}")

    return result


@app.post("/admin/knowledge/upload")
async def upload_knowledge(
    request: Request,
    file: UploadFile = File(...),
    clear_database: bool = Form(False),
    session: AsyncSession = Depends(get_knowledge_session),
    _: None = Depends(auth.ensure_api_auth),
) -> JSONResponse:
    if not auth.is_authenticated_request(request):
        raise HTTPException(status_code=401, detail="Authentication required")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover
        logger.exception("Pandas import failed: %s", exc)
        raise HTTPException(
            status_code=500, detail="Pandas is required on the server"
        ) from exc

    try:
        dataframe = pd.read_excel(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to read Excel: {exc}"
        ) from exc

    if dataframe.empty:
        raise HTTPException(status_code=400, detail="File contains no records")

    normalized = {str(col).strip().lower(): col for col in dataframe.columns}
    question_column = next(
        (
            normalized[key]
            for key in ("question", "������", "questions", "�������")
            if key in normalized
        ),
        None,
    )
    answer_column = next(
        (
            normalized[key]
            for key in ("answer", "�����", "answers", "������")
            if key in normalized
        ),
        None,
    )

    if question_column is None or answer_column is None:
        if len(dataframe.columns) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least two columns with questions and answers are required",
            )
        question_column = dataframe.columns[0]
        answer_column = dataframe.columns[1]

    pairs: list[tuple[str, str]] = []
    for _, row in dataframe.iterrows():
        question_cell = row[question_column]
        answer_cell = row[answer_column]
        if pd.isna(question_cell) or pd.isna(answer_cell):
            continue
        question = str(question_cell).strip()
        answer = str(answer_cell).strip()
        if not question or not answer:
            continue
        pairs.append((question, answer))

    if not pairs:
        raise HTTPException(
            status_code=400, detail="No valid question-answer pairs found"
        )

    if clear_database:
        # Полная очистка всех чанков
        logger.info("Excel upload: clearing all knowledge chunks")
        try:
            from sqlalchemy import delete

            # Удаляем все чанки (теперь все данные только в чанках)
            await session.execute(delete(models.DocumentChunk))
            await session.commit()
            logger.info("Successfully cleared all knowledge chunks before Excel upload")
        except Exception as e:
            logger.warning(f"Error clearing chunks before Excel upload: {e}")
            await session.rollback()

    # НОВАЯ ЛОГИКА: Excel данные тоже превращаем в чанки
    source_file = f"excel_upload_{file.filename}"

    # Удаляем старые чанки от этого файла
    await tickets_crud.delete_chunks_by_source(session, source_file)

    # Создаем чанки из пар вопрос-ответ
    chunks_data = []

    # Получаем embedder из RAG сервиса
    rag_service = request.app.state.rag

    for idx, (question, answer) in enumerate(pairs):
        # Объединяем вопрос и ответ в один текстовый блок
        chunk_text = f"Вопрос: {question}\nОтвет: {answer}"

        # Генерируем embedding для чанка
        embedding_list = rag_service.create_embedding(chunk_text)
        if embedding_list:
            import numpy as np

            embedding_vector = np.array(embedding_list, dtype=np.float32)
            embedding_bytes = embedding_vector.tobytes()
        else:
            # Если не удалось создать embedding, пропускаем этот чанк
            logger.warning(f"Failed to create embedding for chunk {idx}")
            continue

        chunks_data.append(
            (
                chunk_text,  # content
                source_file,  # source_file
                idx,  # chunk_index
                0,  # start_char (для Excel не актуально)
                len(chunk_text),  # end_char
                embedding_bytes,  # embedding
            )
        )

    # Сохраняем чанки в базу
    await tickets_crud.add_document_chunks(session, chunks_data)
    await session.commit()

    # Перезагружаем RAG систему
    await request.app.state.rag.reload()

    return JSONResponse({"success": True, "entries": len(pairs)})


@app.get("/api/knowledge/stats", response_model=KnowledgeStats)
async def knowledge_stats(
    session: AsyncSession = Depends(get_knowledge_session),
    _: None = Depends(auth.ensure_api_auth),
) -> KnowledgeStats:
    # Теперь считаем только чанки - все данные только в чанках
    total = await tickets_crud.count_document_chunks(session)
    return KnowledgeStats(total_entries=total)


@app.post("/api/knowledge/upload-files")
async def upload_knowledge_files(
    files: list[UploadFile] = File(...),
    clear_database: bool = False,  # Добавляем параметр для очистки БД
    request: Request = None,
    session: AsyncSession = Depends(get_knowledge_session),
    _: None = Depends(auth.ensure_api_auth),
) -> JSONResponse:
    """
    Загрузка и обработка файлов для базы знаний
    Поддерживает: .pdf, .docx, .doc, .md
    """
    import tempfile
    import os
    from pathlib import Path
    from app.rag.document_parsers import DocumentParserFactory
    from app.rag.chunker import chunk_document

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Очистка базы знаний если требуется
    if clear_database:
        logger.info("Clearing all knowledge chunks")
        try:
            from sqlalchemy import delete

            # Удаляем все чанки (теперь все данные только в чанках)
            await session.execute(delete(models.DocumentChunk))
            await session.commit()
            logger.info("Successfully cleared all knowledge chunks")
        except Exception as e:
            logger.warning(f"Error clearing chunks: {e}")
            await session.rollback()

    total_chunks = 0
    processed_files = []
    errors = []

    # Получаем RAG сервис
    rag_service: HybridRAGService = request.app.state.rag

    for file in files:
        try:
            # Проверяем расширение
            file_ext = Path(file.filename).suffix.lower()
            supported_exts = DocumentParserFactory.supported_extensions()

            if file_ext not in supported_exts:
                errors.append(
                    f"{file.filename}: Unsupported format. Supported: {', '.join(supported_exts)}"
                )
                continue

            # Сохраняем во временный файл
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            try:
                # Парсим документ
                text = DocumentParserFactory.parse_document(tmp_path)

                if not text or len(text.strip()) < 10:
                    errors.append(f"{file.filename}: No text extracted")
                    continue

                # Разбиваем на чанки
                chunks = chunk_document(
                    text, source_file=file.filename, chunk_size=1000, chunk_overlap=200
                )

                if not chunks:
                    errors.append(f"{file.filename}: No chunks created")
                    continue

                # Удаляем старые чанки этого файла (если есть)
                await tickets_crud.delete_chunks_by_source(session, file.filename)

                # Создаем embeddings и сохраняем в БД
                chunks_data = []
                for chunk in chunks:
                    # Генерируем embedding
                    embedding_list = rag_service.create_embedding(chunk.content)
                    if embedding_list:
                        import numpy as np

                        embedding_vector = np.array(embedding_list, dtype=np.float32)
                        embedding_bytes = embedding_vector.tobytes()

                        chunks_data.append(
                            (
                                chunk.content,
                                chunk.source_file,
                                chunk.chunk_index,
                                chunk.start_char,
                                chunk.end_char,
                                embedding_bytes,
                            )
                        )
                    else:
                        logger.warning(
                            f"Failed to create embedding for chunk from {chunk.source_file}"
                        )
                        continue

                # Сохраняем в БД
                await tickets_crud.add_document_chunks(session, chunks_data)

                total_chunks += len(chunks)
                processed_files.append(
                    {
                        "filename": file.filename,
                        "chunks": len(chunks),
                        "text_length": len(text),
                    }
                )

            finally:
                # Удаляем временный файл
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        except Exception as e:
            logger.exception(f"Error processing file {file.filename}")
            errors.append(f"{file.filename}: {str(e)}")
            continue

    # Обработка завершена - простой сервис не требует перезагрузки

    return JSONResponse(
        {
            "success": True,
            "total_chunks": total_chunks,
            "processed_files": processed_files,
            "errors": errors if errors else None,
        }
    )


@app.get("/api/conversations", response_model=list[TicketRead])
async def api_list_conversations(
    archived: bool = False,
    session: AsyncSession = Depends(get_tickets_session),
    _: None = Depends(auth.ensure_api_auth),
) -> list[dict]:
    tickets = await crud.list_tickets(session, archived=archived)
    # Используем _serialize_tickets для подсчета unread_count
    return _serialize_tickets(tickets)


@app.get("/api/conversations/{conversation_id}", response_model=TicketRead)
async def api_get_conversation(
    conversation_id: int,
    session: AsyncSession = Depends(get_tickets_session),
    _: None = Depends(auth.ensure_api_auth),
) -> dict:
    ticket = await crud.get_ticket_by_id(session, conversation_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    # Используем _serialize_tickets для подсчета unread_count и корректной сериализации
    serialized = _serialize_tickets([ticket])
    return serialized[0] if serialized else {}


@app.get(
    "/api/conversations/{conversation_id}/messages", response_model=list[MessageRead]
)
async def api_list_messages(
    conversation_id: int,
    include_system: bool = False,  # Включать системные сообщения
    session: AsyncSession = Depends(get_tickets_session),
    _: None = Depends(auth.ensure_api_auth),
) -> list[MessageRead]:
    ticket = await crud.get_ticket_by_id(session, conversation_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")

    messages = await crud.list_messages_for_ticket(
        session, conversation_id, include_system=include_system
    )
    return messages


@app.post("/api/conversations/{conversation_id}/finish")
async def api_finish(
    conversation_id: int,
    request: Request,
    session: AsyncSession = Depends(get_tickets_session),
    _: None = Depends(auth.ensure_api_auth),
) -> JSONResponse:
    ticket = await crud.get_ticket_by_id(session, conversation_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    if ticket.status in [models.TicketStatus.CLOSED, models.TicketStatus.ARCHIVED]:
        raise HTTPException(status_code=400, detail="Ticket already closed")

    finish_text = 'Оператор завершил заявку. Если потребуется помощь, напишите снова "Позови оператора".'

    bot: Bot | None = request.app.state.bot
    # Send finish notification to the correct platform
    if isinstance(ticket.telegram_chat_id, str) and ticket.telegram_chat_id.startswith(
        "vk_"
    ):
        try:
            from app.bots.vk_bot import VK_API

            if VK_API is not None:
                peer = int(ticket.telegram_chat_id.split("_", 1)[1])
                try:
                    import vk_api

                    try:
                        random_id = vk_api.utils.get_random_id()
                    except Exception:
                        random_id = 0
                except Exception:
                    random_id = 0
                VK_API.messages.send(
                    peer_id=peer, message=finish_text, random_id=random_id
                )
            else:
                logger.warning(
                    "VK API client not initialized; cannot send finish notification"
                )
        except Exception as e:
            logger.exception(f"Failed to send finish notification via VK: {e}")
    else:
        if bot is not None:
            await bot.send_message(ticket.telegram_chat_id, finish_text)

    # Добавляем финальное сообщение и закрываем заявку
    finish_message = await crud.add_message(
        session, conversation_id, "bot", finish_text, is_system=True
    )
    await crud.update_ticket_status(
        session, conversation_id, models.TicketStatus.CLOSED
    )

    # Отмечаем закрытие заявки в RAG сервисе для правильной сегментации истории
    rag_service: HybridRAGService = request.app.state.rag
    # Закрытие тикета - просто обновляем статус

    manager: ConnectionManager = request.app.state.connection_manager
    await manager.broadcast_message(conversation_id, _serialize_message(finish_message))
    tickets = await crud.list_tickets(session, archived=False)
    await manager.broadcast_conversations(_serialize_tickets(tickets))

    return JSONResponse({"success": True})


@app.post("/api/conversations/{conversation_id}/reply", response_model=MessageRead)
async def api_reply(
    conversation_id: int,
    message: MessageCreate,
    request: Request,
    session: AsyncSession = Depends(get_tickets_session),
    _: None = Depends(auth.ensure_api_auth),
) -> MessageRead:
    ticket = await crud.get_ticket_by_id(session, conversation_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    if ticket.status in [models.TicketStatus.CLOSED, models.TicketStatus.ARCHIVED]:
        raise HTTPException(status_code=400, detail="Cannot reply to closed ticket")

    # Переводим заявку в статус "В работе" если она была открыта
    if ticket.status == models.TicketStatus.OPEN:
        await crud.update_ticket_status(
            session, conversation_id, models.TicketStatus.IN_PROGRESS
        )

    # Формируем ответ для оператора
    await crud.set_first_response_time(session, conversation_id)

    # Получаем текущего пользователя из запроса
    from app.auth import get_current_user_from_request

    current_user = get_current_user_from_request(request)
    operator_name = (
        current_user.get("full_name", "Неизвестный") if current_user else "Неизвестный"
    )

    # Форматируем сообщение для оператора
    formatted_message = f"<b>Оператор {operator_name}:</b>\n{message.text}"

    bot: Bot | None = request.app.state.bot
    # If ticket.telegram_chat_id is a VK chat id (vk_...), send via VK instead of Telegram
    if isinstance(ticket.telegram_chat_id, str) and ticket.telegram_chat_id.startswith(
        "vk_"
    ):
        try:
            from app.bots.vk_bot import VK_API

            if VK_API is not None:
                # peer_id is the numeric part after 'vk_'
                peer = int(ticket.telegram_chat_id.split("_", 1)[1])
                try:
                    import vk_api

                    try:
                        random_id = vk_api.utils.get_random_id()
                    except Exception:
                        random_id = 0
                except Exception:
                    random_id = 0
                VK_API.messages.send(
                    peer_id=peer, message=formatted_message, random_id=random_id
                )
            else:
                logger.warning(
                    "VK API client not initialized; cannot send operator message"
                )
        except Exception as e:
            logger.exception(f"Failed to send operator reply via VK: {e}")
    else:
        if bot is not None:
            await bot.send_message(
                ticket.telegram_chat_id, formatted_message, parse_mode="HTML"
            )

    new_message = await crud.add_message(
        session, conversation_id, "operator", message.text, is_system=False
    )

    manager: ConnectionManager = request.app.state.connection_manager
    await manager.broadcast_message(conversation_id, _serialize_message(new_message))

    # Отправляем обновление списка заявок
    await _broadcast_conversations_update(session, manager)

    return new_message


@app.post("/api/conversations/{conversation_id}/read")
async def mark_conversation_read(
    conversation_id: int,
    request: Request,
    _: None = Depends(auth.ensure_api_auth),
) -> dict:
    """Отметить разговор как прочитанный (для пользователя)"""
    if not auth.is_authenticated_request(request):
        raise HTTPException(status_code=401, detail="Authentication required")

    return {"success": True}


@app.get("/api/conversations/{conversation_id}/summary")
async def get_ticket_summary(
    conversation_id: int,
    request: Request,
    session: AsyncSession = Depends(get_tickets_session),
    _: None = Depends(auth.ensure_api_auth),
) -> dict:
    """Получить сводку по заявке"""
    if not auth.is_authenticated_request(request):
        raise HTTPException(status_code=401, detail="Authentication required")

    # Получаем заявку и связанные с ней сообщения
    ticket_with_messages = await crud.get_ticket_with_messages(session, conversation_id)
    if ticket_with_messages is None:
        raise HTTPException(status_code=404, detail="Ticket not found")

    ticket, messages = ticket_with_messages

    # Получаем summary из сообщения, если оно есть
    if ticket.summary:
        summary = ticket.summary
    else:
        # Простое создание summary из сообщений
        if messages:
            summary = f"Тикет создан: {ticket.created_at.strftime('%d.%m.%Y %H:%M')}\n"
            summary += f"Всего сообщений: {len(messages)}\n"
            if messages:
                summary += f"Первое сообщение: {messages[0].text[:100]}..."
        else:
            summary = "Нет сообщений"
        await crud.update_ticket_summary(session, conversation_id, summary)

    return {
        "ticket_id": conversation_id,
        "summary": summary,
        "classification": ticket.classification,
        "priority": ticket.priority,
        "status": ticket.status,
        "created_at": ticket.created_at.isoformat(),
        "message_count": len(messages),
    }


@app.websocket("/ws/conversations")
async def websocket_conversations(websocket: WebSocket) -> None:
    if not auth.is_authenticated_websocket(websocket):
        await websocket.close(code=auth.WEBSOCKET_UNAUTHORIZED_CLOSE_CODE)
        return
    await websocket.accept()
    await connection_manager.register_conversations(websocket)
    try:
        async with TicketsSessionLocal() as session:
            tickets = await crud.list_tickets(session, archived=False)
            await connection_manager.send_conversations_snapshot(
                websocket, _serialize_tickets(tickets)
            )
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await connection_manager.unregister_conversations(websocket)


@app.websocket("/ws/conversations/{conversation_id}")
async def websocket_messages(websocket: WebSocket, conversation_id: int) -> None:
    if not auth.is_authenticated_websocket(websocket):
        await websocket.close(code=auth.WEBSOCKET_UNAUTHORIZED_CLOSE_CODE)
        return

    async with TicketsSessionLocal() as session:
        ticket = await crud.get_ticket_by_id(session, conversation_id)
        if ticket is None:
            await websocket.close(code=4404)
            return
        messages = await crud.list_messages_for_ticket(
            session, conversation_id, include_system=False
        )
        history_payload = [_serialize_message(item) for item in messages]

    # Принимаем WebSocket соединение и регистрируем чат
    await websocket.accept()
    await connection_manager.register_chat(conversation_id, websocket)

    try:
        # Ждем, пока пользователь не отправит сообщение (пока has_active_chat_connections будет True)
        async with TicketsSessionLocal() as session:
            marked_count = await crud.mark_ticket_messages_as_read(
                session, conversation_id
            )
            if marked_count > 0:
                # Отправляем обновление списка заявок
                await _broadcast_conversations_update(session, connection_manager)

        await connection_manager.send_message_history(
            websocket, conversation_id, history_payload
        )
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await connection_manager.unregister_chat(conversation_id, websocket)
        if websocket.client_state.name == "CONNECTED":
            await websocket.close()


# ==================== Simulator API ====================


@app.get("/api/simulator/characters")
async def get_simulator_characters(request: Request):
    """Получить список доступных персонажей для симуляции"""
    if not auth.is_authenticated_request(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    simulator: SimulatorService = request.app.state.simulator
    return {"characters": simulator.characters}


@app.post("/api/simulator/start")
async def start_simulator_session(request: Request, data: dict = Body(...)):
    """Запустить новую сессию симуляции"""
    try:
        if not auth.is_authenticated_request(request):
            raise HTTPException(status_code=401, detail="Unauthorized")

        # Извлекаем параметры из body
        character = data.get("character", "medium")

        # Получаем session cookie для user_id
        settings = auth.get_settings()
        user_id = request.cookies.get(settings.cookie_name, "anonymous")
        simulator: SimulatorService = request.app.state.simulator

        # Запускаем сессию
        session = simulator.start_session(user_id, character)

        # Генерируем первый вопрос
        question = await simulator.generate_question(session)

        return {
            "session_id": user_id,
            "character": session.character,
            "character_info": simulator.characters.get(session.character, {}),
            "questions_total": session.questions_count,
            "current_question": session.current_question + 1,
            "question": question.question,
        }
    except Exception as e:
        logger.error(f"Failed to start simulator session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"������ ������� ������: {str(e)}")


@app.post("/api/simulator/respond")
async def respond_to_question(request: Request):
    """Отправить ответ на вопрос"""
    if not auth.is_authenticated_request(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    settings = auth.get_settings()
    user_id = request.cookies.get(settings.cookie_name, "anonymous")
    simulator: SimulatorService = request.app.state.simulator

    data = await request.json()
    user_answer = data.get("answer", "")

    session = simulator.get_session(user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Оцениваем ответ
    evaluation = simulator.evaluate_response(session, user_answer)

    # Добавляем ответ
    session.add_response(user_answer, evaluation)

    # Проверяем, завершена ли сессия
    is_complete = session.is_complete()

    response_data = {
        "score": evaluation.score,
        "feedback": evaluation.feedback,
        "ai_suggestion": evaluation.ai_suggestion,
        "is_correct": evaluation.is_correct,
        "session_complete": is_complete,
        "current_question": session.current_question,
        "questions_total": session.questions_count,
    }

    # Если сессия не завершена - генерируем следующий вопрос
    if not is_complete:
        next_question = await simulator.generate_question(session)
        response_data["next_question"] = next_question.question
    else:
        # Сессия завершена - подводим итоги
        response_data["total_score"] = session.total_score
        response_data["average_score"] = session.get_average_score()
        response_data["history"] = session.history

    return response_data


@app.get("/api/simulator/hint")
async def get_hint(request: Request):
    """Получить подсказку для текущего вопроса"""
    if not auth.is_authenticated_request(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    settings = auth.get_settings()
    user_id = request.cookies.get(settings.cookie_name, "anonymous")
    simulator: SimulatorService = request.app.state.simulator

    session = simulator.get_session(user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    hint = simulator.get_hint(session)

    return {"hint": hint}


@app.post("/api/simulator/end")
async def end_simulator_session(request: Request):
    """Завершить сессию симуляции"""
    if not auth.is_authenticated_request(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    settings = auth.get_settings()
    user_id = request.cookies.get(settings.cookie_name, "anonymous")
    simulator: SimulatorService = request.app.state.simulator

    session = simulator.get_session(user_id)
    if session:
        final_stats = {
            "questions_answered": len(session.history),
            "total_score": session.total_score,
            "average_score": session.get_average_score(),
            "history": session.history,
        }
        simulator.end_session(user_id)
        return {"message": "Session ended", "stats": final_stats}

    return {"message": "No active session"}


# ==================== ADMIN: USER MANAGEMENT ====================


@app.get("/admin/users", response_class=HTMLResponse)
@require_admin()
async def users_admin(request: Request):
    """Получить список пользователей для администрирования"""
    return templates.TemplateResponse("admin_users.html", {"request": request})


@app.get("/api/admin/users")
@require_admin(redirect_to_home=False)
async def get_users(request: Request):
    """Получить список всех пользователей"""
    from app.auth import user_manager

    users = user_manager.get_all_users()

    # Удаляем password_hash из ответа
    for user in users:
        user.pop("password_hash", None)

    return {"users": users}


@app.get("/api/user/permissions")
async def get_current_user_permissions(request: Request):
    """Получить права доступа текущего пользователя"""
    if not auth.is_authenticated_request(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    permissions = get_user_permissions(request)
    return {"permissions": permissions}


@app.post("/api/admin/users")
@require_admin(redirect_to_home=False)
async def create_user(request: Request, data: dict = Body(...)):
    """Создать нового пользователя"""

    from app.auth import user_manager

    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    full_name = data.get("full_name", "").strip()
    role_id = data.get("role_id")
    is_admin = data.get("is_admin", False)

    if not username or not password or not full_name or not role_id:
        raise HTTPException(status_code=400, detail="��� ���� �����������")

    user_id = user_manager.create_user(username, password, full_name, role_id, is_admin)

    if user_id is None:
        raise HTTPException(
            status_code=400, detail="Пользователь с такими данными уже существует"
        )

    return {"success": True, "user_id": user_id}


@app.put("/api/admin/users/{user_id}")
@require_admin(redirect_to_home=False)
async def update_user(request: Request, user_id: int, data: dict = Body(...)):
    """Обновить данные пользователя"""
    # Проверяем, что нельзя обновить суперпользователя
    if user_id == 1:
        raise HTTPException(
            status_code=403, detail="Изменение суперпользователя запрещено"
        )

    from app.auth import user_manager

    success = user_manager.update_user(user_id, **data)

    if not success:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    return {"success": True}


@app.delete("/api/admin/users/{user_id}")
@require_admin(redirect_to_home=False)
async def delete_user(request: Request, user_id: int):
    """Удалить пользователя"""
    # Проверяем, что нельзя удалить суперпользователя
    if user_id == 1:
        raise HTTPException(
            status_code=403, detail="Удаление суперпользователя запрещено"
        )

    from app.auth import user_manager

    success = user_manager.delete_user(user_id)

    if not success:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    return {"success": True}


@app.post("/api/admin/users/{user_id}/password")
@require_admin(redirect_to_home=False)
async def change_password(request: Request, user_id: int, data: dict = Body(...)):
    from app.auth import user_manager

    new_password = data.get("password", "").strip()

    if not new_password:
        raise HTTPException(status_code=400, detail="Новый пароль не может быть пустым")

    success = user_manager.update_password(user_id, new_password)

    if not success:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    return {"success": True}


# ==================== ADMIN: PERMISSIONS ====================


@app.get("/api/admin/permissions")
@require_admin(redirect_to_home=False)
async def get_permissions(request: Request):

    from app.auth import user_manager

    permissions = user_manager.get_all_permissions()

    return {"permissions": permissions}


@app.get("/api/admin/users/{user_id}/permissions")
@require_admin(redirect_to_home=False)
async def get_user_permissions_admin(request: Request, user_id: int):
    from app.auth import user_manager

    permissions = user_manager.get_user_permissions(user_id)

    return {"permissions": permissions}


@app.post("/api/admin/users/{user_id}/permissions")
@require_admin(redirect_to_home=False)
async def set_user_permission(request: Request, user_id: int, data: dict = Body(...)):
    from app.auth import user_manager

    page_key = data.get("page_key")
    granted = data.get("granted", True)

    if not page_key:
        raise HTTPException(status_code=400, detail="page_key не указанный")

    success = user_manager.set_user_permission(user_id, page_key, granted)

    if not success:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    return {"success": True}


# ==================== ADMIN: ROLES ====================


@app.get("/api/admin/roles")
@require_admin(redirect_to_home=False)
async def get_roles(request: Request):

    from app.auth import user_manager

    roles = user_manager.get_all_roles()

    return {"roles": roles}


@app.get("/api/admin/roles/{role_id}/permissions")
@require_admin(redirect_to_home=False)
async def get_role_permissions(request: Request, role_id: int):
    from app.auth import user_manager

    permissions = user_manager.get_role_permissions(role_id)

    return {"permissions": permissions}


@app.post("/api/admin/roles/{role_id}/permissions")
@require_admin(redirect_to_home=False)
async def set_role_permissions(request: Request, role_id: int, data: dict = Body(...)):
    from app.auth import user_manager

    page_keys = data.get("page_keys", [])

    success = user_manager.set_role_permissions(role_id, page_keys)

    return {"success": success}


# ==================== ADMIN: SETTINGS ====================


@app.get("/api/admin/settings")
@require_admin(redirect_to_home=False)
async def get_settings_admin(request: Request):
    from app.auth import user_manager

    settings = user_manager.get_all_settings()

    return {"settings": settings}


@app.post("/api/admin/settings")
@require_admin(redirect_to_home=False)
async def update_setting(request: Request, data: dict = Body(...)):
    from app.auth import user_manager

    key = data.get("key")
    value = data.get("value")

    if not key or value is None:
        raise HTTPException(status_code=400, detail="key или value не указаны")

    success = user_manager.set_setting(key, str(value))

    if not success:
        raise HTTPException(status_code=404, detail="Настройка не найдена")

    return {"success": True}


@app.post("/api/agent/chat")
async def agent_chat(request: Request, data: dict = Body(...)):
    """Обработка запроса через RAG агента"""
    query = data.get("query", "").strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    try:
        rag_service: HybridRAGService = request.app.state.rag
        response = await rag_service.process_query(query)

        return {"success": True, "response": response, "query": query}

    except Exception as e:
        logger.error(f"Agent chat error: {e}")
        raise HTTPException(status_code=500, detail="Ошибка обработки запроса")
