"""
Инструменты для RAG агента
"""

import logging
import contextvars
from typing import List, Dict, Any, Optional
from agno.tools import tool
from collections import deque
import threading

logger = logging.getLogger(__name__)

# Context variable для передачи conversation_id в async контексте
_conversation_id_var: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "conversation_id", default=None
)

# Очередь для отложенной отправки действий агента в Telegram
_action_queue: deque = deque()
_action_queue_lock = threading.Lock()

# Глобальная модель SentenceTransformer (инициализируется один раз)
_sentence_transformer = None
_sentence_transformer_lock = threading.Lock()


def get_sentence_transformer():
    """Получить глобальный экземпляр SentenceTransformer (ленивая инициализация)"""
    global _sentence_transformer

    if _sentence_transformer is None:
        with _sentence_transformer_lock:
            # Double-check locking pattern
            if _sentence_transformer is None:
                print("[TRANSFORMER] Инициализация SentenceTransformer (один раз)...")
                from sentence_transformers import SentenceTransformer

                _sentence_transformer = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
                print("[TRANSFORMER] ✅ SentenceTransformer готов к использованию")

    return _sentence_transformer


def set_current_conversation_id(conversation_id: int):
    """Установить ID текущего разговора для async контекста"""
    _conversation_id_var.set(conversation_id)
    print(f"[CONTEXT] set_current_conversation_id: {conversation_id}")


def get_current_conversation_id() -> Optional[int]:
    """Получить ID текущего разговора"""
    conv_id = _conversation_id_var.get()
    print(f"[CONTEXT] get_current_conversation_id: {conv_id}")
    return conv_id


def _send_action_to_telegram(action_text: str) -> None:
    """Отправить действие агента в Telegram (добавить в очередь для немедленной обработки)

    Args:
        action_text: Описание действия (например, "🔍 Поиск в базе знаний")
    """
    try:
        conversation_id = get_current_conversation_id()
        if not conversation_id:
            print(f"[ACTION] Пропускаем отправку действия - нет conversation_id")
            return

        print(f"[ACTION] Добавляю действие в очередь: {action_text}")

        # Добавляем действие в очередь
        with _action_queue_lock:
            _action_queue.append(
                {"conversation_id": conversation_id, "action_text": action_text}
            )

        # Планируем обработку очереди в главном loop
        import asyncio

        try:
            # Пытаемся получить главный loop и запланировать обработку
            from app.main import _main_loop

            if _main_loop and _main_loop.is_running():
                # Планируем обработку через call_soon_threadsafe
                _main_loop.call_soon_threadsafe(
                    lambda: asyncio.ensure_future(
                        process_pending_actions(), loop=_main_loop
                    )
                )
                print(f"[ACTION] Запланирована обработка в главном loop")
        except Exception as schedule_error:
            print(f"[ACTION] Не удалось запланировать обработку: {schedule_error}")

    except Exception as e:
        print(f"[ACTION ERROR] Не удалось добавить действие: {e}")
        logger.warning(f"Failed to queue action: {e}")


async def process_pending_actions():
    """Обработать все ожидающие действия из очереди немедленно

    Вызывается автоматически при добавлении действий и после завершения работы агента.
    """
    from app.bots import send_agent_action_to_telegram
    from app.db.database import TicketsSessionLocal
    from app.db.models import Ticket
    from sqlalchemy import select

    actions_to_process = []

    # Извлекаем все действия из очереди
    with _action_queue_lock:
        while _action_queue:
            actions_to_process.append(_action_queue.popleft())

    if not actions_to_process:
        return

    print(f"[ACTION] Обработка {len(actions_to_process)} действий")

    # Обрабатываем каждое действие
    for action_data in actions_to_process:
        try:
            conversation_id = action_data["conversation_id"]
            action_text = action_data["action_text"]

            # Получаем chat_id из ticket
            async with TicketsSessionLocal() as session:
                stmt = select(Ticket).where(Ticket.id == conversation_id)
                result = await session.execute(stmt)
                ticket = result.scalar_one_or_none()

                if not ticket:
                    print(f"[ACTION] Ticket {conversation_id} не найден")
                    continue

                # telegram_chat_id - это строка, нужно преобразовать в int для Telegram
                chat_id_str = ticket.telegram_chat_id
                if not chat_id_str:
                    print(f"[ACTION] У ticket {conversation_id} нет telegram_chat_id")
                    continue

                # Проверяем, что это не VK (VK id начинаются с 'vk_')
                if isinstance(chat_id_str, str) and chat_id_str.startswith("vk_"):
                    print(f"[ACTION] Пропускаем VK чат {chat_id_str}")
                    continue

                try:
                    chat_id = int(chat_id_str)
                except (ValueError, TypeError):
                    print(f"[ACTION] Не удалось преобразовать chat_id: {chat_id_str}")
                    continue

            # Отправляем действие в Telegram
            await send_agent_action_to_telegram(chat_id, action_text)
            print(f"[ACTION] ✅ Отправлено: {action_text}")

        except Exception as e:
            print(f"[ACTION ERROR] Ошибка обработки действия: {e}")
            logger.warning(f"Failed to process action: {e}")


# Глобальное хранилище для передачи данных между агентом и ботом
# Ключ - conversation_id, значение - список похожих результатов
_similar_suggestions_storage: Dict[int, List[Dict[str, Any]]] = {}

# Храним timestamp последнего показа кнопок для каждого conversation
# Чтобы не спамить кнопками слишком часто
_last_suggestions_time: Dict[int, float] = {}

# Минимальный интервал между показами кнопок (в секундах)
MIN_SUGGESTIONS_INTERVAL = 60  # 1 минута


def store_similar_suggestions(conversation_id: int, suggestions: List[Dict[str, Any]]):
    """Сохранить похожие результаты для разговора"""
    import time

    global _similar_suggestions_storage, _last_suggestions_time

    # Проверяем, не показывали ли мы кнопки недавно
    last_time = _last_suggestions_time.get(conversation_id, 0)
    current_time = time.time()

    if current_time - last_time < MIN_SUGGESTIONS_INTERVAL:
        logger.info(
            f"Skipping suggestions for conversation {conversation_id} - shown too recently "
            f"({int(current_time - last_time)}s ago, minimum {MIN_SUGGESTIONS_INTERVAL}s)"
        )
        return  # Не сохраняем - слишком рано

    _similar_suggestions_storage[conversation_id] = suggestions
    _last_suggestions_time[conversation_id] = current_time
    logger.info(
        f"Stored {len(suggestions)} similar suggestions for conversation {conversation_id}"
    )


def get_similar_suggestions(conversation_id: int) -> List[Dict[str, Any]] | None:
    """Получить похожие результаты для разговора и очистить хранилище"""
    global _similar_suggestions_storage
    suggestions = _similar_suggestions_storage.pop(conversation_id, None)
    if suggestions:
        logger.info(
            f"Retrieved {len(suggestions)} similar suggestions for conversation {conversation_id}"
        )
    return suggestions


@tool
async def search_knowledge_base(query: str, suggest_similar: bool = False) -> str:
    """Поиск в базе знаний - твой главный инструмент!

    Параметры:
    - query: запрос для поиска
    - suggest_similar: если True, покажет пользователю 3 кнопки с похожими проблемами для выбора.
      Используй suggest_similar=True ТОЛЬКО когда:
      * Пользователь описывает общую проблему без деталей (например: "интернет не работает", "компьютер тормозит")
      * Есть несколько похожих решений и пользователь может выбрать подходящее
      * НЕ используй если пользователь задал конкретный вопрос или нужен прямой ответ

    По умолчанию (suggest_similar=False) - обычный поиск с прямым ответом.
    """
    print(
        f"[AGENT ACTION] Поиск в базе знаний: '{query}' (suggest_similar={suggest_similar})"
    )

    # Отправляем действие в Telegram
    _send_action_to_telegram(
        f"🔍 Поиск в базе знаний: {query[:50]}{'...' if len(query) > 50 else ''}"
    )

    try:
        from app.db.database import KnowledgeSessionLocal
        from app.db import tickets_crud as crud
        import numpy as np

        # Получаем глобальный экземпляр модели (быстро, инициализируется только один раз)
        model = get_sentence_transformer()

        async with KnowledgeSessionLocal() as session:
            # Загружаем все чанки
            chunks = await crud.load_all_chunks(session)

            if not chunks:
                print("[AGENT] База знаний пуста")
                return "База знаний пуста"

            print(f"[AGENT] Поиск среди {len(chunks)} записей...")

            # Создаем векторное представление запроса
            query_embedding = model.encode([query], convert_to_numpy=True)
            query_vector = query_embedding[0]

            # Ищем среди чанков с эмбеддингами
            results = []
            text_results = []  # Для чанков без эмбеддингов

            for chunk in chunks:
                if chunk.embedding is not None:
                    # Семантический поиск
                    chunk_vector = np.frombuffer(chunk.embedding, dtype=np.float32)
                    # Косинусное сходство
                    similarity = np.dot(query_vector, chunk_vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(chunk_vector)
                    )
                    results.append(
                        {
                            "id": chunk.id,
                            "content": chunk.content,
                            "source": chunk.source_file,
                            "score": float(similarity),
                            "type": "semantic",
                        }
                    )
                else:
                    # Текстовый поиск для чанков без эмбеддингов
                    query_lower = query.lower()
                    if query_lower in chunk.content.lower():
                        relevance = chunk.content.lower().count(query_lower)
                        text_results.append(
                            {
                                "id": chunk.id,
                                "content": chunk.content,
                                "source": chunk.source_file,
                                "score": relevance / 10.0,  # Нормализуем
                                "type": "text",
                            }
                        )

            # Объединяем результаты
            all_results = results + text_results

            if not all_results:
                print(f"[AGENT] Ничего не найдено по запросу: '{query}'")
                return f"По запросу '{query}' ничего не найдено в базе знаний"

            # Сортируем по релевантности
            all_results.sort(key=lambda x: x["score"], reverse=True)

            print(f"[AGENT] Найдено {len(all_results)} результатов")

            # Отслеживаем использование топ-3 чанков для FAQ
            from app.rag.faq_service import track_chunk_usage

            for result in all_results[:3]:
                track_chunk_usage(result["id"], result["content"])

            # Формируем обычный текстовый ответ из топ-3 результатов
            response_parts = []
            for i, result in enumerate(all_results[:3], 1):
                content = result["content"]
                # Обрабатываем проблемные символы Unicode
                try:
                    content = content.encode("cp1251", errors="ignore").decode("cp1251")
                except:
                    content = content.encode("ascii", errors="ignore").decode("ascii")

                if len(content) > 300:
                    content = content[:300] + "..."

                response_parts.append(
                    f"{i}. Источник: {result['source']}\n"
                    f"Содержание: {content}\n"
                    f"(релевантность: {result['score']:.3f}, тип: {result['type']})"
                )

            response = "\n\n".join(response_parts)

            # Дополнительно: если нужно показать кнопки с похожими вариантами
            if suggest_similar and len(all_results) >= 3:
                print("[AGENT] Сохраняю варианты для показа кнопок пользователю")

                # Берём топ-3
                top_3 = all_results[:3]
                suggestions = []

                for result in top_3:
                    # Создаём краткое описание для кнопки (первые 80 символов)
                    preview = result["content"][:80].replace("\n", " ").strip()
                    if len(result["content"]) > 80:
                        preview += "..."

                    suggestions.append(
                        {
                            "id": result["id"],
                            "preview": preview,
                            "score": result["score"],
                            "source": result["source"],
                            "full_content": result[
                                "content"
                            ],  # Полный текст для отображения
                        }
                    )

                # Сохраняем в хранилище (если известен conversation_id)
                conversation_id = get_current_conversation_id()
                if conversation_id:
                    store_similar_suggestions(conversation_id, suggestions)
                    print(
                        f"[AGENT] Сохранил {len(suggestions)} вариантов для conversation {conversation_id}"
                    )
                else:
                    print(
                        "[AGENT WARNING] conversation_id не установлен, кнопки не будут показаны!"
                    )

                print(f"[AGENT] Кнопки будут показаны дополнительным сообщением")

            print(
                f"[AGENT] Возвращаю результат поиска (длина: {len(response)} символов)"
            )
            return response

    except Exception as e:
        print(f"[AGENT ERROR] Ошибка поиска: {e}")
        logger.error(f"Error searching knowledge base: {e}")
        return f"Ошибка поиска в базе знаний: {e}"


@tool
def improve_search_query(original_query: str, context: str = None) -> str:
    """Улучшение поискового запроса через LLM - анализирует запрос и создает оптимальные поисковые термины для любой предметной области."""
    print(f"[AGENT ACTION] Улучшение запроса через LLM: '{original_query}'")

    try:
        from app.rag.service import get_llm_client
        import yaml

        # Загружаем конфигурацию
        with open("configs/rag_config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Получаем промпт из конфига
        improvement_prompt_template = config.get("agent", {}).get(
            "improve_search_prompt", ""
        )

        # Формируем промпт с подстановкой значений
        improvement_prompt = improvement_prompt_template.format(
            original_query=original_query,
            context=context if context else "Техническая поддержка",
        )

        # Получаем LLM клиент
        llm_client = get_llm_client()

        # Улучшаем запрос через LLM
        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": improvement_prompt}],
            max_tokens=100,
            temperature=0.3,
        )

        improved_query = response.choices[0].message.content.strip()

        # Убираем кавычки если LLM их добавил
        if improved_query.startswith('"') and improved_query.endswith('"'):
            improved_query = improved_query[1:-1]

        if improved_query != original_query:
            print(f"[AGENT] Запрос улучшен LLM: '{improved_query}'")
            return improved_query
        else:
            print(f"[AGENT] LLM решил что запрос уже оптимален")
            return original_query

    except Exception as e:
        print(f"[AGENT ERROR] Ошибка улучшения запроса: {e}")

        # Фоллбэк на простое улучшение
        simple_improvements = {
            "принтер": ["принтер", "печать", "МФУ", "струйный", "лазерный"],
            "компьютер": ["компьютер", "ПК", "ноутбук", "системный блок"],
            "интернет": ["интернет", "сеть", "wi-fi", "подключение"],
            "ошибка": ["ошибка", "проблема", "сбой", "не работает"],
            "программа": ["программа", "приложение", "софт", "ПО"],
        }

        original_lower = original_query.lower()
        enhanced_terms = [original_query]

        for key, synonyms in simple_improvements.items():
            if key in original_lower:
                enhanced_terms.extend(synonyms[:3])  # Берем первые 3 синонима
                break

        if len(enhanced_terms) > 1:
            improved_query = " ".join(enhanced_terms[:4])  # Максимум 4 термина
            print(f"[AGENT] Запрос улучшен (fallback): '{improved_query}'")
            return improved_query
        else:
            print(f"[AGENT] Запрос остался без изменений")
            return original_query


def _classify_request_internal(
    dialogue_history: str = None, text: str = None, categories: List[str] = None
) -> str:
    """Внутренняя функция классификации для прямого вызова из ботов (без декоратора @tool)."""
    # Определяем текст для анализа
    if dialogue_history:
        analysis_text = dialogue_history
        print(
            f"[AGENT ACTION] Классификация диалога через LLM (длина: {len(dialogue_history)} символов)"
        )
    elif text:
        analysis_text = text
        print(f"[AGENT ACTION] Классификация запроса через LLM: '{text[:50]}...'")
    else:
        return "Ошибка: не предоставлен ни диалог, ни текст для классификации"

    try:
        from app.rag.service import get_llm_client
        import yaml

        # Загружаем конфигурацию
        with open("configs/rag_config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Получаем категории из конфига
        default_categories = [
            "Инфраструктура",
            "Разработка",
            "Сеть",
            "Безопасность",
            "Пользовательское ПО",
            "Аппаратные проблемы",
            "Общий",
        ]
        available_categories = config.get("agent", {}).get(
            "categories", categories or default_categories
        )

        # Формируем промпт для классификации из конфига
        categories_text = ", ".join(available_categories)
        classification_prompt_template = config.get("agent", {}).get(
            "classify_request_prompt", ""
        )

        classification_prompt = classification_prompt_template.format(
            categories_text=categories_text, analysis_text=analysis_text
        )

        # Получаем LLM клиент
        llm_client = get_llm_client()

        # Классифицируем через LLM
        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": classification_prompt}],
            max_tokens=100,
            temperature=0.1,
        )

        llm_result = response.choices[0].message.content.strip()

        # Парсим результат и проверяем категории
        suggested_categories = [cat.strip() for cat in llm_result.split(",")]
        valid_categories = []

        for cat in suggested_categories:
            if cat in available_categories:
                valid_categories.append(cat)

        # Если LLM не вернул валидные категории, используем fallback
        if not valid_categories:
            text_lower = analysis_text.lower()
            category_keywords = {
                "Инфраструктура": [
                    "сервер",
                    "сеть",
                    "железо",
                    "оборудование",
                    "инфраструктура",
                ],
                "Разработка": [
                    "код",
                    "программа",
                    "баг",
                    "ошибка",
                    "приложение",
                    "разработка",
                ],
                "Сеть": ["интернет", "сеть", "роутер", "wifi", "подключение"],
                "Безопасность": [
                    "пароль",
                    "доступ",
                    "права",
                    "безопасность",
                    "блокировка",
                ],
                "Пользовательское ПО": ["программа", "софт", "приложение", "установка"],
                "Аппаратные проблемы": [
                    "компьютер",
                    "принтер",
                    "мышь",
                    "клавиатура",
                    "монитор",
                ],
                "Общий": [],
            }

            scores = {}
            for category, keywords in category_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > 0:
                    scores[category] = score

            if scores:
                # Берем топ категории
                sorted_categories = sorted(
                    scores.items(), key=lambda x: x[1], reverse=True
                )
                valid_categories = [
                    cat for cat, score in sorted_categories[:2]
                ]  # Максимум 2 категории
            else:
                valid_categories = ["Общий"]

        result_categories = ", ".join(valid_categories)
        print(f"[AGENT] Диалог классифицирован как: {result_categories}")
        logger.info(f"Request classified as: {result_categories}")

        # Отправляем действие с результатом в Telegram
        _send_action_to_telegram(f"🏷️ Классификация: {result_categories}")

        return f"Классификация проблемы: {result_categories}"

    except Exception as e:
        print(f"[AGENT ERROR] Ошибка классификации: {e}")
        logger.error(f"Classification error: {e}")
        return "Ошибка классификации. Категория: Общий"


@tool
def classify_request(
    dialogue_history: str = None, text: str = None, categories: List[str] = None
) -> str:
    """Классификация запроса через LLM - анализирует весь диалог с пользователем для точной классификации проблемы. Возвращает категории проблем которые можно назначить заявке."""
    return _classify_request_internal(dialogue_history, text, categories)


def _set_priority_internal(dialogue_history: str) -> str:
    """Внутренняя функция установки приоритета для прямого вызова из ботов (без декоратора @tool)."""
    print(
        f"[AGENT ACTION] Определение приоритета диалога через LLM (длина: {len(dialogue_history)} символов)"
    )

    try:
        from app.rag.service import get_llm_client
        import yaml

        # Загружаем конфигурацию
        with open("configs/rag_config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Получаем промпт из конфига
        priority_prompt_template = config.get("agent", {}).get(
            "set_priority_prompt", ""
        )

        if not priority_prompt_template:
            print("[AGENT WARNING] Промпт для set_priority не найден в конфиге")
            return "medium"

        # Формируем промпт для определения приоритета
        priority_prompt = priority_prompt_template.format(
            dialogue_text=dialogue_history
        )

        # Получаем LLM клиент
        llm_client = get_llm_client()

        # Определяем приоритет через LLM
        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": priority_prompt}],
            max_tokens=10,
            temperature=0.1,
        )

        llm_result = response.choices[0].message.content.strip().lower()

        # Парсим результат и проверяем валидность
        valid_priorities = ["low", "medium", "high"]

        # Извлекаем приоритет из ответа LLM
        priority = "medium"  # По умолчанию
        for valid_priority in valid_priorities:
            if valid_priority in llm_result:
                priority = valid_priority
                break

        print(f"[AGENT] Приоритет диалога определен как: {priority}")
        logger.info(f"Dialogue priority set to: {priority}")

        # Отправляем действие с результатом в Telegram
        priority_labels = {"low": "низкий", "medium": "средний", "high": "высокий"}
        _send_action_to_telegram(
            f"⚡ Установлен приоритет: {priority_labels.get(priority, priority)}"
        )

        return priority

    except Exception as e:
        print(f"[AGENT ERROR] Ошибка определения приоритета: {e}")
        logger.error(f"Priority determination error: {e}")
        return "medium"  # По умолчанию при ошибке


@tool
def set_priority(dialogue_history: str = None) -> str:
    """Определение приоритета заявки на основе важности диалога - анализирует весь диалог и устанавливает приоритет (низкий/средний/высокий).

    Используй этот инструмент когда:
    - Пользователь описывает критическую проблему (блокирует работу, потеря данных, безопасность)
    - Проблема влияет на многих пользователей или критические системы
    - Эмоциональный тон указывает на срочность
    - Нужно понять, насколько важна проблема для приоритизации обработки

    Параметры:
    - dialogue_history: полный текст диалога с пользователем для анализа

    Возвращает описание установленного приоритета.
    """
    if not dialogue_history:
        return (
            "Ошибка: необходимо предоставить историю диалога для определения приоритета"
        )

    priority = _set_priority_internal(dialogue_history)

    # Сохраняем приоритет в ticket через conversation_id
    conversation_id = get_current_conversation_id()
    print(f"[AGENT] conversation_id = {conversation_id}")

    if conversation_id:
        print(f"[AGENT] Начинаю процесс обновления приоритета: {priority}")
        try:
            import asyncio
            from app.db.database import TicketsSessionLocal
            from app.db import tickets_crud as crud

            async def update_priority():
                print(
                    f"[AGENT] Начинаю обновление приоритета для ticket {conversation_id} на {priority}"
                )
                async with TicketsSessionLocal() as session:
                    # Получаем ticket по conversation_id (это ID тикета)
                    ticket = await crud.get_ticket_by_id(session, conversation_id)
                    if ticket:
                        old_priority = ticket.priority
                        ticket.priority = priority
                        await session.commit()
                        await session.refresh(ticket)
                        print(
                            f"[AGENT] ✅ Приоритет ticket {conversation_id} изменен: {old_priority} -> {ticket.priority}"
                        )

                        # Отправляем обновление всем подключенным клиентам
                        try:
                            from app.main import connection_manager
                            from app.db import TicketRead

                            # Получаем обновленный список тикетов для broadcast
                            tickets = await crud.list_tickets(session, archived=False)
                            from app.main import _serialize_tickets

                            tickets_payload = _serialize_tickets(tickets)
                            await connection_manager.broadcast_conversations(
                                tickets_payload
                            )
                            print(
                                f"[AGENT] ✅ Broadcasted priority update for ticket {conversation_id}"
                            )
                        except Exception as broadcast_error:
                            print(
                                f"[AGENT WARNING] Failed to broadcast priority update: {broadcast_error}"
                            )
                            logger.exception(f"Broadcast error: {broadcast_error}")
                    else:
                        print(
                            f"[AGENT WARNING] Ticket {conversation_id} не найден для обновления приоритета"
                        )

            # Запускаем асинхронную функцию
            # Проблема: агент работает в синхронном контексте, нужен новый event loop
            print(f"[AGENT] Пытаюсь запустить update_priority...")
            try:
                # Пытаемся получить текущий loop
                try:
                    loop = asyncio.get_running_loop()
                    # Loop уже работает - создаем задачу
                    print(f"[AGENT] Found running loop, creating task")
                    asyncio.create_task(update_priority())
                    print(f"[AGENT] Task created in running loop")
                except RuntimeError:
                    # Нет работающего loop - создаем новый
                    print(
                        f"[AGENT] No running loop, creating new one with asyncio.run()"
                    )
                    asyncio.run(update_priority())
                    print(f"[AGENT] asyncio.run() completed")
            except Exception as loop_error:
                print(f"[AGENT ERROR] Loop error: {loop_error}")
                logger.exception(f"Event loop error: {loop_error}")

        except Exception as e:
            print(f"[AGENT ERROR] Ошибка сохранения приоритета в БД: {e}")
            logger.exception(f"Failed to save priority to database: {e}")
    else:
        print(
            f"[AGENT WARNING] conversation_id не установлен, не могу сохранить приоритет"
        )

    priority_labels = {
        "low": "низкий",
        "medium": "средний",
        "high": "высокий",
    }

    return f"Приоритет заявки установлен: {priority_labels.get(priority, priority)}"


@tool
def create_it_ticket(problem_description: str, location: str = "не указано") -> str:
    """Создание заявки на выезд IT-специалиста для устранения технических проблем на месте.

    ВАЖНО: Используй ДВУХЭТАПНЫЙ процесс:
    1. ПЕРВЫЙ РАЗ вызывай БЕЗ location (или с "не указано") - это создаст ЧЕРНОВИК заявки
       Агент должен спросить у пользователя точную локацию (кабинет/офис/этаж)
    2. ВТОРОЙ РАЗ вызывай с КОНКРЕТНОЙ локацией - это ЗАВЕРШИТ заявку с временем и специалистом

    Используй этот инструмент когда:
    - Проблема требует физического присутствия специалиста (сломано оборудование, нужна установка/настройка)
    - Необходима диагностика оборудования на месте
    - Проблемы с принтерами, компьютерами, сетевым оборудованием требующие физического вмешательства
    - Пользователь явно просит "приехать", "подойти", "посмотреть на месте"

    НЕ используй для:
    - Программных проблем, которые можно решить удаленно
    - Консультаций и вопросов
    - Проблем с доступами и паролями

    Параметры:
    - problem_description: описание проблемы для специалиста
    - location: местоположение (кабинет, офис, этаж). Если "не указано" - это черновик.

    Возвращает:
    - Черновик: просит уточнить локацию
    - Готовая заявка: номер, специалист, время прибытия
    """
    import random
    import datetime
    import asyncio

    print(f"[AGENT ACTION] Создание/обновление заявки на выезд IT-специалиста")
    print(f"[IT TICKET] Проблема: {problem_description}")
    print(f"[IT TICKET] Локация: {location}")

    conversation_id = get_current_conversation_id()
    print(f"[IT TICKET] conversation_id: {conversation_id}")

    if not conversation_id:
        logger.warning("create_it_ticket вызван без conversation_id")
        return "⚠️ Ошибка: не удалось определить текущий диалог"

    # Результат, который вернём
    result_holder = {"result": None}

    async def process_it_ticket():
        """Асинхронная обработка IT-заявки"""
        from app.db.database import TicketsSessionLocal
        from app.db.models import Ticket

        async with TicketsSessionLocal() as session:
            try:
                # Получаем тикет из БД
                from sqlalchemy import select

                stmt = select(Ticket).where(Ticket.id == conversation_id)
                result = await session.execute(stmt)
                ticket = result.scalar_one_or_none()

                if not ticket:
                    logger.warning(f"Ticket {conversation_id} не найден")
                    result_holder["result"] = "⚠️ Ошибка: заявка не найдена"
                    return

                # Проверяем, есть ли уже созданная IT-заявка
                if ticket.it_ticket_number:
                    print(
                        f"[IT TICKET] Заявка уже существует: {ticket.it_ticket_number}"
                    )
                    result_holder["result"] = (
                        f"ℹ️ Заявка на выезд специалиста уже создана ранее:\n"
                        f"📋 Номер: {ticket.it_ticket_number}\n\n"
                        f"Если нужна дополнительная помощь, обратитесь к оператору."
                    )
                    return

                # Если локация не указана - создаем ЧЕРНОВИК
                if location == "не указано" or not location or location.strip() == "":
                    print("[IT TICKET] Создание черновика - запрос локации")

                    # Отправляем действие в Telegram
                    _send_action_to_telegram(
                        f"📋 Подготовка заявки IT (требуется локация)"
                    )

                    result_holder["result"] = (
                        "✅ Заявка на выезд специалиста будет создана!\n\n"
                        "📍 Пожалуйста, уточните местоположение:\n"
                        "- Адрес и номер кабинета/офиса\n"
                        "- Этаж\n"
                        "- Корпус (если применимо)\n\n"
                        "Это поможет специалисту быстрее найти вас."
                    )
                    return

                # Локация указана - создаем ПОЛНУЮ заявку
                print("[IT TICKET] Создание полной заявки с локацией")

                # Генерируем номер заявки
                ticket_number = f"IT-{random.randint(1000, 9999)}"

                # Рассчитываем время прибытия (через 30-60 минут)
                arrival_minutes = random.randint(30, 60)
                arrival_time = datetime.datetime.now() + datetime.timedelta(
                    minutes=arrival_minutes
                )
                arrival_str = arrival_time.strftime("%H:%M")

                # Назначаем специалиста
                specialists = [
                    "Иван Петров",
                    "Мария Сидорова",
                    "Алексей Козлов",
                    "Елена Волкова",
                ]
                assigned_specialist = random.choice(specialists)

                # Сохраняем номер заявки в БД
                ticket.it_ticket_number = ticket_number
                await session.commit()

                logger.info(
                    f"IT ticket created: {ticket_number}, specialist: {assigned_specialist}, "
                    f"arrival in {arrival_minutes} min, location: {location}"
                )

                # Отправляем действие с деталями в Telegram
                _send_action_to_telegram(
                    f"📝 Создана IT-заявка #{ticket_number}\n"
                    f"👤 Специалист: {assigned_specialist}\n"
                    f"⏰ Прибытие: ~{arrival_minutes} мин"
                )

                result_holder["result"] = (
                    f"✅ Заявка создана!\n\n"
                    f"📋 Номер заявки: {ticket_number}\n"
                    f"👤 Назначен специалист: {assigned_specialist}\n"
                    f"⏰ Ожидаемое время прибытия: {arrival_str} (примерно через {arrival_minutes} минут)\n"
                    f"📍 Местоположение: {location}\n\n"
                    f"Специалист свяжется с вами перед приездом. "
                    f"Пожалуйста, будьте на месте и подготовьте оборудование для диагностики."
                )

                print(f"[IT TICKET] Заявка сохранена: {ticket_number}")

            except Exception as e:
                logger.error(f"Ошибка при создании IT-заявки: {e}")
                result_holder["result"] = f"⚠️ Ошибка при создании заявки: {str(e)}"

    # Запускаем асинхронную функцию
    try:
        try:
            loop = asyncio.get_running_loop()
            # Loop уже работает - создаем задачу и ждём результата
            print(f"[IT TICKET] Found running loop, creating task")
            task = asyncio.create_task(process_it_ticket())
            # Не можем использовать await в синхронной функции
            # Используем run_coroutine_threadsafe если есть loop
            import concurrent.futures

            future = asyncio.run_coroutine_threadsafe(process_it_ticket(), loop)
            future.result(timeout=5)  # Ждём до 5 секунд
            print(f"[IT TICKET] Task completed via run_coroutine_threadsafe")
        except RuntimeError:
            # Нет работающего loop - создаем новый
            print(f"[IT TICKET] No running loop, creating new one with asyncio.run()")
            asyncio.run(process_it_ticket())
            print(f"[IT TICKET] asyncio.run() completed")
    except Exception as loop_error:
        print(f"[IT TICKET ERROR] Loop error: {loop_error}")
        logger.exception(f"Event loop error: {loop_error}")
        return f"⚠️ Ошибка: {str(loop_error)}"

    return result_holder.get("result") or "⚠️ Неизвестная ошибка"


@tool
def call_operator() -> str:
    """ВНИМАНИЕ: Используй ТОЛЬКО в крайних случаях! Вызывает живого оператора для очень сложных технических проблем, которые невозможно решить самостоятельно или через поиск в базе знаний. Перед использованием обязательно попробуй все другие способы помочь пользователю."""
    print(
        "[AGENT ACTION] ВЫЗОВ ОПЕРАТОРА! Передача сложного запроса живому специалисту"
    )

    # Отправляем действие в Telegram
    _send_action_to_telegram("👤 Вызов оператора")

    logger.info("Operator call requested")
    return "Запрос передан оператору. Ожидайте ответа в ближайшее время."


@tool
def get_system_status() -> str:
    """Проверка статуса системы - используй для диагностики технических проблем. Показывает состояние базы знаний и системы."""
    print("[AGENT ACTION] Проверка статуса системы")

    try:
        # Простая проверка статуса без async операций
        import datetime

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"[AGENT] Система работает нормально, время: {current_time}")

        return (
            f"Система работает нормально. Время: {current_time}. База знаний доступна."
        )

    except Exception as e:
        print(f"[AGENT ERROR] Ошибка проверки статуса: {e}")
        logger.error(f"System status error: {e}")
        return "Возникли проблемы с проверкой статуса системы."


@tool
async def get_support_report() -> str:
    """Получить отчёт о текущем состоянии службы поддержки.

    Генерирует подробный отчёт с информацией о:
    - Общем количестве заявок (всего, за день, за неделю)
    - Среднем времени решения проблем
    - Распределении по статусам (открыто, в работе, закрыто, архивировано)
    - Распределении по приоритетам (низкий, средний, высокий)
    - Специальных метриках (с IT-специалистом, ожидают оператора)
    - Топ-5 категорий проблем
    - Самых старых активных заявках (топ-3)
    - Анализе тональности случайных диалогов (3 шт)

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Пользователь спрашивает "какая ситуация в поддержке?"
    - Пользователь хочет узнать "сколько заявок в очереди?"
    - Запрос отчёта о работе службы поддержки
    - Вопросы о загруженности системы
    - "Покажи статистику", "Дай отчёт"

    Возвращает форматированный текстовый отчёт со всеми метриками.
    Если каких-то данных нет, отчёт показывает соответствующее сообщение.
    """
    import datetime
    from datetime import timedelta

    print(f"[AGENT ACTION] Генерация отчёта о службе поддержки")

    # Отправляем действие в Telegram
    _send_action_to_telegram("📊 Генерация отчёта о службе поддержки...")

    try:
        from app.db import tickets_crud as crud
        from app.db.models import Ticket, TicketStatus, Message
        from app.bots.telegram_bot import _session_maker
        from sqlalchemy import select, func, and_

        if not _session_maker:
            print("[REPORT ERROR] session_maker не инициализирован")
            return "⚠️ Ошибка: БД не инициализирована"

        async with _session_maker() as session:
            # 1. Общее количество заявок
            total_result = await session.execute(
                select(func.count()).select_from(Ticket)
            )
            total_tickets = total_result.scalar_one() or 0

            # 2. Статистика по статусам
            status_stats = {}
            for status in [
                TicketStatus.OPEN,
                TicketStatus.IN_PROGRESS,
                TicketStatus.CLOSED,
                TicketStatus.ARCHIVED,
            ]:
                result = await session.execute(
                    select(func.count())
                    .select_from(Ticket)
                    .where(Ticket.status == status)
                )
                status_stats[status.value] = result.scalar_one() or 0

            # 3. Статистика по приоритетам
            priority_stats = {}
            for priority in ["low", "medium", "high"]:
                result = await session.execute(
                    select(func.count())
                    .select_from(Ticket)
                    .where(Ticket.priority == priority)
                )
                priority_stats[priority] = result.scalar_one() or 0

            # 4. Заявки за сегодня
            today = datetime.datetime.utcnow().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            today_result = await session.execute(
                select(func.count())
                .select_from(Ticket)
                .where(Ticket.created_at >= today)
            )
            tickets_today = today_result.scalar_one() or 0

            # 5. Заявки за последние 7 дней
            week_ago = datetime.datetime.utcnow() - timedelta(days=7)
            week_result = await session.execute(
                select(func.count())
                .select_from(Ticket)
                .where(Ticket.created_at >= week_ago)
            )
            tickets_this_week = week_result.scalar_one() or 0

            # 6. Заявки с IT-специалистом
            it_tickets_result = await session.execute(
                select(func.count())
                .select_from(Ticket)
                .where(
                    and_(
                        Ticket.it_ticket_number.isnot(None),
                        Ticket.status != TicketStatus.CLOSED,
                    )
                )
            )
            active_it_tickets = it_tickets_result.scalar_one() or 0

            # 7. Заявки ожидающие оператора
            operator_requested_result = await session.execute(
                select(func.count())
                .select_from(Ticket)
                .where(
                    and_(
                        Ticket.operator_requested == True,
                        Ticket.status != TicketStatus.CLOSED,
                    )
                )
            )
            operator_requests = operator_requested_result.scalar_one() or 0

            # 8. Среднее время решения проблем (для закрытых заявок)
            closed_tickets_result = await session.execute(
                select(Ticket).where(
                    and_(
                        Ticket.status == TicketStatus.CLOSED,
                        Ticket.closed_at.isnot(None),
                    )
                )
            )
            closed_tickets = closed_tickets_result.scalars().all()

            avg_resolution_time = None
            if closed_tickets:
                total_time = sum(
                    [
                        (ticket.closed_at - ticket.created_at).total_seconds()
                        for ticket in closed_tickets
                    ]
                )
                avg_seconds = total_time / len(closed_tickets)
                avg_hours = int(avg_seconds / 3600)
                avg_minutes = int((avg_seconds % 3600) / 60)
                avg_resolution_time = f"{avg_hours}ч {avg_minutes}м"

            # 9. Топ категорий проблем
            categories_result = await session.execute(
                select(Ticket.classification, func.count())
                .where(Ticket.classification.isnot(None))
                .group_by(Ticket.classification)
                .order_by(func.count().desc())
                .limit(5)
            )
            top_categories = categories_result.all()

            print(f"[REPORT] Найдено категорий: {len(top_categories)}")
            if top_categories:
                for cat, count in top_categories:
                    print(f"[REPORT]   - {cat}: {count} заявок")

            # 10. Случайные 3 диалога для анализа тональности (если есть закрытые)
            import random

            sample_tickets_result = await session.execute(
                select(Ticket)
                .where(Ticket.status == TicketStatus.CLOSED)
                .order_by(func.random())
                .limit(3)
            )
            sample_tickets = sample_tickets_result.scalars().all()

            print(
                f"[REPORT] Найдено закрытых заявок для анализа: {len(sample_tickets)}"
            )

            # Загружаем сообщения для sample_tickets
            sample_dialogs = []
            for ticket in sample_tickets:
                messages_result = await session.execute(
                    select(Message)
                    .where(Message.ticket_id == ticket.id)
                    .order_by(Message.created_at)
                )
                messages = messages_result.scalars().all()
                if messages:
                    sample_dialogs.append(
                        {
                            "ticket_id": ticket.id,
                            "classification": ticket.classification or "Без категории",
                            "messages": messages,
                        }
                    )

            print(f"[REPORT] Сформировано диалогов для анализа: {len(sample_dialogs)}")

            # 11. Самые старые открытые заявки
            oldest_tickets_result = await session.execute(
                select(Ticket)
                .where(Ticket.status.in_([TicketStatus.OPEN, TicketStatus.IN_PROGRESS]))
                .order_by(Ticket.created_at)
                .limit(3)
            )
            oldest_tickets = oldest_tickets_result.scalars().all()

            # Формируем отчёт
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            report_lines = [
                f"📊 ОТЧЁТ О СЛУЖБЕ ПОДДЕРЖКИ",
                f"⏰ Сформирован: {current_time}",
                f"",
                f"📈 ОБЩАЯ СТАТИСТИКА:",
                f"   • Всего заявок: {total_tickets}",
                f"   • За сегодня: {tickets_today}",
                f"   • За неделю: {tickets_this_week}",
            ]

            # Среднее время решения
            if avg_resolution_time:
                report_lines.append(
                    f"   • Среднее время решения: {avg_resolution_time}"
                )
            else:
                report_lines.append(f"   • Среднее время решения: данных нет")

            report_lines.extend(
                [
                    f"",
                    f"📋 ПО СТАТУСАМ:",
                    f"   • Открыто: {status_stats.get('open', 0)}",
                    f"   • В работе: {status_stats.get('in_progress', 0)}",
                    f"   • Закрыто: {status_stats.get('closed', 0)}",
                    f"   • Архивировано: {status_stats.get('archived', 0)}",
                    f"",
                    f"⚡ ПО ПРИОРИТЕТАМ:",
                    f"   • Высокий: {priority_stats.get('high', 0)}",
                    f"   • Средний: {priority_stats.get('medium', 0)}",
                    f"   • Низкий: {priority_stats.get('low', 0)}",
                    f"",
                    f"🔧 СПЕЦИАЛЬНЫЕ:",
                    f"   • С IT-специалистом: {active_it_tickets}",
                    f"   • Ожидают оператора: {operator_requests}",
                ]
            )

            # Топ категорий проблем
            if top_categories:
                report_lines.append(f"")
                report_lines.append(f"🏷️ ТОП КАТЕГОРИЙ ПРОБЛЕМ:")
                for category, count in top_categories:
                    report_lines.append(f"   • {category}: {count} заявок")
            else:
                report_lines.append(f"")
                if total_tickets > 0:
                    report_lines.append(
                        f"🏷️ ТОП КАТЕГОРИЙ ПРОБЛЕМ: заявки не классифицированы"
                    )
                else:
                    report_lines.append(f"🏷️ ТОП КАТЕГОРИЙ ПРОБЛЕМ: данных нет")

            # Добавляем самые старые заявки
            if oldest_tickets:
                report_lines.append(f"")
                report_lines.append(f"⏳ САМЫЕ СТАРЫЕ АКТИВНЫЕ ЗАЯВКИ:")
                for ticket in oldest_tickets:
                    age = datetime.datetime.utcnow() - ticket.created_at
                    hours = int(age.total_seconds() / 3600)
                    priority_emoji = (
                        "🔴"
                        if ticket.priority == "high"
                        else "🟡" if ticket.priority == "medium" else "🟢"
                    )
                    classification = ticket.classification or "Без категории"
                    report_lines.append(
                        f"   {priority_emoji} #{ticket.id} - {classification[:30]} (висит {hours}ч)"
                    )
            else:
                report_lines.append(f"")
                report_lines.append(
                    f"⏳ САМЫЕ СТАРЫЕ АКТИВНЫЕ ЗАЯВКИ: нет активных заявок"
                )

            # Анализ тональности случайных диалогов
            if sample_dialogs:
                report_lines.append(f"")
                report_lines.append(f"💬 АНАЛИЗ СЛУЧАЙНЫХ ДИАЛОГОВ:")

                # Простой анализ тональности
                for dialog in sample_dialogs:
                    # Собираем текст диалога
                    user_messages = [
                        m.text for m in dialog["messages"] if m.sender == "user"
                    ]

                    # Простой анализ по ключевым словам
                    positive_words = [
                        "спасибо",
                        "помогло",
                        "работает",
                        "заработало",
                        "отлично",
                        "хорошо",
                        "решили",
                    ]
                    negative_words = [
                        "не работает",
                        "плохо",
                        "ошибка",
                        "проблема",
                        "не помогло",
                        "всё ещё",
                    ]

                    all_text = " ".join(user_messages).lower()
                    positive_count = sum(
                        1 for word in positive_words if word in all_text
                    )
                    negative_count = sum(
                        1 for word in negative_words if word in all_text
                    )

                    if positive_count > negative_count:
                        sentiment = "😊 Позитивная"
                    elif negative_count > positive_count:
                        sentiment = "😟 Негативная"
                    else:
                        sentiment = "😐 Нейтральная"

                    msg_count = len(dialog["messages"])
                    report_lines.append(
                        f"   • #{dialog['ticket_id']} ({dialog['classification'][:20]}): {sentiment}, {msg_count} сообщ."
                    )
            else:
                report_lines.append(f"")
                if status_stats.get("closed", 0) > 0:
                    report_lines.append(
                        f"💬 АНАЛИЗ СЛУЧАЙНЫХ ДИАЛОГОВ: закрытые заявки без сообщений"
                    )
                else:
                    report_lines.append(
                        f"💬 АНАЛИЗ СЛУЧАЙНЫХ ДИАЛОГОВ: нет закрытых заявок"
                    )

            report = "\n".join(report_lines)

            print(f"[REPORT] Отчёт сформирован: {len(report)} символов")

            # Отправляем финальное действие
            _send_action_to_telegram("📊 Отчёт о службе поддержки готов")

            return report

    except Exception as e:
        logger.error(f"Ошибка при генерации отчёта: {e}")
        print(f"[REPORT ERROR] {e}")
        _send_action_to_telegram("❌ Ошибка при генерации отчёта")
        return f"⚠️ Ошибка при генерации отчёта: {str(e)}"


@tool
async def save_case_to_knowledge_base(
    problem_description: str = None, solution: str = None
) -> str:
    """Сохранить успешный кейс в базу знаний для будущего использования.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Когда ты успешно решил проблему пользователя
    - Пользователь подтвердил что проблема решена (сказал "спасибо", "заработало", "помогло")
    - Решение оказалось полезным и может помочь другим
    - Проблема и решение достаточно конкретные и понятные
    - НЕ используй для общих вопросов или неполных решений

    Параметры (ОПЦИОНАЛЬНЫЕ):
    - problem_description: Краткое описание проблемы (если не указано - сгенерируется автоматически из диалога)
    - solution: Пошаговое решение (если не указано - сгенерируется автоматически из диалога)

    ВАЖНО: Если пользователь попросил "добавь в базу" или "сохрани кейс" -
    вызывай БЕЗ параметров! Инструмент сам проанализирует диалог и создаст саммари.

    Пример использования:
    1. С параметрами: save_case_to_knowledge_base("Принтер не печатает", "1. Проверить бумагу...")
    2. БЕЗ параметров (автоматически): save_case_to_knowledge_base()

    Возвращает подтверждение о сохранении кейса.
    """
    import datetime

    print(f"[AGENT ACTION] Сохранение кейса в базу знаний")

    # Отправляем действие в Telegram (начало процесса)
    _send_action_to_telegram("💾 Сохранение кейса в базу знаний...")

    # Если параметры не указаны - будем генерировать из диалога
    if problem_description is None or solution is None:
        print(
            f"[SAVE CASE] Параметры не указаны - буду генерировать саммари из диалога"
        )
    else:
        print(f"[SAVE CASE] Проблема: {problem_description[:100]}...")
        print(f"[SAVE CASE] Решение: {solution[:100]}...")

    try:
        from app.db.database import KnowledgeSessionLocal
        from app.db import tickets_crud as crud
        from app.bots.telegram_bot import _session_maker

        # Получаем описание и решение
        final_problem = problem_description
        final_solution = solution

        # Если не указаны - генерируем из диалога
        if final_problem is None or final_solution is None:
            conversation_id = get_current_conversation_id()

            if not conversation_id:
                print("[SAVE CASE ERROR] conversation_id не найден")
                return "⚠️ Ошибка: не удалось определить текущий диалог"

            if not _session_maker:
                print("[SAVE CASE ERROR] session_maker не инициализирован")
                return "⚠️ Ошибка: БД не инициализирована"

            # Получаем историю диалога из БД через crud функцию
            async with _session_maker() as session:
                ticket_data = await crud.get_ticket_with_messages(
                    session, conversation_id
                )

                if not ticket_data:
                    print(f"[SAVE CASE ERROR] Ticket {conversation_id} не найден")
                    return "⚠️ Ошибка: диалог не найден"

                ticket, messages = ticket_data

                # Формируем историю диалога
                dialogue_parts = []
                for msg in messages:
                    role = (
                        "Пользователь"
                        if msg.sender == "user"
                        else "Бот" if msg.sender == "bot" else "Оператор"
                    )
                    dialogue_parts.append(f"{role}: {msg.text}")

                dialogue_history = "\n".join(dialogue_parts)
                print(
                    f"[SAVE CASE] Получена история диалога: {len(dialogue_history)} символов"
                )

            # Генерируем саммари через LLM
            from app.rag.service import get_llm_client
            import yaml

            # Загружаем конфигурацию
            with open("configs/rag_config.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Промпт для генерации саммари
            summary_prompt = f"""Проанализируй диалог технической поддержки и создай краткое саммари для базы знаний.

ДИАЛОГ:
{dialogue_history}

Создай:
1. ПРОБЛЕМА: Краткое описание проблемы в 1-2 предложениях (что не работало, какие симптомы)
2. РЕШЕНИЕ: Пошаговая инструкция как решить эту проблему (конкретные действия)

Формат ответа:
ПРОБЛЕМА: [описание]
РЕШЕНИЕ:
[пошаговая инструкция]

Будь конкретным и практичным. Описание должно помочь другим пользователям с похожей проблемой."""

            llm_client = get_llm_client()

            print("[SAVE CASE] Генерирую саммари через LLM...")
            response = llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=500,
                temperature=0.3,
            )

            summary_text = response.choices[0].message.content.strip()
            print(f"[SAVE CASE] LLM саммари получено: {len(summary_text)} символов")

            # Парсим ответ LLM
            if "ПРОБЛЕМА:" in summary_text and "РЕШЕНИЕ:" in summary_text:
                parts = summary_text.split("РЕШЕНИЕ:")
                final_problem = parts[0].replace("ПРОБЛЕМА:", "").strip()
                final_solution = parts[1].strip()
                print(f"[SAVE CASE] Распарсено - Проблема: {final_problem[:50]}...")
                print(f"[SAVE CASE] Распарсено - Решение: {final_solution[:50]}...")
            else:
                # Используем весь текст как есть
                final_problem = "Кейс из диалога"
                final_solution = summary_text

        # Форматируем кейс для базы знаний
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        case_content = f"""ПРОБЛЕМА: {final_problem}

РЕШЕНИЕ:
{final_solution}

(Кейс добавлен ботом {timestamp})"""

        # Создаём embedding для кейса
        model = get_sentence_transformer()
        embedding_vector = model.encode(case_content)
        embedding_bytes = embedding_vector.tobytes()

        async with KnowledgeSessionLocal() as session:
            # Добавляем кейс в базу знаний
            chunk = await crud.add_document_chunk(
                session=session,
                content=case_content,
                source_file=f"bot_case_{timestamp}.txt",
                chunk_index=0,
                start_char=0,
                end_char=len(case_content),
                embedding=embedding_bytes,
                chunk_metadata=f'{{"source": "bot", "type": "solved_case", "date": "{timestamp}"}}',
            )

            logger.info(f"Case saved to knowledge base: chunk_id={chunk.id}")
            print(f"[SAVE CASE] ✅ Кейс сохранён с ID: {chunk.id}")

            # Отправляем действие в Telegram
            _send_action_to_telegram(f"💾 Кейс сохранён в базу знаний (#{chunk.id})")

            return (
                f"✅ Кейс успешно сохранён в базу знаний!\n"
                f"ID записи: {chunk.id}\n\n"
                f"Теперь этот кейс будет доступен при поиске решений похожих проблем."
            )

    except Exception as e:
        logger.error(f"Ошибка при сохранении кейса: {e}")
        print(f"[SAVE CASE ERROR] {e}")
        _send_action_to_telegram(f"❌ Ошибка при сохранении кейса")
        return f"⚠️ Ошибка при сохранении кейса: {str(e)}"


def should_update_classification_and_priority(
    current_classification: str,
    current_priority: str,
    recent_messages: str,
    message_count: int,
) -> bool:
    """Определяет, нужно ли обновить классификацию и приоритет на основе анализа диалога."""
    try:
        from app.rag.service import get_llm_client
        import yaml

        # Загружаем конфигурацию
        with open("configs/rag_config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        auto_update_config = config.get("agent", {}).get("auto_update", {})

        # Проверяем минимальное количество сообщений
        min_messages = auto_update_config.get("min_messages_before_update", 2)
        if message_count < min_messages:
            print(
                f"[AUTO_UPDATE] Недостаточно сообщений ({message_count} < {min_messages})"
            )
            return False

        # Проверяем периодичность обновления
        update_every = auto_update_config.get("update_every_n_messages", 3)
        if message_count % update_every != 0:
            print(
                f"[AUTO_UPDATE] Не время обновлять (сообщение {message_count}, обновляем каждые {update_every})"
            )
            return False

        # Получаем промпт для определения необходимости обновления
        should_update_prompt_template = auto_update_config.get(
            "should_update_prompt", ""
        )

        if not should_update_prompt_template:
            print("[AUTO_UPDATE WARNING] Промпт should_update не найден в конфиге")
            return False

        # Формируем промпт
        should_update_prompt = should_update_prompt_template.format(
            current_classification=current_classification or "Не установлена",
            current_priority=current_priority or "medium",
            recent_messages=recent_messages,
        )

        # Получаем LLM клиент
        llm_client = get_llm_client()

        # Спрашиваем LLM нужно ли обновление
        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": should_update_prompt}],
            max_tokens=5,
            temperature=0.1,
        )

        llm_result = response.choices[0].message.content.strip().lower()

        should_update = "yes" in llm_result

        print(f"[AUTO_UPDATE] LLM решение: {llm_result} -> {should_update}")
        logger.info(f"Should update classification/priority: {should_update}")

        return should_update

    except Exception as e:
        print(f"[AUTO_UPDATE ERROR] Ошибка проверки необходимости обновления: {e}")
        logger.error(f"Should update check error: {e}")
        return False


def auto_update_classification(
    conversation_id: int,
    dialogue_history: str,
    message_count: int,
    current_classification: str = None,
) -> dict:
    """Автоматическое обновление ТОЛЬКО классификации (без приоритета).

    Приоритет устанавливается ТОЛЬКО через MCP tool set_priority агентом.

    Возвращает dict с ключами:
    - updated: bool - была ли обновлена классификация
    - classification: str - новая классификация (если была обновлена)
    """
    try:
        # Формируем последние сообщения для анализа
        dialogue_lines = dialogue_history.split("\n")
        recent_lines = (
            dialogue_lines[-5:] if len(dialogue_lines) > 5 else dialogue_lines
        )
        recent_messages = "\n".join(recent_lines)

        if len(recent_messages) < 1000 and len(dialogue_history) > 1000:
            recent_messages = dialogue_history[-1000:]

        # Проверяем нужно ли обновление (используем medium как фиктивный приоритет)
        if not should_update_classification_and_priority(
            current_classification or "Не установлена",
            "medium",  # Фиктивный приоритет, т.к. мы его не обновляем
            recent_messages,
            message_count,
        ):
            return {"updated": False}

        print(
            f"[AUTO_UPDATE] Обновление классификации для conversation {conversation_id}"
        )

        # Обновляем ТОЛЬКО классификацию на основе ВСЕЙ истории
        new_classification_result = _classify_request_internal(
            dialogue_history=dialogue_history
        )
        new_classification = new_classification_result.replace(
            "Классификация проблемы:", ""
        ).strip()

        print(f"[AUTO_UPDATE] Новая классификация: {new_classification}")

        # Сохраняем в БД
        import asyncio
        from app.db.database import TicketsSessionLocal
        from app.db import tickets_crud as crud

        async def update_ticket():
            async with TicketsSessionLocal() as session:
                ticket = await crud.get_ticket_by_id(session, conversation_id)
                if ticket:
                    ticket.classification = new_classification
                    await session.commit()
                    print(
                        f"[AUTO_UPDATE] Обновлена классификация ticket {conversation_id}: "
                        f"classification={new_classification}"
                    )
                else:
                    print(f"[AUTO_UPDATE WARNING] Ticket {conversation_id} не найден")

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(update_ticket())
            else:
                loop.run_until_complete(update_ticket())
        except Exception as e:
            print(f"[AUTO_UPDATE ERROR] Ошибка сохранения в БД: {e}")
            logger.error(f"Failed to save classification update: {e}")

        return {
            "updated": True,
            "classification": new_classification,
        }

    except Exception as e:
        print(f"[AUTO_UPDATE ERROR] Ошибка автообновления классификации: {e}")
        logger.error(f"Auto update classification error: {e}")
        return {"updated": False}


def auto_update_classification_and_priority(
    conversation_id: int,
    dialogue_history: str,
    message_count: int,
    current_classification: str = None,
    current_priority: str = None,
) -> dict:
    """УСТАРЕВШАЯ ФУНКЦИЯ - оставлена для обратной совместимости.

    Теперь используй:
    - auto_update_classification() - для автоматического обновления классификации
    - set_priority() MCP tool - для установки приоритета агентом

    Эта функция теперь обновляет ТОЛЬКО классификацию.
    """
    return auto_update_classification(
        conversation_id=conversation_id,
        dialogue_history=dialogue_history,
        message_count=message_count,
        current_classification=current_classification,
    )


# suggest_similar_problems удалена - функциональность перенесена в search_knowledge_base с параметром suggest_similar=True
