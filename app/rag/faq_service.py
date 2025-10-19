"""
Сервис для управления FAQ на основе популярных запросов к базе знаний
"""

import logging
from collections import Counter
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import asyncio
import threading

logger = logging.getLogger(__name__)

# Глобальный счётчик использования чанков
_chunk_usage_counter: Counter = Counter()
_chunk_usage_lock = threading.Lock()

# Кэш FAQ
_faq_cache: List[Dict] = []
_faq_cache_lock = threading.Lock()
_last_faq_update: Optional[datetime] = None
_faq_update_in_progress = False  # Флаг фонового обновления

# Настройки из конфига (будут загружены при инициализации)
_faq_update_interval_minutes = 5
_faq_top_chunks_count = 10


def track_chunk_usage(chunk_id: int, chunk_content: str):
    """Отслеживать использование чанка в поиске

    Args:
        chunk_id: ID чанка из базы данных
        chunk_content: Содержимое чанка (для последующей генерации FAQ)
    """
    with _chunk_usage_lock:
        _chunk_usage_counter[chunk_id] = _chunk_usage_counter.get(chunk_id, 0) + 1
        # Сохраняем также содержимое чанка для генерации FAQ
        if not hasattr(_chunk_usage_counter, "_chunk_contents"):
            _chunk_usage_counter._chunk_contents = {}
        _chunk_usage_counter._chunk_contents[chunk_id] = chunk_content

    logger.debug(
        f"[FAQ] Chunk {chunk_id} usage tracked: {_chunk_usage_counter[chunk_id]} times"
    )


def get_top_chunks(limit: int = 10) -> List[tuple]:
    """Получить топ самых используемых чанков

    Returns:
        List of tuples: (chunk_id, usage_count, chunk_content)
    """
    with _chunk_usage_lock:
        top_chunks = _chunk_usage_counter.most_common(limit)

        # Добавляем содержимое чанков
        result = []
        chunk_contents = getattr(_chunk_usage_counter, "_chunk_contents", {})
        for chunk_id, count in top_chunks:
            content = chunk_contents.get(chunk_id, "")
            result.append((chunk_id, count, content))

        logger.info(f"[FAQ] Top {limit} chunks retrieved: {len(result)} items")
        return result


def reset_chunk_counter():
    """Сбросить счётчик использования чанков (после генерации FAQ)"""
    with _chunk_usage_lock:
        _chunk_usage_counter.clear()
        if hasattr(_chunk_usage_counter, "_chunk_contents"):
            _chunk_usage_counter._chunk_contents.clear()
    logger.info("[FAQ] Chunk usage counter reset")


async def generate_faq_from_chunks(top_chunks: List[tuple], config: Dict) -> List[Dict]:
    """Сгенерировать FAQ на основе топ-чанков через LLM

    Args:
        top_chunks: Список (chunk_id, usage_count, chunk_content)
        config: Конфигурация приложения

    Returns:
        List of FAQ items: [{"question": "...", "answer": "..."}]
    """
    if not top_chunks:
        logger.warning("[FAQ] No top chunks provided for FAQ generation")
        return []

    try:
        from app.rag.service import get_llm_client

        # Формируем промпт для LLM
        chunks_text = "\n\n".join(
            [
                f"Чанк #{chunk_id} (использован {count} раз):\n{content[:500]}"
                for chunk_id, count, content in top_chunks
            ]
        )

        prompt = f"""На основе следующих популярных фрагментов из базы знаний создай 10 самых важных вопросов с ответами для FAQ.

ПОПУЛЯРНЫЕ ФРАГМЕНТЫ:
{chunks_text}

Создай 10 вопросов в формате JSON:
[
  {{"question": "Вопрос 1", "answer": "Подробный ответ 1"}},
  {{"question": "Вопрос 2", "answer": "Подробный ответ 2"}},
  ...
]

ТРЕБОВАНИЯ:
- Вопросы должны быть понятными и конкретными
- Ответы должны быть практичными и полезными
- Охватывай разные темы из предоставленных фрагментов
- Используй простой понятный язык
- Вопросы должны отражать реальные проблемы пользователей
- Ответ должен быть 2-5 предложений

Верни ТОЛЬКО JSON массив, без дополнительного текста."""

        llm_client = get_llm_client()

        logger.info("[FAQ] Generating FAQ through LLM...")
        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.7,
        )

        response_text = response.choices[0].message.content.strip()
        logger.info(f"[FAQ] LLM response received: {len(response_text)} characters")

        # Парсим JSON
        import json
        import re

        # Извлекаем JSON из ответа (может быть в кодовом блоке)
        json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
            faq_items = json.loads(json_text)
            logger.info(f"[FAQ] Generated {len(faq_items)} FAQ items")
            return faq_items[:10]  # Ограничиваем 10 вопросами
        else:
            logger.error("[FAQ] Failed to extract JSON from LLM response")
            return []

    except Exception as e:
        logger.error(f"[FAQ] Error generating FAQ: {e}")
        return []


def get_faq_cache() -> List[Dict]:
    """Получить кэшированный FAQ"""
    with _faq_cache_lock:
        return _faq_cache.copy()


def set_faq_cache(faq_items: List[Dict]):
    """Установить кэш FAQ"""
    global _last_faq_update
    with _faq_cache_lock:
        _faq_cache.clear()
        _faq_cache.extend(faq_items)
        _last_faq_update = datetime.now()
    logger.info(
        f"[FAQ] Cache updated with {len(faq_items)} items at {_last_faq_update}"
    )


def should_update_faq() -> bool:
    """Проверить нужно ли обновить FAQ"""
    if _last_faq_update is None:
        return True

    elapsed = datetime.now() - _last_faq_update
    should_update = elapsed >= timedelta(minutes=_faq_update_interval_minutes)

    if should_update:
        logger.info(
            f"[FAQ] Update needed. Last update: {_last_faq_update}, elapsed: {elapsed}"
        )

    return should_update


def set_faq_config(update_interval_minutes: int = 5, top_chunks_count: int = 10):
    """Установить настройки FAQ из конфига"""
    global _faq_update_interval_minutes, _faq_top_chunks_count
    _faq_update_interval_minutes = update_interval_minutes
    _faq_top_chunks_count = top_chunks_count
    logger.info(
        f"[FAQ] Config set: update_interval={update_interval_minutes}m, top_chunks={top_chunks_count}"
    )


async def _perform_faq_update(config: Dict):
    """Внутренняя функция для фонового обновления FAQ"""
    global _faq_update_in_progress

    try:
        logger.info("[FAQ] Starting FAQ update...")

        # Получаем топ чанков
        top_chunks = get_top_chunks(_faq_top_chunks_count)

        if not top_chunks:
            logger.warning("[FAQ] No chunk usage data, keeping old FAQ")
            return

        # Генерируем новый FAQ
        new_faq = await generate_faq_from_chunks(top_chunks, config)

        if new_faq:
            set_faq_cache(new_faq)
            # Сбрасываем счётчик после успешного обновления
            reset_chunk_counter()
            logger.info(f"[FAQ] Successfully updated with {len(new_faq)} items")
        else:
            logger.warning("[FAQ] Failed to generate new FAQ, keeping old")
    except Exception as e:
        logger.error(f"[FAQ] Error during background update: {e}")
    finally:
        _faq_update_in_progress = False


def start_faq_update_background(config: Dict):
    """Запустить обновление FAQ в фоне (неблокирующе)

    Args:
        config: Конфигурация приложения
    """
    global _faq_update_in_progress

    # Проверяем нужно ли обновлять
    if not should_update_faq():
        logger.debug("[FAQ] Update not needed yet")
        return False

    # Проверяем не идёт ли уже обновление
    if _faq_update_in_progress:
        logger.debug("[FAQ] Update already in progress")
        return False

    # Устанавливаем флаг
    _faq_update_in_progress = True

    # Запускаем обновление в фоне
    import asyncio

    try:
        loop = asyncio.get_running_loop()
        asyncio.create_task(_perform_faq_update(config))
        logger.info("[FAQ] Background update task created")
        return True
    except RuntimeError:
        logger.error("[FAQ] No running event loop for background update")
        _faq_update_in_progress = False
        return False


async def update_faq_if_needed(config: Dict):
    """Обновить FAQ если прошло достаточно времени (устаревшая, используйте start_faq_update_background)

    Оставлено для совместимости, но рекомендуется использовать start_faq_update_background
    для неблокирующего обновления.

    Args:
        config: Конфигурация приложения
    """
    if not should_update_faq():
        logger.debug("[FAQ] Update not needed yet")
        return

    await _perform_faq_update(config)


async def search_faq(query: str) -> List[Dict]:
    """Поиск по FAQ (поиск по ключевым словам)

    Args:
        query: Поисковый запрос

    Returns:
        Отфильтрованный список FAQ items, отсортированный по релевантности
    """
    faq_items = get_faq_cache()

    if not query or len(query) < 2:
        return faq_items

    query_lower = query.lower()

    # Разбиваем запрос на отдельные слова (ключевые слова)
    # Удаляем стоп-слова и слова короче 2 символов
    stop_words = {
        "и",
        "в",
        "на",
        "с",
        "по",
        "для",
        "как",
        "что",
        "это",
        "or",
        "an",
        "the",
        "a",
    }
    keywords = [
        word.strip()
        for word in query_lower.split()
        if len(word.strip()) >= 2 and word.strip() not in stop_words
    ]

    if not keywords:
        # Если после фильтрации не осталось слов, используем исходный запрос
        keywords = [query_lower]

    logger.debug(f"[FAQ] Search keywords: {keywords}")

    # Поиск с подсчётом релевантности
    results = []
    for item in faq_items:
        question_lower = item.get("question", "").lower()
        answer_lower = item.get("answer", "").lower()

        # Считаем сколько ключевых слов найдено
        matches = 0
        for keyword in keywords:
            if keyword in question_lower:
                matches += 2  # Вопрос важнее
            if keyword in answer_lower:
                matches += 1  # Ответ менее важен

        if matches > 0:
            results.append(
                {**item, "_relevance": matches}  # Временное поле для сортировки
            )

    # Сортируем по релевантности (больше совпадений = выше)
    results.sort(key=lambda x: x["_relevance"], reverse=True)

    # Убираем временное поле релевантности
    filtered = [
        {k: v for k, v in item.items() if k != "_relevance"} for item in results
    ]

    logger.debug(
        f"[FAQ] Search '{query}' (keywords: {keywords}) returned {len(filtered)} results"
    )
    return filtered


async def load_qa_xlsx_if_empty(session, qa_file_path: str) -> int:
    """Загрузить QA.xlsx в document_chunks если база пустая

    Args:
        session: Async database session
        qa_file_path: Путь к файлу QA.xlsx

    Returns:
        Количество загруженных записей
    """
    try:
        import pandas as pd
        from app.db.tickets_crud import count_document_chunks, add_document_chunk
        from app.rag.service import get_sentence_transformer
        import os

        # Проверяем, пуста ли база
        chunk_count = await count_document_chunks(session)
        if chunk_count > 0:
            logger.info(
                f"[QA Loader] Knowledge base already has {chunk_count} chunks, skipping auto-load"
            )
            return 0

        # Проверяем наличие файла
        if not os.path.exists(qa_file_path):
            logger.warning(f"[QA Loader] QA file not found: {qa_file_path}")
            return 0

        logger.info(f"[QA Loader] Knowledge base is empty, loading {qa_file_path}...")

        # Читаем Excel файл
        df = pd.read_excel(qa_file_path)

        # Ожидаем колонки: Question, Answer (или Вопрос, Ответ)
        # Попробуем найти подходящие колонки
        question_col = None
        answer_col = None

        for col in df.columns:
            col_lower = col.lower()
            if "question" in col_lower or "вопрос" in col_lower or "q" == col_lower:
                question_col = col
            elif "answer" in col_lower or "ответ" in col_lower or "a" == col_lower:
                answer_col = col

        if not question_col or not answer_col:
            logger.error(
                f"[QA Loader] Could not find question/answer columns in {qa_file_path}"
            )
            logger.error(f"[QA Loader] Available columns: {df.columns.tolist()}")
            return 0

        logger.info(
            f"[QA Loader] Found columns: Question='{question_col}', Answer='{answer_col}'"
        )

        # Получаем модель для генерации эмбеддингов
        model = get_sentence_transformer()

        loaded_count = 0
        for idx, row in df.iterrows():
            try:
                question = str(row[question_col]).strip()
                answer = str(row[answer_col]).strip()

                # Пропускаем пустые строки
                if not question or not answer or question == "nan" or answer == "nan":
                    continue

                # Объединяем вопрос и ответ
                content = f"Вопрос: {question}\n\nОтвет: {answer}"

                # Генерируем эмбеддинг
                embedding = model.encode(content).tolist()

                # Добавляем в базу
                await add_document_chunk(
                    session=session,
                    content=content,
                    source_file="QA.xlsx",
                    chunk_index=idx,
                    embedding=embedding,
                    metadata={
                        "question": question,
                        "answer": answer,
                        "auto_loaded": True,
                    },
                )

                loaded_count += 1

            except Exception as e:
                logger.error(f"[QA Loader] Error processing row {idx}: {e}")
                continue

        await session.commit()
        logger.info(
            f"[QA Loader] Successfully loaded {loaded_count} QA pairs from {qa_file_path}"
        )
        return loaded_count

    except ImportError as e:
        logger.error(
            f"[QA Loader] Missing dependency: {e}. Install with: pip install pandas openpyxl"
        )
        return 0
    except Exception as e:
        logger.error(f"[QA Loader] Error loading QA file: {e}")
        return 0
