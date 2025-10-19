from __future__ import annotations

import json
import logging
import os
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import faiss
import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import BertForSequenceClassification, BertTokenizer
import torch

from app.db import tickets_crud as crud
from app.db import KnowledgeSessionLocal

logger = logging.getLogger(__name__)


def get_llm_client():
    """Создает и возвращает клиент для LLM"""
    import yaml

    # Загружаем конфигурацию
    with open("configs/rag_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    llm_cfg = config.get("llm", {})
    base_url = (
        llm_cfg.get("base_url")
        or os.getenv("LLM_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
    )
    api_key = (
        llm_cfg.get("api_key")
        or os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )

    client_kwargs = {"api_key": api_key or "EMPTY"}
    if base_url:
        client_kwargs["base_url"] = base_url

    return OpenAI(**client_kwargs)


def _strip_thinking_tags(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _parse_float(text: str) -> float | None:
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return float(match.group()) if match else None


def _preprocess(text: str) -> str:
    """Простая очистка текста от лишних символов"""
    cleaned = re.sub(r"[^\w\s?!]", " ", text)
    words = cleaned.split()
    return " ".join(words)


@dataclass
class ChatMessage:
    """Сообщение в чате для временного хранения"""

    message: str
    is_user: bool
    timestamp: datetime


@dataclass
class RAGResult:
    final_answer: str
    operator_requested: bool = False
    filter_info: dict[str, Any] | None = None
    confidence_score: float = 1.0  # Добавляем оценку уверенности (чем выше, тем хуже)


class ToxicityClassifier:
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested for toxicity model but not available. Falling back to CPU."
            )
            device = "cpu"
        self.device = torch.device(device)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path).to(
            self.device
        )

    def infer(self, text: str) -> float:
        inputs = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        return probs[0][1].item()


class SpeechToTextService:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Инициализация сервиса speech-to-text с локальной моделью Whisper"""
        from app.rag.whisper_service import WhisperService

        # Получаем настройки из конфигурации
        speech_cfg = (config or {}).get("speech", {})
        model_name = speech_cfg.get("whisper_model", "medium")
        ffmpeg_path = speech_cfg.get("ffmpeg_path", "") or None

        logger.info(f"🔧 Инициализация SpeechToTextService")
        logger.info(f"   - Модель: {model_name}")
        logger.info(f"   - FFmpeg путь из конфига: {ffmpeg_path}")

        # Используем локальную модель Whisper
        self.whisper = WhisperService(model_name=model_name, ffmpeg_path=ffmpeg_path)
        logger.info(
            f"SpeechToTextService инициализирован с моделью Whisper '{model_name}'"
        )

    async def transcribe_audio(
        self, audio_file_path: str, language: str = "ru-RU"
    ) -> str:
        """
        Преобразование аудио в текст с помощью локальной модели Whisper

        Args:
            audio_file_path: Путь к аудио файлу
            language: Язык аудио (по умолчанию русский)

        Returns:
            Транскрибированный текст
        """
        # Конвертируем формат языка из ru-RU в ru для Whisper
        lang_code = language.split("-")[0] if "-" in language else language

        # Whisper работает локально и поддерживает все популярные форматы
        return await self.whisper.transcribe_audio(audio_file_path, language=lang_code)


class RAGService:
    def __init__(
        self,
        config: dict[str, Any],
    ) -> None:
        self.config = config

        llm_cfg = config.get("llm", {})

        base_url = llm_cfg.get("base_url", "")
        env_base_url = os.getenv("LLM_API_BASE") or os.getenv("OPENAI_BASE_URL")
        if env_base_url:
            base_url = env_base_url

        api_key = llm_cfg.get("api_key", "")
        env_api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if env_api_key:
            api_key = env_api_key

        self.llm_model = llm_cfg.get("model", "")
        env_model = os.getenv("LLM_MODEL")
        if env_model:
            self.llm_model = env_model

        self.strip_thinking_tags_enabled = llm_cfg.get("strip_thinking_tags", False)

        if not self.llm_model:
            raise ValueError(
                "LLM model name must be provided in config.llm.model or via environment variable LLM_MODEL"
            )

        client_kwargs: dict[str, str] = {"api_key": api_key or "EMPTY"}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.llm_client = OpenAI(**client_kwargs)

        embedding_cfg = config.get("embeddings", {})
        embedding_model_name = embedding_cfg.get(
            "model_name", "ai-forever/sbert_large_nlu_ru"
        )
        embedding_device = embedding_cfg.get("device", "cpu")
        self.embedder = SentenceTransformer(
            embedding_model_name, device=embedding_device
        )

        toxicity_cfg = config.get("toxicity", {})
        tox_model = toxicity_cfg.get("model_path")
        if tox_model:
            self.toxicity_classifier = ToxicityClassifier(
                tox_model,
                toxicity_cfg.get("device", "cpu"),
            )
            self.toxicity_threshold = float(toxicity_cfg.get("threshold", 1.0))
        else:
            self.toxicity_classifier = None
            self.toxicity_threshold = 1.0

        rag_cfg = config.get("rag", {})
        self.top_n = int(rag_cfg.get("top_n", 20))
        self.top_m = int(rag_cfg.get("top_m", 20))
        self.top_n_tokens = int(rag_cfg.get("top_n_tokens", 250))
        self.top_m_tokens = int(rag_cfg.get("top_m_tokens", 250))
        self.filter_threshold = float(rag_cfg.get("filter_threshold", 1.0))
        self.output_threshold = float(rag_cfg.get("output_threshold", 1.0))
        self.operator_threshold = float(rag_cfg.get("operator_threshold", 0.8))
        self.history_window = int(rag_cfg.get("history_window", 3))
        self.documents_history_window = int(rag_cfg.get("documents_history_window", 1))
        self.filter_prompt = rag_cfg.get("filter_prompt", "")
        self.evaluation_prompt = rag_cfg.get("evaluation_prompt", "")
        self.persona_prompt = rag_cfg.get("persona_prompt", "")
        self.operator_intent_prompt = rag_cfg.get("operator_intent_prompt", "")

        self.filter_classification_error_message = rag_cfg.get(
            "filter_classification_error_message", []
        )
        self.filter_threshold_message = rag_cfg.get("filter_threshold_message", [])
        self.toxicity_filter_message = rag_cfg.get("toxicity_filter_message", [])
        self.operator_intent_message = rag_cfg.get("operator_intent_message", [])
        self.evaluation_failure_message = rag_cfg.get("evaluation_failure_message", [])

        # Topics / related quick-questions (UI buttons)
        self.topics_count = int(rag_cfg.get("topics_count", 3))
        self.topics_max_len = int(rag_cfg.get("topics_max_len", 30))
        self.topics_system_prompt = rag_cfg.get(
            "topics_system_prompt",
            "Ты — генератор кратких тем и вопросов для кнопок. На входе — исходный вопрос и ответ ассистента. Верни {count} коротких тезисных тем (не больше {max_len} символов каждая), по одной на строку.",
        )
        self.topics_user_template = rag_cfg.get(
            "topics_user_template",
            "Вопрос: {question}\nОтвет: {answer}\nСформируй {count} коротких вариантов тем (не больше {max_len} символов) — по одной теме на строку. Только сами короткие заголовки.",
        )
        self.main_response_template = rag_cfg.get(
            "main_response_template",
            "Промпт персоны:\n{persona_prompt}\n\nИстория диалога:\n{history_text}\n\nДокументы (название и выдержка):\n{doc_payload}\n\nИспользуя документы и историю, ответь на вопрос пользователя:\n{preprocessed_query}",
        )
        self.ticket_summary_prompt = rag_cfg.get(
            "ticket_summary_prompt",
            "Проанализируй переписку в службе поддержки и создай краткое саммари в 1-2 предложения для оператора.\nСаммари должно отражать суть проблемы пользователя и текущий статус обращения.\n\nПереписка:\n{conversation_text}\n\nКраткое саммари:",
        )
        # Temporary storage for topic references when callback_data would be too long
        self._topic_refs: dict[str, str] = {}

        self.histories: dict[int, list[dict[str, str]]] = defaultdict(list)
        # Хранение временной истории чатов (до создания заявки)
        self.chat_histories: dict[int, list[ChatMessage]] = defaultdict(list)
        # Хранение истории документов для каждого пользователя
        self.documents_histories: dict[int, list[list[dict[str, Any]]]] = defaultdict(
            list
        )

        # Инициализация Speech-to-Text сервиса
        self.speech_to_text = SpeechToTextService(
            api_key=api_key, base_url=base_url, config=config
        )

        # Кеш для саммари тикетов (простой in-memory кеш)
        self._summary_cache: dict[int, str] = {}

        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: list[list[str]] = []
        self._faiss_index: faiss.IndexFlatIP | None = None
        self._faiss_matrix: np.ndarray | None = None
        self._documents: list[dict[str, str]] = []

    async def prepare(self) -> None:
        async with KnowledgeSessionLocal() as session:
            entries = await crud.load_all_chunks(session)
        self._load_documents(entries)

    async def reload(self) -> None:
        await self.prepare()

    def reset_history(self, conversation_id: int) -> None:
        self.histories.pop(conversation_id, None)

    def _normalize_user_key(self, user_key: Any) -> Any:
        """Normalize conversation/user keys so that 'vk_123' and 123 map to the same key (int 123) when possible."""
        try:
            if isinstance(user_key, str):
                if "_" in user_key:
                    suffix = user_key.split("_", 1)[1]
                    if suffix.isdigit():
                        return int(suffix)
                if user_key.isdigit():
                    return int(user_key)
        except Exception:
            pass
        return user_key

    def _load_documents(self, entries: Iterable[Any]) -> None:
        documents: list[dict[str, str]] = []
        vectors: list[np.ndarray] = []
        bm25_corpus: list[list[str]] = []
        for idx, entry in enumerate(entries):
            # DocumentChunk содержит только content, используем его как question и answer
            content = entry.content.strip()
            documents.append(
                {
                    "id": idx,
                    "question": content,  # Используем content как вопрос
                    "answer": content,  # И как ответ
                    "content": content,
                }
            )

            # Пропускаем записи без embedding'ов
            if entry.embedding is not None:
                vec = np.frombuffer(entry.embedding, dtype=np.float32)
                vectors.append(vec)

            bm25_corpus.append(_tokenize(content))
        if not documents:
            self._documents = []
            self._faiss_matrix = None
            self._faiss_index = None
            self._bm25 = None
            self._bm25_corpus = []
            logger.warning(
                "Knowledge base is empty; RAG answers will fallback to default message."
            )
            return

        self._documents = documents
        # Если нет векторов (все embedding'и NULL), создаем только BM25 индекс
        if not vectors:
            self._faiss_matrix = None
            self._faiss_index = None
            logger.warning("No embeddings found; using only BM25 for search.")
        else:
            matrix = np.stack(vectors).astype("float32")
            faiss.normalize_L2(matrix)
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            self._faiss_matrix = matrix
            self._faiss_index = index

        self._bm25_corpus = bm25_corpus
        self._bm25 = BM25Okapi(bm25_corpus)
        logger.info("Loaded %s knowledge documents for RAG", len(documents))

    def _call_llm(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        print(1, messages)
        print(2, response)
        content = response.choices[0].message.content.strip()
        if self.strip_thinking_tags_enabled:
            content = _strip_thinking_tags(content)
        print(3, content)
        return content

    def _check_toxicity(self, query: str) -> float:
        if not self.toxicity_classifier:
            return 0.0
        return self.toxicity_classifier.infer(query)

    def _retrieve_documents(self, query: str) -> list[dict[str, str]]:
        if not self._documents or self._faiss_index is None:
            return []
        query_vector = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0].astype("float32")
        distances, indices = self._faiss_index.search(
            query_vector.reshape(1, -1), min(self.top_n, len(self._documents))
        )
        selected = {
            int(idx): float(distances[0][pos])
            for pos, idx in enumerate(indices[0])
            if idx != -1
        }

        if self._bm25 is not None and self._bm25_corpus:
            bm25_scores = self._bm25.get_scores(_tokenize(query))
            for idx, score in enumerate(bm25_scores):
                if idx in selected:
                    selected[idx] = max(selected[idx], float(score))
                else:
                    selected[idx] = float(score)

        sorted_idx = sorted(selected.items(), key=lambda item: item[1], reverse=True)
        docs: list[dict[str, str]] = []
        total_tokens = 0
        for doc_idx, _ in sorted_idx:
            doc = self._documents[doc_idx]
            token_count = len(doc["content"].split())
            if total_tokens + token_count > self.top_n_tokens + self.top_m_tokens:
                break
            # Do NOT expose internal doc identifiers to the LLM or user-facing text.
            # Provide only a human-friendly title and content.
            docs.append(
                {
                    "title": doc["question"][:80] or "Документ",
                    "content": doc["content"],
                }
            )
            total_tokens += token_count
        return docs

    def _format_history(self, conversation_id: int) -> str:
        # Use chat_histories (where add_chat_message writes messages) so the LLM sees recent chat
        key = self._normalize_user_key(conversation_id)
        history_msgs = self.chat_histories.get(key, [])
        if not history_msgs:
            return "История пуста."
        # take last N messages (history_window pairs -> window*2 messages)
        relevant = history_msgs[-(self.history_window * 2) :]
        parts: list[str] = []
        for msg in relevant:
            role = "Пользователь" if msg.is_user else "Ассистент"
            parts.append(f"{role}: {msg.message}")
        return "\n".join(parts)

    def _apply_filter(self, user_query: str) -> tuple[str, dict[str, Any]]:
        details: dict[str, Any] = {}
        if self.filter_threshold >= 1.0 or not self.filter_prompt:
            return "", details
        messages = [
            {"role": "system", "content": self.filter_prompt},
            {"role": "user", "content": user_query},
        ]
        try:
            score_str = self._call_llm(messages, temperature=0.0, max_tokens=64)
            score = _parse_float(score_str)
            if score is None:
                raise ValueError(f"Could not parse filter score from: {score_str!r}")
            details["filter_probability"] = score
        except Exception as exc:
            logger.exception("Filter classification failed: %s", exc)
            message = (
                random.choice(self.filter_classification_error_message)
                if self.filter_classification_error_message
                else "Не удалось классифицировать запрос."
            )
            return message, details
        filter_score = details["filter_probability"]
        if filter_score > self.filter_threshold:
            message = (
                random.choice(self.filter_threshold_message)
                if self.filter_threshold_message
                else "Запрос не по теме."
            )
            return message, details
        return "", details

    def _check_operator_intent(self, user_query: str) -> tuple[bool, float]:
        if not self.operator_intent_prompt:
            return False, 0.0
        messages = [
            {
                "role": "user",
                "content": (
                    f"{self.operator_intent_prompt}\n\n"
                    f"Определи вероятность (0–1), что это запрос оператора.\n"
                    f"Ответь только числом.\n\n"
                    f"Текст: {user_query}"
                ),
            }
        ]
        try:
            score_str = self._call_llm(messages, temperature=0.0, max_tokens=64)
            score = _parse_float(score_str)
            if score is None:
                raise ValueError(f"Could not parse operator score from: {score_str!r}")
        except Exception as exc:
            logger.exception("Operator intent classification failed: %s", exc)
            score = 0.0
        return score >= self.operator_threshold, score

    def _evaluate_answer(self, answer: str, dialog_text: str) -> tuple[str, float]:
        if not self.evaluation_prompt:
            return answer, 0.0
        messages = [
            {
                "role": "user",
                "content": (
                    f"{self.evaluation_prompt}\n\n"
                    f"Вот ответ, который нужно оценить:\n{answer}\n\n"
                    f"Верни число от 0 до 1. Ответь только числом."
                ),
            }
        ]
        try:
            score_str = self._call_llm(messages, temperature=0.0, max_tokens=64)
            score = _parse_float(score_str)
            if score is None:
                raise ValueError(
                    f"Could not parse evaluation score from: {score_str!r}"
                )
        except Exception as exc:
            logger.exception("Evaluation failed: %s", exc)
            score = math.inf
        if score > self.output_threshold:
            failure_message = (
                random.choice(self.evaluation_failure_message)
                if self.evaluation_failure_message
                else "Ответ не прошёл проверку."
            )
            return failure_message, score
        return answer, score

    def _store_history(
        self, conversation_id: int, user_text: str, answer_text: str
    ) -> None:
        history = self.histories[conversation_id]
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": answer_text})
        if self.history_window > 0:
            max_messages = self.history_window * 2
            if len(history) > max_messages:
                self.histories[conversation_id] = history[-max_messages:]

    def add_chat_message(self, user_id: int, message: str, is_user: bool):
        """Добавляет сообщение в историю чата"""
        chat_message = ChatMessage(
            message=message, is_user=is_user, timestamp=datetime.now()
        )
        self.chat_histories[user_id].append(chat_message)
        sender_type = "USER" if is_user else "BOT"
        print(
            f"RAG DEBUG: Added message to user {user_id}: [{sender_type}] {message[:50]}..."
        )
        print(
            f"RAG DEBUG: Total messages for user {user_id}: {len(self.chat_histories[user_id])}"
        )

    def add_to_history(self, conversation_id: str, message: str, is_user: bool):
        """Добавляет сообщение в историю чата для VK/Telegram ботов"""
        # Извлекаем user_id из conversation_id (например, "vk_306608478" -> 306608478)
        if "_" in conversation_id:
            user_id_str = conversation_id.split("_", 1)[1]
            try:
                user_id = int(user_id_str)
                self.add_chat_message(user_id, message, is_user)
            except ValueError:
                print(
                    f"RAG DEBUG: Failed to parse user_id from conversation_id: {conversation_id}"
                )
        else:
            # Если conversation_id это просто число
            try:
                user_id = int(conversation_id)
                self.add_chat_message(user_id, message, is_user)
            except ValueError:
                print(f"RAG DEBUG: Invalid conversation_id: {conversation_id}")

    def get_chat_history(self, user_id: int) -> list[ChatMessage]:
        """Получает историю чата для пользователя"""
        return self.chat_histories[user_id]

    def get_chat_history_since_last_ticket(self, user_id: int) -> list[ChatMessage]:
        """Получает историю чата с момента последнего закрытия заявки или с начала"""
        history = self.chat_histories[user_id]
        print(
            f"RAG DEBUG: Getting chat history for user {user_id}, total messages: {len(history)}"
        )
        logging.info(
            f"Getting chat history for user {user_id}, total messages: {len(history)}"
        )

        # Показываем всю историю для отладки
        print(f"RAG DEBUG: Full history for user {user_id}:")
        for i, msg in enumerate(history):
            sender_type = "USER" if msg.is_user else "BOT"
            print(f"  {i+1}. [{sender_type}] {msg.message[:80]}...")

        # Ищем последнее системное сообщение о закрытии заявки
        last_closure_index = -1
        for i in range(len(history) - 1, -1, -1):
            msg = history[i]
            if not msg.is_user and any(
                phrase in msg.message.lower()
                for phrase in [
                    "завершил заявку",
                    "заявка завершена",
                    "если потребуется помощь",
                ]
            ):
                last_closure_index = i
                print(
                    f"RAG DEBUG: Found last ticket closure at index {i}: {msg.message[:50]}..."
                )
                logging.info(
                    f"Found last ticket closure at index {i}: {msg.message[:50]}..."
                )
                break

        # Если нашли закрытие заявки, берем сообщения после него
        if last_closure_index >= 0:
            relevant_history = history[last_closure_index + 1 :]
            print(
                f"RAG DEBUG: Using history after closure: {len(relevant_history)} messages"
            )
            logging.info(
                f"Using history after closure: {len(relevant_history)} messages"
            )
        else:
            # Если не было закрытий, берем всю историю
            relevant_history = history
            print(
                f"RAG DEBUG: No previous closures found, using full history: {len(relevant_history)} messages"
            )
            logging.info(
                f"No previous closures found, using full history: {len(relevant_history)} messages"
            )

        # Фильтруем служебные команды и системные сообщения
        filtered_history = []
        for msg in relevant_history:
            # Пропускаем служебные команды
            if msg.is_user and msg.message.startswith("/"):
                print(f"RAG DEBUG: Filtering out command: {msg.message}")
                logging.debug(f"Filtering out command: {msg.message}")
                continue
            # Пропускаем системные сообщения о создании заявки
            if not msg.is_user and any(
                phrase in msg.message.lower()
                for phrase in ["уведомили оператора", "ожидайте ответа", "мы уведомили"]
            ):
                print(f"RAG DEBUG: Filtering out system message: {msg.message[:30]}...")
                logging.debug(f"Filtering out system message: {msg.message[:30]}...")
                continue
            filtered_history.append(msg)

        print(
            f"RAG DEBUG: Final filtered history for user {user_id}: {len(filtered_history)} messages"
        )
        logging.info(
            f"Final filtered history for user {user_id}: {len(filtered_history)} messages"
        )

        # Показываем финальную отфильтрованную историю
        print(f"RAG DEBUG: Final filtered messages:")
        for i, msg in enumerate(filtered_history):
            sender_type = "USER" if msg.is_user else "BOT"
            print(f"  {i+1}. [{sender_type}] {msg.message[:80]}...")

        return filtered_history

    def mark_ticket_created(self, user_id: int):
        """Отмечает момент создания заявки (для будущего использования)"""
        # Можно добавить специальный маркер в историю если нужно
        pass

    def mark_ticket_closed(self, user_id: int):
        """Отмечает момент закрытия заявки"""
        # Добавляем специальное сообщение-маркер о закрытии заявки
        # Это поможет при следующем создании заявки найти точку сегментации
        closure_message = 'Оператор завершил заявку. Если потребуется помощь, напишите снова или нажмите кнопку "Позвать оператора".'
        self.add_chat_message(user_id, closure_message, is_user=False)

    def clear_chat_history(self, user_id: int):
        """Очищает историю чата после создания заявки"""
        if user_id in self.chat_histories:
            del self.chat_histories[user_id]

    def generate_reply(self, conversation_id: int, user_message: str) -> RAGResult:
        print(
            f"RAG DEBUG: generate_reply called for user {conversation_id}, message: {user_message[:50]}..."
        )
        filter_info: dict[str, Any] = {}
        query = user_message.strip()
        if not query:
            return RAGResult(
                "Пока не вижу вопроса. Напишите подробнее?", False, filter_info, 0.0
            )

        # Сохраняем сообщение пользователя в историю чата
        try:
            self.add_chat_message(conversation_id, query, is_user=True)
            print(f"RAG DEBUG: Successfully saved user message")
        except Exception as e:
            print(f"RAG DEBUG: Failed to save user message: {e}")
            logging.warning(f"Failed to save user message to chat history: {e}")

        toxicity_prob = self._check_toxicity(query)
        filter_info["toxicity_probability"] = toxicity_prob
        if toxicity_prob > self.toxicity_threshold:
            message = (
                random.choice(self.toxicity_filter_message)
                if self.toxicity_filter_message
                else "Сообщение слишком резкое. Переформулируйте, пожалуйста."
            )
            # Сохраняем ответ бота в историю чата
            try:
                self.add_chat_message(conversation_id, message, is_user=False)
                print(f"RAG DEBUG: Successfully saved bot toxicity message")
            except Exception as e:
                print(f"RAG DEBUG: Failed to save bot toxicity message: {e}")
                logging.warning(f"Failed to save bot message to chat history: {e}")
            return RAGResult(message, False, filter_info, 0.0)

        preprocessed_query = _preprocess(query)
        filter_error, filter_details = self._apply_filter(preprocessed_query)
        filter_info.update(filter_details)
        if filter_error:
            # Сохраняем ответ бота в историю чата
            try:
                self.add_chat_message(conversation_id, filter_error, is_user=False)
            except Exception as e:
                logging.warning(f"Failed to save bot message to chat history: {e}")
            return RAGResult(filter_error, False, filter_info, 0.0)

        operator_flag, operator_score = self._check_operator_intent(preprocessed_query)
        filter_info["operator_probability"] = operator_score
        if operator_flag:
            message = (
                random.choice(self.operator_intent_message)
                if self.operator_intent_message
                else "Могу подключить оператора."
            )
            self._store_history(conversation_id, preprocessed_query, message)
            # Сохраняем ответ бота в историю чата
            try:
                self.add_chat_message(conversation_id, message, is_user=False)
            except Exception as e:
                logging.warning(f"Failed to save bot message to chat history: {e}")
            return RAGResult(message, True, filter_info, 0.0)

        documents = self._retrieve_documents(preprocessed_query)

        # Сохраняем текущие документы в историю
        if documents:
            user_doc_history = self.documents_histories[conversation_id]
            user_doc_history.append(documents)
            # Ограничиваем размер истории документов
            if len(user_doc_history) > self.documents_history_window:
                self.documents_histories[conversation_id] = user_doc_history[
                    -self.documents_history_window :
                ]

        # Объединяем текущие документы с документами из истории
        all_documents = documents.copy()
        if (
            self.documents_history_window > 0
            and conversation_id in self.documents_histories
        ):
            # Добавляем документы из прошлых запросов (кроме текущего)
            past_docs = (
                self.documents_histories[conversation_id][:-1]
                if len(self.documents_histories[conversation_id]) > 1
                else []
            )
            for doc_set in past_docs:
                # Добавляем только уникальные документы
                for doc in doc_set:
                    if doc not in all_documents:
                        all_documents.append(doc)
                        # Ограничиваем общее количество документов
                        if len(all_documents) >= self.top_n + self.top_m:
                            break
                if len(all_documents) >= self.top_n + self.top_m:
                    break

        if not all_documents:
            message = "Не нашла инструкций по этому вопросу. Попробуйте уточнить или попросите оператора."
            self._store_history(conversation_id, preprocessed_query, message)
            print(f"RAG DEBUG: No documents found, returning with score=1.0")
            return RAGResult(
                message, False, filter_info, 1.0
            )  # Высокий score = низкая уверенность

        history_text = self._format_history(conversation_id)
        # Build a clean, human-readable documents payload (titles + short excerpts)
        safe_docs = []
        for d in all_documents:
            title = d.get("title") or "Документ"
            content = d.get("content") or ""
            # Truncate content to a reasonable length for the prompt
            excerpt = " ".join(content.split()[:120])
            safe_docs.append({"title": title, "excerpt": excerpt})
        doc_payload = json.dumps(safe_docs, ensure_ascii=False)

        combined_prompt = self.main_response_template.format(
            persona_prompt=self.persona_prompt,
            history_text=history_text,
            doc_payload=doc_payload,
            preprocessed_query=preprocessed_query,
        )
        messages = [
            {"role": "system", "content": self.persona_prompt},
            {"role": "user", "content": combined_prompt},
        ]
        final_answer_raw = self._call_llm(messages, temperature=0.1, max_tokens=512)
        # Post-process final answer: strip any internal doc_x tokens or debug traces
        cleaned_raw = re.sub(r"doc_\d+", "", final_answer_raw)
        cleaned_raw = re.sub(r"\[doc_\d+\]", "", cleaned_raw)
        final_answer, eval_score = self._evaluate_answer(cleaned_raw, history_text)
        print(f"RAG DEBUG: Generated answer: {final_answer[:100]}...")
        print(f"RAG DEBUG: Evaluation score: {eval_score}")
        # Greeting suppression is handled via persona prompt; do not post-process greetings here.
        filter_info["evaluation_probability"] = eval_score
        self._store_history(conversation_id, preprocessed_query, final_answer)

        # Сохраняем ответ бота в историю чата
        try:
            self.add_chat_message(conversation_id, final_answer, is_user=False)
            print(f"RAG DEBUG: Successfully saved final bot answer")
        except Exception as e:
            print(f"RAG DEBUG: Failed to save final bot answer: {e}")
            logging.warning(f"Failed to save bot message to chat history: {e}")

        return RAGResult(final_answer, False, filter_info, eval_score)

    def suggest_topics(
        self, conversation_id: int, user_query: str, answer_text: str
    ) -> list[str]:
        """Return a list of short topic strings (labels) to show as quick buttons.
        This calls the LLM synchronously (wrap in thread when calling async code).
        """
        try:
            user_prompt = self.topics_user_template.format(
                question=user_query,
                answer=answer_text or "",
                count=self.topics_count,
                max_len=self.topics_max_len,
            )
            messages = [
                {"role": "system", "content": self.topics_system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            raw = self._call_llm(messages, temperature=0.2, max_tokens=128)
            # Parse lines; tolerate several formats
            lines = [l.strip() for l in re.split(r"\r?\n|\t|-\s*", raw) if l.strip()]
            topics: list[str] = []
            for ln in lines:
                # remove leading numbering like '1.' or '1)'
                ln = re.sub(r"^[0-9]+[\.)\-]*\s*", "", ln).strip()
                if not ln:
                    continue
                # Truncate to max length
                if len(ln) > self.topics_max_len:
                    ln = ln[: self.topics_max_len].rsplit(" ", 1)[0]
                topics.append(ln)
                if len(topics) >= self.topics_count:
                    break
            # Fallback: if nothing parsed, produce simple splits of answer
            if not topics:
                fallback = re.findall(
                    r"[A-Za-zА-Яа-я0-9\s]{3,}", answer_text or user_query
                )[: self.topics_count]
                topics = [t.strip()[: self.topics_max_len] for t in fallback]
            return topics
        except Exception:
            logger.exception("Failed to suggest topics")
            return []

    async def generate_ticket_summary(
        self, messages: list, ticket_id: int = None
    ) -> str:
        """
        Генерирует краткое саммари тикета в 1-2 предложения для операторов

        Args:
            messages: Список сообщений тикета (объекты Message из БД)
            ticket_id: ID тикета для кеширования (опционально)

        Returns:
            Краткое описание проблемы
        """
        # Проверяем кеш если есть ticket_id
        if ticket_id and ticket_id in self._summary_cache:
            return self._summary_cache[ticket_id]

        if not messages:
            summary = "Нет сообщений в тикете"
            if ticket_id:
                self._summary_cache[ticket_id] = summary
            return summary

        # Собираем историю переписки
        conversation = []
        for msg in messages:
            role = (
                "Пользователь"
                if msg.sender == "user"
                else ("Бот" if msg.sender == "bot" else "Оператор")
            )
            conversation.append(f"{role}: {msg.text}")

        conversation_text = "\n".join(conversation)

        summary_prompt = self.ticket_summary_prompt.format(
            conversation_text=conversation_text
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=150,
                temperature=0.3,
            )

            summary = response.choices[0].message.content.strip()
            if self.strip_thinking_tags_enabled:
                summary = _strip_thinking_tags(summary)

            result = summary or "Не удалось создать саммари"

            # Сохраняем в кеш если есть ticket_id
            if ticket_id:
                self._summary_cache[ticket_id] = result

            return result

        except Exception as e:
            logger.error(f"Error generating ticket summary: {e}")
            error_summary = "Ошибка при создании саммари"
            if ticket_id:
                self._summary_cache[ticket_id] = error_summary
            return error_summary

    async def generate_ticket_summary_from_chat_history(self, user_id: int) -> str:
        """
        Генерирует краткое саммари на основе истории чата с момента последнего тикета

        Используется при переключении на оператора для создания контекста

        Args:
            user_id: ID пользователя

        Returns:
            Краткое описание проблемы на основе истории чата
        """
        # Получаем историю чата с момента последнего закрытия тикета
        chat_history = self.get_chat_history_since_last_ticket(user_id)

        if not chat_history:
            return "Нет истории диалога"

        # Собираем переписку из истории чата
        conversation = []
        for msg in chat_history:
            role = "Пользователь" if msg.is_user else "Бот"
            conversation.append(f"{role}: {msg.message}")

        conversation_text = "\n".join(conversation)

        summary_prompt = self.ticket_summary_prompt.format(
            conversation_text=conversation_text
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=150,
                temperature=0.3,
            )

            summary = response.choices[0].message.content.strip()
            if self.strip_thinking_tags_enabled:
                summary = _strip_thinking_tags(summary)

            return summary or "Не удалось создать саммари"

        except Exception as e:
            logger.error(f"Error generating summary from chat history: {e}")
            return "Ошибка при создании саммари"


__all__ = ["RAGService", "RAGResult"]
