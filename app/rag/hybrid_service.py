"""
Интегрированный RAG сервис с агентом и классической функциональностью
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np

from app.db import get_knowledge_session
from app.db import tickets_crud as knowledge_crud
from app.config import load_config

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Результат RAG обработки"""

    final_answer: str
    operator_requested: bool = False
    filter_info: dict[str, Any] | None = None
    confidence_score: float = 0.3  # Низкий score = высокая уверенность
    similar_suggestions: List[Dict[str, Any]] | None = (
        None  # Список похожих решений для кнопок
    )


class HybridRAGService:
    """Гибридный RAG сервис с агентом и классической функциональностью"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.agent = None
        self.use_agent = config.get("rag", {}).get("use_agent", True)

        print(f"HybridRAGService INIT: use_agent from config = {self.use_agent}")

        # Создаем базовый RAG сервис для совместимости
        try:
            from app.rag.service import RAGService

            self.rag_service = RAGService(config)
            logger.info("Base RAG service initialized")
            print("HybridRAGService: Base RAG service initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize base RAG service: {e}")
            print(f"HybridRAGService: Failed to init base RAG service: {e}")
            self.rag_service = None

        self._initialize_model()
        if self.use_agent:
            print("HybridRAGService: Initializing agent...")
            self._initialize_agent()
        else:
            print("HybridRAGService: Agent disabled by config")

    def _initialize_model(self):
        """Инициализация модели для создания embeddings"""
        try:
            model_name = self.config.get("rag", {}).get(
                "model", "sentence-transformers/all-MiniLM-L6-v2"
            )
            self.model = SentenceTransformer(model_name)
            logger.info(f"RAG model loaded: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load RAG model: {e}")
            self.model = None

    def _initialize_agent(self):
        """Инициализация RAG агента"""
        try:
            from app.rag.agent import RAGAgent

            self.agent = RAGAgent(self.config)
            logger.info("RAG Agent initialized")
            print(
                f"HybridRAGService: RAG Agent initialized successfully, agent={self.agent is not None}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize RAG Agent: {e}")
            print(f"HybridRAGService: Failed to initialize agent: {e}")
            self.agent = None
            self.use_agent = False

    def create_embedding(self, text: str) -> Optional[List[float]]:
        """Создание эмбеддинга для текста"""
        if not self.model:
            return None

        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            return None

    async def process_query(self, query: str) -> str:
        """Основной метод обработки запроса"""
        # Проверка на токсичность (если доступна)
        if self._is_toxic(query):
            return "Пожалуйста, используйте корректные формулировки."

        if self.use_agent and self.agent:
            # Используем агента для обработки
            try:
                return await self.agent.process_query(query)
            except Exception as e:
                logger.error(f"Agent processing failed: {e}")
                return await self._fallback_search(query)
        else:
            # Используем классический поиск
            return await self._fallback_search(query)

    async def generate_reply(self, conversation_id: int, user_text: str) -> RAGResult:
        """Совместимость со старым интерфейсом с сохранением истории"""
        print(
            f"HybridRAG DEBUG: generate_reply called, use_agent={self.use_agent}, agent={self.agent is not None}"
        )

        # Сначала сохраняем сообщение пользователя в историю (если есть базовый сервис)
        if self.rag_service and hasattr(self.rag_service, "add_chat_message"):
            try:
                self.rag_service.add_chat_message(
                    conversation_id, user_text, is_user=True
                )
                print(f"HybridRAG DEBUG: Saved user message to history")
            except Exception as e:
                logger.warning(f"Failed to save user message: {e}")

        # Если используем агента - пусть он обрабатывает
        if self.use_agent and self.agent:
            print(f"HybridRAG DEBUG: Using agent to process query")
            try:
                # Устанавливаем conversation_id для текущего потока
                from app.rag.agent_tools import (
                    set_current_conversation_id,
                    get_similar_suggestions,
                )

                set_current_conversation_id(conversation_id)
                print(f"HybridRAG DEBUG: Set conversation_id = {conversation_id}")

                # Получаем историю диалога для контекста
                chat_history = []
                if self.rag_service and hasattr(self.rag_service, "get_chat_history"):
                    try:
                        history_msgs = self.rag_service.get_chat_history(
                            conversation_id
                        )
                        # Берем последние N сообщений согласно history_window
                        history_window = self.config.get("rag", {}).get(
                            "history_window", 3
                        )
                        chat_history = (
                            history_msgs[-history_window:] if history_msgs else []
                        )
                        print(
                            f"HybridRAG DEBUG: Passing {len(chat_history)} history messages to agent"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to get chat history: {e}")

                # Передаем запрос с историей и conversation_id агенту
                response_text = await self.agent.process_query(
                    user_text,
                    chat_history=chat_history,
                    conversation_id=conversation_id,
                )

                # Получаем похожие предложения из хранилища (если есть)
                similar_suggestions = get_similar_suggestions(conversation_id)
                if similar_suggestions:
                    print(
                        f"HybridRAG DEBUG: Got {len(similar_suggestions)} similar suggestions for buttons"
                    )

                # Сохраняем ответ бота в историю
                if self.rag_service and hasattr(self.rag_service, "add_chat_message"):
                    try:
                        self.rag_service.add_chat_message(
                            conversation_id, response_text, is_user=False
                        )
                        print(f"HybridRAG DEBUG: Saved bot response to history")
                    except Exception as e:
                        logger.warning(f"Failed to save bot message: {e}")

                return RAGResult(
                    final_answer=response_text,
                    operator_requested="оператор" in response_text.lower(),
                    confidence_score=0.3,  # Низкий score = высокая уверенность
                    similar_suggestions=similar_suggestions,  # Добавляем предложения
                )
            except Exception as e:
                logger.error(f"Agent processing failed: {e}")
                # Fallback на базовый сервис

        # Если агент недоступен или упал - используем базовый RAG сервис
        if self.rag_service and hasattr(self.rag_service, "generate_reply"):
            print(f"HybridRAG DEBUG: Using base RAG service")
            # Вызываем синхронный метод базового сервиса через to_thread
            result = await asyncio.to_thread(
                self.rag_service.generate_reply, conversation_id, user_text
            )
            return result
        else:
            # Последний fallback - простой поиск без истории
            print(f"HybridRAG DEBUG: Using fallback search")
            response_text = await self._fallback_search(user_text)

            # Сохраняем ответ в историю
            if self.rag_service and hasattr(self.rag_service, "add_chat_message"):
                try:
                    self.rag_service.add_chat_message(
                        conversation_id, response_text, is_user=False
                    )
                except Exception as e:
                    logger.warning(f"Failed to save bot message: {e}")

            return RAGResult(
                final_answer=response_text,
                operator_requested="оператор" in response_text.lower(),
                confidence_score=0.3,  # Высокая уверенность
            )

    async def suggest_topics(
        self, conversation_id: int, user_text: str, answer_text: str
    ) -> List[str]:
        """Предложение связанных тем"""
        try:
            async with get_knowledge_session()() as session:
                chunks = await knowledge_crud.load_all_chunks(session)

                if not chunks:
                    return []

                # Простой поиск связанных тем
                words = user_text.lower().split()
                topics = []

                for chunk in chunks[:10]:  # Ограничиваем поиск
                    if any(word in chunk.content.lower() for word in words):
                        # Извлекаем первое предложение как тему
                        sentences = chunk.content.split(".")
                        if sentences:
                            topic = sentences[0].strip()[:100]
                            if topic and topic not in topics:
                                topics.append(topic)

                return topics[:5]  # Максимум 5 тем

        except Exception as e:
            logger.error(f"Error suggesting topics: {e}")
            return []

    def mark_ticket_closed(self, chat_id: int):
        """Заглушка для совместимости"""
        logger.info(f"Ticket closed for chat {chat_id}")

    async def generate_ticket_summary(
        self, messages: List[Any], ticket_id: int = None
    ) -> str:
        """Генерация краткого описания тикета через прямой вызов LLM"""
        if not messages:
            return "Нет сообщений"

        try:
            # Формируем текст из сообщений
            message_texts = []
            for msg in messages:
                sender = getattr(msg, "sender", "unknown")
                content = (
                    getattr(msg, "text", "") or getattr(msg, "content", "") or str(msg)
                )
                message_texts.append(f"{sender}: {content}")

            conversation = "\n".join(message_texts)

            # Создаем промпт для генерации summary
            prompt = f"""Создай краткое резюме диалога между пользователем и поддержкой.
Основная проблема или запрос, ключевые моменты обсуждения.

Диалог:
{conversation}

Краткое резюме (максимум 2-3 предложения):"""

            # Используем LLM клиент напрямую
            if (
                hasattr(self, "rag_service")
                and self.rag_service
                and hasattr(self.rag_service, "llm_client")
            ):
                response = self.rag_service.llm_client.chat.completions.create(
                    model=self.rag_service.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=150,
                )
                summary = response.choices[0].message.content.strip()
                return summary or f"Тикет содержит {len(messages)} сообщений"
            else:
                # Fallback если нет LLM клиента
                return f"Тикет содержит {len(messages)} сообщений"

        except Exception as e:
            logger.error(f"Ошибка генерации summary: {e}")
            # Fallback к простому описанию
            summary = f"Тикет содержит {len(messages)} сообщений."
            if messages:
                first_msg = (
                    getattr(messages[0], "text", "")
                    or getattr(messages[0], "content", "")
                    or str(messages[0])
                )
                summary += f" Первое сообщение: {first_msg[:100]}..."
            return summary

    def get_chat_history(self, user_id: int):
        """Получает историю чата для пользователя"""
        # Делегируем к базовому RAG сервису
        if hasattr(self, "rag_service") and self.rag_service:
            return self.rag_service.get_chat_history(user_id)
        return []

    def get_chat_history_since_last_ticket(self, user_id: int):
        """Получает историю чата с момента последней заявки"""
        # Делегируем к базовому RAG сервису
        if hasattr(self, "rag_service") and self.rag_service:
            return self.rag_service.get_chat_history_since_last_ticket(user_id)
        return []

    def mark_ticket_created(self, user_id: int):
        """Отмечает создание заявки для пользователя"""
        # Делегируем к базовому RAG сервису
        if hasattr(self, "rag_service") and self.rag_service:
            return self.rag_service.mark_ticket_created(user_id)

    async def generate_ticket_summary_from_chat_history(self, user_id: int) -> str:
        """Генерация саммари на основе истории чата"""
        # Делегируем к базовому RAG сервису
        if hasattr(self, "rag_service") and self.rag_service:
            return await self.rag_service.generate_ticket_summary_from_chat_history(
                user_id
            )

        # Fallback если нет базового сервиса
        return "Запрос помощи от пользователя"

    def reset_history(self, conversation_id: int) -> None:
        """Сброс истории беседы"""
        if hasattr(self, "rag_service") and self.rag_service:
            self.rag_service.reset_history(conversation_id)
        # Если нет базового сервиса - ничего не делаем (нет истории для сброса)

    async def _fallback_search(self, query: str) -> str:
        """Простой поиск по базе знаний без агента"""
        try:
            async with get_knowledge_session()() as session:
                chunks = await knowledge_crud.load_all_chunks(session)

                if not chunks:
                    return "База знаний пуста. Обратитесь к оператору."

                # Простой поиск по ключевым словам
                query_lower = query.lower()
                best_match = None
                best_score = 0

                for chunk in chunks:
                    score = 0
                    for word in query_lower.split():
                        score += chunk.content.lower().count(word)

                    if score > best_score:
                        best_score = score
                        best_match = chunk

                if best_match and best_score > 0:
                    content = (
                        best_match.content[:300] + "..."
                        if len(best_match.content) > 300
                        else best_match.content
                    )
                    return f"Найдено в базе знаний:\n\nИсточник: {best_match.source_file}\n\n{content}\n\nЕсли нужна дополнительная помощь, обратитесь к оператору."
                else:
                    return "Информация не найдена в базе знаний. Обратитесь к оператору для получения помощи."

        except Exception as e:
            logger.error(f"Fallback search error: {e}")
            return "Произошла ошибка при поиске. Обратитесь к оператору."

    def _is_toxic(self, text: str) -> bool:
        """Простая проверка на токсичность"""
        # Можно добавить более сложную проверку
        toxic_words = ["дурак", "идиот", "тупой"]  # Базовый список
        text_lower = text.lower()
        return any(word in text_lower for word in toxic_words)

    async def prepare(self):
        """Подготовка сервиса"""
        logger.info("Hybrid RAG Service prepared")

    async def reload(self):
        """Перезагрузка сервиса"""
        logger.info("Hybrid RAG Service reloaded")


# Создаем глобальный экземпляр сервиса
_hybrid_rag_service = None


def get_hybrid_rag_service() -> HybridRAGService:
    """Получение экземпляра гибридного RAG сервиса"""
    global _hybrid_rag_service
    if _hybrid_rag_service is None:
        config = load_config()
        _hybrid_rag_service = HybridRAGService(config)
    return _hybrid_rag_service
