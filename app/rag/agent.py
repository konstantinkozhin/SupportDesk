"""
RAG Agent - интеллектуальный помощник с инструментами
"""

import logging
import os
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class RAGAgent:
    """Простой RAG агент с инструментами"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.categories = config.get("agent", {}).get("categories", [])

        # Инициализируем только если agno доступно
        self.agent = None
        try:
            self._init_agent()
        except ImportError:
            logger.warning("agno not available, using fallback mode")

    def _init_agent(self):
        """Инициализация agno агента"""
        from agno.agent import Agent
        from agno.models.llama_cpp import LlamaCpp
        from app.rag.agent_tools import (
            search_knowledge_base,
            classify_request,
            set_priority,
            create_it_ticket,
            call_operator,
            get_system_status,
            save_case_to_knowledge_base,
            get_support_report,
        )

        # Настройка модели
        llm = LlamaCpp(
            id="gemma-3-27b-it",
            base_url="https://demo.ai.sfu-kras.ru/v1",
        )

        # Получаем системные инструкции из конфига
        system_instructions = self.config.get("agent", {}).get(
            "system_instructions",
            "Ты - помощник технической поддержки. Отвечай полезно и дружелюбно.",
        )

        self.agent = Agent(
            name="RAG Support Agent",
            description="Агент технической поддержки с доступом к базе знаний",
            model=llm,
            tools=[
                search_knowledge_base,
                classify_request,
                set_priority,
                create_it_ticket,
                call_operator,
                get_system_status,
                save_case_to_knowledge_base,
                get_support_report,
            ],
            instructions=system_instructions,
            debug_mode=False,
            store_history_messages=False,
            store_tool_messages=False,
            store_media=False,
        )

    async def process_query(
        self,
        query: str,
        chat_history: List[Dict[str, Any]] = None,
        conversation_id: int = None,
    ) -> str:
        """Обработка запроса через LLM агента с учетом истории

        Args:
            query: Текущий запрос пользователя
            chat_history: История диалога (список ChatMessage объектов с полями message, is_user, timestamp)
            conversation_id: ID текущего диалога для инструментов агента
        """
        print(f"\n[AGENT START] Обрабатываю запрос: '{query}'")

        # Устанавливаем conversation_id в контексте перед выполнением
        if conversation_id is not None:
            from app.rag.agent_tools import set_current_conversation_id

            set_current_conversation_id(conversation_id)
            print(f"[AGENT] Установлен conversation_id: {conversation_id}")
        else:
            print(f"[AGENT WARNING] conversation_id не передан в process_query!")

        if chat_history:
            print(f"[AGENT] История диалога: {len(chat_history)} сообщений")

        if not self.agent:
            print("⚠️ [AGENT] Агент недоступен, использую fallback")
            return self._fallback_process(query)

        try:
            print("[AGENT] Отправляю запрос в LLM...")

            # Формируем контекст с историей если она есть
            if chat_history and len(chat_history) > 0:
                # Формируем текстовый контекст из истории
                context_parts = []
                for msg in chat_history:
                    role = "Пользователь" if msg.is_user else "Ассистент"
                    context_parts.append(f"{role}: {msg.message}")

                # Объединяем историю и добавляем текущий запрос
                full_context = "\n".join(context_parts) + f"\nПользователь: {query}"

                print(
                    f"[AGENT] Передаю контекст с {len(chat_history)} предыдущими сообщениями"
                )

                # Отправляем весь контекст агенту
                result = await self.agent.arun(full_context)
            else:
                # Нет истории - просто отправляем запрос
                result = await self.agent.arun(query)

            # Обрабатываем отложенные действия агента
            print("[AGENT] Обрабатываю отложенные действия...")
            from app.rag.agent_tools import process_pending_actions

            await process_pending_actions()

            # Извлекаем текст ответа
            if hasattr(result, "content"):
                response = result.content.strip()
            else:
                response = str(result).strip()

            print(f"✅ [AGENT COMPLETE] Ответ получен, длина: {len(response)} символов")
            return response

        except Exception as e:
            print(f"💥 [AGENT ERROR] Ошибка агента: {e}")
            logger.error(f"Agent error: {e}")
            return self._fallback_process(query)

    def _fallback_process(self, query: str) -> str:
        """Простая обработка без агента - возвращаем стандартный ответ"""
        print(f"🔄 [FALLBACK] Обрабатываю запрос без агента: '{query}'")
        logger.info(f"Fallback processing query: {query}")
        return "Сейчас сервис недоступен. Попробуйте позже или обратитесь к оператору."
