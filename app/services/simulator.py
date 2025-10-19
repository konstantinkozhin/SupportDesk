"""
Сервис для симулятора обучения операторов
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Optional

from app.config import load_simulator_prompts
from app.rag import RAGService
from app.db import tickets_crud as crud, database

logger = logging.getLogger(__name__)


@dataclass
class SimulatorQuestion:
    """Вопрос от симулированного пользователя"""

    question: str
    context: str  # Контекст из базы знаний
    difficulty: str  # easy, medium, hard


@dataclass
class SimulatorResponse:
    """Результат оценки ответа оператора"""

    score: int  # 0-100
    feedback: str  # Обратная связь
    ai_suggestion: str  # Эталонный ответ от AI
    is_correct: bool  # Достаточно ли хорош ответ


class SimulatorSession:
    """Сессия обучения"""

    def __init__(self, character: str, questions_count: int = 5):
        self.character = character  # easy, medium, hard
        self.questions_count = questions_count
        self.current_question = 0
        self.total_score = 0
        self.history: list[dict] = []
        self.current_question_data: Optional[SimulatorQuestion] = None

    def add_response(self, user_answer: str, evaluation: SimulatorResponse):
        """Добавить ответ оператора и оценку"""
        self.history.append(
            {
                "question": (
                    self.current_question_data.question
                    if self.current_question_data
                    else ""
                ),
                "user_answer": user_answer,
                "score": evaluation.score,
                "feedback": evaluation.feedback,
                "ai_suggestion": evaluation.ai_suggestion,
            }
        )
        self.total_score += evaluation.score
        self.current_question += 1

    def is_complete(self) -> bool:
        """Завершена ли сессия"""
        return self.current_question >= self.questions_count

    def get_average_score(self) -> float:
        """Средний балл"""
        if not self.history:
            return 0.0
        return self.total_score / len(self.history)


class SimulatorService:
    """Сервис симулятора"""

    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.simulator_prompts = load_simulator_prompts()
        self.characters = self.simulator_prompts.get("characters", {})
        self.sessions: dict[str, SimulatorSession] = {}  # user_id -> session

    def start_session(self, user_id: str, character: str) -> SimulatorSession:
        """Начать новую сессию"""
        if character not in self.characters:
            character = "medium"

        session = SimulatorSession(character=character, questions_count=5)
        self.sessions[user_id] = session
        return session

    def get_session(self, user_id: str) -> Optional[SimulatorSession]:
        """Получить текущую сессию"""
        return self.sessions.get(user_id)

    def end_session(self, user_id: str):
        """Завершить сессию"""
        if user_id in self.sessions:
            del self.sessions[user_id]

    async def generate_question(self, session: SimulatorSession) -> SimulatorQuestion:
        """
        Генерирует вопрос из чанков базы знаний

        Новый алгоритм:
        1. Получаем случайный чанк из БД
        2. Извлекаем из него проблему/ситуацию
        3. Используем LLM для создания вопроса в стиле персонажа
        """
        # Fallback вопросы на случай ошибок
        fallback_questions = {
            "easy": [
                "Привет! Я забыл пароль от компьютера, что делать?",
                "Помогите, у меня не работает интернет!",
                "Как мне установить программу 1С?",
            ],
            "medium": [
                "Не могу подключиться к корпоративному VPN. Ошибка подключения.",
                "У меня не открывается файл Excel, пишет ошибку совместимости.",
                "Нужно настроить почту на новом компьютере. Какие параметры?",
            ],
            "hard": [
                "Мне СРОЧНО нужен доступ к системе 1С. Когда будет готово?",
                "Почему мой отдел до сих пор не получил обновление ПО? Это недопустимо!",
                "Требую немедленно восстановить доступ к базе данных. У нас простой!",
            ],
        }

        # Получаем чанк из системы DocumentChunk
        try:
            async for kb_session in database.get_knowledge_session():
                # Получаем случайный чанк
                chunk = await crud.get_random_chunk(kb_session)

                if chunk and chunk.content:
                    # Генерируем вопрос из чанка
                    character_info = self.characters[session.character]

                    # Получаем промпт из конфига и форматируем
                    extraction_prompt_template = self.simulator_prompts.get(
                        "question_generation", {}
                    ).get("extraction_prompt", "")

                    extraction_prompt = extraction_prompt_template.format(
                        character_name=character_info["name"],
                        character_description=character_info["description"],
                        chunk_content=chunk.content,
                    )

                    # Генерируем вопрос через LLM
                    generated_question = self._generate_with_llm(extraction_prompt)

                    # Проверяем что получили нормальный вопрос
                    if (
                        generated_question
                        and len(generated_question) >= 10
                        and "ошибка" not in generated_question.lower()
                    ):
                        question = SimulatorQuestion(
                            question=generated_question,
                            context=chunk.content,  # Используем сам чанк как контекст
                            difficulty=session.character,
                        )

                        session.current_question_data = question
                        return question

                # Если чанков нет
                raise ValueError("No chunks found in database")

        except Exception as e:
            logger.error(f"Failed to generate question from chunks: {e}")
            # Fallback на предопределенные вопросы
            questions_list = fallback_questions.get(
                session.character, fallback_questions["medium"]
            )
            question_text = random.choice(questions_list)

            question = SimulatorQuestion(
                question=question_text,
                context="Контекст недоступен - используйте свои знания для ответа",
                difficulty=session.character,
            )
            session.current_question_data = question
            return question

    def evaluate_response(
        self, session: SimulatorSession, user_answer: str
    ) -> SimulatorResponse:
        """
        Оценивает ответ оператора через LLM

        Критерии оценки:
        - Правильность (есть ли решение)
        - Полнота (достаточно ли деталей)
        - Тон (вежливость, профессионализм)
        - Структурированность
        """
        if not session.current_question_data:
            return SimulatorResponse(
                score=0,
                feedback="Ошибка: нет текущего вопроса",
                ai_suggestion="",
                is_correct=False,
            )

        question = session.current_question_data

        # Предварительная проверка ответа
        word_count = len(user_answer.split())
        char_count = len(user_answer.strip())

        # Очень короткие ответы
        if char_count < 10:
            return SimulatorResponse(
                score=0,
                feedback="Ответ слишком короткий и не содержит полезной информации. Необходимо предоставить развернутое решение проблемы.",
                ai_suggestion=self._generate_ai_answer(
                    question.question, question.context
                ),
                is_correct=False,
            )

        # Односложные ответы
        if word_count <= 3:
            return SimulatorResponse(
                score=5,
                feedback="Односложный ответ не решает проблему пользователя. Нужно дать подробное объяснение и конкретные шаги решения.",
                ai_suggestion=self._generate_ai_answer(
                    question.question, question.context
                ),
                is_correct=False,
            )

        # Получаем эталонный ответ от AI
        ai_answer = self._generate_ai_answer(question.question, question.context)

        # Оцениваем ответ оператора
        evaluation_prompt = (
            self.simulator_prompts.get("evaluation", {})
            .get("evaluation_prompt", "")
            .format(
                question=question.question,
                context=question.context,
                user_answer=user_answer,
                ai_answer=ai_answer,
            )
        )

        evaluation_text = self._generate_with_llm(evaluation_prompt)

        # Парсим оценку и отзыв
        score, feedback = self._parse_evaluation(evaluation_text)

        # Базовые проверки качества ответа
        score = self._validate_basic_quality(score, user_answer)

        return SimulatorResponse(
            score=score,
            feedback=feedback,
            ai_suggestion=ai_answer,
            is_correct=score >= 60,
        )

    def _parse_evaluation(self, evaluation_text: str) -> tuple[int, str]:
        """Парсит оценку и отзыв из ответа LLM"""
        import re

        score = 50
        feedback = "Не удалось оценить ответ"

        try:
            # Ищем SCORE: число
            score_match = re.search(r"SCORE[:\s]*(\d+)", evaluation_text, re.IGNORECASE)
            if score_match:
                score = max(0, min(100, int(score_match.group(1))))
            else:
                # Ищем первое число от 0 до 100
                numbers = re.findall(r"\b(\d+)\b", evaluation_text)
                for num_str in numbers:
                    num = int(num_str)
                    if 0 <= num <= 100:
                        score = num
                        break

            # Ищем FEEDBACK: текст
            feedback_match = re.search(
                r"FEEDBACK[:\s]+(.*?)(?:\n\n|$)",
                evaluation_text,
                re.IGNORECASE | re.DOTALL,
            )
            if feedback_match:
                feedback = feedback_match.group(1).strip()
            else:
                # Берём текст после первого числа
                lines = evaluation_text.strip().split("\n")
                feedback_lines = [line.strip() for line in lines[1:] if line.strip()]
                if feedback_lines:
                    feedback = " ".join(feedback_lines)
        except Exception as e:
            logger.error(f"Failed to parse evaluation: {e}")

        return score, feedback

    def _validate_basic_quality(self, initial_score: int, user_answer: str) -> int:
        """Базовая валидация качества ответа"""
        if len(user_answer.strip()) < 10:
            return min(initial_score, 20)

        # Попытка обмана системы оценки
        scam_keywords = ["баллов", "балл", "100", "оцени", "поставь"]
        if (
            any(kw in user_answer.lower() for kw in scam_keywords)
            and len(user_answer.split()) < 15
        ):
            return 0

        return initial_score

    def get_hint(self, session: SimulatorSession) -> str:
        """Получить подсказку для текущего вопроса"""
        if not session.current_question_data:
            return "Нет активного вопроса"

        question = session.current_question_data

        hint_prompt = (
            self.simulator_prompts.get("hints", {})
            .get("hint_prompt", "")
            .format(question=question.question, context=question.context)
        )

        return self._generate_with_llm(hint_prompt)

    def _generate_with_llm(self, prompt: str) -> str:
        """Генерация текста через LLM"""
        try:
            from app.rag.service import get_llm_client
            import os

            # Получаем LLM клиент через функцию
            llm_client = get_llm_client()

            # Получаем модель из переменных окружения или конфига
            llm_model = os.getenv("LLM_MODEL") or self.rag_service.llm_model

            messages = [{"role": "user", "content": prompt}]
            response = llm_client.chat.completions.create(
                model=llm_model,
                messages=messages,
                temperature=0.7,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return f"Ошибка генерации: {str(e)}"

    def _generate_ai_answer(self, question: str, context: str) -> str:
        """Генерирует эталонный ответ от AI"""
        prompt = (
            self.simulator_prompts.get("ai_answers", {})
            .get("ai_answer_prompt", "")
            .format(question=question, context=context)
        )

        return self._generate_with_llm(prompt)
