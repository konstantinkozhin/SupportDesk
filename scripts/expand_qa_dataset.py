"""
Скрипт для расширения QA.xlsx дополнительными вопрос-ответами из Обращения.txt
Использует LLM для генерации ответов на основе обращений
"""

import pandas as pd
import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Загружаем переменные окружения
load_dotenv()


def read_appeals(file_path: str) -> list:
    """Читает файл с обращениями и возвращает список обращений"""
    logger.info(f"Читаем обращения из {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        appeals = [line.strip() for line in f if line.strip()]

    logger.info(f"Найдено {len(appeals)} обращений")
    return appeals


def generate_answer_for_appeal(client: OpenAI, appeal: str) -> str:
    """
    Генерирует ответ на обращение используя LLM

    Args:
        client: OpenAI клиент
        appeal: Текст обращения/проблемы

    Returns:
        Сгенерированный ответ
    """
    system_prompt = """Ты - помощник службы технической поддержки. 
Твоя задача - дать краткий, конкретный и профессиональный ответ на обращение пользователя.

Правила ответа:
1. Ответ должен быть практичным и содержать конкретные шаги решения
2. Если проблема требует доступа/разрешений - укажи куда обратиться
3. Если это технический вопрос - дай пошаговую инструкцию
4. Если нужна документация - укажи где её найти
5. Ответ должен быть на русском языке
6. Длина ответа: 2-4 предложения, максимум 300 символов

Примеры:
Вопрос: "Не могу войти в систему, пишет ошибку авторизации"
Ответ: "Попробуйте сбросить пароль через портал self-service.company.ru. Если не помогает, проверьте что учетная запись не заблокирована - обратитесь к системному администратору через заявку. При повторной ошибке приложите скриншот."

Вопрос: "Нужен доступ к GitLab репозиторию"
Ответ: "Создайте заявку в ServiceDesk с указанием репозитория и требуемой роли (Developer/Maintainer). Ваш руководитель должен согласовать доступ. Обычно заявки обрабатываются в течение 1 рабочего дня."
"""

    user_prompt = f"Обращение пользователя:\n{appeal}\n\nСформулируй краткий профессиональный ответ:"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=200,
        )

        answer = response.choices[0].message.content.strip()
        return answer

    except Exception as e:
        logger.error(f"Ошибка при генерации ответа: {e}")
        return "Обратитесь в службу поддержки для решения данного вопроса. Создайте заявку с подробным описанием проблемы."


def expand_qa_dataset(
    original_qa_path: str, appeals_path: str, output_path: str, num_new_pairs: int = 200
):
    """
    Расширяет датасет QA новыми парами вопрос-ответ

    Args:
        original_qa_path: Путь к исходному QA.xlsx
        appeals_path: Путь к файлу с обращениями
        output_path: Путь для сохранения расширенного датасета
        num_new_pairs: Количество новых пар для добавления
    """
    logger.info("=" * 80)
    logger.info("РАСШИРЕНИЕ ДАТАСЕТА QA")
    logger.info("=" * 80)

    # Читаем исходный QA
    logger.info(f"Загружаем исходный датасет: {original_qa_path}")
    original_df = pd.read_excel(original_qa_path)
    logger.info(f"Исходный датасет содержит {len(original_df)} записей")

    # Читаем обращения
    appeals = read_appeals(appeals_path)

    # Ограничиваем количество обращений
    appeals_to_process = appeals[:num_new_pairs]
    logger.info(f"Будет обработано {len(appeals_to_process)} обращений")

    # Инициализируем OpenAI клиент
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY не найден в переменных окружения")

    client = OpenAI(api_key=api_key)

    # Генерируем ответы
    logger.info("Начинаем генерацию ответов...")
    new_qa_pairs = []

    for idx, appeal in enumerate(appeals_to_process, 1):
        logger.info(f"Обрабатываем {idx}/{len(appeals_to_process)}: {appeal[:60]}...")

        answer = generate_answer_for_appeal(client, appeal)

        new_qa_pairs.append({"Вопрос": appeal, "Ответ": answer})

        # Логируем прогресс каждые 20 записей
        if idx % 20 == 0:
            logger.info(
                f"Прогресс: {idx}/{len(appeals_to_process)} ({idx*100//len(appeals_to_process)}%)"
            )

    # Создаем новый DataFrame
    new_df = pd.DataFrame(new_qa_pairs)

    # Объединяем с исходными данными
    logger.info("Объединяем датасеты...")
    combined_df = pd.concat([original_df, new_df], ignore_index=True)

    logger.info(f"Итоговый датасет содержит {len(combined_df)} записей")
    logger.info(f"  - Исходных: {len(original_df)}")
    logger.info(f"  - Новых: {len(new_df)}")

    # Сохраняем результат
    logger.info(f"Сохраняем расширенный датасет в {output_path}")
    combined_df.to_excel(output_path, index=False, engine="openpyxl")

    logger.info("=" * 80)
    logger.info("✅ ДАТАСЕТ УСПЕШНО РАСШИРЕН!")
    logger.info("=" * 80)

    # Показываем примеры новых записей
    logger.info("\nПримеры новых Q&A пар:")
    logger.info("-" * 80)
    for i in range(min(3, len(new_df))):
        logger.info(f"\nПример {i+1}:")
        logger.info(f"Q: {new_df.iloc[i]['Вопрос'][:100]}...")
        logger.info(f"A: {new_df.iloc[i]['Ответ'][:100]}...")


if __name__ == "__main__":
    # Пути к файлам
    ORIGINAL_QA = "QA.xlsx"
    APPEALS_FILE = "Обращения.txt"
    OUTPUT_FILE = "QA_expanded.xlsx"
    NUM_NEW_PAIRS = 200

    try:
        expand_qa_dataset(
            original_qa_path=ORIGINAL_QA,
            appeals_path=APPEALS_FILE,
            output_path=OUTPUT_FILE,
            num_new_pairs=NUM_NEW_PAIRS,
        )
    except Exception as e:
        logger.error(f"❌ Ошибка при расширении датасета: {e}")
        raise
