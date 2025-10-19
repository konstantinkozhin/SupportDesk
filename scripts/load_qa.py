import pandas as pd
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import KnowledgeSessionLocal
from app import models

async def load_qa_data():
    """Загружает данные из QA.xlsx в базу знаний"""
    
    # Читаем Excel файл
    try:
        df = pd.read_excel('QA.xlsx')
        print(f"Загружен файл с {len(df)} записями")
        print("Колонки:", df.columns.tolist())
    except Exception as e:
        print(f"Ошибка чтения файла: {e}")
        return

    # Создаем записи в базе знаний
    async with KnowledgeSessionLocal() as session:
        for index, row in df.iterrows():
            # Предполагаем, что есть колонки 'Вопрос' и 'Ответ' или подобные
            question_col = None
            answer_col = None
            
            for col in df.columns:
                col_lower = str(col).lower()
                if any(word in col_lower for word in ['вопрос', 'question', 'q']):
                    question_col = col
                elif any(word in col_lower for word in ['ответ', 'answer', 'a']):
                    answer_col = col
            
            if question_col is None or answer_col is None:
                print("Не найдены колонки для вопросов и ответов")
                print("Доступные колонки:", df.columns.tolist())
                break
                
            question = str(row[question_col]).strip()
            answer = str(row[answer_col]).strip()
            
            if question and answer and question != 'nan' and answer != 'nan':
                entry = models.KnowledgeEntry(
                    question=question,
                    answer=answer
                )
                session.add(entry)
                
        await session.commit()
        print(f"Загружено записей в базу знаний")

if __name__ == "__main__":
    asyncio.run(load_qa_data())