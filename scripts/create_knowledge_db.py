import asyncio
from app.database import knowledge_engine
from app.models import KnowledgeBase

async def create_knowledge_db():
    """Создает базу знаний с новой схемой"""
    async with knowledge_engine.begin() as conn:
        await conn.run_sync(KnowledgeBase.metadata.create_all)
    print("База знаний создана!")

if __name__ == "__main__":
    asyncio.run(create_knowledge_db())