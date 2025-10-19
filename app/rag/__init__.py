"""
RAG (Retrieval-Augmented Generation) подсистема

Включает в себя:
- RAGService - основной сервис для работы с RAG
- KnowledgeBase - работа с базой знаний
- Retrieval функции
- WhisperService - локальное распознавание речи
"""

from .service import (
    RAGService,
    RAGResult,
    ChatMessage,
    ToxicityClassifier,
    SpeechToTextService,
)
from .whisper_service import WhisperService

__all__ = [
    "RAGService",
    "RAGResult",
    "ChatMessage",
    "ToxicityClassifier",
    "SpeechToTextService",
    "KnowledgeBase",
    "WhisperService",
]
