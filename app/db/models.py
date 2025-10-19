from __future__ import annotations

from datetime import datetime
from enum import Enum

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class TicketStatus(str, Enum):
    OPEN = "open"  # Открыта
    IN_PROGRESS = "in_progress"  # В работе
    CLOSED = "closed"  # Закрыта
    ARCHIVED = "archived"  # Архивирована


class Ticket(Base):
    """Заявка в службу поддержки"""

    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    telegram_chat_id = Column(
        String(255), index=True, nullable=False
    )  # Изменено на String для поддержки VK
    title = Column(String(255), nullable=True)
    summary = Column(Text, nullable=True)  # Краткое описание заявки (авто-генерируется)
    classification = Column(Text, nullable=True)  # Классификация проблемы от AI агента
    operator_requested = Column(
        Boolean, default=False, nullable=False
    )  # Флаг запроса оператора
    status = Column(String(20), default=TicketStatus.OPEN, nullable=False)
    priority = Column(String(10), default="medium", nullable=False)  # low, medium, high
    it_ticket_number = Column(String(50), nullable=True)  # Номер заявки IT-специалисту
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    first_response_at = Column(
        DateTime, nullable=True
    )  # Время первого ответа оператора
    closed_at = Column(DateTime, nullable=True)  # Время закрытия заявки
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Связи
    messages = relationship(
        "Message",
        back_populates="ticket",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


class Message(Base):
    """Сообщение в заявке"""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(Integer, ForeignKey("tickets.id"), nullable=False, index=True)
    sender = Column(String(32), nullable=False)  # user, bot, operator
    text = Column(Text, nullable=False)
    telegram_message_id = Column(BigInteger, nullable=True)
    vk_message_id = Column(BigInteger, nullable=True)
    is_system = Column(Boolean, default=False, nullable=False)  # Системное сообщение
    is_read = Column(Boolean, default=False, nullable=False)  # Прочитано оператором
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Связи
    ticket = relationship("Ticket", back_populates="messages")


# Отдельная база для знаний (чанки документов для агента)
KnowledgeBase = declarative_base()


class DocumentChunk(KnowledgeBase):
    """Чанк документа в новой системе базы знаний"""

    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)  # Текст чанка
    source_file = Column(String(512), nullable=False)  # Имя исходного файла
    chunk_index = Column(Integer, nullable=False)  # Порядковый номер чанка в документе
    start_char = Column(Integer, default=0)  # Начальная позиция в документе
    end_char = Column(Integer, default=0)  # Конечная позиция в документе
    embedding = Column(LargeBinary, nullable=True)  # Векторное представление
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    # Дополнительные метаданные (JSON как текст)
    chunk_metadata = Column(Text, nullable=True)  # Переименовано из metadata
