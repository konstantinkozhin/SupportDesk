from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict

from app.db.models import TicketStatus


class TicketRead(BaseModel):
    id: int
    # Store chat id as string to support Telegram numeric ids and VK style ids like 'vk_123456'
    telegram_chat_id: str
    title: Optional[str]
    summary: Optional[str]
    status: TicketStatus
    priority: str
    operator_requested: bool = False  # Флаг запроса оператора
    created_at: datetime
    first_response_at: Optional[datetime]
    closed_at: Optional[datetime]
    updated_at: datetime
    unread_count: int = 0  # Количество непрочитанных сообщений

    model_config = ConfigDict(from_attributes=True)

    @property
    def is_archived(self) -> bool:
        """Проверяет, является ли заявка архивной"""
        from app.db.models import TicketStatus

        return self.status in [TicketStatus.CLOSED, TicketStatus.ARCHIVED]

    @property
    def chat_id_numeric(self) -> int | None:
        """If the telegram_chat_id represents a numeric id, return it as int, otherwise None."""
        try:
            return int(self.telegram_chat_id)
        except Exception:
            return None


class MessageRead(BaseModel):
    id: int
    ticket_id: int
    sender: str
    text: str
    is_system: bool = False
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

    # Для обратной совместимости с API
    @property
    def conversation_id(self) -> int:
        return self.ticket_id


class MessageCreate(BaseModel):
    text: str


class KnowledgeStats(BaseModel):
    total_entries: int
