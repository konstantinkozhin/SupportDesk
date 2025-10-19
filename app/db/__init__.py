"""Database module - models, schemas, CRUD operations, and connections."""

# Database connections and sessions
from app.db.database import (
    get_tickets_session,
    get_knowledge_session,
    init_db,
    TicketsSessionLocal,
    KnowledgeSessionLocal,
)

# SQLAlchemy models
from app.db.models import (
    Ticket,
    Message,
    DocumentChunk,
    TicketStatus,
)

# Pydantic schemas
from app.db.schemas import (
    TicketRead,
    MessageRead,
    MessageCreate,
    KnowledgeStats,
)

# CRUD operations
from app.db import tickets_crud

__all__ = [
    # Database
    "get_tickets_session",
    "get_knowledge_session",
    "init_db",
    "TicketsSessionLocal",
    "KnowledgeSessionLocal",
    # Models
    "Ticket",
    "Message",
    "DocumentChunk",
    "TicketStatus",
    # Schemas
    "TicketRead",
    "MessageRead",
    "MessageCreate",
    "KnowledgeStats",
    # CRUD module
    "tickets_crud",
]
