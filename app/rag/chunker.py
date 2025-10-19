"""
Простая и быстрая разбивка текста на чанки
Без зависимостей, работает сразу
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Чанк текста с метаданными"""
    content: str
    chunk_index: int
    source_file: str | None = None
    start_char: int = 0
    end_char: int = 0
    metadata: dict | None = None


class SimpleTextChunker:
    """Простая и быстрая разбивка текста"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Args:
            chunk_size: Размер чанка в символах
            chunk_overlap: Перекрытие между чанками
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(
        self, 
        text: str, 
        source_file: str | None = None
    ) -> List[TextChunk]:
        """
        Быстрая разбивка текста на чанки с учетом границ предложений
        
        Args:
            text: Исходный текст
            source_file: Имя исходного файла
            
        Returns:
            Список чанков
        """
        chunks = []
        current_pos = 0
        chunk_idx = 0
        
        while current_pos < len(text):
            # Берем чанк с перекрытием
            end_pos = min(current_pos + self.chunk_size, len(text))
            
            # Пытаемся найти границу предложения
            if end_pos < len(text):
                # Ищем ближайшую точку, восклицательный или вопросительный знак
                for sep in [". ", "! ", "? ", "\n\n", "\n", "; "]:
                    last_sep = text.rfind(sep, current_pos, end_pos)
                    if last_sep != -1:
                        end_pos = last_sep + len(sep)
                        break
                else:
                    # Если не нашли знаки препинания, ищем пробел
                    last_space = text.rfind(" ", current_pos, end_pos)
                    if last_space != -1:
                        end_pos = last_space
            
            chunk_text = text[current_pos:end_pos].strip()
            
            if chunk_text and len(chunk_text) > 10:  # Минимальная длина чанка
                chunk = TextChunk(
                    content=chunk_text,
                    chunk_index=chunk_idx,
                    source_file=source_file,
                    start_char=current_pos,
                    end_char=end_pos,
                )
                chunks.append(chunk)
                chunk_idx += 1
            
            # Сдвигаемся с учетом перекрытия
            if current_pos == end_pos:  # Если не продвинулись, принудительно сдвигаемся
                current_pos += 1
            else:
                current_pos = max(end_pos - self.chunk_overlap, current_pos + 1)

        logger.info(f"Split text into {len(chunks)} chunks (simple)")
        return chunks


def chunk_document(
    text: str,
    source_file: str | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    method: str = "simple"
) -> List[TextChunk]:
    """
    Быстрая функция для разбивки документа на чанки
    
    Args:
        text: Текст документа
        source_file: Имя файла источника
        chunk_size: Размер чанка
        chunk_overlap: Перекрытие между чанками
        method: Метод разбивки (только 'simple' пока)
        
    Returns:
        Список чанков
    """
    chunker = SimpleTextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    return chunker.split_text(text, source_file)
