# Используем официальный Python образ
FROM python:3.11-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей ОТДЕЛЬНО
# Этот слой кэшируется и не пересобирается если requirements.txt не изменился
COPY requirements.txt .

# Устанавливаем Python зависимости с использованием pip cache
# --mount=type=cache позволяет кэшировать скачанные пакеты между сборками
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Копируем все файлы проекта
# Этот слой пересобирается только если изменились файлы проекта
COPY . .

# Создаем директории для данных
RUN mkdir -p /app/data

# Переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose порт для FastAPI
EXPOSE 8000

# Команда запуска
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
