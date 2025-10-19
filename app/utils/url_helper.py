"""
Утилита для автоматического определения базового URL приложения
Поддерживает ngrok, переменные окружения и автоопределение
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_base_url() -> str:
    """
    Получает базовый URL приложения с автоматическим определением

    Приоритет:
    1. BASE_URL из переменной окружения
    2. Автоматическое определение ngrok URL
    3. Конфиг из app_config.yaml
    4. Fallback на localhost:8000

    Returns:
        Базовый URL приложения (без trailing slash)
    """
    # 1. Проверяем переменную окружения BASE_URL
    base_url = os.getenv("BASE_URL")
    if base_url:
        base_url = base_url.rstrip("/")
        logger.info(f"Using BASE_URL from environment: {base_url}")
        return base_url

    # 2. Пробуем получить ngrok URL
    ngrok_url = get_ngrok_url()
    if ngrok_url:
        logger.info(f"Using ngrok URL: {ngrok_url}")
        return ngrok_url

    # 3. Проверяем конфиг
    try:
        from app.config import load_app_config

        app_config = load_app_config()
        config_url = app_config.get("base_url")
        if config_url:
            config_url = config_url.rstrip("/")
            logger.info(f"Using BASE_URL from config: {config_url}")
            return config_url
    except Exception as e:
        logger.warning(f"Could not load app config: {e}")

    # 4. Fallback
    fallback = "http://localhost:8000"
    logger.info(f"Using fallback BASE_URL: {fallback}")
    return fallback


def get_ngrok_url() -> Optional[str]:
    """
    Автоматически получает ngrok URL если ngrok запущен

    Returns:
        ngrok URL или None если ngrok не активен
    """
    try:
        import requests

        # ngrok API endpoint (локальный)
        ngrok_api = "http://127.0.0.1:4040/api/tunnels"

        response = requests.get(ngrok_api, timeout=2)
        if response.status_code == 200:
            data = response.json()
            tunnels = data.get("tunnels", [])

            # Ищем https туннель
            for tunnel in tunnels:
                if tunnel.get("proto") == "https":
                    public_url = tunnel.get("public_url")
                    if public_url:
                        logger.info(f"Found ngrok HTTPS tunnel: {public_url}")
                        return public_url.rstrip("/")

            # Если нет https, берем первый доступный
            if tunnels:
                public_url = tunnels[0].get("public_url")
                if public_url:
                    logger.info(f"Found ngrok tunnel: {public_url}")
                    return public_url.rstrip("/")

        return None

    except ImportError:
        # requests не установлен
        return None
    except Exception as e:
        # ngrok не запущен или недоступен
        logger.debug(f"Could not get ngrok URL: {e}")
        return None


def set_base_url_env(url: str) -> None:
    """
    Устанавливает BASE_URL в переменную окружения

    Args:
        url: Базовый URL для установки
    """
    os.environ["BASE_URL"] = url.rstrip("/")
    logger.info(f"Set BASE_URL environment variable to: {url}")


if __name__ == "__main__":
    # Тестирование утилиты
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("BASE URL Detection Test")
    print("=" * 60)

    # Тест 1: Проверка ngrok
    ngrok = get_ngrok_url()
    print(f"\nngrok URL: {ngrok or 'Not detected'}")

    # Тест 2: Получение финального URL
    final_url = get_base_url()
    print(f"\nFinal BASE_URL: {final_url}")

    print("\n" + "=" * 60)
