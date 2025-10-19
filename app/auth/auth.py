from __future__ import annotations

import hashlib
import os
import secrets
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from fastapi import HTTPException, Request, Response, WebSocket, status


@dataclass(frozen=True)
class AuthSettings:
    cookie_name: str
    cookie_secure: bool
    secret_key: str


@lru_cache(maxsize=1)
def get_settings() -> AuthSettings:
    cookie_name = os.getenv("SESSION_COOKIE_NAME", "support_session")
    cookie_secure = os.getenv("SESSION_COOKIE_SECURE", "false").lower() == "true"
    secret_key = os.getenv("SECRET_KEY", "default-secret-key-change-in-production")
    return AuthSettings(
        cookie_name=cookie_name, cookie_secure=cookie_secure, secret_key=secret_key
    )


def _create_session_token(user_id: int) -> str:
    """Создать токен сессии для пользователя"""
    settings = get_settings()
    # Формат: user_id:random_token:signature
    random_part = secrets.token_urlsafe(32)
    payload = f"{user_id}:{random_part}"
    signature = hashlib.sha256(f"{payload}:{settings.secret_key}".encode()).hexdigest()
    return f"{payload}:{signature}"


def _verify_session_token(token: str) -> Optional[int]:
    """Проверить токен и вернуть user_id"""
    try:
        settings = get_settings()
        parts = token.split(":")
        if len(parts) != 3:
            return None

        user_id_str, random_part, signature = parts
        payload = f"{user_id_str}:{random_part}"
        expected_signature = hashlib.sha256(
            f"{payload}:{settings.secret_key}".encode()
        ).hexdigest()

        if signature != expected_signature:
            return None

        return int(user_id_str)
    except (ValueError, AttributeError):
        return None


def validate_credentials(username: str, password: str) -> Optional[dict]:
    """Проверить учетные данные и вернуть пользователя"""
    from .user_manager import user_manager

    return user_manager.verify_password(username, password)


def issue_session_cookie(response: Response, user_id: int) -> None:
    """Установить cookie сессии для пользователя"""
    settings = get_settings()
    token = _create_session_token(user_id)
    response.set_cookie(
        settings.cookie_name,
        token,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        max_age=7 * 24 * 60 * 60,  # 7 дней
    )


def clear_session_cookie(response: Response) -> None:
    settings = get_settings()
    response.delete_cookie(settings.cookie_name)


def get_user_id_from_request(request: Request) -> Optional[int]:
    """Получить user_id из cookie сессии"""
    settings = get_settings()
    token = request.cookies.get(settings.cookie_name)
    if not token:
        return None
    return _verify_session_token(token)


def is_authenticated_request(request: Request) -> bool:
    """Проверить, авторизован ли пользователь"""
    return get_user_id_from_request(request) is not None


def ensure_api_auth(request: Request) -> None:
    if not is_authenticated_request(request):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )


def is_authenticated_websocket(websocket: WebSocket) -> bool:
    """Проверить авторизацию WebSocket"""
    settings = get_settings()
    token = websocket.cookies.get(settings.cookie_name)
    if not token:
        return False
    return _verify_session_token(token) is not None


WEBSOCKET_UNAUTHORIZED_CLOSE_CODE = 4401
