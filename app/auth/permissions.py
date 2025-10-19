"""
Модуль для проверки прав доступа к страницам и функционалу
"""

from functools import wraps
from typing import Optional

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse

from .user_manager import user_manager


def get_current_user_from_request(request: Request) -> Optional[dict]:
    """
    Получить текущего пользователя из сессии.
    Возвращает None если не авторизован.
    """
    from .auth import get_user_id_from_request

    user_id = get_user_id_from_request(request)
    if not user_id:
        return None

    # Получаем пользователя из БД
    user = user_manager.get_user_by_id(user_id)

    # Проверяем, что пользователь активен
    if user and not user.get("is_active"):
        return None

    return user


def get_current_user_id(request: Request) -> Optional[int]:
    """Получить ID текущего пользователя"""
    user = get_current_user_from_request(request)
    return user["id"] if user else None


def has_permission(request: Request, page_key: str) -> bool:
    """
    Проверить, есть ли у текущего пользователя право на страницу
    """
    user = get_current_user_from_request(request)

    if not user:
        return False

    # Админы имеют все права
    if user.get("is_admin"):
        return True

    return user_manager.has_permission(user["id"], page_key)


def require_permission(page_key: str, redirect_to_login: bool = True):
    """
    Декоратор для проверки прав доступа к endpoint

    Использование:
        @app.get("/dashboard")
        @require_permission("dashboard")
        async def dashboard_page(request: Request):
            ...
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            user = get_current_user_from_request(request)

            if not user:
                if redirect_to_login:
                    return RedirectResponse(url="/login", status_code=303)
                raise HTTPException(status_code=401, detail="Authentication required")

            # Админы имеют все права
            if user.get("is_admin"):
                return await func(request, *args, **kwargs)

            # Проверяем право доступа
            if not user_manager.has_permission(user["id"], page_key):
                raise HTTPException(
                    status_code=403, detail=f"У вас нет доступа к разделу '{page_key}'"
                )

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator


def require_admin(redirect_to_home: bool = True):
    """
    Декоратор для проверки прав администратора
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            user = get_current_user_from_request(request)

            if not user:
                return RedirectResponse(url="/login", status_code=303)

            if not user.get("is_admin"):
                if redirect_to_home:
                    return RedirectResponse(url="/", status_code=303)
                raise HTTPException(
                    status_code=403, detail="Требуются права администратора"
                )

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator


def get_user_permissions(request: Request) -> list[str]:
    """
    Получить список всех прав текущего пользователя
    Полезно для фронтенда, чтобы скрывать недоступные разделы
    """
    user = get_current_user_from_request(request)

    if not user:
        return []

    # Админы имеют все права
    if user.get("is_admin"):
        all_permissions = user_manager.get_all_permissions()
        return [p["page_key"] for p in all_permissions] + ["admin"]

    return user_manager.get_user_permissions(user["id"])
