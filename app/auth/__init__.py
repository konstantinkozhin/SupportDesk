"""
Подсистема аутентификации и авторизации

Включает в себя:
- auth - базовая аутентификация (сессии, токены)
- permissions - система прав доступа
- user_manager - управление пользователями
"""

from .auth import (
    get_settings,
    validate_credentials,
    issue_session_cookie,
    clear_session_cookie,
    get_user_id_from_request,
    is_authenticated_request,
    ensure_api_auth,
    is_authenticated_websocket,
    WEBSOCKET_UNAUTHORIZED_CLOSE_CODE,
)
from .permissions import (
    get_current_user_from_request,
    get_current_user_id,
    has_permission,
    require_permission,
    require_admin,
    get_user_permissions,
)
from .user_manager import user_manager

__all__ = [
    # auth
    "get_settings",
    "validate_credentials",
    "issue_session_cookie",
    "clear_session_cookie",
    "get_user_id_from_request",
    "is_authenticated_request",
    "ensure_api_auth",
    "is_authenticated_websocket",
    "WEBSOCKET_UNAUTHORIZED_CLOSE_CODE",
    # permissions
    "get_current_user_from_request",
    "get_current_user_id",
    "has_permission",
    "require_permission",
    "require_admin",
    "get_user_permissions",
    # user_manager
    "user_manager",
]
