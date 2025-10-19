"""
Скрипт для создания системы управления пользователями и правами доступа.
Создает таблицы: users, roles, permissions, user_permissions
"""

import sqlite3
from datetime import datetime
import bcrypt


def create_users_database():
    conn = sqlite3.connect("db/users.db")
    cursor = conn.cursor()

    # Таблица пользователей
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT,
        role_id INTEGER,
        is_active BOOLEAN DEFAULT 1,
        is_admin BOOLEAN DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP,
        FOREIGN KEY (role_id) REFERENCES roles(id)
    )
    """
    )

    # Таблица ролей
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS roles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    )

    # Таблица прав доступа (страниц)
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS permissions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        page_key TEXT UNIQUE NOT NULL,
        page_name TEXT NOT NULL,
        description TEXT
    )
    """
    )

    # Связующая таблица: роли и права
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS role_permissions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role_id INTEGER NOT NULL,
        permission_id INTEGER NOT NULL,
        FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE,
        FOREIGN KEY (permission_id) REFERENCES permissions(id) ON DELETE CASCADE,
        UNIQUE(role_id, permission_id)
    )
    """
    )

    # Связующая таблица: пользователи и индивидуальные права
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS user_permissions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        permission_id INTEGER NOT NULL,
        granted BOOLEAN DEFAULT 1,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (permission_id) REFERENCES permissions(id) ON DELETE CASCADE,
        UNIQUE(user_id, permission_id)
    )
    """
    )

    # Таблица настроек системы
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS system_settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        setting_key TEXT UNIQUE NOT NULL,
        setting_value TEXT,
        setting_type TEXT DEFAULT 'string',
        description TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    )

    # Вставка базовых прав доступа (страниц)
    pages = [
        ("tickets", "Заявки", "Управление заявками пользователей"),
        ("dashboard", "Дашборд", "Просмотр статистики"),
        ("knowledge", "База знаний", "Управление базой знаний"),
        ("simulator", "Симулятор", "Доступ к симулятору обучения"),
        ("admin", "Администрирование", "Доступ к админ-панели)"),
    ]

    cursor.executemany(
        """
    INSERT OR IGNORE INTO permissions (page_key, page_name, description)
    VALUES (?, ?, ?)
    """,
        pages,
    )

    # Создание базовых ролей
    roles = [
        ("admin", "Администратор - Полный доступ ко всем разделам"),
        ("operator", "Оператор - Доступ к заявкам и симулятору"),
        ("analyst", "Аналитик - Доступ к дашборду и базе знаний"),
        ("trainee", "Стажер - Доступ только к симулятору"),
    ]

    cursor.executemany(
        """
    INSERT OR IGNORE INTO roles (name, description)
    VALUES (?, ?)
    """,
        roles,
    )

    # Назначение прав для роли "Администратор" (все права)
    cursor.execute(
        """
    INSERT OR IGNORE INTO role_permissions (role_id, permission_id)
    SELECT r.id, p.id
    FROM roles r, permissions p
    WHERE r.name = 'admin'
    """
    )

    # Назначение прав для роли "Оператор" (заявки и симулятор)
    cursor.execute(
        """
    INSERT OR IGNORE INTO role_permissions (role_id, permission_id)
    SELECT r.id, p.id
    FROM roles r, permissions p
    WHERE r.name = 'operator' AND p.page_key IN ('tickets', 'simulator')
    """
    )

    # Назначение прав для роли "Аналитик" (дашборд и база знаний)
    cursor.execute(
        """
    INSERT OR IGNORE INTO role_permissions (role_id, permission_id)
    SELECT r.id, p.id
    FROM roles r, permissions p
    WHERE r.name = 'analyst' AND p.page_key IN ('dashboard', 'knowledge')
    """
    )

    # Назначение прав для роли "Стажер" (только симулятор)
    cursor.execute(
        """
    INSERT OR IGNORE INTO role_permissions (role_id, permission_id)
    SELECT r.id, p.id
    FROM roles r, permissions p
    WHERE r.name = 'trainee' AND p.page_key = 'simulator'
    """
    )

    # Создание администратора по умолчанию
    admin_password = "admin123"  # ВАЖНО: Сменить после первого входа!
    password_hash = bcrypt.hashpw(
        admin_password.encode("utf-8"), bcrypt.gensalt()
    ).decode("utf-8")

    cursor.execute(
        """
    INSERT OR IGNORE INTO users (username, password_hash, full_name, role_id, is_admin)
    SELECT 'admin', ?, 'Администратор', id, 1
    FROM roles WHERE name = 'admin'
    """,
        (password_hash,),
    )

    # Вставка базовых настроек системы
    settings = [
        (
            "registration_enabled",
            "false",
            "boolean",
            "Разрешить регистрацию новых пользователей",
        ),
        ("auto_bot_responses", "true", "boolean", "Автоматические ответы бота"),
        ("ticket_moderation", "false", "boolean", "Модерация заявок перед публикацией"),
        (
            "max_file_size_mb",
            "10",
            "number",
            "Максимальный размер загружаемых файлов (МБ)",
        ),
        ("session_timeout_minutes", "60", "number", "Время жизни сессии (минуты)"),
    ]

    cursor.executemany(
        """
    INSERT OR IGNORE INTO system_settings (setting_key, setting_value, setting_type, description)
    VALUES (?, ?, ?, ?)
    """,
        settings,
    )

    conn.commit()
    conn.close()

    print("✅ База данных пользователей создана успешно!")
    print("📝 Учетные данные администратора:")
    print("   Логин: admin")
    print("   Пароль: admin123")
    print("⚠️  ВАЖНО: Смените пароль после первого входа!")


if __name__ == "__main__":
    create_users_database()
