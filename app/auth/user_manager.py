"""
Модуль для управления пользователями, ролями и правами доступа
"""

import sqlite3
from typing import Optional, List, Dict, Any
import bcrypt
from datetime import datetime


class UserManager:
    def __init__(self, db_path: str = "db/users.db"):
        self.db_path = db_path
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        """Проверить существование БД и создать если нужно"""
        import os

        # Создаём директорию если её нет
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        # Проверяем существует ли БД
        db_exists = os.path.exists(self.db_path)

        if not db_exists:
            print(f"⚠️ Users database not found, creating: {self.db_path}")
            self._create_tables()
            self._create_default_admin()

    def _create_tables(self):
        """Создать таблицы БД пользователей"""
        conn = self._get_conn()
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

        # Таблица прав доступа
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

        # Связь ролей и прав
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

        # Индивидуальные права пользователей
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

        # Системные настройки
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

        # Вставка базовых прав
        pages = [
            ("tickets", "Заявки", "Управление заявками пользователей"),
            ("dashboard", "Дашборд", "Просмотр статистики"),
            ("knowledge", "База знаний", "Управление базой знаний"),
            ("simulator", "Симулятор", "Доступ к симулятору обучения"),
            ("admin", "Администрирование", "Доступ к админ-панели"),
        ]

        cursor.executemany(
            """
            INSERT OR IGNORE INTO permissions (page_key, page_name, description)
            VALUES (?, ?, ?)
        """,
            pages,
        )

        # Создание ролей
        roles = [
            ("admin", "Администратор - Полный доступ"),
            ("operator", "Оператор - Доступ к заявкам"),
        ]

        cursor.executemany(
            """
            INSERT OR IGNORE INTO roles (name, description)
            VALUES (?, ?)
        """,
            roles,
        )

        # Права для админа (все)
        cursor.execute(
            """
            INSERT OR IGNORE INTO role_permissions (role_id, permission_id)
            SELECT r.id, p.id
            FROM roles r, permissions p
            WHERE r.name = 'admin'
        """
        )

        # Права для оператора (заявки + симулятор)
        cursor.execute(
            """
            INSERT OR IGNORE INTO role_permissions (role_id, permission_id)
            SELECT r.id, p.id
            FROM roles r, permissions p
            WHERE r.name = 'operator' AND p.page_key IN ('tickets', 'simulator')
        """
        )

        conn.commit()
        conn.close()
        print("✅ Users database tables created")

    def _create_default_admin(self):
        """Создать пользователя admin по умолчанию"""
        import os

        conn = self._get_conn()
        cursor = conn.cursor()

        # Получаем ID роли admin
        cursor.execute("SELECT id FROM roles WHERE name = 'admin'")
        role_row = cursor.fetchone()

        if role_row:
            role_id = role_row[0]

            # Получаем пароль из переменных окружения или используем admin по умолчанию
            admin_username = os.getenv("ADMIN_USERNAME", "admin")
            admin_password = os.getenv("ADMIN_PASSWORD", "admin")

            # Создаём хеш пароля
            password_hash = bcrypt.hashpw(
                admin_password.encode("utf-8"), bcrypt.gensalt()
            ).decode("utf-8")

            try:
                cursor.execute(
                    """
                    INSERT INTO users (username, password_hash, full_name, role_id, is_admin)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (admin_username, password_hash, "Администратор", role_id, True),
                )

                conn.commit()
                print(
                    f"✅ Default admin user created (username: {admin_username}, password: {admin_password})"
                )
                if admin_password == "admin":
                    print("⚠️  PLEASE CHANGE THE DEFAULT PASSWORD!")
            except sqlite3.IntegrityError:
                # Пользователь уже существует
                pass

        conn.close()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    # ==================== USERS ====================

    def create_user(
        self,
        username: str,
        password: str,
        full_name: str,
        role_id: int,
        is_admin: bool = False,
    ) -> Optional[int]:
        """Создать нового пользователя"""
        password_hash = bcrypt.hashpw(
            password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
            INSERT INTO users (username, password_hash, full_name, role_id, is_admin)
            VALUES (?, ?, ?, ?, ?)
            """,
                (username, password_hash, full_name, role_id, is_admin),
            )

            user_id = cursor.lastrowid
            conn.commit()
            return user_id
        except sqlite3.IntegrityError:
            conn.rollback()
            return None
        except sqlite3.OperationalError:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Получить пользователя по имени"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            """
        SELECT u.*, r.name as role_name, r.description as role_description
        FROM users u
        LEFT JOIN roles r ON u.role_id = r.id
        WHERE u.username = ?
        """,
            (username,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Получить пользователя по ID"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            """
        SELECT u.*, r.name as role_name, r.description as role_description
        FROM users u
        LEFT JOIN roles r ON u.role_id = r.id
        WHERE u.id = ?
        """,
            (user_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    def get_all_users(self) -> List[Dict[str, Any]]:
        """Получить всех пользователей"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            """
        SELECT u.*, r.name as role_name
        FROM users u
        LEFT JOIN roles r ON u.role_id = r.id
        ORDER BY u.created_at DESC
        """
        )

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def update_user(self, user_id: int, **kwargs) -> bool:
        """Обновить данные пользователя"""
        allowed_fields = ["full_name", "role_id", "is_active", "is_admin"]

        # Обрабатываем пароль отдельно
        password = kwargs.pop("password", None)
        if password:
            self.update_password(user_id, password)

        update_fields = {k: v for k, v in kwargs.items() if k in allowed_fields}

        if not update_fields:
            # Если был обновлен только пароль, возвращаем True
            return password is not None

        set_clause = ", ".join([f"{k} = ?" for k in update_fields.keys()])
        values = list(update_fields.values()) + [user_id]

        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            f"""
        UPDATE users SET {set_clause}
        WHERE id = ?
        """,
            values,
        )

        affected = cursor.rowcount
        conn.commit()
        conn.close()

        return affected > 0

    def update_password(self, user_id: int, new_password: str) -> bool:
        """Обновить пароль пользователя"""
        password_hash = bcrypt.hashpw(
            new_password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")

        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            """
        UPDATE users SET password_hash = ?
        WHERE id = ?
        """,
            (password_hash, user_id),
        )

        affected = cursor.rowcount
        conn.commit()
        conn.close()

        return affected > 0

    def delete_user(self, user_id: int) -> bool:
        """Удалить пользователя"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))

        affected = cursor.rowcount
        conn.commit()
        conn.close()

        return affected > 0

    def verify_password(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Проверить пароль и вернуть данные пользователя"""
        user = self.get_user_by_username(username)

        if not user or not user.get("is_active"):
            return None

        if bcrypt.checkpw(
            password.encode("utf-8"), user["password_hash"].encode("utf-8")
        ):
            # Обновить время последнего входа
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET last_login = ? WHERE id = ?",
                (datetime.now(), user["id"]),
            )
            conn.commit()
            conn.close()

            return user

        return None

    # ==================== PERMISSIONS ====================

    def get_user_permissions(self, user_id: int) -> List[str]:
        """Получить список прав доступа пользователя (ключи страниц)"""
        conn = self._get_conn()
        cursor = conn.cursor()

        # Права от роли
        cursor.execute(
            """
        SELECT DISTINCT p.page_key
        FROM users u
        JOIN roles r ON u.role_id = r.id
        JOIN role_permissions rp ON r.id = rp.role_id
        JOIN permissions p ON rp.permission_id = p.id
        WHERE u.id = ?
        """,
            (user_id,),
        )

        role_perms = set(row[0] for row in cursor.fetchall())

        # Индивидуальные права (могут отменять или добавлять)
        cursor.execute(
            """
        SELECT p.page_key, up.granted
        FROM user_permissions up
        JOIN permissions p ON up.permission_id = p.id
        WHERE up.user_id = ?
        """,
            (user_id,),
        )

        for page_key, granted in cursor.fetchall():
            if granted:
                role_perms.add(page_key)
            else:
                role_perms.discard(page_key)

        conn.close()
        return list(role_perms)

    def has_permission(self, user_id: int, page_key: str) -> bool:
        """Проверить, есть ли у пользователя право на страницу"""
        # Админы имеют все права
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT is_admin FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()

        if row and row[0]:
            conn.close()
            return True

        conn.close()
        permissions = self.get_user_permissions(user_id)
        return page_key in permissions

    def set_user_permission(self, user_id: int, page_key: str, granted: bool) -> bool:
        """Установить индивидуальное право пользователя"""
        conn = self._get_conn()
        cursor = conn.cursor()

        # Получить permission_id
        cursor.execute("SELECT id FROM permissions WHERE page_key = ?", (page_key,))
        row = cursor.fetchone()

        if not row:
            conn.close()
            return False

        permission_id = row[0]

        cursor.execute(
            """
        INSERT OR REPLACE INTO user_permissions (user_id, permission_id, granted)
        VALUES (?, ?, ?)
        """,
            (user_id, permission_id, granted),
        )

        conn.commit()
        conn.close()
        return True

    def remove_user_permission(self, user_id: int, page_key: str) -> bool:
        """Удалить индивидуальное право (вернуться к правам роли)"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            """
        DELETE FROM user_permissions
        WHERE user_id = ? AND permission_id = (
            SELECT id FROM permissions WHERE page_key = ?
        )
        """,
            (user_id, page_key),
        )

        affected = cursor.rowcount
        conn.commit()
        conn.close()

        return affected > 0

    # ==================== ROLES ====================

    def get_all_roles(self) -> List[Dict[str, Any]]:
        """Получить все роли"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM roles ORDER BY name")
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_role_permissions(self, role_id: int) -> List[str]:
        """Получить права роли"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            """
        SELECT p.page_key
        FROM role_permissions rp
        JOIN permissions p ON rp.permission_id = p.id
        WHERE rp.role_id = ?
        """,
            (role_id,),
        )

        rows = cursor.fetchall()
        conn.close()

        return [row[0] for row in rows]

    def set_role_permissions(self, role_id: int, page_keys: List[str]) -> bool:
        """Установить права роли (замещает существующие)"""
        conn = self._get_conn()
        cursor = conn.cursor()

        # Удалить старые права
        cursor.execute("DELETE FROM role_permissions WHERE role_id = ?", (role_id,))

        # Добавить новые
        for page_key in page_keys:
            cursor.execute(
                """
            INSERT INTO role_permissions (role_id, permission_id)
            SELECT ?, id FROM permissions WHERE page_key = ?
            """,
                (role_id, page_key),
            )

        conn.commit()
        conn.close()
        return True

    # ==================== PERMISSIONS (PAGES) ====================

    def get_all_permissions(self) -> List[Dict[str, Any]]:
        """Получить все доступные права (страницы)"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM permissions ORDER BY page_name")
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    # ==================== SETTINGS ====================

    def get_setting(self, key: str) -> Optional[str]:
        """Получить значение настройки"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT setting_value FROM system_settings WHERE setting_key = ?", (key,)
        )
        row = cursor.fetchone()
        conn.close()

        return row[0] if row else None

    def set_setting(self, key: str, value: str) -> bool:
        """Установить значение настройки"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            """
        UPDATE system_settings 
        SET setting_value = ?, updated_at = ?
        WHERE setting_key = ?
        """,
            (value, datetime.now(), key),
        )

        affected = cursor.rowcount
        conn.commit()
        conn.close()

        return affected > 0

    def get_all_settings(self) -> List[Dict[str, Any]]:
        """Получить все настройки"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM system_settings ORDER BY setting_key")
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]


# Глобальный экземпляр
user_manager = UserManager()
