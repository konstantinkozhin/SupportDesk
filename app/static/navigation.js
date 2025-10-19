/**
 * Общий скрипт для управления навигацией на основе прав пользователя
 */

async function loadUserPermissions() {
    try {
        const response = await fetch('/api/user/permissions');
        if (!response.ok) {
            console.warn('Не удалось загрузить права пользователя');
            return [];
        }
        const data = await response.json();
        return data.permissions || [];
    } catch (error) {
        console.error('Ошибка загрузки прав:', error);
        return [];
    }
}

// Скрыть пункты меню, к которым нет доступа
function filterNavigationByPermissions(permissions) {
    const menuItems = {
        tickets: 'a[href="/"]',
        dashboard: 'a[href="/dashboard"]',
        knowledge: 'a[href="/admin/knowledge"]',
        simulator: 'a[href="/simulator"]',
        admin: 'a[href="/admin/users"]'
    };

    Object.entries(menuItems).forEach(([permission, selector]) => {
        const menuItem = document.querySelector(`.main-nav ${selector}`);
        if (!menuItem) {
            return;
        }

        if (permissions.includes(permission)) {
            menuItem.style.display = 'flex';
        } else {
            menuItem.style.display = 'none';
        }
    });
}

document.addEventListener('DOMContentLoaded', async () => {
    const permissions = await loadUserPermissions();
    filterNavigationByPermissions(permissions);
});
