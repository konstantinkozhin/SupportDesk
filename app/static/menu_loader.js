/**
 * Универсальный загрузчик меню на основе прав доступа
 * Предотвращает мигание недоступных пунктов меню
 * Использует Material Icons вместо эмодзи
 */

const MENU_ITEMS = [
    { id: 'tickets', name: 'Заявки', icon: 'description', url: '/tickets', permission: 'tickets' },
    { id: 'dashboard', name: 'Дашборд', icon: 'dashboard', url: '/dashboard', permission: 'dashboard' },
    { id: 'knowledge', name: 'База знаний', icon: 'menu_book', url: '/admin/knowledge', permission: 'knowledge' },
    { id: 'simulator', name: 'Симулятор', icon: 'smart_toy', url: '/simulator', permission: 'simulator' },
    { id: 'admin_users', name: 'Пользователи', icon: 'group', url: '/admin/users', permission: 'admin' }
];

/**
 * Загружает меню навигации на основе прав пользователя
 */
async function loadNavigationMenu() {
    try {
        // Получаем доступные страницы
        const response = await fetch('/api/permissions');
        if (!response.ok) {
            console.error('Failed to load permissions');
            return;
        }
        
        const permissions = await response.json();
        const availablePages = permissions.available_pages || [];
        
        console.log('User permissions:', availablePages);
        
        // Сохраняем в глобальную область для notification.js
        window.userPermissions = availablePages;

        // Находим контейнер меню
        const menuContainer = document.getElementById('mainNav');
        if (!menuContainer) {
            console.warn('Menu container #mainNav not found');
            return;
        }

        // Очищаем меню
        menuContainer.innerHTML = '';

        // Формируем меню только из доступных пунктов
        MENU_ITEMS.forEach(item => {
            if (availablePages.includes(item.permission)) {
                const link = document.createElement('a');
                link.href = item.url;
                link.className = 'main-nav-btn';
                
                // Отмечаем текущую страницу как активную
                if (window.location.pathname === item.url) {
                    link.classList.add('active');
                }
                
                // Используем Material Icons вместо эмодзи
                link.innerHTML = `
                    <span class="material-icons">${item.icon}</span>
                    <span class="btn-text">${item.name}</span>
                `;
                menuContainer.appendChild(link);
            }
        });
    } catch (error) {
        console.error('Error loading menu:', error);
    }
}

// Автоматически загружаем меню при загрузке страницы
document.addEventListener('DOMContentLoaded', loadNavigationMenu);
