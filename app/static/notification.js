/**
 * Глобальная система уведомлений для всех страниц
 * Воспроизводит звук при новых сообщениях
 */

(function() {
    console.log('🔔 Global notification system initializing...');
    
    // Инициализация аудио
    let audioContext = null;
    let notificationBuffer = null;
    let isAudioInitialized = false;
    
    function initAudio() {
        if (isAudioInitialized) return;
        
        try {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Создаем звук программно
            const duration = 0.15;
            const sampleRate = audioContext.sampleRate;
            const buffer = audioContext.createBuffer(1, duration * sampleRate, sampleRate);
            const data = buffer.getChannelData(0);
            
            const frequency = 800;
            for (let i = 0; i < buffer.length; i++) {
                const t = i / sampleRate;
                data[i] = Math.sin(2 * Math.PI * frequency * t) * Math.exp(-t * 8);
            }
            
            notificationBuffer = buffer;
            isAudioInitialized = true;
            console.log('Global notification audio initialized');
        } catch (e) {
            console.error('Audio initialization failed:', e);
        }
    }
    
    function playNotificationSound() {
        console.log('🔊 playNotificationSound() called');
        console.log('🔊 Current state:', {
            audioContext: !!audioContext,
            audioContextState: audioContext?.state,
            notificationBuffer: !!notificationBuffer,
            isAudioInitialized: isAudioInitialized
        });
        
        try {
            // Инициализируем если еще не инициализировано
            if (!audioContext || !notificationBuffer) {
                console.log('⚠️ Audio not initialized, initializing now...');
                initAudio();
            }
            
            // Проверяем что AudioContext в рабочем состоянии
            if (audioContext && audioContext.state === 'suspended') {
                console.log('⚠️ AudioContext suspended, resuming...');
                audioContext.resume().then(() => {
                    console.log('✅ AudioContext resumed, retrying...');
                    playNotificationSound();
                });
                return;
            }
            
            if (audioContext && notificationBuffer) {
                console.log('🎵 Creating sound source...');
                const source = audioContext.createBufferSource();
                source.buffer = notificationBuffer;
                
                const gainNode = audioContext.createGain();
                gainNode.gain.value = 0.5; // Громкость 50%
                
                source.connect(gainNode);
                gainNode.connect(audioContext.destination);
                
                source.onended = () => {
                    console.log('✅ Sound finished playing');
                };
                
                source.start(0);
                console.log('🔔 Notification sound STARTED!');
            } else {
                console.error('❌ Audio not ready:', {
                    audioContext: !!audioContext,
                    notificationBuffer: !!notificationBuffer
                });
            }
        } catch (e) {
            console.error('❌ Sound play failed:', e);
        }
    }
    
    // Инициализация при первом взаимодействии с любым элементом
    const initOnInteraction = () => {
        initAudio();
        if (audioContext && audioContext.state === 'suspended') {
            audioContext.resume();
        }
    };
    
    document.addEventListener('click', initOnInteraction, { once: true });
    document.addEventListener('keydown', initOnInteraction, { once: true });
    
    // Подключаемся к WebSocket для глобальных уведомлений
    let globalSocket = null;
    let lastUnreadCounts = {}; // Храним предыдущие значения непрочитанных
    
    function connectGlobalNotifications() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/conversations`;
        
        globalSocket = new WebSocket(wsUrl);
        
        globalSocket.onopen = () => {
            console.log('✅ Global notifications WebSocket connected');
        };
        
        globalSocket.onmessage = (event) => {
            try {
                const payload = JSON.parse(event.data);
                const currentPath = window.location.pathname;
                const currentConvId = currentPath.match(/\/tickets\/(\d+)/)?.[1];
                
                // Проверяем глобальную переменную activeConversationId
                const activeConvId = window.getActiveConversationId ? window.getActiveConversationId() : null;
                
                console.log('📦 WebSocket received:', {
                    type: payload.type,
                    conversationsCount: payload.conversations?.length,
                    currentPath: currentPath,
                    currentConvId: currentConvId,
                    activeConvId: activeConvId,
                    isInConversationPage: currentPath.includes('/tickets/') && currentConvId,
                    timestamp: new Date().toISOString()
                });
                
                // Обрабатываем обновления списка заявок
                if (payload.type === 'conversations' && payload.conversations) {
                    let shouldPlaySound = false;
                    let changedConversations = [];
                    
                    console.log('📊 Processing conversations:', payload.conversations.map(c => ({
                        id: c.id,
                        unread: c.unread_count,
                        prevUnread: lastUnreadCounts[c.id]
                    })));
                    
                    payload.conversations.forEach(conversation => {
                        const prevCount = lastUnreadCounts[conversation.id];
                        const newCount = conversation.unread_count || 0;
                        
                        console.log(`📍 Conv #${conversation.id}: prev=${prevCount}, new=${newCount}, defined=${prevCount !== undefined}`);
                        
                        // Звук не должен играть если пользователь реально находится в чате
                        const isInThisChatByUrl = currentConvId && String(conversation.id) === currentConvId;
                        const isInThisChatByVar = activeConvId && String(conversation.id) === String(activeConvId);
                        const isInThisChat = isInThisChatByUrl || isInThisChatByVar;
                        const isStrictlyInChat = Boolean(isInThisChatByUrl && isInThisChatByVar);
                        // Временное подавление при клике на карточку
                        const suppressedId = window.suppressNotificationFor || null;
                        const isSuppressed = suppressedId && String(suppressedId) === String(conversation.id);
                        if (isSuppressed) {
                            console.log(`� Suppressed notifications for Conv #${conversation.id} due to recent open`);
                        }

                        console.log(`�📍 Chat check for Conv #${conversation.id}:`, {
                            currentPath: currentPath,
                            currentConvId: currentConvId,
                            activeConvId: activeConvId,
                            isInThisChatByUrl: isInThisChatByUrl,
                            isInThisChatByVar: isInThisChatByVar,
                            isInThisChat: isInThisChat,
                            isStrictlyInChat: isStrictlyInChat,
                            onListPage: !currentConvId && !activeConvId
                        });

                        // Воспроизводим звук только если:
                        // 1. Мы не в этом конкретном чате (ни по URL, ни по переменной)
                        // 2. Мы уже видели эту заявку раньше (prevCount !== undefined)
                        // 3. Счетчик увеличился (новое сообщение от пользователя)
                        if (!isStrictlyInChat && !isSuppressed && prevCount !== undefined && newCount > prevCount) {
                            console.log(`✅ TRIGGER: Conversation #${conversation.id}: ${prevCount} -> ${newCount} (sound will play)`);
                            changedConversations.push(conversation.id);
                            shouldPlaySound = true;
                        } else if ((isStrictlyInChat || isSuppressed) && newCount > prevCount) {
                            console.log(`🔇 MUTED: Strictly in active chat #${conversation.id}, NOT playing sound`);
                        } else if (!isStrictlyInChat && prevCount === undefined) {
                            console.log(`⏭️ SKIP: First load for Conv #${conversation.id}, not playing sound`);
                        }
                        
                        // Обновляем счетчик для этой заявки
                        lastUnreadCounts[conversation.id] = newCount;
                    });
                    
                    if (shouldPlaySound) {
                        console.log(`🔊 ATTEMPTING TO PLAY SOUND for: ${changedConversations.join(', ')}`);
                        console.log(`🔊 AudioContext state: ${audioContext?.state}, Buffer ready: ${!!notificationBuffer}`);
                        playNotificationSound();
                    } else {
                        console.log('❌ Sound not triggered - no conditions met');
                    }
                }
                
                // Также обрабатываем прямые сообщения из чата (если пришли)
                if (payload.type === 'message' && payload.message) {
                    const currentPath = window.location.pathname;
                    const messageConvId = payload.conversation_id || payload.message.conversation_id;
                    const currentConvId = currentPath.match(/\/tickets\/(\d+)/)?.[1];
                    const activeConvId = window.getActiveConversationId ? window.getActiveConversationId() : null;
                    const isFromUser = payload.message.sender === 'user' || payload.message.sender === 'bot';
                    
                    // КРИТИЧЕСКАЯ ПРОВЕРКА: используем обе проверки для надежности
                    const isInThisChatByUrl = currentConvId && String(messageConvId) === currentConvId;
                    const isInThisChatByVar = activeConvId && String(messageConvId) === String(activeConvId);
                    const isInThisChat = isInThisChatByUrl || isInThisChatByVar;
                    const suppressedIdMsg = window.suppressNotificationFor || null;
                    const isSuppressedMsg = suppressedIdMsg && String(suppressedIdMsg) === String(messageConvId);
                    if (isSuppressedMsg) {
                        console.log(`🔕 Suppressed direct-message notifications for Conv #${messageConvId} due to recent open`);
                    }
                    
                    console.log('📨 Direct message received:', {
                        sender: payload.message.sender,
                        convId: messageConvId,
                        currentConvId: currentConvId,
                        activeConvId: activeConvId,
                        isInThisChatByUrl: isInThisChatByUrl,
                        isInThisChatByVar: isInThisChatByVar,
                        isInThisChat: isInThisChat,
                        isFromUser: isFromUser
                    });
                    
                    // Воспроизводим звук только если:
                    // 1. Сообщение от пользователя/бота
                    // 2. Мы не в этом конкретном чате
                    if (isFromUser && !isInThisChat && !isSuppressedMsg) {
                        console.log('🔔 Playing sound: new message from', payload.message.sender, 'in conversation', messageConvId);
                        playNotificationSound();
                    } else if (isFromUser && isInThisChat) {
                        console.log('🔇 MUTED: Direct message in active chat, NOT playing sound');
                    }
                }
            } catch (e) {
                console.error('Failed to parse WebSocket message:', e);
            }
        };
        
        globalSocket.onclose = () => {
            console.log('❌ Global notifications WebSocket closed, reconnecting...');
            setTimeout(connectGlobalNotifications, 3000);
        };
        
        globalSocket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
    
    // Запускаем подключение
    connectGlobalNotifications();
    
    // Экспортируем в глобальную область для ручного управления и отладки
    window.globalNotifications = {
        playSound: playNotificationSound,
        reconnect: connectGlobalNotifications,
        testSound: () => {
            console.log('🧪 Manual test triggered');
            initAudio();
            playNotificationSound();
        },
        getState: () => ({
            audioContext: !!audioContext,
            audioContextState: audioContext?.state,
            notificationBuffer: !!notificationBuffer,
            isAudioInitialized: isAudioInitialized,
            lastUnreadCounts: lastUnreadCounts,
            websocketConnected: globalSocket?.readyState === WebSocket.OPEN
        })
    };
    
    console.log('💡 Debug commands available:');
    console.log('  window.globalNotifications.testSound() - test notification sound');
    console.log('  window.globalNotifications.getState() - get current state');
})();
