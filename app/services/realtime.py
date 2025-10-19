from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Iterable

from fastapi import WebSocket
from starlette.websockets import WebSocketState


class ConnectionManager:
    def __init__(self) -> None:
        self._conversation_clients: set[WebSocket] = set()
        self._chat_clients: dict[int, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def register_conversations(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._conversation_clients.add(websocket)

    async def unregister_conversations(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._conversation_clients.discard(websocket)

    async def register_chat(self, conversation_id: int, websocket: WebSocket) -> None:
        async with self._lock:
            self._chat_clients[conversation_id].add(websocket)

    async def unregister_chat(self, conversation_id: int, websocket: WebSocket) -> None:
        async with self._lock:
            sockets = self._chat_clients.get(conversation_id)
            if sockets is None:
                return
            sockets.discard(websocket)
            if not sockets:
                self._chat_clients.pop(conversation_id, None)

    def has_active_chat_connections(self, conversation_id: int) -> bool:
        """Проверяет, есть ли активные подключения к данному чату"""
        return (
            conversation_id in self._chat_clients
            and len(self._chat_clients[conversation_id]) > 0
        )

    async def send_conversations_snapshot(
        self, websocket: WebSocket, conversations: list[dict]
    ) -> None:
        await self._safe_send(
            websocket, {"type": "conversations", "conversations": conversations}
        )

    async def broadcast_conversations(self, conversations: list[dict]) -> None:
        payload = {"type": "conversations", "conversations": conversations}
        async with self._lock:
            recipients = list(self._conversation_clients)
        await self._broadcast(recipients, payload, conversation_id=None)

    async def send_message_history(
        self, websocket: WebSocket, conversation_id: int, messages: list[dict]
    ) -> None:
        await self._safe_send(
            websocket,
            {
                "type": "history",
                "conversation_id": conversation_id,
                "messages": messages,
            },
        )

    async def broadcast_message(self, conversation_id: int, message: dict) -> None:
        payload = {
            "type": "message",
            "conversation_id": conversation_id,
            "message": message,
        }
        async with self._lock:
            recipients = list(self._chat_clients.get(conversation_id, set()))
        await self._broadcast(recipients, payload, conversation_id=conversation_id)

    async def close_all(self) -> None:
        async with self._lock:
            conversation_clients = list(self._conversation_clients)
            chat_clients = {
                cid: list(clients) for cid, clients in self._chat_clients.items()
            }
            self._conversation_clients.clear()
            self._chat_clients.clear()
        for websocket in conversation_clients:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()
        for sockets in chat_clients.values():
            for websocket in sockets:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.close()

    async def _broadcast(
        self,
        websockets: Iterable[WebSocket],
        payload: dict,
        conversation_id: int | None,
    ) -> None:
        stale: list[WebSocket] = []
        for websocket in list(websockets):
            if websocket.client_state != WebSocketState.CONNECTED:
                stale.append(websocket)
                continue
            try:
                await websocket.send_json(payload)
            except Exception:
                stale.append(websocket)
        if stale:
            async with self._lock:
                for websocket in stale:
                    if conversation_id is None:
                        self._conversation_clients.discard(websocket)
                    else:
                        sockets = self._chat_clients.get(conversation_id)
                        if sockets is None:
                            continue
                        sockets.discard(websocket)
                        if not sockets:
                            self._chat_clients.pop(conversation_id, None)

    async def _safe_send(self, websocket: WebSocket, payload: dict) -> None:
        if websocket.client_state != WebSocketState.CONNECTED:
            return
        try:
            await websocket.send_json(payload)
        except Exception:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()
