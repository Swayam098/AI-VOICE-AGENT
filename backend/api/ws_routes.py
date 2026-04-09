"""
WebSocket Routes for Live Dashboard Updates.
Provides real-time system status and call monitoring.
"""

import asyncio
import json
import time

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger


class LiveStatusManager:
    """
    Manages WebSocket connections for live dashboard updates.
    Broadcasts system status, active calls, and routing events.
    """

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept and register a new dashboard WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Dashboard WS connected — {len(self.active_connections)} active")

    def disconnect(self, websocket: WebSocket):
        """Remove a disconnected WebSocket."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Dashboard WS disconnected — {len(self.active_connections)} active")

    async def broadcast(self, message: dict):
        """Broadcast a message to all connected dashboard clients."""
        if not self.active_connections:
            return

        dead = []
        payload = json.dumps(message)

        for ws in self.active_connections:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)

        for ws in dead:
            self.disconnect(ws)

    async def broadcast_status(self, components: dict):
        """Broadcast current system status."""
        media_handler = components.get("media_handler")
        router_engine = components.get("router")

        status = {
            "type": "status_update",
            "timestamp": time.time(),
            "active_calls": len(media_handler.get_active_sessions()) if media_handler else 0,
            "sessions": media_handler.get_active_sessions() if media_handler else [],
            "routing_stats": router_engine.get_stats() if router_engine else {},
        }

        await self.broadcast(status)

    async def broadcast_routing_event(self, query: str, route: str, latency_ms: int):
        """Broadcast a routing event to dashboard."""
        await self.broadcast({
            "type": "routing_event",
            "timestamp": time.time(),
            "query_preview": query[:100],
            "route": route,
            "latency_ms": latency_ms,
        })

    async def broadcast_call_event(self, event: str, session_id: str, data: dict = None):
        """Broadcast a call lifecycle event."""
        await self.broadcast({
            "type": "call_event",
            "timestamp": time.time(),
            "event": event,  # 'started', 'message', 'ended'
            "session_id": session_id,
            "data": data or {},
        })

    async def status_loop(self, components: dict, interval: float = 5.0):
        """Background loop that periodically broadcasts status updates."""
        while True:
            try:
                if self.active_connections:
                    await self.broadcast_status(components)
            except Exception as e:
                logger.error(f"Status broadcast error: {e}")

            await asyncio.sleep(interval)


# Global instance
live_status = LiveStatusManager()
