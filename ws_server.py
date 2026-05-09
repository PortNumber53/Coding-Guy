#!/usr/bin/env python3
"""WebSocket server for broadcasting Coding Guy agent activity to the frontend.

This module provides a standalone asyncio WebSocket server using the `websockets`
library. It manages client connections and broadcasts agent activity events
in real-time, including:
  - Agent thinking / text streaming
  - Tool calls and their results
  - Conversation round progress
  - Status changes (complete, error, blocked, max_rounds)
  - Session lifecycle events

Usage:
    python ws_server.py [--port 8765] [--host 0.0.0.0]

Integration with coding_agent.py:
    The server exposes an `ActivityBroadcaster` singleton that the agent loop
    calls via an `activity_callback` parameter.
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional, Set

try:
    import websockets
    from websockets.server import serve
except ImportError:
    print("Error: 'websockets' package not found. Install with: pip install websockets", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

EVENT_THINKING = "thinking"           # Agent is generating text
EVENT_TOOL_CALL = "tool_call"         # Agent is calling a tool
EVENT_TOOL_RESULT = "tool_result"     # Tool execution result
EVENT_TEXT_CHUNK = "text_chunk"       # Streaming text delta from LLM
EVENT_STATUS = "status"               # Agent loop status change
EVENT_SESSION_START = "session_start" # New conversation session started
EVENT_SESSION_END = "session_end"     # Conversation session ended
EVENT_ERROR = "error"                 # Error occurred
EVENT_CONNECTED = "connected"         # Client connected (server -> client)
EVENT_PING = "ping"                   # Keepalive ping


# ---------------------------------------------------------------------------
# Activity Broadcaster (singleton used by coding_agent)
# ---------------------------------------------------------------------------

class ActivityBroadcaster:
    """Broadcasts agent activity events to all connected WebSocket clients.

    Also maintains a rolling history of recent events so newly connected
    clients can catch up.
    """

    MAX_HISTORY = 500  # Keep last N events

    def __init__(self):
        self._clients: Set[Any] = set()
        self._history: Deque[dict] = deque(maxlen=self.MAX_HISTORY)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def add_client(self, websocket):
        self._clients.add(websocket)

    def remove_client(self, websocket):
        self._clients.discard(websocket)

    def add_to_history(self, event: dict):
        self._history.append(event)

    def get_history(self) -> list:
        return list(self._history)

    async def broadcast(self, event_type: str, data: dict, meta: Optional[dict] = None):
        """Broadcast an event to all connected clients."""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": time.time(),
        }
        if meta:
            event["meta"] = meta

        self.add_to_history(event)
        message = json.dumps(event)

        if self._clients:
            # Send to all clients concurrently, remove any that fail
            clients = list(self._clients)
            results = await asyncio.gather(*(c.send(message) for c in clients), return_exceptions=True)
            for client, result in zip(clients, results):
                if isinstance(result, Exception):
                    self._clients.discard(client)

    def broadcast_sync(self, event_type: str, data: dict, meta: Optional[dict] = None):
        """Synchronous wrapper: schedules broadcast on the event loop."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(event_type, data, meta), self._loop)
        else:
            # No running loop – just add to history
            event = {
                "type": event_type,
                "data": data,
                "timestamp": time.time(),
            }
            if meta:
                event["meta"] = meta
            self.add_to_history(event)


# Module-level singleton
_broadcaster: Optional[ActivityBroadcaster] = None


def get_broadcaster() -> ActivityBroadcaster:
    global _broadcaster
    if _broadcaster is None:
        _broadcaster = ActivityBroadcaster()
    return _broadcaster


# ---------------------------------------------------------------------------
# Make activity callback for agent_loop
# ---------------------------------------------------------------------------

def make_activity_callback(broadcaster: Optional[ActivityBroadcaster] = None) -> Callable:
    """Create a callback function suitable for passing to agent_loop's
    `activity_callback` parameter.

    The callback signature is: ``callback(event_type, data, meta=None)``
    """
    if broadcaster is None:
        broadcaster = get_broadcaster()

    def callback(event_type: str, data: dict, meta: Optional[dict] = None):
        broadcaster.broadcast_sync(event_type, data, meta)

    return callback


# ---------------------------------------------------------------------------
# WebSocket server handler
# ---------------------------------------------------------------------------

async def client_handler(websocket, path=None):
    """Handle a single WebSocket client connection."""
    broadcaster = get_broadcaster()
    broadcaster.add_client(websocket)

    # Send connection confirmation with history
    connected_event = {
        "type": EVENT_CONNECTED,
        "data": {
            "message": "Connected to Coding Guy agent activity feed",
            "client_count": broadcaster.client_count,
        },
        "timestamp": time.time(),
    }
    await websocket.send(json.dumps(connected_event))

    # Send recent history
    history = broadcaster.get_history()
    if history:
        history_event = {
            "type": "history",
            "data": {"events": history, "total": len(history)},
            "timestamp": time.time(),
        }
        await websocket.send(json.dumps(history_event))

    try:
        # Listen for client messages (e.g., ping, task submission)
        async for raw_message in websocket:
            try:
                msg = json.loads(raw_message)
                msg_type = msg.get("type", "")

                if msg_type == "ping":
                    await websocket.send(json.dumps({
                        "type": "pong",
                        "data": {"timestamp": time.time()},
                        "timestamp": time.time(),
                    }))
                elif msg_type == "request_history":
                    history = broadcaster.get_history()
                    await websocket.send(json.dumps({
                        "type": "history",
                        "data": {"events": history, "total": len(history)},
                        "timestamp": time.time(),
                    }))

            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "type": "error",
                    "data": {"message": "Invalid JSON"},
                    "timestamp": time.time(),
                }))
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        broadcaster.remove_client(websocket)


async def start_server(host: str = "0.0.0.0", port: int = 8765):
    """Start the WebSocket server."""
    print(f"WebSocket server starting on ws://{host}:{port}", file=sys.stderr)
    broadcaster = get_broadcaster()
    broadcaster._loop = asyncio.get_running_loop()

    async with serve(client_handler, host, port):
        await asyncio.Future()  # Run forever


def run_ws_server(host: str = "0.0.0.0", port: int = 8765):
    """Entry point: start the WebSocket server (blocking)."""
    asyncio.run(start_server(host, port))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WebSocket server for Coding Guy agent activity")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    args = parser.parse_args()
    run_ws_server(args.host, args.port)


if __name__ == "__main__":
    main()
