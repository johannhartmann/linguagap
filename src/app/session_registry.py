"""
Session registry for managing WebSocket sessions and viewers.

This module implements a token-based session registry that allows:
    1. Recording sessions to be shared via URL
    2. Read-only viewers to connect before recording starts
    3. Real-time broadcast of updates to all connected viewers

Session lifecycle:
    1. Client generates a token on page load
    2. reserve(token) - Creates pending entry (for viewers connecting early)
    3. activate(token) - Associates session and WebSocket when recording starts
    4. add_viewer(token) - Adds read-only viewer connections
    5. unregister(token) - Cleanup when recording ends

The WeakSet for viewers ensures connections are automatically cleaned up
when WebSocket objects are garbage collected.
"""

import asyncio
import secrets
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from weakref import WeakSet

from fastapi import WebSocket

if TYPE_CHECKING:
    from app.streaming import StreamingSession


@dataclass
class SessionEntry:
    """A registered session with its main WebSocket and viewers."""

    token: str
    session: "StreamingSession | None"  # None when pending (waiting for recording to start)
    main_ws: WebSocket | None  # None when pending
    viewers: WeakSet[WebSocket] = field(default_factory=WeakSet)

    @property
    def is_active(self) -> bool:
        """Session is active when recording has started."""
        return self.session is not None


class SessionRegistry:
    """Registry for active streaming sessions and their viewers."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sessions: dict[str, SessionEntry] = {}

    def generate_token(self) -> str:
        """Generate a unique session token."""
        return secrets.token_urlsafe(24)  # 192 bits, URL-safe

    async def reserve(self, token: str) -> bool:
        """Reserve a token for a pending session. Returns False if token exists."""
        async with self._lock:
            if token in self._sessions:
                return False
            self._sessions[token] = SessionEntry(
                token=token,
                session=None,
                main_ws=None,
            )
            return True

    async def activate(self, token: str, session: "StreamingSession", main_ws: WebSocket) -> bool:
        """Activate a pending session when recording starts. Returns False if token not found."""
        async with self._lock:
            entry = self._sessions.get(token)
            if entry is None:
                # Token wasn't reserved, create new entry
                self._sessions[token] = SessionEntry(
                    token=token,
                    session=session,
                    main_ws=main_ws,
                )
                return True
            # Activate existing pending session
            entry.session = session
            entry.main_ws = main_ws
            return True

    async def get(self, token: str) -> "SessionEntry | None":
        """Get a session entry by token (may be pending or active)."""
        async with self._lock:
            return self._sessions.get(token)

    async def unregister(self, token: str) -> None:
        """Unregister a session."""
        async with self._lock:
            self._sessions.pop(token, None)

    async def add_viewer(self, token: str, viewer_ws: WebSocket) -> bool:
        """Add a viewer to a session. Returns True if successful."""
        async with self._lock:
            entry = self._sessions.get(token)
            if entry is None:
                return False
            entry.viewers.add(viewer_ws)
            return True

    async def remove_viewer(self, token: str, viewer_ws: WebSocket) -> None:
        """Remove a viewer from a session."""
        async with self._lock:
            entry = self._sessions.get(token)
            if entry is not None:
                entry.viewers.discard(viewer_ws)


# Global registry instance
registry = SessionRegistry()
