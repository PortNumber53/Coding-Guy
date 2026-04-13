"""Memory session management for the coding agent.

Each chat/user gets a default memory session. New sessions are automatically
started on the first message. Sessions have UUIDs and can be renamed to
friendlier names.
"""

import json
import logging
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, List
from collections import defaultdict

from settings_db import get_settings_db, Setting

logger = logging.getLogger(__name__)

CATEGORY_MEMORY = "memory"
CATEGORY_MEMORY_SESSION = "memory_session"

# Prefixes for settings keys
MEMORY_ACTIVE_PREFIX = "memory.active."  # memory.active.<chat_id> = session_uuid
MEMORY_SESSION_PREFIX = "memory.session."  # memory.session.<uuid> = session_data
MEMORY_INDEX_PREFIX = "memory.index."        # memory.index.<chat_id> = [uuid1, uuid2, ...]


@dataclass
class MemorySession:
    """Represents a memory session."""
    uuid: str
    chat_id: str
    name: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    message_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MemorySession":
        return cls(**data)

    @property
    def display_name(self) -> str:
        """Return the friendly name or a shortened UUID."""
        if self.name:
            return self.name
        return self.uuid[:8]


class MemoryManager:
    """Manages memory sessions for chats/users."""

    def __init__(self):
        self.db = get_settings_db()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _get_active_session_key(self, chat_id: str) -> str:
        return f"{MEMORY_ACTIVE_PREFIX}{chat_id}"

    def _get_session_data_key(self, session_uuid: str) -> str:
        return f"{MEMORY_SESSION_PREFIX}{session_uuid}"

    def _get_index_key(self, chat_id: str) -> str:
        return f"{MEMORY_INDEX_PREFIX}{chat_id}"
    
    def _add_to_chat_index(self, chat_id: str, session_uuid: str):
        """Add a session UUID to the chat's index."""
        index_key = self._get_index_key(chat_id)
        index = self.db.get(index_key) or []
        if session_uuid not in index:
            index.append(session_uuid)
            self.db.set(
                index_key,
                index,
                value_type="json",
                category=CATEGORY_MEMORY,
                description=f"Session index for chat {chat_id}"
            )
    
    def _remove_from_chat_index(self, chat_id: str, session_uuid: str):
        """Remove a session UUID from the chat's index."""
        index_key = self._get_index_key(chat_id)
        index = self.db.get(index_key) or []
        if session_uuid in index:
            index.remove(session_uuid)
            if index:
                self.db.set(
                    index_key,
                    index,
                    value_type="json",
                    category=CATEGORY_MEMORY,
                    description=f"Session index for chat {chat_id}"
                )
            else:
                self.db.delete(index_key)

    def get_active_session(self, chat_id: str) -> Optional[MemorySession]:
        """Get the currently active session for a chat."""
        active_uuid = self.db.get(self._get_active_session_key(chat_id))
        if not active_uuid:
            return None
        return self.get_session(active_uuid)

    def get_session(self, session_uuid: str) -> Optional[MemorySession]:
        """Get a session by UUID."""
        data = self.db.get(self._get_session_data_key(session_uuid))
        if not data:
            return None
        if isinstance(data, str):
            data = json.loads(data)
        return MemorySession.from_dict(data)

    def create_session(self, chat_id: str, name: Optional[str] = None) -> MemorySession:
        """Create a new memory session for a chat.
        
        Automatically sets it as the active session.
        """
        session_uuid = str(uuid.uuid4())
        now = self._now()
        
        session = MemorySession(
            uuid=session_uuid,
            chat_id=chat_id,
            name=name,
            created_at=now,
            updated_at=now,
            message_count=0,
        )

        # Store session data
        self.db.set(
            self._get_session_data_key(session_uuid),
            session.to_dict(),
            value_type="json",
            category=CATEGORY_MEMORY_SESSION,
            description=f"Memory session for chat {chat_id}"
        )

        # Set as active session
        self.db.set(
            self._get_active_session_key(chat_id),
            session_uuid,
            value_type="string",
            category=CATEGORY_MEMORY,
            description=f"Active memory session for chat {chat_id}"
        )

        # Add to chat's session index
        self._add_to_chat_index(chat_id, session_uuid)

        logger.info(f"Created memory session {session_uuid[:8]} for chat {chat_id}")
        return session

    def get_or_create_session(self, chat_id: str, auto_create: bool = True) -> Optional[MemorySession]:
        """Get active session or create a new one."""
        session = self.get_active_session(chat_id)
        if session:
            return session
        if auto_create:
            return self.create_session(chat_id)
        return None

    def switch_session(self, chat_id: str, session_uuid: str) -> bool:
        """Switch to an existing session by UUID."""
        session = self.get_session(session_uuid)
        if not session:
            return False

        # Update active session
        self.db.set(
            self._get_active_session_key(chat_id),
            session_uuid,
            value_type="string",
            category=CATEGORY_MEMORY,
            description=f"Active memory session for chat {chat_id}"
        )

        # Update session's chat_id if it was from another chat
        if session.chat_id != chat_id:
            old_chat_id = session.chat_id
            session.chat_id = chat_id
            session.updated_at = self._now()
            self.db.set(
                self._get_session_data_key(session_uuid),
                session.to_dict(),
                value_type="json",
                category=CATEGORY_MEMORY_SESSION,
                description=f"Memory session for chat {chat_id}"
            )
            # Update the chat indexes
            self._remove_from_chat_index(old_chat_id, session_uuid)
            self._add_to_chat_index(chat_id, session_uuid)

        logger.info(f"Switched chat {chat_id} to session {session_uuid[:8]}")
        return True

    def rename_session(self, session_uuid: str, new_name: str) -> bool:
        """Rename a session."""
        session = self.get_session(session_uuid)
        if not session:
            return False

        session.name = new_name
        session.updated_at = self._now()

        self.db.set(
            self._get_session_data_key(session_uuid),
            session.to_dict(),
            value_type="json",
            category=CATEGORY_MEMORY_SESSION,
            description=f"Memory session for chat {session.chat_id}"
        )

        logger.info(f"Renamed session {session_uuid[:8]} to '{new_name}'")
        return True

    def delete_session(self, session_uuid: str) -> bool:
        """Delete a session permanently."""
        session = self.get_session(session_uuid)
        if not session:
            return False

        chat_id = session.chat_id

        # Delete session data
        self.db.delete(self._get_session_data_key(session_uuid))

        # Remove from chat's session index
        self._remove_from_chat_index(chat_id, session_uuid)

        # If this was the active session for the chat, clear it
        active_key = self._get_active_session_key(chat_id)
        current_active = self.db.get(active_key)
        if current_active == session_uuid:
            self.db.delete(active_key)

        logger.info(f"Deleted session {session_uuid[:8]}")
        return True

    def list_sessions(self, chat_id: Optional[str] = None) -> List[MemorySession]:
        """List all sessions, optionally filtered by chat_id.
        
        Uses a per-chat index for efficient lookups when filtering by chat_id.
        """
        if chat_id is not None:
            # Use the chat's session index for efficient lookup
            index_key = self._get_index_key(chat_id)
            session_uuids = self.db.get(index_key) or []
            sessions = []
            for uuid in session_uuids:
                session = self.get_session(uuid)
                if session:
                    sessions.append(session)
            return sorted(sessions, key=lambda s: s.updated_at or s.created_at, reverse=True)
        else:
            # Get all session settings when no chat_id filter
            all_settings = self.db.get_all_settings(category=CATEGORY_MEMORY_SESSION)
            sessions = []
            for setting in all_settings:
                try:
                    if isinstance(setting.value, str):
                        data = json.loads(setting.value)
                    else:
                        data = setting.value
                    session = MemorySession.from_dict(data)
                    sessions.append(session)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse session data: {e}")
                    continue
            return sorted(sessions, key=lambda s: s.updated_at or s.created_at, reverse=True)

    def update_session_stats(self, session_uuid: str, message_count: int):
        """Update session statistics."""
        session = self.get_session(session_uuid)
        if session:
            session.message_count = message_count
            session.updated_at = self._now()
            self.db.set(
                self._get_session_data_key(session_uuid),
                session.to_dict(),
                value_type="json",
                category=CATEGORY_MEMORY_SESSION,
            )

    def get_session_by_name(self, chat_id: str, name: str) -> Optional[MemorySession]:
        """Find a session by name for a specific chat."""
        sessions = self.list_sessions(chat_id)
        for session in sessions:
            if session.name == name or session.display_name == name:
                return session
        # Also try partial match
        for session in sessions:
            if session.name and name.lower() in session.name.lower():
                return session
            if name.lower() in session.uuid.lower():
                return session
        return None

    def export_session(self, session_uuid: str) -> Optional[str]:
        """Export a session to JSON."""
        session = self.get_session(session_uuid)
        if not session:
            return None
        
        data = {
            "type": "memory_session_export",
            "exported_at": self._now(),
            "session": session.to_dict(),
        }
        return json.dumps(data, indent=2)

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        all_sessions = self.list_sessions()
        by_chat: Dict[str, int] = {}
        for s in all_sessions:
            by_chat[s.chat_id] = by_chat.get(s.chat_id, 0) + 1

        return {
            "total_sessions": len(all_sessions),
            "total_chats": len(by_chat),
            "by_chat": by_chat,
        }


# Global instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get or create the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
