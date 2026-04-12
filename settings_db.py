#!/usr/bin/env python3
"""SQLite database module for storing application settings."""

import json
import os
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

# Default database path - stored in workspace directory
DEFAULT_DB_PATH = os.path.join(
    os.path.expanduser("~"),
    "coding-guy-workspace",
    "settings.db"
)

# Environment override
DB_PATH = os.getenv("CODING_GUY_SETTINGS_DB", DEFAULT_DB_PATH)


@dataclass
class Setting:
    """Represents a single setting record."""
    key: str
    value: Any
    value_type: str = "string"
    category: str = "general"
    description: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert setting to dictionary."""
        return asdict(self)


class SettingsDatabase:
    """SQLite database manager for application settings."""
    
    def __init__(self, db_path: str = None):
        """Initialize the settings database.
        
        Args:
            db_path: Path to SQLite database file. Uses CODING_GUY_SETTINGS_DB env var or default.
        """
        self.db_path = db_path or DB_PATH
        self._ensure_directory()
        self._init_db()
    
    def _ensure_directory(self):
        """Ensure the database directory exists."""
        directory = os.path.dirname(self.db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    value_type TEXT DEFAULT 'string',
                    category TEXT DEFAULT 'general',
                    description TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS settings_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT NOT NULL,
                    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_settings_category ON settings(category)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_settings_history_key ON settings_history(key)
            """)
            conn.commit()
    
    def _serialize_value(self, value: Any, value_type: str) -> str:
        """Serialize a value based on its type."""
        if value_type == "json":
            return json.dumps(value)
        elif value_type == "boolean":
            return "1" if value else "0"
        elif value_type == "integer":
            return str(int(value))
        elif value_type == "float":
            return str(float(value))
        else:
            return str(value)
    
    def _deserialize_value(self, value: str, value_type: str) -> Any:
        """Deserialize a value based on its type."""
        if value_type == "json":
            return json.loads(value)
        elif value_type == "boolean":
            return value == "1" or value.lower() == "true"
        elif value_type == "integer":
            return int(value)
        elif value_type == "float":
            return float(value)
        else:
            return value
    
    def set(self, key: str, value: Any, value_type: str = "string",
            category: str = "general", description: str = "") -> bool:
        """Set or update a setting.
        
        Args:
            key: Unique setting key
            value: Setting value
            value_type: Type of value (string, integer, float, boolean, json)
            category: Grouping category
            description: Description of the setting
            
        Returns:
            True if successful
        """
        serialized = self._serialize_value(value, value_type)
        now = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            # Get old value for history
            old_row = conn.execute(
                "SELECT value FROM settings WHERE key = ?",
                (key,)
            ).fetchone()
            
            old_value = old_row[0] if old_row else None
            
            # Insert or update
            conn.execute("""
                INSERT INTO settings (key, value, value_type, category, description, updated_at)
                VALUES (:key, :value, :value_type, :category, :description, :updated_at)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    value_type = excluded.value_type,
                    category = excluded.category,
                    description = excluded.description,
                    updated_at = excluded.updated_at
            """, {
                "key": key,
                "value": serialized,
                "value_type": value_type,
                "category": category,
                "description": description,
                "updated_at": now
            })
            
            # Record in history
            conn.execute("""
                INSERT INTO settings_history (key, old_value, new_value)
                VALUES (:key, :old_value, :new_value)
            """, {
                "key": key,
                "old_value": old_value,
                "new_value": serialized
            })
            
            conn.commit()
        
        return True
    
    def get(self, key: str, default: Any = None) -> Optional[Any]:
        """Get a setting value.
        
        Args:
            key: Setting key
            default: Default value if not found
            
        Returns:
            Setting value or default
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT value, value_type FROM settings WHERE key = ?",
                (key,)
            ).fetchone()
            
            if row:
                return self._deserialize_value(row["value"], row["value_type"])
            return default
    
    def get_setting(self, key: str) -> Optional[Setting]:
        """Get full setting object including metadata.
        
        Args:
            key: Setting key
            
        Returns:
            Setting object or None
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM settings WHERE key = ?",
                (key,)
            ).fetchone()
            
            if row:
                return Setting(
                    key=row["key"],
                    value=self._deserialize_value(row["value"], row["value_type"]),
                    value_type=row["value_type"],
                    category=row["category"],
                    description=row["description"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"]
                )
            return None
    
    def get_all(self, category: Optional[str] = None) -> Dict[str, Any]:
        """Get all settings, optionally filtered by category.
        
        Args:
            category: Optional category filter
            
        Returns:
            Dictionary of key-value pairs
        """
        with self._get_connection() as conn:
            if category:
                rows = conn.execute(
                    "SELECT key, value, value_type FROM settings WHERE category = ? ORDER BY key",
                    (category,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT key, value, value_type FROM settings ORDER BY category, key"
                ).fetchall()
            
            return {
                row["key"]: self._deserialize_value(row["value"], row["value_type"])
                for row in rows
            }
    
    def get_all_settings(self, category: Optional[str] = None) -> List[Setting]:
        """Get full setting objects.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of Setting objects
        """
        with self._get_connection() as conn:
            if category:
                rows = conn.execute(
                    "SELECT * FROM settings WHERE category = ? ORDER BY key",
                    (category,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM settings ORDER BY category, key"
                ).fetchall()
            
            return [
                Setting(
                    key=row["key"],
                    value=self._deserialize_value(row["value"], row["value_type"]),
                    value_type=row["value_type"],
                    category=row["category"],
                    description=row["description"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"]
                )
                for row in rows
            ]
    
    def delete(self, key: str) -> bool:
        """Delete a setting.
        
        Args:
            key: Setting key to delete
            
        Returns:
            True if setting existed and was deleted
        """
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM settings WHERE key = ?", (key,))
            conn.commit()
            return cursor.rowcount > 0
    
    def get_categories(self) -> List[str]:
        """Get all unique categories.
        
        Returns:
            List of category names
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT category FROM settings ORDER BY category"
            ).fetchall()
            return [row[0] for row in rows]
    
    def get_history(self, key: str, limit: int = 10) -> List[Dict]:
        """Get change history for a setting.
        
        Args:
            key: Setting key
            limit: Maximum number of entries
            
        Returns:
            List of history entries
        """
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT key, old_value, new_value, changed_at
                FROM settings_history
                WHERE key = ?
                ORDER BY changed_at DESC
                LIMIT ?
            """, (key, limit)).fetchall()
            
            return [
                {
                    "key": row["key"],
                    "old_value": row["old_value"],
                    "new_value": row["new_value"],
                    "changed_at": row["changed_at"]
                }
                for row in rows
            ]
    
    def export_to_json(self, category: Optional[str] = None) -> str:
        """Export settings to JSON format.
        
        Args:
            category: Optional category filter
            
        Returns:
            JSON string of settings
        """
        settings_list = self.get_all_settings(category)
        data = {
            "exported_at": datetime.now().isoformat(),
            "category": category,
            "settings": [s.to_dict() for s in settings_list]
        }
        return json.dumps(data, indent=2)
    
    def import_from_json(self, json_str: str, overwrite: bool = True) -> int:
        """Import settings from JSON.
        
        Args:
            json_str: JSON string containing settings
            overwrite: Whether to overwrite existing settings
            
        Returns:
            Number of settings imported
        """
        data = json.loads(json_str)
        count = 0
        
        for setting_data in data.get("settings", []):
            key = setting_data.get("key")
            if not key:
                continue
            
            existing = self.get_setting(key)
            if existing and not overwrite:
                continue
            
            self.set(
                key=key,
                value=setting_data.get("value"),
                value_type=setting_data.get("value_type", "string"),
                category=setting_data.get("category", "general"),
                description=setting_data.get("description", "")
            )
            count += 1
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM settings").fetchone()[0]
            categories = conn.execute("SELECT COUNT(DISTINCT category) FROM settings").fetchone()[0]
            history = conn.execute("SELECT COUNT(*) FROM settings_history").fetchone()[0]
            
            category_counts = conn.execute("""
                SELECT category, COUNT(*) as count
                FROM settings
                GROUP BY category
                ORDER BY count DESC
            """).fetchall()
            
            return {
                "total_settings": total,
                "total_categories": categories,
                "history_entries": history,
                "by_category": {row[0]: row[1] for row in category_counts},
                "database_path": self.db_path
            }


# Global instance for convenience
_settings_db: Optional[SettingsDatabase] = None


def get_settings_db() -> SettingsDatabase:
    """Get or create the global settings database instance."""
    global _settings_db
    if _settings_db is None:
        _settings_db = SettingsDatabase()
    return _settings_db


def init_settings_db(db_path: str = None) -> SettingsDatabase:
    """Initialize and return the global settings database instance."""
    global _settings_db
    _settings_db = SettingsDatabase(db_path)
    return _settings_db


# Convenience functions that use the global instance
def set_setting(key: str, value: Any, value_type: str = "string",
                category: str = "general", description: str = "") -> bool:
    """Set a setting using the global database."""
    return get_settings_db().set(key, value, value_type, category, description)


def get_setting(key: str, default: Any = None) -> Any:
    """Get a setting using the global database."""
    return get_settings_db().get(key, default)


def delete_setting(key: str) -> bool:
    """Delete a setting using the global database."""
    return get_settings_db().delete(key)


# Predefined setting categories
CATEGORY_AGENT = "agent"
CATEGORY_TELEGRAM = "telegram"
CATEGORY_SLACK = "slack"
CATEGORY_DOCKER = "docker"
CATEGORY_API = "api"
CATEGORY_UI = "ui"

# Predefined settings with defaults
DEFAULT_SETTINGS = {
    # Agent behavior
    ("agent.max_rounds", 250, "integer", CATEGORY_AGENT, "Maximum tool rounds per request"),
    ("agent.auto_save", True, "boolean", CATEGORY_AGENT, "Auto-save conversation history"),
    ("agent.confirm_destructive", True, "boolean", CATEGORY_AGENT, "Confirm destructive operations"),
    
    # Telegram settings
    ("telegram.enabled", True, "boolean", CATEGORY_TELEGRAM, "Enable Telegram bot"),
    ("telegram.webhook_port", 21031, "integer", CATEGORY_TELEGRAM, "Webhook server port"),
    
    # Slack settings
    ("slack.enabled", True, "boolean", CATEGORY_SLACK, "Enable Slack bot"),
    ("slack.socket_mode", True, "boolean", CATEGORY_SLACK, "Use Socket Mode for Slack"),
    
    # Docker settings
    ("docker.auto_cleanup", True, "boolean", CATEGORY_DOCKER, "Auto-cleanup containers on exit"),
    ("docker.timeout", 300, "integer", CATEGORY_DOCKER, "Default command timeout in seconds"),
    
    # API settings
    ("api.timeout", 300, "integer", CATEGORY_API, "API request timeout in seconds"),
    ("api.max_retries", 5, "integer", "Maximum API retry attempts"),
    ("api.stream_by_default", True, "boolean", CATEGORY_API, "Stream responses by default"),
    
    # UI settings
    ("ui.show_tool_calls", True, "boolean", CATEGORY_UI, "Show tool calls in output"),
    ("ui.show_progress", True, "boolean", CATEGORY_UI, "Show progress indicators"),
}


def init_default_settings():
    """Initialize default settings if they don't exist."""
    db = get_settings_db()
    for key, value, value_type, category, description in DEFAULT_SETTINGS:
        if db.get_setting(key) is None:
            db.set(key, value, value_type, category, description)


if __name__ == "__main__":
    # Test/demo
    db = SettingsDatabase()
    print(f"Settings DB initialized at: {db.db_path}")
    print(f"\nStatistics:")
    stats = db.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
