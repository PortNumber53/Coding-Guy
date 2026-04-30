"""Task tracking and management for the coding agent.

Provides structured task tracking with:
- Task creation with ordered steps
- Progress tracking through steps
- Automatic resume after errors
- Human-in-the-loop blocking (ask_human)
- Persistent state via the settings database
"""

import json
import logging
import threading
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Optional, Dict, List

from settings_db import get_settings_db

logger = logging.getLogger(__name__)

CATEGORY_TASK = "task"

# Settings key prefixes
TASK_DATA_PREFIX = "task.data."       # task.data.<uuid> = task dict
TASK_INDEX_KEY = "task.index"         # list of active task uuids
TASK_ACTIVE_PREFIX = "task.active."   # task.active.<session_key> = task_uuid


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaskStep:
    """A single step within a task."""
    description: str
    status: str = "pending"       # pending | in_progress | completed | failed | skipped
    result: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TaskStep":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Task:
    """A tracked task with ordered steps."""
    uuid: str
    description: str
    status: str = "pending"        # pending | in_progress | completed | failed | blocked
    steps: List[Dict] = field(default_factory=list)
    current_step_index: int = 0
    error: Optional[str] = None
    blocker: Optional[str] = None  # Human intervention needed
    human_response: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        # Handle steps as list of dicts
        steps = data.get("steps", [])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def display_id(self) -> str:
        return self.uuid[:8]

    def get_step_objects(self) -> List[TaskStep]:
        return [TaskStep.from_dict(s) if isinstance(s, dict) else s for s in self.steps]

    def summary(self) -> str:
        """Return a human-readable summary of the task state."""
        lines = [f"Task [{self.display_id}]: {self.description}"]
        lines.append(f"Status: {self.status}")
        if self.error:
            lines.append(f"Error: {self.error}")
        if self.blocker:
            lines.append(f"Blocker (needs human): {self.blocker}")
        steps = self.get_step_objects()
        for i, step in enumerate(steps):
            marker = "→" if i == self.current_step_index and step.status == "in_progress" else " "
            status_icon = {"completed": "✓", "failed": "✗", "skipped": "⊘", "in_progress": "…", "pending": "○"}.get(step.status, "?")
            line = f"  {marker} {status_icon} Step {i+1}: {step.description}"
            if step.status == "completed" and step.result:
                line += f" → {step.result[:80]}"
            elif step.status == "failed" and step.error:
                line += f" → {step.error[:80]}"
            lines.append(line)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# TaskManager
# ---------------------------------------------------------------------------

class TaskManager:
    """Manages task lifecycle with persistence via the settings database."""

    def __init__(self):
        self.db = get_settings_db()
        self._cache: Dict[str, Task] = {}
        self._index_lock = threading.Lock()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # -- Persistence helpers --

    def _task_data_key(self, task_uuid: str) -> str:
        return f"{TASK_DATA_PREFIX}{task_uuid}"

    def _save_task(self, task: Task):
        """Persist a task to the database."""
        task.updated_at = self._now()
        self.db.set(
            self._task_data_key(task.uuid),
            task.to_dict(),
            value_type="json",
            category=CATEGORY_TASK,
            description=f"Task: {task.description[:80]}",
        )
        self._cache[task.uuid] = task
        # Ensure it's in the index
        self._add_to_index(task.uuid)

    def _load_task(self, task_uuid: str) -> Optional[Task]:
        """Load a task from the database (or cache)."""
        if task_uuid in self._cache:
            return self._cache[task_uuid]
        data = self.db.get(self._task_data_key(task_uuid))
        if not data:
            return None
        if isinstance(data, str):
            data = json.loads(data)
        task = Task.from_dict(data)
        self._cache[task.uuid] = task
        return task

    def _add_to_index(self, task_uuid: str):
        """Add a session UUID to the task index (thread-safe)."""
        with self._index_lock:
            index_key = TASK_INDEX_KEY
            index = self.db.get(index_key) or []
            if task_uuid not in index:
                index.append(task_uuid)
                self.db.set(
                    index_key,
                    index,
                    value_type="json",
                    category=CATEGORY_TASK,
                    description="Active task index"
                )

    def _remove_from_index(self, task_uuid: str):
        """Remove a session UUID from the task index (thread-safe)."""
        with self._index_lock:
            index_key = TASK_INDEX_KEY
            index = self.db.get(index_key) or []
            if task_uuid in index:
                index.remove(task_uuid)
                if index:
                    self.db.set(
                        index_key,
                        index,
                        value_type="json",
                        category=CATEGORY_TASK,
                        description="Active task index"
                    )
                else:
                    self.db.delete(index_key)

    # -- Active task per session --

    def _active_task_key(self, session_key: str) -> str:
        return f"{TASK_ACTIVE_PREFIX}{session_key}"

    def set_active_task(self, session_key: str, task_uuid: str):
        self.db.set(
            self._active_task_key(session_key), task_uuid,
            value_type="string", category=CATEGORY_TASK,
            description=f"Active task for session {session_key}",
        )

    def get_active_task_id(self, session_key: str) -> Optional[str]:
        return self.db.get(self._active_task_key(session_key))

    def get_active_task(self, session_key: str) -> Optional[Task]:
        task_uuid = self.get_active_task_id(session_key)
        if task_uuid:
            return self._load_task(task_uuid)
        return None

    def clear_active_task(self, session_key: str):
        key = self._active_task_key(session_key)
        val = self.db.get(key)
        if val:
            self.db.delete(key)

    # -- Public API --

    def create_task(self, description: str, steps: Optional[List[str]] = None,
                    session_key: Optional[str] = None) -> Task:
        """Create a new task with optional step descriptions."""
        task_uuid = str(uuid.uuid4())
        now = self._now()

        step_list = []
        if steps:
            for s in steps:
                step_list.append(TaskStep(description=s).to_dict())

        task = Task(
            uuid=task_uuid,
            description=description,
            status="in_progress" if step_list else "pending",
            steps=step_list,
            current_step_index=0,
            created_at=now,
            updated_at=now,
        )
        self._save_task(task)

        if session_key:
            self.set_active_task(session_key, task_uuid)

        logger.info(f"Created task {task.display_id}: {description}")
        return task

    def get_task(self, task_uuid: str) -> Optional[Task]:
        return self._load_task(task_uuid)

    def update_step(self, task_uuid: str, step_index: int,
                    status: str, result: Optional[str] = None,
                    error: Optional[str] = None) -> Optional[Task]:
        """Update a step's status within a task."""
        task = self._load_task(task_uuid)
        if not task:
            return None

        steps = task.get_step_objects()
        if step_index < 0 or step_index >= len(steps):
            return None

        steps[step_index].status = status
        if result is not None:
            steps[step_index].result = result
        if error is not None:
            steps[step_index].error = error

        task.steps = [s.to_dict() for s in steps]

        # Auto-advance current_step_index on completion
        if status == "completed" and task.current_step_index == step_index:
            task.current_step_index = min(step_index + 1, len(steps))

        # If step failed, mark task as failed too
        if status == "failed":
            task.status = "failed"
            task.error = error or steps[step_index].error

        # Check if all steps are done
        if all(s.status in ("completed", "skipped") for s in steps):
            task.status = "completed"

        self._save_task(task)
        return task

    def advance_step(self, task_uuid: str) -> Optional[Task]:
        """Move to the next pending step and mark it in_progress."""
        task = self._load_task(task_uuid)
        if not task:
            return None

        steps = task.get_step_objects()
        for i, step in enumerate(steps):
            if step.status == "pending":
                step.status = "in_progress"
                task.current_step_index = i
                task.steps = [s.to_dict() for s in steps]
                task.status = "in_progress"
                self._save_task(task)
                return task

        # No pending steps left
        if all(s.status in ("completed", "skipped") for s in steps):
            task.status = "completed"
            self._save_task(task)
        return task

    def complete_task(self, task_uuid: str, result: Optional[str] = None) -> Optional[Task]:
        """Mark a task as completed."""
        task = self._load_task(task_uuid)
        if not task:
            return None

        task.status = "completed"
        task.error = None
        task.blocker = None

        # Mark remaining steps as completed
        steps = task.get_step_objects()
        for step in steps:
            if step.status in ("pending", "in_progress"):
                step.status = "completed"
                if result:
                    step.result = result
        task.steps = [s.to_dict() for s in steps]
        self._save_task(task)
        return task

    def fail_task(self, task_uuid: str, error: str) -> Optional[Task]:
        """Mark a task as failed with an error message."""
        task = self._load_task(task_uuid)
        if not task:
            return None

        task.status = "failed"
        task.error = error
        self._save_task(task)
        return task

    def block_task(self, task_uuid: str, blocker: str) -> Optional[Task]:
        """Block a task pending human intervention."""
        task = self._load_task(task_uuid)
        if not task:
            return None

        task.status = "blocked"
        task.blocker = blocker
        task.human_response = None
        self._save_task(task)
        logger.info(f"Task {task.display_id} blocked: {blocker}")
        return task

    def unblock_task(self, task_uuid: str, human_response: str) -> Optional[Task]:
        """Unblock a task after human provides input."""
        task = self._load_task(task_uuid)
        if not task:
            return None

        task.status = "in_progress"
        task.blocker = None
        task.human_response = human_response
        self._save_task(task)
        logger.info(f"Task {task.display_id} unblocked with response: {human_response[:80]}")
        return task

    def delete_task(self, task_uuid: str) -> bool:
        """Delete a task permanently."""
        task = self._load_task(task_uuid)
        if not task:
            return False

        self.db.delete(self._task_data_key(task_uuid))
        self._remove_from_index(task_uuid)
        self._cache.pop(task_uuid, None)
        return True

    def list_tasks(self, status: Optional[str] = None) -> List[Task]:
        """List all tasks, optionally filtered by status."""
        index = self.db.get(TASK_INDEX_KEY) or []
        tasks = []
        for task_uuid in index:
            task = self._load_task(task_uuid)
            if task and (status is None or task.status == status):
                tasks.append(task)
        return sorted(tasks, key=lambda t: t.updated_at or t.created_at, reverse=True)

    def get_resume_context(self, session_key: str) -> Optional[str]:
        """Get context string for resuming an active/blocked/failed task."""
        task = self.get_active_task(session_key)
        if not task:
            return None
        if task.status == "completed":
            return None

        lines = [f"[RESUMING TASK]"]
        lines.append(task.summary())

        if task.status == "failed" and task.error:
            lines.append(f"\nThe task previously failed with error: {task.error}")
            lines.append("Try a different approach to work around this error.")

        if task.human_response:
            lines.append(f"\nHuman provided response: {task.human_response}")
            lines.append("Use this information to continue the task.")

        if task.status == "blocked" and task.blocker:
            lines.append(f"\nBLOCKED — waiting for human input on: {task.blocker}")

        # Find the next step to work on
        steps = task.get_step_objects()
        next_pending = None
        for i, step in enumerate(steps):
            if step.status in ("pending", "failed", "in_progress"):
                next_pending = (i, step)
                break

        if next_pending:
            i, step = next_pending
            if step.status == "failed":
                lines.append(f"\nRetry or find a workaround for failed step {i+1}: {step.description}")
                if step.error:
                    lines.append(f"  Previous error: {step.error}")
            else:
                lines.append(f"\nContinue from step {i+1}: {step.description}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Global instance
# ---------------------------------------------------------------------------

_task_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    """Get or create the global task manager instance."""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager
