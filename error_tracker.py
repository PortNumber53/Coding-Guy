#!/usr/bin/env python3
"""Error tracking and self-healing module for Coding Guy.

Provides:
- 'errors' table in the SQLite settings database for persistent error tracking
- Captures exceptions, API failures, stack traces, and context
- Tracks agent calls (LLM requests, tool calls) for debugging
- Auto-generates fix/heal/improve tasks from recurring errors
- Deduplication of similar errors to avoid noise
"""

import json
import logging
import os
import sqlite3
import sys
import traceback
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from settings_db import get_settings_db

logger = logging.getLogger(__name__)

# Category for error-related settings
CATEGORY_ERROR = "error"

# Error types for classification
ERROR_TYPE_EXCEPTION = "exception"
ERROR_TYPE_API_FAILURE = "api_failure"
ERROR_TYPE_TOOL_FAILURE = "tool_failure"
ERROR_TYPE_AGENT_CALL = "agent_call"  # Tracked agent calls (successful or failed)
ERROR_TYPE_HTTP_ERROR = "http_error"
ERROR_TYPE_DOCKER_ERROR = "docker_error"
ERROR_TYPE_MCP_ERROR = "mcp_error"

# Severity levels
SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"
SEVERITY_CRITICAL = "critical"

# Task auto-generation thresholds (setting keys)
SETTING_AUTO_HEAL_ENABLED = "error_tracker.auto_heal_enabled"
SETTING_AUTO_HEAL_THRESHOLD = "error_tracker.auto_heal_threshold"  # occurrences before generating task
SETTING_AUTO_HEAL_COOLDOWN = "error_tracker.auto_heal_cooldown"  # seconds between auto-heal tasks for same error


@dataclass
class ErrorRecord:
    """Represents a single error record in the database."""
    id: int = 0
    error_type: str = ERROR_TYPE_EXCEPTION
    severity: str = SEVERITY_MEDIUM
    source_module: str = ""
    source_function: str = ""
    error_class: str = ""
    error_message: str = ""
    stack_trace: str = ""
    context: str = ""  # JSON blob with extra context (request details, tool args, etc.)
    request_url: str = ""
    request_method: str = ""
    request_payload_summary: str = ""  # Truncated payload for debugging
    response_status_code: int = 0
    response_body_summary: str = ""  # Truncated response body
    session_key: str = ""
    conversation_round: int = -1
    task_id: str = ""  # Link to auto-generated fix task
    fingerprint: str = ""  # Dedup hash
    occurrence_count: int = 1
    first_seen_at: str = ""
    last_seen_at: str = ""
    resolved: bool = False
    resolved_at: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class ErrorTracker:
    """Manages error tracking, storage, and self-healing task generation."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the error tracker.

        Args:
            db_path: Path to the SQLite database. If None, uses the shared
                     settings database via get_settings_db().
        """
        self.db_path = db_path
        self._own_db = db_path is not None
        self._init_errors_table()

    def _get_db(self):
        """Get the database connection or the shared SettingsDatabase."""
        if self._own_db:
            return None  # Use direct connection methods
        return get_settings_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a raw SQLite connection to the shared settings database."""
        from settings_db import DB_PATH, SettingsDatabase
        path = self.db_path or DB_PATH
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_errors_table(self):
        """Create the errors table and indexes if they don't exist."""
        conn = self._get_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_type TEXT NOT NULL DEFAULT 'exception',
                    severity TEXT NOT NULL DEFAULT 'medium',
                    source_module TEXT DEFAULT '',
                    source_function TEXT DEFAULT '',
                    error_class TEXT DEFAULT '',
                    error_message TEXT NOT NULL DEFAULT '',
                    stack_trace TEXT DEFAULT '',
                    context TEXT DEFAULT '',
                    request_url TEXT DEFAULT '',
                    request_method TEXT DEFAULT '',
                    request_payload_summary TEXT DEFAULT '',
                    response_status_code INTEGER DEFAULT 0,
                    response_body_summary TEXT DEFAULT '',
                    session_key TEXT DEFAULT '',
                    conversation_round INTEGER DEFAULT -1,
                    task_id TEXT DEFAULT '',
                    fingerprint TEXT NOT NULL DEFAULT '',
                    occurrence_count INTEGER DEFAULT 1,
                    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved INTEGER DEFAULT 0,
                    resolved_at TIMESTAMP DEFAULT NULL
                )
            """)

            # Indexes for fast queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_errors_type ON errors(error_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_errors_severity ON errors(severity)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_errors_fingerprint ON errors(fingerprint)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_errors_resolved ON errors(resolved)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_errors_source_module ON errors(source_module)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_errors_last_seen ON errors(last_seen_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_errors_session_key ON errors(session_key)
            """)
            conn.commit()
        finally:
            conn.close()

    def _now(self) -> str:
        """Return current UTC timestamp as ISO string."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def compute_fingerprint(error_type: str, error_class: str, error_message: str,
                            source_module: str, source_function: str,
                            request_url: str = "") -> str:
        """Compute a dedup fingerprint for an error.

        Errors with the same fingerprint are considered the same root cause
        and are counted rather than inserted as new rows.
        """
        import hashlib
        # Normalize: strip variable parts (line numbers, timestamps, UUIDs, etc.)
        msg = error_message or ""
        # Remove specific numbers, UUIDs, timestamps from message for fingerprinting
        import re
        clean_msg = re.sub(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', '<UUID>', msg)
        clean_msg = re.sub(r'\b0x[0-9a-f]+\b', '<HEX>', clean_msg)
        clean_msg = re.sub(r'line \d+', 'line N', clean_msg)
        clean_msg = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', '<TIMESTAMP>', clean_msg)
        # Truncate to avoid huge fingerprints from long variable content
        clean_msg = clean_msg[:500]

        raw = f"{error_type}|{error_class}|{clean_msg}|{source_module}|{source_function}|{request_url}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def record_error(
        self,
        error_type: str = ERROR_TYPE_EXCEPTION,
        severity: str = SEVERITY_MEDIUM,
        source_module: str = "",
        source_function: str = "",
        error_class: str = "",
        error_message: str = "",
        stack_trace: str = "",
        context: Optional[Dict[str, Any]] = None,
        request_url: str = "",
        request_method: str = "",
        request_payload_summary: str = "",
        response_status_code: int = 0,
        response_body_summary: str = "",
        session_key: str = "",
        conversation_round: int = -1,
    ) -> ErrorRecord:
        """Record an error, deduplicating by fingerprint.

        If an error with the same fingerprint already exists, increments
        occurrence_count and updates last_seen_at. Otherwise inserts a new row.

        Returns:
            The ErrorRecord (either existing or new).
        """
        fingerprint = self.compute_fingerprint(
            error_type, error_class, error_message,
            source_module, source_function, request_url
        )
        context_json = json.dumps(context or {}, default=str)
        now = self._now()

        conn = self._get_connection()
        try:
            # Check for existing unresolved error with same fingerprint
            existing = conn.execute(
                "SELECT * FROM errors WHERE fingerprint = ? AND resolved = 0 ORDER BY last_seen_at DESC LIMIT 1",
                (fingerprint,)
            ).fetchone()

            if existing:
                # Update existing record: increment count, update last_seen and latest details
                new_count = existing["occurrence_count"] + 1
                conn.execute("""
                    UPDATE errors SET
                        occurrence_count = ?,
                        last_seen_at = ?,
                        error_message = ?,
                        stack_trace = ?,
                        context = ?,
                        session_key = ?,
                        conversation_round = ?,
                        response_status_code = ?,
                        response_body_summary = ?,
                        request_payload_summary = ?,
                        severity = ?
                    WHERE id = ?
                """, (
                    new_count,
                    now,
                    error_message[:2000],
                    stack_trace[:10000],
                    context_json[:5000],
                    session_key,
                    conversation_round,
                    response_status_code,
                    response_body_summary[:2000],
                    request_payload_summary[:2000],
                    severity,
                    existing["id"],
                ))
                conn.commit()

                record = ErrorRecord(
                    id=existing["id"],
                    error_type=existing["error_type"],
                    severity=severity,
                    source_module=existing["source_module"],
                    source_function=existing["source_function"],
                    error_class=existing["error_class"],
                    error_message=error_message[:2000],
                    stack_trace=stack_trace[:10000],
                    context=context_json[:5000],
                    request_url=existing["request_url"],
                    request_method=existing["request_method"],
                    request_payload_summary=request_payload_summary[:2000],
                    response_status_code=response_status_code,
                    response_body_summary=response_body_summary[:2000],
                    session_key=session_key,
                    conversation_round=conversation_round,
                    task_id=existing["task_id"] or "",
                    fingerprint=fingerprint,
                    occurrence_count=new_count,
                    first_seen_at=existing["first_seen_at"],
                    last_seen_at=now,
                    resolved=False,
                )
                logger.info(f"Updated existing error #{existing['id']} (fingerprint={fingerprint}), count={new_count}")
            else:
                # Insert new error record
                cursor = conn.execute("""
                    INSERT INTO errors (
                        error_type, severity, source_module, source_function,
                        error_class, error_message, stack_trace, context,
                        request_url, request_method, request_payload_summary,
                        response_status_code, response_body_summary,
                        session_key, conversation_round,
                        fingerprint, occurrence_count, first_seen_at, last_seen_at, resolved
                    ) VALUES (
                        ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, 1, ?, ?, 0
                    )
                """, (
                    error_type, severity, source_module, source_function,
                    error_class, error_message[:2000], stack_trace[:10000], context_json[:5000],
                    request_url, request_method, request_payload_summary[:2000],
                    response_status_code, response_body_summary[:2000],
                    session_key, conversation_round,
                    fingerprint, now, now,
                ))
                conn.commit()

                record = ErrorRecord(
                    id=cursor.lastrowid,
                    error_type=error_type,
                    severity=severity,
                    source_module=source_module,
                    source_function=source_function,
                    error_class=error_class,
                    error_message=error_message[:2000],
                    stack_trace=stack_trace[:10000],
                    context=context_json[:5000],
                    request_url=request_url,
                    request_method=request_method,
                    request_payload_summary=request_payload_summary[:2000],
                    response_status_code=response_status_code,
                    response_body_summary=response_body_summary[:2000],
                    session_key=session_key,
                    conversation_round=conversation_round,
                    fingerprint=fingerprint,
                    occurrence_count=1,
                    first_seen_at=now,
                    last_seen_at=now,
                    resolved=False,
                )
                logger.info(f"Recorded new error #{record.id} (fingerprint={fingerprint})")

            # Check if we should auto-generate a healing task
            self._maybe_auto_heal(record, conn)

            return record

        except Exception as e:
            logger.error(f"Failed to record error: {e}", exc_info=True)
            # Return a minimal record rather than crashing
            return ErrorRecord(
                error_type=error_type,
                severity=severity,
                error_message=error_message[:2000],
                fingerprint=fingerprint,
                first_seen_at=now,
                last_seen_at=now,
            )
        finally:
            conn.close()

    def record_exception(
        self,
        exc: BaseException,
        source_module: str = "",
        source_function: str = "",
        context: Optional[Dict[str, Any]] = None,
        session_key: str = "",
        conversation_round: int = -1,
        severity: str = SEVERITY_MEDIUM,
    ) -> ErrorRecord:
        """Convenience method to record a caught exception with full stack trace.

        Args:
            exc: The exception instance.
            source_module: Module where the exception occurred.
            source_function: Function where the exception occurred.
            context: Additional context as a dict.
            session_key: Session identifier.
            conversation_round: Current agent loop round.
            severity: Error severity level.

        Returns:
            The recorded ErrorRecord.
        """
        error_class = type(exc).__name__
        error_message = str(exc)
        stack_trace = traceback.format_exception(type(exc), exc, exc.__traceback__)
        stack_trace_str = "".join(stack_trace)[-10000:]  # Truncate

        return self.record_error(
            error_type=ERROR_TYPE_EXCEPTION,
            severity=severity,
            source_module=source_module,
            source_function=source_function,
            error_class=error_class,
            error_message=error_message,
            stack_trace=stack_trace_str,
            context=context,
            session_key=session_key,
            conversation_round=conversation_round,
        )

    def record_api_failure(
        self,
        url: str,
        method: str = "POST",
        status_code: int = 0,
        error_message: str = "",
        request_payload_summary: str = "",
        response_body_summary: str = "",
        source_module: str = "",
        source_function: str = "",
        context: Optional[Dict[str, Any]] = None,
        session_key: str = "",
        conversation_round: int = -1,
        severity: str = SEVERITY_HIGH,
    ) -> ErrorRecord:
        """Convenience method to record an API/HTTP failure.

        Args:
            url: The request URL.
            method: HTTP method.
            status_code: HTTP status code (0 if no response).
            error_message: Error description.
            request_payload_summary: Truncated request body.
            response_body_summary: Truncated response body.
            source_module: Module making the request.
            source_function: Function making the request.
            context: Additional context.
            session_key: Session identifier.
            conversation_round: Current agent loop round.
            severity: Error severity.

        Returns:
            The recorded ErrorRecord.
        """
        return self.record_error(
            error_type=ERROR_TYPE_API_FAILURE if status_code == 0 else ERROR_TYPE_HTTP_ERROR,
            severity=severity,
            source_module=source_module,
            source_function=source_function,
            error_class="HTTPError",
            error_message=error_message,
            context=context,
            request_url=url,
            request_method=method,
            request_payload_summary=request_payload_summary,
            response_status_code=status_code,
            response_body_summary=response_body_summary,
            session_key=session_key,
            conversation_round=conversation_round,
        )

    def record_tool_failure(
        self,
        tool_name: str,
        tool_args: str,
        error_message: str,
        stack_trace: str = "",
        context: Optional[Dict[str, Any]] = None,
        session_key: str = "",
        conversation_round: int = -1,
        severity: str = SEVERITY_MEDIUM,
    ) -> ErrorRecord:
        """Convenience method to record a tool execution failure.

        Args:
            tool_name: Name of the tool that failed.
            tool_args: The arguments passed to the tool.
            error_message: Error description.
            stack_trace: Stack trace if available.
            context: Additional context.
            session_key: Session identifier.
            conversation_round: Current agent loop round.
            severity: Error severity.

        Returns:
            The recorded ErrorRecord.
        """
        ctx = context or {}
        ctx["tool_name"] = tool_name
        ctx["tool_args_preview"] = tool_args[:500]

        return self.record_error(
            error_type=ERROR_TYPE_TOOL_FAILURE,
            severity=severity,
            source_module="tools",
            source_function=tool_name,
            error_class="ToolExecutionError",
            error_message=error_message,
            stack_trace=stack_trace,
            context=ctx,
            session_key=session_key,
            conversation_round=conversation_round,
        )

    def track_agent_call(
        self,
        url: str,
        method: str = "POST",
        model: str = "",
        request_payload_summary: str = "",
        response_status_code: int = 200,
        error_message: str = "",
        context: Optional[Dict[str, Any]] = None,
        session_key: str = "",
        conversation_round: int = -1,
    ) -> Optional[ErrorRecord]:
        """Track an agent LLM API call. Only records failures (not successes).

        Successful calls are logged at debug level. Failed calls are recorded
        as errors in the database.

        Args:
            url: API endpoint URL.
            method: HTTP method.
            model: Model name used.
            request_payload_summary: Truncated request body.
            response_status_code: HTTP status code.
            error_message: Error message if failed.
            context: Additional context.
            session_key: Session identifier.
            conversation_round: Agent loop round number.

        Returns:
            ErrorRecord if the call failed, None if successful.
        """
        ctx = context or {}
        ctx["model"] = model
        ctx["agent_call"] = True

        if response_status_code >= 400 or error_message:
            severity = SEVERITY_CRITICAL if response_status_code >= 500 else SEVERITY_HIGH
            return self.record_error(
                error_type=ERROR_TYPE_AGENT_CALL,
                severity=severity,
                source_module="coding_agent",
                source_function="call_llm_api",
                error_class="AgentCallFailure",
                error_message=error_message or f"HTTP {response_status_code}",
                context=ctx,
                request_url=url,
                request_method=method,
                request_payload_summary=request_payload_summary,
                response_status_code=response_status_code,
                session_key=session_key,
                conversation_round=conversation_round,
            )
        else:
            logger.debug(f"Agent call succeeded: {method} {url} -> {response_status_code}")
            return None

    # ----- Query methods -----

    def get_error(self, error_id: int) -> Optional[ErrorRecord]:
        """Get a single error by ID."""
        conn = self._get_connection()
        try:
            row = conn.execute("SELECT * FROM errors WHERE id = ?", (error_id,)).fetchone()
            if row:
                return self._row_to_record(row)
            return None
        finally:
            conn.close()

    def get_errors(
        self,
        error_type: Optional[str] = None,
        severity: Optional[str] = None,
        source_module: Optional[str] = None,
        resolved: Optional[bool] = None,
        session_key: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "last_seen_at DESC",
    ) -> List[ErrorRecord]:
        """Query errors with optional filters."""
        conditions = []
        params: list = []

        if error_type:
            conditions.append("error_type = ?")
            params.append(error_type)
        if severity:
            conditions.append("severity = ?")
            params.append(severity)
        if source_module:
            conditions.append("source_module = ?")
            params.append(source_module)
        if resolved is not None:
            conditions.append("resolved = ?")
            params.append(1 if resolved else 0)
        if session_key:
            conditions.append("session_key = ?")
            params.append(session_key)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        query = f"SELECT * FROM errors {where} ORDER BY {order_by} LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        conn = self._get_connection()
        try:
            rows = conn.execute(query, tuple(params)).fetchall()
            return [self._row_to_record(row) for row in rows]
        finally:
            conn.close()

    def get_unresolved_errors(self, limit: int = 50) -> List[ErrorRecord]:
        """Get all unresolved errors, most recent first."""
        return self.get_errors(resolved=False, limit=limit)

    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of error statistics."""
        conn = self._get_connection()
        try:
            total = conn.execute("SELECT COUNT(*) FROM errors").fetchone()[0]
            unresolved = conn.execute("SELECT COUNT(*) FROM errors WHERE resolved = 0").fetchone()[0]

            by_type = conn.execute("""
                SELECT error_type, COUNT(*) as count
                FROM errors WHERE resolved = 0
                GROUP BY error_type ORDER BY count DESC
            """).fetchall()

            by_severity = conn.execute("""
                SELECT severity, COUNT(*) as count
                FROM errors WHERE resolved = 0
                GROUP BY severity ORDER BY
                CASE severity
                    WHEN 'critical' THEN 1
                    WHEN 'high' THEN 2
                    WHEN 'medium' THEN 3
                    WHEN 'low' THEN 4
                END
            """).fetchall()

            by_module = conn.execute("""
                SELECT source_module, COUNT(*) as count
                FROM errors WHERE resolved = 0
                GROUP BY source_module ORDER BY count DESC
            """).fetchall()

            top_recurring = conn.execute("""
                SELECT fingerprint, error_class, error_message, occurrence_count, source_module
                FROM errors WHERE resolved = 0
                ORDER BY occurrence_count DESC LIMIT 10
            """).fetchall()

            return {
                "total_errors": total,
                "unresolved_errors": unresolved,
                "unresolved_by_type": {row[0]: row[1] for row in by_type},
                "unresolved_by_severity": {row[0]: row[1] for row in by_severity},
                "unresolved_by_module": {row[0]: row[1] for row in by_module},
                "top_recurring": [
                    {
                        "fingerprint": row[0],
                        "error_class": row[1],
                        "error_message": row[2][:100],
                        "occurrence_count": row[3],
                        "source_module": row[4],
                    }
                    for row in top_recurring
                ],
            }
        finally:
            conn.close()

    def resolve_error(self, error_id: int) -> bool:
        """Mark an error as resolved."""
        conn = self._get_connection()
        try:
            now = self._now()
            cursor = conn.execute(
                "UPDATE errors SET resolved = 1, resolved_at = ? WHERE id = ?",
                (now, error_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def resolve_by_fingerprint(self, fingerprint: str) -> int:
        """Mark all unresolved errors with a fingerprint as resolved."""
        conn = self._get_connection()
        try:
            now = self._now()
            cursor = conn.execute(
                "UPDATE errors SET resolved = 1, resolved_at = ? WHERE fingerprint = ? AND resolved = 0",
                (now, fingerprint),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def cleanup_old_errors(self, days: int = 90) -> int:
        """Delete resolved errors older than N days."""
        conn = self._get_connection()
        try:
            cutoff = (datetime.now(timezone.utc) - __import__('datetime').timedelta(days=days)).isoformat()
            cursor = conn.execute(
                "DELETE FROM errors WHERE resolved = 1 AND last_seen_at < ?",
                (cutoff,),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    # ----- Auto-heal task generation -----

    def _maybe_auto_heal(self, record: ErrorRecord, conn: sqlite3.Connection):
        """Check if an error should trigger an auto-heal task, and create one if so."""
        try:
            # Check if auto-heal is enabled
            db = get_settings_db()
            enabled = db.get(SETTING_AUTO_HEAL_ENABLED, True)
            if not enabled:
                return

            threshold = db.get(SETTING_AUTO_HEAL_THRESHOLD, 3)
            cooldown_seconds = db.get(SETTING_AUTO_HEAL_COOLDOWN, 3600)  # Default 1 hour

            # Only auto-heal if the error has occurred enough times
            if record.occurrence_count < threshold:
                return

            # Only auto-heal non-resolved errors
            if record.resolved:
                return

            # Check if there's already a pending auto-heal task for this error
            if record.task_id:
                from task_manager import get_task_manager
                tm = get_task_manager()
                existing_task = tm.get_task(record.task_id)
                if existing_task and existing_task.status in ("pending", "in_progress", "blocked"):
                    return  # Task already exists and is active

            # Check cooldown: don't generate a new task if the last one was created recently
            now_iso = self._now()
            if record.task_id:
                # Existing task was completed/failed; check how recently
                from task_manager import get_task_manager
                tm = get_task_manager()
                existing_task = tm.get_task(record.task_id)
                if existing_task and existing_task.updated_at:
                    last_time = datetime.fromisoformat(existing_task.updated_at)
                    elapsed = (datetime.now(timezone.utc) - last_time).total_seconds()
                    if elapsed < cooldown_seconds:
                        return  # Still in cooldown

            # Generate the auto-heal task
            self._create_heal_task(record, conn)

        except Exception as e:
            logger.error(f"Auto-heal check failed: {e}", exc_info=True)

    def _create_heal_task(self, record: ErrorRecord, conn: sqlite3.Connection):
        """Create a self-healing task from an error record."""
        try:
            from task_manager import get_task_manager
            tm = get_task_manager()

            # Build a descriptive task
            error_desc = record.error_message[:200] if record.error_message else "Unknown error"
            source_info = f"{record.source_module}"
            if record.source_function:
                source_info += f".{record.source_function}"

            description = (
                f"[Self-Heal] Fix recurring {record.error_type} in {source_info}: "
                f"{error_desc} (occurred {record.occurrence_count} times)"
            )

            # Build steps tailored to the error type
            steps = self._build_heal_steps(record)

            task = tm.create_task(description, steps=steps)

            # Update the error record with the task ID
            conn.execute(
                "UPDATE errors SET task_id = ? WHERE id = ?",
                (task.uuid, record.id),
            )
            conn.commit()

            logger.info(
                f"Auto-generated heal task {task.display_id} for error #{record.id} "
                f"(fingerprint={record.fingerprint}, count={record.occurrence_count})"
            )

            # Store the task ID in our record for reference
            record.task_id = task.uuid

        except Exception as e:
            logger.error(f"Failed to create heal task: {e}", exc_info=True)

    def _build_heal_steps(self, record: ErrorRecord) -> List[str]:
        """Build appropriate heal/fix steps based on error type and context."""
        steps = [
            f"Read error details: error #{record.id}, type={record.error_type}, "
            f"source={record.source_module}.{record.source_function}",
        ]

        if record.error_type == ERROR_TYPE_API_FAILURE or record.error_type == ERROR_TYPE_HTTP_ERROR:
            steps.extend([
                f"Check API endpoint {record.request_url} - verify it's accessible and responding",
                "Review the request payload and headers for issues",
                "Check API key validity and rate limit status",
                "Add retry logic or fallback handling in the API call code",
                "Test the fix by making a sample request",
            ])
        elif record.error_type == ERROR_TYPE_TOOL_FAILURE:
            ctx = {}
            try:
                ctx = json.loads(record.context) if record.context else {}
            except (json.JSONDecodeError, TypeError):
                pass
            tool_name = ctx.get("tool_name", record.source_function)
            steps.extend([
                f"Read the tool implementation for '{tool_name}' in tools.py",
                f"Analyze the stack trace to identify the root cause",
                "Fix the tool implementation to handle the error gracefully",
                f"Test the fix by simulating a call to '{tool_name}'",
            ])
        elif record.error_type == ERROR_TYPE_DOCKER_ERROR:
            steps.extend([
                "Check Docker daemon status and container health",
                "Review the Dockerfile for issues",
                "Rebuild the container if needed",
                "Verify workspace mount and environment variables",
            ])
        elif record.error_type == ERROR_TYPE_MCP_ERROR:
            steps.extend([
                "Check MCP server configuration and connectivity",
                "Verify the MCP server process is running",
                "Review the MCP tool call parameters",
                "Add error handling for MCP communication failures",
            ])
        else:
            # Generic exception
            steps.extend([
                "Read the source code where the error originates",
                "Analyze the stack trace to understand the failure path",
                "Add proper error handling or fix the root cause",
                "Test the fix to ensure the error is resolved",
            ])

        steps.extend([
            "Mark the error as resolved in the error tracker",
            "Verify no similar errors recur",
        ])

        return steps

    def _row_to_record(self, row: sqlite3.Row) -> ErrorRecord:
        """Convert a database row to an ErrorRecord."""
        return ErrorRecord(
            id=row["id"],
            error_type=row["error_type"],
            severity=row["severity"],
            source_module=row["source_module"],
            source_function=row["source_function"],
            error_class=row["error_class"],
            error_message=row["error_message"],
            stack_trace=row["stack_trace"],
            context=row["context"],
            request_url=row["request_url"],
            request_method=row["request_method"],
            request_payload_summary=row["request_payload_summary"],
            response_status_code=row["response_status_code"],
            response_body_summary=row["response_body_summary"],
            session_key=row["session_key"],
            conversation_round=row["conversation_round"],
            task_id=row["task_id"] or "",
            fingerprint=row["fingerprint"],
            occurrence_count=row["occurrence_count"],
            first_seen_at=row["first_seen_at"],
            last_seen_at=row["last_seen_at"],
            resolved=bool(row["resolved"]),
            resolved_at=row["resolved_at"],
        )


# ----- Global instance -----

_error_tracker: Optional[ErrorTracker] = None


def get_error_tracker() -> ErrorTracker:
    """Get or create the global error tracker instance."""
    global _error_tracker
    if _error_tracker is None:
        _error_tracker = ErrorTracker()
    return _error_tracker


def init_error_tracker(db_path: Optional[str] = None) -> ErrorTracker:
    """Initialize and return the global error tracker instance."""
    global _error_tracker
    _error_tracker = ErrorTracker(db_path)
    return _error_tracker


# ----- Decorator for automatic error tracking -----

def track_errors(source_module: str = "", source_function: str = "",
                 severity: str = SEVERITY_MEDIUM):
    """Decorator that automatically catches and records exceptions.

    Usage:
        @track_errors(source_module="coding_agent", source_function="agent_loop")
        def agent_loop(...):
            ...

    The decorator re-raises the exception after recording it, preserving
    the original behavior. Use this on functions where you want every
    exception tracked without modifying the function body.
    """
    def decorator(func):
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                tracker = get_error_tracker()
                tracker.record_exception(
                    exc,
                    source_module=source_module or func.__module__ or "",
                    source_function=source_function or func.__qualname__ or "",
                    severity=severity,
                )
                raise  # Re-raise to preserve original behavior
        return wrapper
    return decorator


if __name__ == "__main__":
    # Demo / test
    tracker = ErrorTracker()
    print(f"Error tracker initialized")
    print(f"Summary: {json.dumps(tracker.get_error_summary(), indent=2)}")

    # Test recording an error
    try:
        raise ValueError("Test error for demonstration")
    except ValueError as e:
        record = tracker.record_exception(
            e,
            source_module="error_tracker",
            source_function="__main__",
            context={"demo": True},
        )
        print(f"Recorded error #{record.id}, fingerprint={record.fingerprint}")
