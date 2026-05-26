"""Telegram bot integration for the coding agent, with GitHub webhook support."""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from urllib.parse import urlparse

import tornado.httpserver
import tornado.ioloop
import tornado.web
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from coding_agent import agent_loop, STATUS_COMPLETE, STATUS_MAX_ROUNDS, STATUS_ERROR, STATUS_BLOCKED, COMMIT_HASH
from settings_db import get_settings_db, init_default_settings, CATEGORY_AGENT, CATEGORY_TELEGRAM
from memory_manager import get_memory_manager, MemorySession
from error_tracker import get_error_tracker

logger = logging.getLogger(__name__)

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "21031"))
PROGRESS_REPORT_INTERVAL = 1 # send a progress update every N tool rounds

# GitHub webhook configuration
GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "")
GITHUB_WEBHOOK_PATH = os.getenv("GITHUB_WEBHOOK_PATH", "/github-webhook")

# Per-session conversation history: session_uuid -> list of message dicts
_chat_histories: dict[str, list] = {}
# Per-chat locks to serialize message processing (keyed by chat_id)
_chat_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
# Memory manager instance
_memory_manager = get_memory_manager()

# Track active message processing tasks for graceful shutdown
_active_tasks: set[asyncio.Task] = set()
_shutdown_event: asyncio.Event = asyncio.Event()

# Server start time for /status
_start_time: float = 0.0

# Progress reporting - send update every tool round for more verbose feedback


def split_message(text: str, max_len: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> list[str]:
    """Split text into chunks that fit within Telegram's message limit."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        # Try to split at a newline
        split_pos = text.rfind("\n", 0, max_len)
        if split_pos == -1 or split_pos < max_len // 2:
            split_pos = max_len
        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip("\n")
    return chunks


def _get_webhook_url() -> str:
    """Build the full GitHub webhook URL from the Telegram webhook base."""
    telegram_url = os.getenv("TELEGRAM_WEBHOOK_URL", "")
    if "://" in telegram_url:
        parsed = urlparse(telegram_url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        return f"{base}{GITHUB_WEBHOOK_PATH}"
    return f"http://your-server:{WEBHOOK_PORT}{GITHUB_WEBHOOK_PATH}"


# ---------------------------------------------------------------------------
# Telegram command handlers
# ---------------------------------------------------------------------------

async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    await update.message.reply_text(
        "Coding Agent ready. Send me a message and I'll help you with coding tasks.\n\n"
        "I'll send real-time tool call reports with 👍/👎 reactions\n"
        "so you can see what I'm doing as I work.\n\n"
        "Commands:\n"
        "/clear - Reset the conversation\n"
        "/memory - Manage memory sessions\n"
        "/webhook - Show GitHub webhook configuration\n"
        "/status - Show server status\n"
        "/settings - Show/manage settings\n"
        "/errors - View tracked errors and self-heal status"
    )


async def handle_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /clear command - creates a new memory session."""
    chat_id = update.effective_chat.id
    
    # Create a new session for this chat (memory-based clear)
    session = _memory_manager.create_session(str(chat_id))
    
    # Clear old session's history from memory
    old_sessions = _memory_manager.list_sessions(str(chat_id))
    for s in old_sessions:
        if s.uuid != session.uuid:
            _chat_histories.pop(s.uuid, None)
    
    await update.message.reply_text(f"New session started: `{session.uuid[:8]}`", parse_mode="HTML")


async def handle_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /memory command - manage memory sessions."""
    chat_id = update.effective_chat.id
    args = context.args or []
    
    if not args:
        # Show current session and list
        active_session = _memory_manager.get_active_session(str(chat_id))
        all_sessions = _memory_manager.list_sessions(str(chat_id))
        
        lines = ["*Memory Sessions*"]
        
        if active_session:
            lines.append(f"\n*Active:* {active_session.display_name} (`{active_session.uuid[:8]}`)")
            if active_session.message_count:
                lines.append(f"  Messages: {active_session.message_count}")
        
        if all_sessions:
            lines.append(f"\n*All sessions ({len(all_sessions)}):*")
            for s in all_sessions[:10]:  # Limit to 10
                marker = " ●" if active_session and s.uuid == active_session.uuid else " ○"
                name = s.name or s.uuid[:8]
                lines.append(f"{marker} {name} ({s.uuid[:8]})")
        else:
            lines.append("\nNo saved sessions. Start chatting to create one!")
        
        lines.extend([
            "\n*Commands:*",
            "/memory list - List all sessions",
            "/memory switch <uuid_or_name> - Switch to session",
            "/memory new [name] - Create new session",
            "/memory rename <uuid> <name> - Rename session",
            "/memory delete <uuid> - Delete session",
            "/memory export <uuid> - Export session data",
        ])
        
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        return
    
    subcommand = args[0].lower()
    
    if subcommand == "list":
        all_sessions = _memory_manager.list_sessions(str(chat_id))
        if not all_sessions:
            await update.message.reply_text("No memory sessions found. Start chatting to create one!")
            return
        
        active = _memory_manager.get_active_session(str(chat_id))
        lines = [f"*Memory Sessions ({len(all_sessions)}):*"]
        
        for s in all_sessions:
            marker = "●" if active and s.uuid == active.uuid else "○"
            name = s.name or s.uuid[:8]
            created = s.created_at[:10] if s.created_at else "?"
            msg_count = s.message_count or 0
            lines.append(f"{marker} `{s.uuid[:8]}` *{name}* ({msg_count} msgs) - {created}")
        
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    
    elif subcommand == "new":
        name = " ".join(args[1:]) if len(args) > 1 else None
        session = _memory_manager.create_session(str(chat_id), name=name)
        name_str = f" named '{name}'" if name else ""
        await update.message.reply_text(f"Created new session{name_str}: `{session.uuid[:8]}" + "`", parse_mode="Markdown")
    
    elif subcommand == "switch" and len(args) >= 2:
        query = " ".join(args[1:])
        # Try to find by exact UUID, partial UUID, or name
        session = _memory_manager.get_session(query)
        if not session:
            session = _memory_manager.get_session_by_name(str(chat_id), query)
        
        if session:
            success = _memory_manager.switch_session(str(chat_id), session.uuid)
            if success:
                await update.message.reply_text(f"Switched to session: {session.display_name} (`{session.uuid[:8]}`)", parse_mode="Markdown")
            else:
                await update.message.reply_text("Failed to switch session.")
        else:
            await update.message.reply_text(f"Session not found: '{query}'. Use /memory list to see available sessions.")
    
    elif subcommand == "rename" and len(args) >= 3:
        session_id = args[1]
        new_name = " ".join(args[2:])
        
        # Find session
        session = _memory_manager.get_session(session_id)
        if not session:
            session = _memory_manager.get_session_by_name(str(chat_id), session_id)
        
        if session:
            success = _memory_manager.rename_session(session.uuid, new_name)
            if success:
                await update.message.reply_text(f"Renamed session to: '{new_name}'")
            else:
                await update.message.reply_text("Failed to rename session.")
        else:
            await update.message.reply_text(f"Session not found: '{session_id}'")
    
    elif subcommand == "delete" and len(args) >= 2:
        session_id = " ".join(args[1:])
        session = _memory_manager.get_session(session_id)
        if not session:
            session = _memory_manager.get_session_by_name(str(chat_id), session_id)
        
        if session:
            success = _memory_manager.delete_session(session.uuid)
            if success:
                _chat_histories.pop(session.uuid, None)
                await update.message.reply_text(f"Deleted session: {session.display_name}")
            else:
                await update.message.reply_text("Failed to delete session.")
        else:
            await update.message.reply_text(f"Session not found: '{session_id}'")
    
    elif subcommand == "export" and len(args) >= 2:
        session_id = " ".join(args[1:])
        session = _memory_manager.get_session(session_id)
        if not session:
            session = _memory_manager.get_session_by_name(str(chat_id), session_id)
        
        if session:
            data = _memory_manager.export_session(session.uuid)
            if data:
                # Truncate if too long
                if len(data) > 4000:
                    data = data[:3900] + "\n... (truncated)"
                await update.message.reply_text(f"```json\n{data}\n```", parse_mode="Markdown")
            else:
                await update.message.reply_text("Failed to export session.")
        else:
            await update.message.reply_text(f"Session not found: '{session_id}'")
    
    else:
        await update.message.reply_text(
            "Unknown subcommand. Usage:\n"
            "/memory list\n"
            "/memory new [name]\n"
            "/memory switch <uuid_or_name>\n"
            "/memory rename <uuid> <name>\n"
            "/memory delete <uuid>\n"
            "/memory export <uuid>"
        )


async def handle_webhook(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /webhook command - show GitHub webhook configuration."""
    url = _get_webhook_url()
    lines = [
        "GitHub Webhook Configuration:",
        f"  URL: {url}",
        f"  Content type: application/json",
        f"  Events: push (recommended)",
        "",
    ]
    if GITHUB_WEBHOOK_SECRET:
        lines.append("  Secret: configured")
    else:
        lines.append("  Secret: not configured (set GITHUB_WEBHOOK_SECRET env var)")

    lines.extend([
        "",
        "When a push event is received, the server will:",
        "  1. Verify the webhook signature (if secret is set)",
        "  2. Run git pull to fetch the latest code",
        "  3. Restart the server process",
    ])
    await update.message.reply_text("\n".join(lines))


async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command - show server status."""
    uptime_secs = int(time.time() - _start_time)
    hours, remainder = divmod(uptime_secs, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{hours}h {minutes}m {seconds}s"

    # Get settings db stats
    db_stats = get_settings_db().get_stats()
    memory_stats = _memory_manager.get_stats()

    lines = [
        f"Build: {COMMIT_HASH}",
        f"Uptime: {uptime_str}",
        f"Active chats: {len(_chat_histories)}",
        f"Memory sessions: {memory_stats['total_sessions']}",
        f"Webhook port: {WEBHOOK_PORT}",
        f"GitHub webhook: {GITHUB_WEBHOOK_PATH}",
        f"GitHub secret: {'configured' if GITHUB_WEBHOOK_SECRET else 'not set'}",
        f"Settings DB: {db_stats['total_settings']} settings, {db_stats['total_categories']} categories",
    ]
    await update.message.reply_text("\n".join(lines))


async def handle_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /settings command - show or modify settings."""
    db = get_settings_db()
    args = context.args or []

    if not args:
        # Show categories overview
        categories = db.get_categories()
        lines = ["Settings Categories:"]
        for cat in categories:
            settings = db.get_all(category=cat)
            lines.append(f"  {cat}: {len(settings)} setting(s)")
        lines.extend([
            "",
            "Usage:",
            "/settings get <key> - Get a specific setting",
            "/settings set <key> <value> - Set a setting",
            "/settings list [category] - List settings in category",
            "/settings categories - List all categories",
            "/settings export - Export all settings as JSON",
        ])
        await update.message.reply_text("\n".join(lines))
        return

    subcommand = args[0].lower()

    if subcommand == "get" and len(args) >= 2:
        key = args[1]
        setting = db.get_setting(key)
        if setting:
            lines = [
                f"Key: {setting.key}",
                f"Value: {setting.value}",
                f"Type: {setting.value_type}",
                f"Category: {setting.category}",
                f"Description: {setting.description}",
                f"Updated: {setting.updated_at}",
            ]
            await update.message.reply_text("\n".join(lines))
        else:
            await update.message.reply_text(f"Setting '{key}' not found.")

    elif subcommand == "set" and len(args) >= 3:
        key = args[1]
        value = " ".join(args[2:])
        # Try to infer type
        value_type = "string"
        if value.lower() in ("true", "false"):
            value_type = "boolean"
            value = value.lower() == "true"
        elif value.isdigit():
            value_type = "integer"
            value = int(value)
        elif "." in value:
            try:
                value = float(value)
                value_type = "float"
            except ValueError:
                pass

        db.set(key, value, value_type)
        await update.message.reply_text(f"Set '{key}' = {value} (type: {value_type})")

    elif subcommand == "list":
        category = args[1] if len(args) > 1 else None
        settings = db.get_all(category=category)
        if not settings:
            await update.message.reply_text(f"No settings found" + (f" in category '{category}'" if category else ""))
            return
        lines = [f"Settings:" + (f" (category: {category})" if category else "")]
        for key, val in list(settings.items())[:20]:  # Limit to 20
            lines.append(f"  {key}: {val}")
        if len(settings) > 20:
            lines.append(f"  ... and {len(settings) - 20} more")
        await update.message.reply_text("\n".join(lines))

    elif subcommand == "categories":
        cats = db.get_categories()
        await update.message.reply_text("Categories: " + ", ".join(cats) if cats else "No categories")

    elif subcommand == "export":
        json_data = db.export_to_json()
        # Truncate if too long - ensure valid JSON structure
        if len(json_data) > 3000:
            data = json.loads(json_data)
            settings = data.get("settings", [])
            # Keep truncating until it fits
            truncated = False
            while len(json.dumps(data, indent=2)) > 2900 and settings:
                settings.pop()
                truncated = True
            data["truncated"] = truncated
            data["total_settings"] = len(db.get_all_settings())
            json_data = json.dumps(data, indent=2)
        await update.message.reply_text(f"<pre>{json_data}</pre>", parse_mode="HTML")

    else:
        await update.message.reply_text(
            "Unknown subcommand. Usage:\n"
            "/settings get <key>\n"
            "/settings set <key> <value>\n"
            "/settings list [category]\n"
            "/settings categories\n"
            "/settings export"
        )



async def handle_errors(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /errors command - show tracked errors and self-heal status."""
    from error_tracker import get_error_tracker
    args = context.args or []
    tracker = get_error_tracker()

    if not args:
        # Show summary
        summary = tracker.get_error_summary()
        lines = [
            "*Error Tracker Summary*",
            f"Total errors: {summary['total_errors']}",
            f"Unresolved: {summary['unresolved_errors']}",
        ]
        if summary.get('unresolved_by_type'):
            lines.append("\n*By type:*")
            for etype, count in summary['unresolved_by_type'].items():
                lines.append(f"  {etype}: {count}")
        if summary.get('unresolved_by_severity'):
            lines.append("\n*By severity:*")
            for sev, count in summary['unresolved_by_severity'].items():
                lines.append(f"  {sev}: {count}")
        if summary.get('top_recurring'):
            lines.append("\n*Top recurring:*")
            for rec in summary['top_recurring'][:5]:
                lines.append(f"  #{rec.get('fingerprint', '?')[:8]}: {rec.get('error_class', '?')} - {rec.get('error_message', '')[:50]} (x{rec.get('occurrence_count', 0)})")
        lines.extend([
            "",
            "Commands:",
            "/errors list - List unresolved errors",
            "/errors <id> - Get details of a specific error",
            "/errors resolve <id> - Mark an error as resolved",
            "/errors all - List all errors including resolved",
        ])
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        return

    subcommand = args[0].lower()

    if subcommand == "list":
        errors = tracker.get_unresolved_errors(limit=20)
        if not errors:
            await update.message.reply_text("No unresolved errors! Everything is clean.")
            return
        lines = [f"*Unresolved Errors ({len(errors)}):*"]
        for e in errors[:20]:
            src = f"{e.source_module}.{e.source_function}" if e.source_function else e.source_module
            lines.append(f"  #{e.id} [{e.severity}] {e.error_class} in {src}: {e.error_message[:60]} (x{e.occurrence_count})")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    elif subcommand == "all":
        errors = tracker.get_errors(limit=30)
        if not errors:
            await update.message.reply_text("No errors tracked yet.")
            return
        lines = [f"*All Errors ({len(errors)}):*"]
        for e in errors[:30]:
            status_mark = "✓" if e.resolved else "✗"
            lines.append(f"  {status_mark} #{e.id} [{e.severity}] {e.error_class}: {e.error_message[:50]} (x{e.occurrence_count})")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    elif subcommand == "resolve" and len(args) >= 2:
        try:
            error_id = int(args[1])
        except ValueError:
            await update.message.reply_text("Error ID must be a number.")
            return
        success = tracker.resolve_error(error_id)
        if success:
            await update.message.reply_text(f"Error #{error_id} marked as resolved.")
        else:
            await update.message.reply_text(f"Error #{error_id} not found.")

    else:
        # Try to parse as error ID
        try:
            error_id = int(subcommand)
            record = tracker.get_error(error_id)
            if not record:
                await update.message.reply_text(f"Error #{error_id} not found.")
                return
            lines = [
                f"*Error #{record.id}*",
                f"Type: {record.error_type}",
                f"Severity: {record.severity}",
                f"Class: `{record.error_class}`",
                f"Source: {record.source_module}.{record.source_function}" if record.source_function else f"Source: {record.source_module}",
                f"Message: {record.error_message[:500]}",
                f"Occurrences: {record.occurrence_count}",
                f"First seen: {record.first_seen_at}",
                f"Last seen: {record.last_seen_at}",
                f"Resolved: {'Yes' if record.resolved else 'No'}",
            ]
            if record.task_id:
                lines.append(f"Heal task: `{record.task_id[:8]}`")
            if record.stack_trace:
                trace_preview = record.stack_trace[:1000]
                lines.append(f"\n*Stack trace:*\n```\n{trace_preview}\n```")
            if record.request_url:
                lines.append(f"\n*Request:* {record.request_method} {record.request_url}")
                if record.response_status_code:
                    lines.append(f"Response status: {record.response_status_code}")
            await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        except ValueError:
            await update.message.reply_text(
                "Unknown subcommand. Usage:\n"
                "/errors - Summary\n"
                "/errors list - Unresolved errors\n"
                "/errors all - All errors\n"
                "/errors <id> - Error details\n"
                "/errors resolve <id> - Mark resolved"
            )


def make_progress_callback(chat, loop):
    """Return a callback that sends progress updates to a Telegram chat.

    The callback runs in the agent_loop thread and schedules sends on the
    bot's event loop via run_coroutine_threadsafe.
    """
    last_reported = [0]

    def callback(round_num, max_rounds, tool_names):
        if round_num - last_reported[0] < PROGRESS_REPORT_INTERVAL:
            return
        last_reported[0] = round_num

        tools_str = ", ".join(tool_names)
        text = f"Round {round_num}/{max_rounds}: {tools_str}"

        future = asyncio.run_coroutine_threadsafe(
            chat.send_message(text), loop
        )
        try:
            future.result(timeout=10)
        except Exception as e:
            logger.warning(f"Failed to send progress update: {e}")

    return callback


def make_tool_call_callback(chat, loop, bot):
    """Return a callback that sends tool call reports to a Telegram chat
    with 👍/👎 emoji reactions.

    The callback runs in the agent_loop thread and schedules sends on the
    bot's event loop via run_coroutine_threadsafe.

    Args:
        chat: The Telegram chat object to send messages to.
        loop: The asyncio event loop for the bot.
        bot: The Telegram bot instance (needed for set_message_reaction).

    Returns:
        A callable(tool_name, args_summary, result_summary, is_error) that
        sends a Telegram message and adds a 👍 or 👎 reaction.
    """

    async def _send_tool_report(tool_name, args_summary, result_summary, is_error):
        """Send a tool call message and add a reaction emoji."""
        # Truncate summaries for readability
        args_display = args_summary[:150] + ("…" if len(args_summary) > 150 else "")
        result_display = result_summary[:150] + ("…" if len(result_summary) > 150 else "")

        status_icon = "❌" if is_error else "✅"
        text = (
            f"{status_icon} *Tool:* `{tool_name}`\n"
            f"📋 *Args:* `{args_display}`\n"
            f"📤 *Result:* `{result_display}`"
        )

        try:
            msg = await chat.send_message(text, parse_mode="Markdown")
            # Add 👍 or 👎 reaction
            reaction = "👎" if is_error else "👍"
            try:
                await bot.set_message_reaction(
                    chat_id=chat.id,
                    message_id=msg.message_id,
                    reaction=[{"type": "emoji", "emoji": reaction}],
                )
            except Exception as react_err:
                logger.debug(f"Could not set reaction on tool call message: {react_err}")
        except Exception as e:
            logger.warning(f"Failed to send tool call report: {e}")

    def callback(tool_name, args_summary, result_summary, is_error):
        asyncio.run_coroutine_threadsafe(
            _send_tool_report(tool_name, args_summary, result_summary, is_error),
            loop,
        ).add_done_callback(
            lambda f: f.exception() and logger.warning(f"Failed to schedule tool call report: {f.exception()}")
        )


async def _track_task(coro):
    """Track a coroutine as an active task for graceful shutdown."""
    task = asyncio.current_task()
    _active_tasks.add(task)
    try:
        await coro
    finally:
        _active_tasks.discard(task)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming text messages."""
    if _shutdown_event.is_set():
        return
    # Wrap the actual message handling in a task for tracking
    await _track_task(_handle_message_impl(update, context))


async def _handle_message_impl(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Actual implementation of message handling."""
    chat_id = update.effective_chat.id
    lock = _chat_locks[chat_id]

    async with lock:
        user_text = update.message.text.strip()

        if not user_text:
            return

        api_key = context.bot_data["api_key"]
        invoke_url = context.bot_data["invoke_url"]
        model = context.bot_data["model"]

        # Get or create a session for this chat (auto-creates on first message)
        session = _memory_manager.get_or_create_session(str(chat_id), auto_create=True)
        session_key = session.uuid

        # Use the session's history
        history = _chat_histories.setdefault(session_key, [])

        # Send typing indicator
        await update.effective_chat.send_action("typing")

        # Send initial working message that we'll update with progress
        working_msg = await update.message.reply_text("🔧 Working on your request...")
        working_msg_id = working_msg.message_id

        # Build progress callback for Telegram updates that edits the same message
        loop = asyncio.get_running_loop()
        last_update_time = [0]  # Track last update time to avoid rate limiting

        def make_edit_progress_callback():
            def callback(round_num, max_rounds, tool_names):
                # Rate limit progress updates to avoid hitting Telegram limits
                current_time = time.time()
                if current_time - last_update_time[0] < 1.0:  # Max 1 update per second
                    return
                last_update_time[0] = current_time

                tools_str = ", ".join(tool_names)
                text = f"🔧 Round {round_num}/{max_rounds}: {tools_str}"

                # Schedule the coroutine without blocking
                asyncio.run_coroutine_threadsafe(
                    context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=working_msg_id,
                        text=text
                    ),
                    loop
                ).add_done_callback(
                    lambda f: f.exception() and logger.warning(f"Failed to edit progress message: {f.exception()}")
                )
            return callback

        progress_cb = make_edit_progress_callback()
        tool_call_cb = make_tool_call_callback(update.effective_chat, loop, context.bot)

        # Unblock any blocked task with the user's response
        from task_manager import get_task_manager
        tm = get_task_manager()
        active_task = tm.get_active_task(session_key)
        if active_task and active_task.status == "blocked" and active_task.blocker:
            tm.unblock_task(active_task.uuid, user_text)

        # Run the blocking agent_loop in a thread
        try:
            reply, status = await asyncio.to_thread(
                agent_loop, user_text, history, api_key, invoke_url, model,
                progress_callback=progress_cb, session_key=session_key,
                tool_call_callback=tool_call_cb,
            )
        except Exception as e:
            logger.error(f"Error in agent_loop for chat {chat_id}, session {session_key[:8]}: {e}", exc_info=True)
            # Track the unhandled exception from agent_loop
            try:
                tracker = get_error_tracker()
                tracker.record_exception(
                    e,
                    source_module="telegram_bot",
                    source_function="_handle_message_impl",
                    context={"chat_id": str(chat_id), "session_key": session_key},
                    session_key=session_key,
                    severity="critical",
                )
            except Exception:
                pass  # Don't let error tracking break the handler
            reply = None
            status = STATUS_ERROR

        if reply is None:
            # Edit the working message to show error
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=working_msg_id,
                    text="❌ Sorry, an error occurred while processing your request."
                )
            except Exception:
                pass  # If we can't edit, just send a new message
                await update.message.reply_text("Sorry, an error occurred while processing your request.")
            return

        if not reply.strip():
            # Edit the working message to show no response
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=working_msg_id,
                    text="⚠️ (No response generated.)"
                )
            except Exception:
                pass  # If we can't edit, just send a new message
                await update.message.reply_text("(No response generated.)")
            return

        # Append status indicator for incomplete results
        if status == STATUS_MAX_ROUNDS:
            reply += "\n\n-- Reached maximum tool rounds. The task may be incomplete."
        elif status == STATUS_BLOCKED:
            reply += "\n\n-- Task paused. Reply to this message to continue."

        # Update conversation history
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})

        # Update session message count
        _memory_manager.update_session_stats(session_key, len(history))

        # Edit the working message with the final result, prefixed with build hash and session reference
        final_reply = f"[build {COMMIT_HASH}] 👤 {session.display_name}\n{reply}"
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=working_msg_id,
                text=final_reply
            )
        except Exception as e:
            logger.warning(f"Failed to edit final message, sending new one: {e}")
            # If editing fails (e.g., message too old), send as new message
            for chunk in split_message(final_reply):
                if chunk.strip():
                    await update.message.reply_text(chunk)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Tornado request handlers
# ---------------------------------------------------------------------------

class TelegramWebhookHandler(tornado.web.RequestHandler):
    """Handle incoming Telegram updates."""

    def initialize(self, tg_app: Application) -> None:
        self.tg_app = tg_app

    async def post(self) -> None:
        try:
            data = json.loads(self.request.body)
            update = Update.de_json(data, self.tg_app.bot)
            # Queue the update for the Application to process
            await self.tg_app.update_queue.put(update)
        except json.JSONDecodeError:
            logger.warning("Failed to decode Telegram update JSON", exc_info=True)
        except Exception:
            logger.exception("Error processing Telegram update")
        self.set_status(200)
        self.finish()


class GitHubWebhookHandler(tornado.web.RequestHandler):
    """Handle incoming GitHub webhook events."""

    async def post(self) -> None:
        # Verify HMAC signature if secret is configured
        if GITHUB_WEBHOOK_SECRET:
            signature = self.request.headers.get("X-Hub-Signature-256", "")
            expected = "sha256=" + hmac.new(
                GITHUB_WEBHOOK_SECRET.encode(),
                self.request.body,
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(signature, expected):
                logger.warning("GitHub webhook: invalid signature")
                self.set_status(403)
                self.write({"error": "invalid signature"})
                self.finish()
                return

        # Parse the event
        event = self.request.headers.get("X-GitHub-Event", "unknown")
        logger.info(f"GitHub webhook received: {event}")

        if event == "ping":
            self.write({"status": "pong"})
            self.finish()
            return

        # Only handle push events (and only restart on master/main branch)
        if event == "push":
            try:
                payload = json.loads(self.request.body)
                ref = payload.get("ref", "") if isinstance(payload, dict) else ""
            except json.JSONDecodeError:
                logger.warning("GitHub webhook: failed to decode push payload")
                ref = ""

            # Only restart for pushes to master or main branch
            if ref not in ("refs/heads/master", "refs/heads/main"):
                logger.info(f"GitHub webhook: ignoring push to {ref}")
                self.write({"status": "ok", "event": event, "branch": ref, "action": "ignored"})
                self.finish()
                return

            try:
                # Use shield to ensure git pull completes even if the request is
                # cancelled during a restart.
                result = await asyncio.shield(asyncio.to_thread(
                    subprocess.run,
                    ["git", "pull", "--ff-only"],
                    capture_output=True, text=True, timeout=30,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                ))
                pull_output = result.stdout.strip()
                logger.info(f"git pull: {pull_output}")
                if result.returncode != 0:
                    logger.error(f"git pull stderr: {result.stderr.strip()}")
            except asyncio.CancelledError:
                # Silent during shutdown - hot-reload is doing its job.
                logger.info("GitHub webhook: git pull task shielded during shutdown")
                return
            except Exception as e:
                logger.error(f"git pull failed: {e}")
                pull_output = f"error: {e}"

            self.write({"status": "ok", "action": "restarting", "pull": pull_output})
            self.finish()

            # Schedule a graceful shutdown - the hot-reload watcher (or process
            # manager) will restart the server with the new code. If git pull
            # changed files under .git the hot-reload watcher picks it up
            # automatically. We also send SIGTERM as a fallback for non-reload
            # deployments.
            loop = tornado.ioloop.IOLoop.current()
            loop.call_later(1.0, lambda: os.kill(os.getpid(), signal.SIGTERM))
            return

        # Unhandled event types - just acknowledge
        self.write({"status": "ok", "event": event, "action": "ignored"})
        self.finish()


class HealthHandler(tornado.web.RequestHandler):
    """Simple health-check endpoint."""

    def get(self) -> None:
        self.write({"status": "ok", "build": COMMIT_HASH})
        self.finish()


# ---------------------------------------------------------------------------
# Server startup
# ---------------------------------------------------------------------------

def run_telegram_bot(api_key: str, invoke_url: str, model: str) -> None:
    """Start the combined Telegram + GitHub webhook server."""
    global _start_time
    _start_time = time.time()
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN not found in environment or .env file.", file=sys.stderr)
        sys.exit(1)

    webhook_url = os.getenv("TELEGRAM_WEBHOOK_URL")
    if not webhook_url:
        print(
            "Error: TELEGRAM_WEBHOOK_URL not found. Set it to your public URL "
            "(e.g. https://yourdomain.com/telegram).",
            file=sys.stderr,
        )
        sys.exit(1)

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    # Build the Telegram Application (but don't start its built-in server)
    tg_app = Application.builder().token(token).build()
    tg_app.bot_data["api_key"] = api_key
    tg_app.bot_data["invoke_url"] = invoke_url
    tg_app.bot_data["model"] = model

    tg_app.add_handler(CommandHandler("start", handle_start))
    tg_app.add_handler(CommandHandler("clear", handle_clear))
    tg_app.add_handler(CommandHandler("memory", handle_memory))
    tg_app.add_handler(CommandHandler("webhook", handle_webhook))
    tg_app.add_handler(CommandHandler("status", handle_status))
    tg_app.add_handler(CommandHandler("settings", handle_settings))
    tg_app.add_handler(CommandHandler("errors", handle_errors))
    tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Derive the Telegram webhook path from the URL
    tg_path = urlparse(webhook_url).path or "/telegram"

    # Build the tornado application with all routes
    tornado_app = tornado.web.Application([
        (tg_path, TelegramWebhookHandler, {"tg_app": tg_app}),
        (GITHUB_WEBHOOK_PATH, GitHubWebhookHandler),
        (r"/health", HealthHandler),
    ])

    print(f"Starting webhook server on 0.0.0.0:{WEBHOOK_PORT} [build {COMMIT_HASH}]", file=sys.stderr)
    print(f"  Telegram webhook: {webhook_url}", file=sys.stderr)
    print(f"  GitHub webhook: {_get_webhook_url()}", file=sys.stderr)
    print(f"  Health check: http://0.0.0.0:{WEBHOOK_PORT}/health", file=sys.stderr)

    asyncio.run(_run_server(tg_app, tornado_app, webhook_url))


async def _run_server(
    tg_app: Application,
    tornado_app: tornado.web.Application,
    webhook_url: str,
) -> None:
    """Async entry point: initialise the Telegram Application, start tornado, and
    block until a shutdown signal is received."""

    # Initialize settings database with defaults
    init_default_settings()
    logger.info(f"Settings DB initialized with {get_settings_db().get_stats()['total_settings']} settings")

    # Initialise and start the Telegram Application (processing pipeline)
    await tg_app.initialize()
    await tg_app.start()

    # Register the webhook URL with Telegram's API
    await tg_app.bot.set_webhook(url=webhook_url)

    # Start the tornado HTTP server
    server = tornado_app.listen(WEBHOOK_PORT, "0.0.0.0")

    # Wait for shutdown signal
    shutdown_event = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)

    logger.info("Server is ready")

    await shutdown_event.wait()

    logger.info("Shutting down...")
    # Signal shutdown to prevent new tasks
    _shutdown_event.set()

    # Wait for active message processing tasks to complete (with timeout)
    if _active_tasks:
        logger.info(f"Waiting for {len(_active_tasks)} active task(s) to complete...")
        try:
            await asyncio.wait_for(
                asyncio.gather(*_active_tasks, return_exceptions=True),
                timeout=30.0
            )
            logger.info("All active tasks completed")
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for tasks to complete, forcing shutdown")

    server.stop()
    await tg_app.stop()
    await tg_app.shutdown()
