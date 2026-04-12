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

from coding_agent import agent_loop, STATUS_COMPLETE, STATUS_MAX_ROUNDS, STATUS_ERROR, COMMIT_HASH
from settings_db import get_settings_db, init_default_settings, CATEGORY_AGENT, CATEGORY_TELEGRAM

logger = logging.getLogger(__name__)

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "21031"))
PROGRESS_REPORT_INTERVAL = 3 # send a progress update every N tool rounds

# GitHub webhook configuration
GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "")
GITHUB_WEBHOOK_PATH = os.getenv("GITHUB_WEBHOOK_PATH", "/github-webhook")

# Per-chat conversation history: chat_id -> list of message dicts
_chat_histories: dict[int, list] = {}
# Per-chat locks to serialize message processing
_chat_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

# Track active message processing tasks for graceful shutdown
_active_tasks: set[asyncio.Task] = set()
_shutdown_event: asyncio.Event = asyncio.Event()

# Server start time for /status
_start_time: float = 0.0


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
        "Commands:\n"
        "/clear - Reset the conversation\n"
        "/webhook - Show GitHub webhook configuration\n"
        "/status - Show server status\n"
        "/settings - Show/manage settings"
    )


async def handle_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /clear command."""
    chat_id = update.effective_chat.id
    _chat_histories.pop(chat_id, None)
    _chat_locks.pop(chat_id, None)
    await update.message.reply_text("Conversation cleared.")


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

    lines = [
        f"Build: {COMMIT_HASH}",
        f"Uptime: {uptime_str}",
        f"Active chats: {len(_chat_histories)}",
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
        history = _chat_histories.setdefault(chat_id, [])

        # Send typing indicator
        await update.effective_chat.send_action("typing")

        # Build progress callback for Telegram updates
        loop = asyncio.get_running_loop()
        progress_cb = make_progress_callback(update.effective_chat, loop)

        # Run the blocking agent_loop in a thread
        try:
            reply, status = await asyncio.to_thread(
                agent_loop, user_text, history, api_key, invoke_url, model,
                progress_callback=progress_cb,
            )
        except Exception as e:
            logger.error(f"Error in agent_loop for chat {chat_id}: {e}", exc_info=True)
            reply = None
            status = STATUS_ERROR

        if reply is None:
            await update.message.reply_text("Sorry, an error occurred while processing your request.")
            return

        if not reply.strip():
            await update.message.reply_text("(No response generated.)")
            return

        # Append status indicator for incomplete results
        if status == STATUS_MAX_ROUNDS:
            reply += "\n\n-- Reached maximum tool rounds. The task may be incomplete."

        # Update conversation history
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})

        # Send reply, prefixed with build hash, splitting if necessary
        reply = f"[build {COMMIT_HASH}]\n{reply}"
        for chunk in split_message(reply):
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
                result = await asyncio.to_thread(
                    subprocess.run,
                    ["git", "pull", "--ff-only"],
                    capture_output=True, text=True, timeout=30,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                )
                pull_output = result.stdout.strip()
                logger.info(f"git pull: {pull_output}")
                if result.returncode != 0:
                    logger.error(f"git pull stderr: {result.stderr.strip()}")
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
    tg_app.add_handler(CommandHandler("webhook", handle_webhook))
    tg_app.add_handler(CommandHandler("status", handle_status))
    tg_app.add_handler(CommandHandler("settings", handle_settings))
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
