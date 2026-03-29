"""Telegram bot integration for the coding agent."""

import asyncio
import logging
import os
import sys
from collections import defaultdict

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from coding_agent import agent_loop, STATUS_COMPLETE, STATUS_MAX_ROUNDS, STATUS_ERROR

logger = logging.getLogger(__name__)

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "21031"))
PROGRESS_REPORT_INTERVAL = 3  # send a progress update every N tool rounds

# Per-chat conversation history: chat_id -> list of message dicts
_chat_histories: dict[int, list] = {}
# Per-chat locks to serialize message processing
_chat_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)


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


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    await update.message.reply_text(
        "Coding Agent ready. Send me a message and I'll help you with coding tasks.\n"
        "Use /clear to reset the conversation."
    )


async def handle_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /clear command."""
    chat_id = update.effective_chat.id
    _chat_histories.pop(chat_id, None)
    _chat_locks.pop(chat_id, None)
    await update.message.reply_text("Conversation cleared.")


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


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming text messages."""
    chat_id = update.effective_chat.id
    lock = _chat_locks[chat_id]

    async with lock:
        user_text = update.message.text.strip()

        if not user_text:
            return

        api_key = context.bot_data["api_key"]
        history = _chat_histories.setdefault(chat_id, [])

        # Send typing indicator
        await update.effective_chat.send_action("typing")

        # Build progress callback for Telegram updates
        loop = asyncio.get_running_loop()
        progress_cb = make_progress_callback(update.effective_chat, loop)

        # Run the blocking agent_loop in a thread
        try:
            reply, status = await asyncio.to_thread(
                agent_loop, user_text, history, api_key,
                progress_callback=progress_cb,
            )
        except Exception as e:
            logger.error(f"Error in agent_loop for chat {chat_id}: {e}", exc_info=True)
            reply = None
            status = STATUS_ERROR

        if reply is None:
            await update.message.reply_text("Sorry, an error occurred while processing your request.")
            return

        # Append status indicator for incomplete results
        if status == STATUS_MAX_ROUNDS:
            reply += "\n\n-- Reached maximum tool rounds. The task may be incomplete."

        # Update conversation history
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})

        # Send reply, splitting if necessary
        for chunk in split_message(reply):
            await update.message.reply_text(chunk)


def run_telegram_bot(api_key: str) -> None:
    """Start the Telegram bot webhook server."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN not found in environment or .env file.", file=sys.stderr)
        sys.exit(1)

    webhook_url = os.getenv("TELEGRAM_WEBHOOK_URL")
    if not webhook_url:
        print(
            "Error: TELEGRAM_WEBHOOK_URL not found. Set it to your public URL "
            "(e.g. https://yourdomain.com/bot-webhook).",
            file=sys.stderr,
        )
        sys.exit(1)

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    app = Application.builder().token(token).build()
    app.bot_data["api_key"] = api_key

    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(CommandHandler("clear", handle_clear))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print(f"Starting Telegram webhook server on 0.0.0.0:{WEBHOOK_PORT}", file=sys.stderr)
    print(f"Webhook URL: {webhook_url}", file=sys.stderr)

    app.run_webhook(
        listen="0.0.0.0",
        port=WEBHOOK_PORT,
        webhook_url=webhook_url,
    )
