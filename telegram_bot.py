"""Telegram bot integration for the coding agent."""

import asyncio
import logging
import os
import sys

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from coding_agent import agent_loop

logger = logging.getLogger(__name__)

TELEGRAM_MAX_MESSAGE_LENGTH = 4096

# Per-chat conversation history: chat_id -> list of message dicts
_chat_histories: dict[int, list] = {}


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
    await update.message.reply_text("Conversation cleared.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming text messages."""
    chat_id = update.effective_chat.id
    user_text = update.message.text.strip()

    if not user_text:
        return

    api_key = context.bot_data["api_key"]
    history = _chat_histories.setdefault(chat_id, [])

    # Send typing indicator
    await update.effective_chat.send_action("typing")

    # Run the blocking agent_loop in a thread
    reply = await asyncio.to_thread(agent_loop, user_text, history, api_key)

    if reply is None:
        await update.message.reply_text("Sorry, an error occurred while processing your request.")
        return

    # Update conversation history
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})

    # Send reply, splitting if necessary
    for chunk in split_message(reply):
        await update.message.reply_text(chunk)


def run_telegram_bot(api_key: str, docker_manager) -> None:
    """Start the Telegram bot webhook server on 0.0.0.0:21031."""
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

    print(f"Starting Telegram webhook server on 0.0.0.0:21031", file=sys.stderr)
    print(f"Webhook URL: {webhook_url}", file=sys.stderr)

    app.run_webhook(
        listen="0.0.0.0",
        port=21031,
        webhook_url=webhook_url,
    )
