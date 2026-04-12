"""Slack bot integration for the coding agent."""

import asyncio
import json
import logging
import os
import re
import signal
import sys
from collections import defaultdict
from typing import Optional

from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_internal_client import AsyncWebClient
from slack_sdk.errors import SlackApiError

from coding_agent import agent_loop, STATUS_COMPLETE, STATUS_MAX_ROUNDS, STATUS_ERROR, COMMIT_HASH
from settings_db import get_settings_db, init_default_settings

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 40000  # Slack has higher limits than Telegram
PROGRESS_REPORT_INTERVAL = 3  # send a progress update every N tool rounds

# Per-channel conversation history: channel_id -> list of message dicts
_channel_histories: dict[str, list] = {}
# Per-channel locks to serialize message processing
_channel_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

# Server start time for status
_start_time: float = 0.0


def split_message(text: str, max_len: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split text into chunks that fit within Slack's message limit."""
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


def format_slack_message(text: str) -> str:
    """Format text for Slack, ensuring code blocks are preserved."""
    # Code blocks are already in Slack-compatible format
    return text


def sanitize_command_text(text: str) -> str:
    """Remove Slack formatting artifacts from command text."""
    # Remove Slack user mentions like <@U12345678>
    text = re.sub(r'<@[A-Z0-9]+>', '', text)
    # Remove Slack channel mentions like <#C12345678|channel-name>
    text = re.sub(r'<#[A-Z0-9]+\|[^>]+>', lambda m: m.group(0).split('|')[1].rstrip('>'), text)
    # Remove Slack URLs like <http://example.com|example.com>
    text = re.sub(r'<(https?://[^>|]+)\|[^>]+>', r'\1', text)
    # Remove plain URL brackets
    text = re.sub(r'<(https?://[^>]+)>', r'\1', text)
    # Remove special Slack characters
    text = text.replace('\xa0', ' ')  # Non-breaking space
    text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
    return text.strip()


def make_progress_callback(say, channel_id, loop):
    """Return a callback that sends progress updates to a Slack channel.

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

        async def send_progress():
            try:
                await say(text=text)
            except Exception as e:
                logger.warning(f"Failed to send progress update: {e}")

        future = asyncio.run_coroutine_threadsafe(send_progress(), loop)
        try:
            future.result(timeout=10)
        except Exception as e:
            logger.warning(f"Failed to send progress update: {e}")

    return callback


class SlackBot:
    """Slack bot wrapper for the coding agent."""

    def __init__(self, api_key: str, invoke_url: str, model: str):
        self.api_key = api_key
        self.invoke_url = invoke_url
        self.model = model

        # OAuth tokens for socket mode
        self.bot_token = os.getenv("SLACK_BOT_TOKEN")
        self.signing_secret = os.getenv("SLACK_SIGNING_SECRET")
        self.socket_mode_token = os.getenv("SLACK_SOCKET_MODE_TOKEN")

        if not self.bot_token:
            logger.error("SLACK_BOT_TOKEN not found in environment")
            sys.exit(1)

        if not self.socket_mode_token:
            logger.error("SLACK_SOCKET_MODE_TOKEN not found in environment. Socket mode requires an app-level token.")
            sys.exit(1)

        # Initialize the Slack app with socket mode
        self.app = AsyncApp(
            token=self.bot_token,
            signing_secret=self.signing_secret,
        )

        # Cached bot user ID (fetched on first mention)
        self._bot_user_id: Optional[str] = None

        # Setup event handlers
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up event handlers for Slack interactions."""

        @self.app.event("app_mention")
        async def handle_app_mention(event, say, client):
            """Handle when the bot is mentioned in a channel."""
            channel_id = event.get("channel")
            user = event.get("user")
            text = event.get("text", "")

            # Remove bot mention from text - fetch bot user ID once and cache it
            if self._bot_user_id is None:
                self._bot_user_id = (await client.auth_test()).get("user_id", "")
            text = re.sub(f"<@{self._bot_user_id}>", "", text).strip()

            await self._process_message(channel_id, user, text, say)

        @self.app.event("message")
        async def handle_message(event, say, client):
            """Handle direct messages to the bot."""
            # Skip bot messages and system messages
            if event.get("subtype") or event.get("bot_id"):
                return

            channel_id = event.get("channel")
            user = event.get("user")
            text = event.get("text", "")

            # Check if this is a direct message (IM) - check event payload first to avoid API call
            if event.get("channel_type") != "im" and not channel_id.startswith("D"):
                return  # Only process DMs, mentions are handled separately

            await self._process_message(channel_id, user, text, say)

        @self.app.command("/coding-guy")
        async def handle_command(ack, say, command, client):
            """Handle slash command."""
            await ack()

            channel_id = command.get("channel_id")
            user = command.get("user_id")
            text = command.get("text", "").strip()

            # Parse command subcommands
            if text.lower() == "clear":
                _channel_histories.pop(channel_id, None)
                await say(text="*Conversation cleared.*", mrkdwn=True)
                return

            if text.lower() == "status":
                uptime_secs = int(asyncio.get_event_loop().time() - _start_time)
                hours, remainder = divmod(uptime_secs, 3600)
                minutes, seconds = divmod(remainder, 60)
                uptime_str = f"{hours}h {minutes}m {seconds}s"

                # Get settings db stats
                db_stats = get_settings_db().get_stats()

                status_text = (
                    f"*Coding Guy Status*\n"
                    f"• Build: `{COMMIT_HASH}`\n"
                    f"• Uptime: {uptime_str}\n"
                    f"• Active conversations: {len(_channel_histories)}\n"
                    f"• Model: `{self.model}`\n"
                    f"• Settings: {db_stats.get('total_settings', 0)} settings, {db_stats.get('total_categories', 0)} categories"
                )
                await say(text=status_text, mrkdwn=True)
                return

            if text.lower() == "settings":
                db = get_settings_db()
                args = text.split()[1:] if len(text.split()) > 1 else []

                if not args:
                    cats = db.get_categories()
                    lines = ["*Settings Categories*"]
                    for cat in cats:
                        settings = db.get_all(category=cat)
                        lines.append(f"• {cat}: {len(settings)} setting(s)")
                    lines.extend(["", "Use `/coding-guy settings list [category]` or `/coding-guy settings get <key>`"])
                    await say(text="\n".join(lines), mrkdwn=True)
                    return

                subcommand = args[0].lower() if args else ""

                if subcommand == "get" and len(args) >= 2:
                    key = args[1]
                    setting = db.get_setting(key)
                    if setting:
                        text = f"*{key}*\n• Value: `{setting.value}`\n• Type: {setting.value_type}\n• Category: {setting.category}" + (f"\n• Description: {setting.description}" if setting.description else "")
                        await say(text=text, mrkdwn=True)
                    else:
                        await say(text=f"Setting '{key}' not found.", mrkdwn=True)
                    return

                if subcommand == "list":
                    category = args[1] if len(args) > 1 else None
                    settings = db.get_all(category=category)
                    if not settings:
                        await say(text=f"No settings found" + (f" in category '{category}'" if category else ""), mrkdwn=True)
                        return
                    lines = [f"*Settings*" + (f" (category: {category})" if category else "")]
                    for key, val in list(settings.items())[:15]:  # Limit to 15
                        lines.append(f"• `{key}`: {val}")
                    if len(settings) > 15:
                        lines.append(f"• ... and {len(settings) - 15} more")
                    await say(text="\n".join(lines), mrkdwn=True)
                    return

                await say(text="Unknown subcommand. Try `list`, `get <key>`, `set <key> <value>`", mrkdwn=True)
                return

            if text.lower() == "help" or not text:
                help_text = (
                    "*Coding Guy - Your AI coding assistant*\n\n"
                    "*Commands:*\n"
                    "• `/coding-guy <question>` - Ask me anything about coding\n"
                    "• `/coding-guy clear` - Reset the conversation\n"
                    "• `/coding-guy status` - Show server status\n"
                    "• `/coding-guy settings` - Manage settings\n"
                    "• `/coding-guy help` - Show this help message\n\n"
                    "• *DM me* for private conversations\n"
                    "• *Mention me* (@Coding Guy) in channels"
                )
                await say(text=help_text, mrkdwn=True)
                return

            await self._process_message(channel_id, user, text, say)

    async def _process_message(self, channel_id: str, user: str, text: str, say):
        """Process an incoming message through the agent."""
        # Sanitize the text
        text = sanitize_command_text(text)

        if not text:
            return

        lock = _channel_locks[channel_id]
        async with lock:
            history = _channel_histories.setdefault(channel_id, [])

            # Send typing indicator (https://api.slack.com/methods/users.setPresence)
            try:
                await self.app.client.users_setPresence(presence="auto")
            except Exception:
                pass  # Typing indicator is best effort

            # Build progress callback for Slack updates
            loop = asyncio.get_running_loop()
            progress_cb = make_progress_callback(say, channel_id, loop)

            # Run the blocking agent_loop in a thread
            try:
                reply, status = await asyncio.to_thread(
                    agent_loop,
                    text,
                    history,
                    self.api_key,
                    self.invoke_url,
                    self.model,
                    progress_callback=progress_cb,
                )
            except Exception as e:
                logger.error(f"Error in agent_loop for channel {channel_id}: {e}", exc_info=True)
                await say(
                    text="Sorry, an error occurred while processing your request.",
                    thread_ts=None
                )
                return

            if reply is None:
                await say(
                    text="Sorry, an error occurred while processing your request.",
                    thread_ts=None
                )
                return

            if not reply.strip():
                await say(text="_(No response generated.)_", mrkdwn=True)
                return

            # Append status indicator for incomplete results
            if status == STATUS_MAX_ROUNDS:
                reply += "\n\n---\n_Reached maximum tool rounds. The task may be incomplete._"

            # Update conversation history
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": reply})

            # Send reply, formatted for Slack
            reply = f"[build `{COMMIT_HASH}`]\n{reply}"

            # Format and send message chunks
            formatted_reply = format_slack_message(reply)
            for chunk in split_message(formatted_reply):
                if chunk.strip():
                    try:
                        await say(text=chunk, mrkdwn=True)
                    except SlackApiError as e:
                        logger.error(f"Slack API error: {e}")
                        # Try sending without markdown if rich formatting fails
                        await say(text=re.sub(r'(`|\*|_)', '', chunk), mrkdwn=False)

    async def start(self):
        """Start the Slack bot with socket mode."""
        global _start_time
        _start_time = asyncio.get_event_loop().time()

        # Initialize settings database with defaults
        init_default_settings()
        logger.info(f"Settings DB initialized with {get_settings_db().get_stats().get('total_settings', 0)} settings")

        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

        from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

        handler = AsyncSocketModeHandler(
            self.app,
            self.socket_mode_token,
        )

        logger.info(f"Starting Slack bot [build {COMMIT_HASH}]")
        logger.info(f"Socket mode active - will maintain persistent WebSocket connection")

        await handler.start_async()


def run_slack_bot(api_key: str, invoke_url: str, model: str) -> None:
    """Start the Slack bot."""
    bot = SlackBot(api_key, invoke_url, model)
    asyncio.run(bot.start())
