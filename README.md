# Coding-Guy

An AI-powered coding agent that integrates with Telegram and Slack, powered by Nvidia's API (Kimi K2.5 model).

## Features

- 🤖 AI-powered coding assistant with tool use capabilities
- 💬 Telegram bot integration with GitHub webhooks
- 💼 Slack bot integration with Socket Mode
- 🐳 Docker sandbox for safe code execution
- 🔧 Multiple tools: file operations, command execution, web requests, and more
- 🔄 Hot-reload for development
- ⏰ Rate limiting to minimize 429 errors from the LLM API

## Quick Start

1. Clone the repository:
```bash
git clone <your-repo>
cd Coding-Guy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and configure your API keys:
```bash
cp .env.example .env
```

### Nvidia API Key

Get your free key here:
https://build.nvidia.com/moonshotai/kimi-k2.5

## Rate Limiting

The agent includes built-in rate limiting to minimize 429 (Rate Limit) errors from the LLM API. Configure via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `RATE_LIMIT_STRATEGY` | Rate limiting strategy: `adaptive`, `fixed`, `token_bucket`, or `none` | `adaptive` |
| `RATE_LIMIT_INITIAL_DELAY` | Initial delay between requests in seconds | `0.5` |
| `RATE_LIMIT_MIN_DELAY` | Minimum delay (for adaptive strategy) | `0.1` |
| `RATE_LIMIT_MAX_DELAY` | Maximum delay (for adaptive strategy) | `60.0` |

### Rate Limiting Strategies

- **`adaptive`** (default): Automatically adjusts delay based on 429 errors. Starts at `INITIAL_DELAY` and increases after 429s, decreasing slowly on success.
- **`fixed`**: Enforces a constant delay between requests (uses `INITIAL_DELAY`).
- **`token_bucket`**: Token bucket algorithm for burst handling.
- **`none`**: Disables rate limiting entirely.

### Command Line (Interactive)
```bash
python coding_agent.py
```

### Telegram Bot
```bash
python coding_agent.py --serve
```

For hot-reload during development:
```bash
python coding_agent.py --serve --reload
```

### Slack Bot
```bash
python coding_agent.py --slack
```

## Configuration

### Slack Bot Setup

1. Go to https://api.slack.com/apps and create a new app
2. Go to **OAuth & Permissions** and add these scopes:
   - `chat:write` - Send messages
   - `im:history` - Read direct message history
   - `app_mentions:read` - Receive app mentions
   - `commands` - Add slash commands
   - `mentions:read` - Read mentions

3. Go to **Slash Commands** and create:
   - Command: `/coding-guy`
   - Request URL: (leave empty for Socket Mode)
   - Description: "Ask the coding agent a question"

4. Go to **Socket Mode** and enable it
5. Go to **Basic Information** and generate an **App-Level Token** with `connections:write` scope
6. Add these to your `.env` file:
```env
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_SIGNING_SECRET=your-signing-secret
SLACK_SOCKET_MODE_TOKEN=xapp-your-app-level-token
```

7. Go to **App Home** and enable:
   - ✅ Messages Tab
   - ✅ Home Tab

8. Go to **Event Subscriptions** and enable:
   - ✅ App mentions (`app_mention`)
   - ✅ Messages (Bot DM) (`message.im`)

9. Install/Reinstall the app to your workspace

## Tools

The agent has access to these tools:

- `read_file` - Read file contents
- `write_file` - Create new files
- `patch_file` - Apply search-and-replace patches
- `grep_file` - Search for patterns in files
- `ls_file` - List directory contents
- `execute_command` - Run shell commands
- `multi_read_file` / `multi_write_file` - Batch file operations
- `read_dockerfile` / `write_dockerfile` / `rebuild_container` - Docker management
- `web` - HTTP requests
- `ask_ollama` - Local LLM queries

## Commands

### Telegram Commands
- `/start` - Welcome message
- `/clear` - Reset conversation
- `/webhook` - Show GitHub webhook configuration
- `/status` - Server status

### Slack Commands
- `/coding-guy <question>` - Ask a coding question
- `/coding-guy clear` - Reset conversation
- `/coding-guy status` - Show server status
- `/coding-guy help` - Show help

You can also:
- **DM the bot** directly for private conversations
- **Mention the bot** with `@Coding Guy` in channels

## Git Integration

The agent can clone, pull, and push to GitHub repositories. SSH keys are automatically forwarded or you can use `GIT_TOKEN` for HTTPS authentication.

## Development

For development with hot-reload, use `docker_manager.py` and `telegram_bot.py` as examples for extending the bot's functionality.

## License

MIT
