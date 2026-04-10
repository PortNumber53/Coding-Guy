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
- 🔑 API Key Pool for load balancing across multiple keys with automatic cooldown

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

## API Key Pool

The agent supports using multiple API keys for load balancing and improved reliability. When multiple keys are configured, the agent automatically:

- **Selects the key with the lowest current usage** for each request
- **Tracks rate limit hits independently** for each key
- **Cools down keys** that hit rate limits (configurable duration)
- **Automatically recovers** keys after cooldown period

### Configuration Methods

Choose one of these methods to configure multiple keys:

**Method 1: Comma-separated list**
```env
NVIDIA_API_KEYS=key1,key2,key3
```

**Method 2: Numbered keys** (any number of keys)
```env
NVIDIA_API_KEY_0=your-first-api-key
NVIDIA_API_KEY_1=your-second-api-key
NVIDIA_API_KEY_2=your-third-api-key
```

**Method 3: Single key** (backward compatible)
```env
NVIDIA_API_KEY=your-api-key
```

### API Key Pool Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `API_KEY_COOLDOWN` | Seconds to cooldown a key after rate limit hit | `60` |

### Key Health Monitoring

When a key hits a rate limit (429 error), it enters cooldown mode:

- **First rate limit**: Key continues to be used but is flagged
- **Consecutive rate limits**: Key enters cooldown with exponential backoff
- **Cooldown period**: 60s, 120s, 240s, etc. (max 10 minutes)
- **Auto-recovery**: Key becomes available again after cooldown

You'll see messages like:
```
[Pool] Key key_1 hit rate limit, recorded cooldown
API key pool initialized with 3 keys
```

## Rate Limiting

The agent includes built-in rate limiting to minimize 429 (Rate Limit) errors from the LLM API. This is particularly important when making multiple sequential requests or using the agent interactively.

### Rate Limiting Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **`adaptive`** (default) | Automatically adjusts delay based on 429 errors. Increases delay after rate limit hits, decreases on success. | Best for most use cases - balances performance and reliability |
| **`fixed`** | Enforces a constant delay between requests. | Predictable timing, good for batch operations |
| **`token_bucket`** | Token bucket algorithm allows request bursts. | Good when you need occasional bursts of requests |
| **`none`** | Disables rate limiting entirely. | Not recommended - will likely hit 429 errors |

### Configuration via Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RATE_LIMIT_STRATEGY` | Rate limiting strategy: `adaptive`, `fixed`, `token_bucket`, or `none` | `adaptive` |
| `RATE_LIMIT_INITIAL_DELAY` | Initial delay between requests in seconds | `0.5` |
| `RATE_LIMIT_MIN_DELAY` | Minimum delay (adaptive only) | `0.1` |
| `RATE_LIMIT_MAX_DELAY` | Maximum delay (adaptive only) | `60.0` |

### Command-Line Arguments

| Flag | Description | Example |
|------|-------------|---------|
| `--rate-limit-strategy` | Choose rate limiting strategy | `--rate-limit-strategy fixed` |
| `--rate-limit-initial-delay` | Initial delay between requests (seconds) | `--rate-limit-initial-delay 1.0` |
| `--rate-limit-min-delay` | Minimum delay for adaptive mode (seconds) | `--rate-limit-min-delay 0.2` |
| `--rate-limit-max-delay` | Maximum delay for adaptive mode (seconds) | `--rate-limit-max-delay 30.0` |

### Usage Examples

```bash
# Use fixed delay of 1 second between requests
python coding_agent.py --rate-limit-strategy fixed --rate-limit-initial-delay 1.0

# Use adaptive rate limiting with custom delays
python coding_agent.py --rate-limit-strategy adaptive --rate-limit-initial-delay 0.5 --rate-limit-min-delay 0.2 --rate-limit-max-delay 30.0

# Disable rate limiting entirely (not recommended)
python coding_agent.py --rate-limit-strategy none

# Use token bucket for burst-friendly operations
python coding_agent.py --rate-limit-strategy token_bucket

# Configure for Slack bot with conservative limits
python coding_agent.py --slack --rate-limit-strategy adaptive --rate-limit-initial-delay 1.0

# Configure for Telegram bot with aggressive limits
python coding_agent.py --serve --rate-limit-strategy adaptive --rate-limit-initial-delay 0.5 --rate-limit-min-delay 0.1
```

### Rate Limiting Output

When rate limiting is active, you'll see messages in stderr:

```
[Rate limit] Waiting 0.50s before next request
[Rate limit] Recorded 429 error. Adaptive delay may increase.
```

The agent will automatically handle retries for 429 errors with exponential backoff.

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
