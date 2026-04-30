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
https://build.nvidia.com/moonshotai/kimi-k2-thinking

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

## Settings Database

The agent uses SQLite to store application settings with full support for queryable, categorized configuration. This replaces scattered environment variables with a persistent, type-safe settings API.

### Features

- **Persistent storage** - SQLite database survives restarts
- **Type-safe values** - Support for string, integer, float, boolean, and JSON types
- **Categorized organization** - Group settings by category (agent, telegram, slack, docker, api, ui)
- **Change history** - Track all setting modifications over time
- **Import/Export** - JSON format for backup and migration
- **Bot integration** - Access settings via `/settings` command in Telegram and Slack

### Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `CODING_GUY_SETTINGS_DB` | Path to SQLite database file | `~/coding-guy-workspace/settings.db` |

### Telegram Commands

- `/settings` - Show categories and usage
- `/settings get <key>` - Get a specific setting
- `/settings set <key> <value>` - Set a setting (auto-detects type)
- `/settings list [category]` - List all settings
- `/settings categories` - List all categories
- `/settings export` - Export as JSON

### Slack Commands

- `/coding-guy settings` - Show categories
- `/coding-guy settings list [category]` - List settings
- `/coding-guy settings get <key>` - Get a specific setting

### Usage in Code

```python
from settings_db import get_settings_db, set_setting, get_setting

# Initialize with defaults
from settings_db import init_default_settings
init_default_settings()

# Set a value
db = get_settings_db()
db.set("agent.max_rounds", 500, "integer", "agent", "Maximum tool rounds per request")

# Get a value
value = db.get("agent.max_rounds", default=250)

# Get all in category
settings = db.get_all(category="telegram")
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
- `suno_generate_song` - Generate AI music with custom lyrics
- `suno_get_job_status` - Check song generation progress
- `suno_get_song_data` - Retrieve song metadata and URLs
- `suno_list_songs` - Browse generated songs
- `suno_delete_song` - Delete a generated song

## Suno API Integration

The agent can generate AI music using the Suno API! Configure your `SUNO_API_KEY` in `.env` to enable music generation features.

### Configuration

Add to your `.env` file:
```env
SUNO_API_KEY=your-suno-api-key-here
```

### Music Generation Tools

| Tool | Description | Example |
|------|-------------|---------|
| `suno_generate_song` | Generate a song with lyrics and style | `suno_generate_song(lyrics="Hello world...", style="Pop", title="Coding Anthem")` |
| `suno_get_job_status` | Check generation progress | `suno_get_job_status(job_id="job-abc-123")` |
| `suno_get_song_data` | Get completed song details | `suno_get_song_data(song_id="song-xyz-789")` |
| `suno_list_songs` | Browse your generated songs | `suno_list_songs(limit=10)` |
| `suno_delete_song` | Remove a song | `suno_delete_song(song_id="song-xyz-789")` |

### Workflow
1. Call `suno_generate_song` with your lyrics and desired style
2. Receive a `job_id` - save this!
3. Poll `suno_get_job_status` with the job ID until status is "completed"
4. Call `suno_get_song_data` with the returned `song_id` to get download URLs
5. Download your song from the provided URLs (mp3, wav formats available)

## Commands

### Telegram Commands
- `/start` - Welcome message
- `/clear` - Reset conversation
- `/webhook` - Show GitHub webhook configuration
- `/status` - Server status
- `/settings` - Manage settings (get, set, list, export)

### Slack Commands
- `/coding-guy <question>` - Ask a coding question
- `/coding-guy clear` - Reset conversation
- `/coding-guy status` - Show server status
- `/coding-guy settings` - Manage settings (list, get)
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
