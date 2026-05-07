"""Semantic tool registry — rich metadata for every tool the agent can use.

This module provides:
  - TOOL_REGISTRY: list of ToolEntry dicts with name, description, keywords,
    usage_examples, and capability_tags for each tool
  - get_registry(): returns the full registry (including dynamic MCP tools)
  - get_tool_entry(name): returns a single ToolEntry or None
  - build_registry_from_tools(): builds registry from TOOL_DEFINITIONS when
    the full registry isn't available (e.g. MCP tools added at runtime)

The registry is the foundation for semantic tool search — each entry contains
enriched metadata that makes embedding-based similarity search effective.
"""

from tools import TOOL_DEFINITIONS, TOOL_HANDLERS


# ---------------------------------------------------------------------------
# Tool registry entries — each tool gets rich semantic metadata
# ---------------------------------------------------------------------------

# Capability tags are small ontology terms that describe WHAT a tool does,
# independent of its name. This enables semantic search to match "load image"
# to "browser_navigate" even though the names share no words.

TOOL_REGISTRY = [
    {
        "name": "read_file",
        "description": "Read the contents of a file at the given path.",
        "keywords": ["file", "read", "contents", "open", "load", "cat", "view", "inspect"],
        "capability_tags": ["file_io", "read", "inspect_code"],
        "usage_examples": [
            "read_file(path='src/main.py')",
            "read_file(path='/workspace/config.json')",
        ],
        "category": "file_operations",
    },
    {
        "name": "write_file",
        "description": "Write content to a file, creating it and parent directories if needed.",
        "keywords": ["file", "write", "create", "save", "store", "new file", "output"],
        "capability_tags": ["file_io", "write", "create_file"],
        "usage_examples": [
            "write_file(path='src/app.py', content='print(\"hello\")')",
            "write_file(path='config.yaml', content='key: value')",
        ],
        "category": "file_operations",
    },
    {
        "name": "patch_file",
        "description": (
            "Apply search-and-replace patches to an existing file. "
            "Each patch replaces only the *first* occurrence of an 'old' string with a 'new' string. "
            "Patches are applied in order. Use this for targeted edits instead of rewriting entire files."
        ),
        "keywords": ["file", "edit", "patch", "modify", "update", "replace", "fix", "change", "search-replace"],
        "capability_tags": ["file_io", "edit", "modify_code", "patch"],
        "usage_examples": [
            "patch_file(path='src/main.py', patches=[{'old': 'def hello():', 'new': 'def greet():'}])",
            "patch_file(path='config.py', patches=[{'old': 'DEBUG = False', 'new': 'DEBUG = True'}])",
        ],
        "category": "file_operations",
    },
    {
        "name": "grep_file",
        "description": (
            "Search for a regex pattern in files. Returns matching lines with "
            "line numbers. Useful for finding code, functions, variables, or "
            "any text pattern across the project."
        ),
        "keywords": ["search", "find", "grep", "pattern", "regex", "locate", "scan", "text search"],
        "capability_tags": ["search", "find", "code_navigation"],
        "usage_examples": [
            "grep_file(pattern='def main', path='src/')",
            "grep_file(pattern='TODO|FIXME', recursive=True)",
        ],
        "category": "search",
    },
    {
        "name": "ls_file",
        "description": "List the contents of a directory with details (permissions, size, dates).",
        "keywords": ["list", "directory", "ls", "files", "folder", "browse", "explore", "contents"],
        "capability_tags": ["file_io", "list", "explore"],
        "usage_examples": [
            "ls_file(path='/workspace')",
            "ls_file(path='src/components')",
        ],
        "category": "file_operations",
    },
    {
        "name": "execute_command",
        "description": (
            "Execute a shell command inside the Docker sandbox. Use this to "
            "run builds, tests, scripts, install packages, or any shell command. "
            "Returns stdout, stderr, and exit code."
        ),
        "keywords": ["command", "shell", "bash", "execute", "run", "build", "test", "install", "script"],
        "capability_tags": ["execution", "shell", "build", "test", "install", "run_command"],
        "usage_examples": [
            "execute_command(command='npm test')",
            "execute_command(command='pip install flask', working_dir='/workspace')",
            "execute_command(command='go run main.go')",
        ],
        "category": "execution",
    },
    {
        "name": "multi_read_file",
        "description": "Read multiple files at once. Returns an array of results, each with path and content (or error).",
        "keywords": ["read", "multiple files", "batch read", "several files", "read many"],
        "capability_tags": ["file_io", "read", "batch"],
        "usage_examples": [
            "multi_read_file(paths=['src/a.py', 'src/b.py', 'src/c.py'])",
        ],
        "category": "file_operations",
    },
    {
        "name": "multi_write_file",
        "description": "Write multiple files at once. Each entry needs a path and content. Creates parent directories as needed.",
        "keywords": ["write", "multiple files", "batch write", "create files", "save many"],
        "capability_tags": ["file_io", "write", "batch", "create_file"],
        "usage_examples": [
            "multi_write_file(files=[{'path': 'a.py', 'content': '...'}, {'path': 'b.py', 'content': '...'}])",
        ],
        "category": "file_operations",
    },
    {
        "name": "rebuild_container",
        "description": (
            "Rebuild and restart the Docker sandbox. Use this after calling "
            "write_dockerfile to apply Dockerfile changes. The workspace "
            "files are preserved across rebuilds."
        ),
        "keywords": ["docker", "rebuild", "container", "restart", "sandbox", "rebuild sandbox"],
        "capability_tags": ["docker", "rebuild", "environment"],
        "usage_examples": [
            "rebuild_container()",
        ],
        "category": "docker",
    },
    {
        "name": "read_dockerfile",
        "description": (
            "Read the current Dockerfile used to build the sandbox. "
            "Returns the content whether it's a custom Dockerfile or the "
            "embedded default. This reads from the host, not inside the container."
        ),
        "keywords": ["dockerfile", "docker", "read", "config", "sandbox", "container config"],
        "capability_tags": ["docker", "read", "configuration"],
        "usage_examples": [
            "read_dockerfile()",
        ],
        "category": "docker",
    },
    {
        "name": "write_dockerfile",
        "description": (
            "Write or update the Dockerfile used to build the sandbox. "
            "This writes to the host filesystem. After writing, call "
            "rebuild_container to apply the changes."
        ),
        "keywords": ["dockerfile", "docker", "write", "update", "config", "container config", "add package"],
        "capability_tags": ["docker", "write", "configuration", "add_dependency"],
        "usage_examples": [
            "write_dockerfile(content='FROM ubuntu:22.04\\nRUN apt-get install nodejs')",
        ],
        "category": "docker",
    },
    {
        "name": "web",
        "description": (
            "Make an HTTP request. Use this to fetch web pages, call APIs, "
            "download documentation, etc."
        ),
        "keywords": ["http", "request", "fetch", "url", "api", "download", "web page", "rest", "get", "post"],
        "capability_tags": ["network", "http", "fetch", "api_call", "download"],
        "usage_examples": [
            "web(url='https://api.example.com/data')",
            "web(url='https://api.example.com/submit', method='POST', body='{\"key\": \"value\"}')",
        ],
        "category": "network",
    },
    {
        "name": "ask_ollama",
        "description": (
            "Delegate a sub-task or ask a question to a local Ollama model. "
            "Useful for text generation, summarizing, or analyzing data using a local open-source model."
        ),
        "keywords": ["ollama", "local model", "ai", "ask", "subtask", "summarize", "analyze", "delegate", "llm"],
        "capability_tags": ["ai", "llm", "subtask", "summarize", "analyze"],
        "usage_examples": [
            "ask_ollama(prompt='Summarize this code: ...')",
            "ask_ollama(prompt='Write a unit test for function X', model='gemma4:e4b')",
        ],
        "category": "ai",
    },
    {
        "name": "browser_navigate",
        "description": "Navigate to a URL using a headless browser.",
        "keywords": ["browser", "navigate", "url", "web page", "open", "go to", "visit", "load page"],
        "capability_tags": ["browser", "navigate", "web_interaction"],
        "usage_examples": [
            "browser_navigate(url='https://example.com')",
            "browser_navigate(url='https://docs.python.org', wait_until='networkidle')",
        ],
        "category": "browser",
    },
    {
        "name": "browser_action",
        "description": "Perform an action like click, type, or press on the current page.",
        "keywords": ["browser", "click", "type", "press", "interact", "button", "form", "input", "fill"],
        "capability_tags": ["browser", "interact", "click", "type", "form_fill"],
        "usage_examples": [
            "browser_action(action='click', selector='#submit-btn')",
            "browser_action(action='type', selector='#search', text='hello world')",
            "browser_action(action='press', selector='body', key='Enter')",
        ],
        "category": "browser",
    },
    {
        "name": "browser_get_content",
        "description": (
            "Get the text content of the current page, cleaned of HTML tags and scripts."
        ),
        "keywords": ["browser", "content", "text", "extract", "scrape", "page text", "read page"],
        "capability_tags": ["browser", "extract", "scrape", "read"],
        "usage_examples": [
            "browser_get_content()",
            "browser_get_content(include_images=True)",
        ],
        "category": "browser",
    },
    {
        "name": "browser_get_elements",
        "description": "Extract text and attributes from elements matching a CSS selector.",
        "keywords": ["browser", "elements", "css", "selector", "extract", "scrape", "find elements", "attributes"],
        "capability_tags": ["browser", "extract", "scrape", "css_selector"],
        "usage_examples": [
            "browser_get_elements(selector='.product-title', attributes=['href'])",
            "browser_get_elements(selector='a.download-link')",
        ],
        "category": "browser",
    },
    {
        "name": "browser_close",
        "description": "Close the browser and release resources.",
        "keywords": ["browser", "close", "cleanup", "release", "stop browser"],
        "capability_tags": ["browser", "cleanup"],
        "usage_examples": [
            "browser_close()",
        ],
        "category": "browser",
    },
    {
        "name": "suno_generate_song",
        "description": (
            "Generate an AI song using Suno. Creates a musical composition with vocals based on provided lyrics and style. "
            "Returns a job ID that can be used to check generation status. "
            "Use wait_for_completion=True to poll until the song is ready."
        ),
        "keywords": ["suno", "music", "song", "generate", "ai music", "audio", "vocals", "compose", "create song"],
        "capability_tags": ["music", "generate", "audio", "ai_creation"],
        "usage_examples": [
            "suno_generate_song(lyrics='Verse 1: ...', style='Upbeat pop with female vocals')",
            "suno_generate_song(lyrics='...', style='Rap with heavy beat', title='My Song')",
        ],
        "category": "music",
    },
    {
        "name": "suno_get_job_status",
        "description": (
            "Check the status of a song generation job submitted to Suno. "
            "Returns current status (pending, processing, completed, failed) and progress information."
        ),
        "keywords": ["suno", "status", "job", "progress", "check", "song status", "music generation"],
        "capability_tags": ["music", "status_check"],
        "usage_examples": [
            "suno_get_job_status(job_id='abc-123')",
        ],
        "category": "music",
    },
    {
        "name": "suno_get_song_data",
        "description": (
            "Retrieve complete metadata and URLs for a generated song from Suno. "
            "Includes title, artist, duration, audio download URLs, and cover image. "
            "Use this after a job status shows 'completed'."
        ),
        "keywords": ["suno", "song data", "download", "audio", "metadata", "mp3", "get song"],
        "capability_tags": ["music", "download", "metadata"],
        "usage_examples": [
            "suno_get_song_data(song_id='song-456')",
        ],
        "category": "music",
    },
    {
        "name": "suno_list_songs",
        "description": (
            "List all songs generated by the user in Suno. "
            "Supports pagination and status filtering."
        ),
        "keywords": ["suno", "list", "songs", "music", "history", "browse songs"],
        "capability_tags": ["music", "list", "history"],
        "usage_examples": [
            "suno_list_songs(status='completed')",
            "suno_list_songs(limit=10)",
        ],
        "category": "music",
    },
    {
        "name": "suno_delete_song",
        "description": (
            "Delete a generated song from Suno. "
            "This permanently removes the song and all associated data."
        ),
        "keywords": ["suno", "delete", "song", "remove", "music"],
        "capability_tags": ["music", "delete"],
        "usage_examples": [
            "suno_delete_song(song_id='song-789')",
        ],
        "category": "music",
    },
    {
        "name": "create_task",
        "description": (
            "Create a tracked task with ordered steps. Use this at the start of any "
            "non-trivial task to plan your work. Each step should be a concrete action. "
            "Update steps as you progress. This helps you resume if errors occur."
        ),
        "keywords": ["task", "create", "plan", "track", "steps", "todo", "organize", "workflow"],
        "capability_tags": ["task_management", "create", "plan", "organize"],
        "usage_examples": [
            "create_task(description='Fix login bug', steps=['Reproduce bug', 'Find root cause', 'Fix code', 'Test'])",
        ],
        "category": "task_management",
    },
    {
        "name": "update_task_step",
        "description": (
            "Update the status of a task step. Mark steps in_progress when you start them, "
            "completed when done, failed if an error occurs, or skipped if no longer needed."
        ),
        "keywords": ["task", "update", "step", "progress", "status", "mark", "complete", "fail", "skip"],
        "capability_tags": ["task_management", "update", "progress"],
        "usage_examples": [
            "update_task_step(task_id='abc', step_index=0, status='in_progress')",
            "update_task_step(task_id='abc', step_index=1, status='completed', result='Done')",
        ],
        "category": "task_management",
    },
    {
        "name": "complete_task",
        "description": (
            "Mark a task as completed. Call this when all work is done and verified. "
            "Include a brief result summary."
        ),
        "keywords": ["task", "complete", "finish", "done", "mark done", "close task"],
        "capability_tags": ["task_management", "complete", "finish"],
        "usage_examples": [
            "complete_task(task_id='abc', result='Login bug fixed and tested')",
        ],
        "category": "task_management",
    },
    {
        "name": "ask_human",
        "description": (
            "Ask the human a question and pause the task until they respond. "
            "Use this when you need: a decision requiring human judgment, "
            "credentials or access only the human can provide, guidance after "
            "failed workarounds, or approval before destructive changes. "
            "The task will be blocked until the human responds."
        ),
        "keywords": ["ask", "human", "question", "help", "input", "decision", "approval", "clarification", "blocked"],
        "capability_tags": ["human_interaction", "ask", "pause", "approval"],
        "usage_examples": [
            "ask_human(question='Should I delete the old migration files?')",
            "ask_human(question='What is the API key for the staging server?')",
        ],
        "category": "human_interaction",
    },
    {
        "name": "list_tasks",
        "description": "List tracked tasks, optionally filtered by status (pending, in_progress, completed, failed, blocked).",
        "keywords": ["task", "list", "tasks", "show", "view", "status", "filter"],
        "capability_tags": ["task_management", "list", "view"],
        "usage_examples": [
            "list_tasks(status='in_progress')",
            "list_tasks()",
        ],
        "category": "task_management",
    },
    {
        "name": "list_errors",
        "description": "List tracked errors from the error database, optionally filtered by type, severity, or resolved status.",
        "keywords": ["error", "list", "bugs", "issues", "track", "debug", "find errors"],
        "capability_tags": ["error_tracking", "list", "debug"],
        "usage_examples": [
            "list_errors(severity='high')",
            "list_errors(resolved=False, limit=10)",
        ],
        "category": "error_tracking",
    },
    {
        "name": "get_error_details",
        "description": "Get full details of a specific tracked error, including stack trace and context.",
        "keywords": ["error", "details", "stack trace", "debug", "investigate", "error info"],
        "capability_tags": ["error_tracking", "details", "debug", "investigate"],
        "usage_examples": [
            "get_error_details(error_id=42)",
        ],
        "category": "error_tracking",
    },
    {
        "name": "resolve_error",
        "description": "Mark a tracked error as resolved after fixing it.",
        "keywords": ["error", "resolve", "fix", "mark resolved", "close error", "acknowledge"],
        "capability_tags": ["error_tracking", "resolve", "fix"],
        "usage_examples": [
            "resolve_error(error_id=42)",
        ],
        "category": "error_tracking",
    },
    {
        "name": "get_error_summary",
        "description": "Get a summary of tracked errors, including counts by type and severity, and top recurring errors.",
        "keywords": ["error", "summary", "stats", "overview", "error count", "health"],
        "capability_tags": ["error_tracking", "summary", "stats"],
        "usage_examples": [
            "get_error_summary()",
        ],
        "category": "error_tracking",
    },
]


# ---------------------------------------------------------------------------
# Registry access helpers
# ---------------------------------------------------------------------------

_registry_by_name: dict | None = None


def _build_name_index() -> dict:
    """Build a name→entry lookup dict from the registry."""
    return {entry["name"]: entry for entry in TOOL_REGISTRY}


def get_registry() -> list[dict]:
    """Return the full tool registry, including any dynamically added MCP tools.

    For tools not in the static registry (e.g. MCP tools), a minimal entry
    is synthesized from TOOL_DEFINITIONS.
    """
    entries = list(TOOL_REGISTRY)
    known_names = {e["name"] for e in entries}

    # Add entries for tools in TOOL_DEFINITIONS that aren't in the static registry
    for tdef in TOOL_DEFINITIONS:
        name = tdef["function"]["name"]
        if name not in known_names:
            entries.append({
                "name": name,
                "description": tdef["function"].get("description", ""),
                "keywords": name.replace("_", " ").split(),
                "capability_tags": [name.replace("_", " ")],
                "usage_examples": [],
                "category": "dynamic",
            })
            known_names.add(name)

    return entries


def get_tool_entry(name: str) -> dict | None:
    """Get a single tool entry by name, or None if not found."""
    global _registry_by_name
    if _registry_by_name is None:
        _registry_by_name = _build_name_index()

    entry = _registry_by_name.get(name)
    if entry:
        return entry

    # Check dynamic tools (MCP, etc.)
    for tdef in TOOL_DEFINITIONS:
        if tdef["function"]["name"] == name:
            return {
                "name": name,
                "description": tdef["function"].get("description", ""),
                "keywords": name.replace("_", " ").split(),
                "capability_tags": [name.replace("_", " ")],
                "usage_examples": [],
                "category": "dynamic",
            }
    return None


def build_search_text(entry: dict) -> str:
    """Build a single text blob from a tool entry for embedding/search.

    Combines name, description, keywords, capability_tags, and category
    into one searchable string with strategic repetition for TF-IDF weight.
    """
    parts = [
        entry["name"],
        entry["name"].replace("_", " "),
        entry.get("description", ""),
        " ".join(entry.get("keywords", [])),
        " ".join(entry.get("capability_tags", [])),
        entry.get("category", ""),
    ]
    # Add usage examples (they contain realistic parameter names)
    for ex in entry.get("usage_examples", []):
        parts.append(ex)

    return " ".join(parts)


def reset_name_index():
    """Reset the name index (call after MCP tools are loaded)."""
    global _registry_by_name
    _registry_by_name = None
