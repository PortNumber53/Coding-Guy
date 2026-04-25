#!/usr/bin/env python3
"""Coding agent powered by Nvidia API (Kimi K2.5 model) with tool use."""

import argparse
import json
import os
import subprocess
import sys
import time

import requests
from dotenv import load_dotenv

from openrouter_client import (
    get_openrouter_api_key,
    get_openrouter_model,
)

from docker_manager import DockerManager
from tools import TOOL_DEFINITIONS, TOOL_HANDLERS, set_docker_manager
from mcp_client import (
    MCPClient,
 init_mcp,
 
 
)
from rate_limiter import (
    RateLimitManager,
    AdaptiveRateLimiter,
    init_global_limiter,
    get_global_limiter,
)
from api_key_pool import (
    APIKeyPoolManager,
    init_key_pool,
    get_global_pool,
    parse_api_keys_from_env,
)

load_dotenv()

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "moonshotai/kimi-k2.5"

# OpenRouter configuration
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_TOOL_ROUNDS = 250

# Rate limiting configuration
DEFAULT_RATE_LIMIT_STRATEGY = os.getenv("RATE_LIMIT_STRATEGY", "adaptive")
DEFAULT_RATE_LIMIT_INITIAL_DELAY = float(os.getenv("RATE_LIMIT_INITIAL_DELAY", "0.5"))
DEFAULT_RATE_LIMIT_MIN_DELAY = float(os.getenv("RATE_LIMIT_MIN_DELAY", "0.1"))
DEFAULT_RATE_LIMIT_MAX_DELAY = float(os.getenv("RATE_LIMIT_MAX_DELAY", "60.0"))

# Dedicated workspace directory for the agent's Docker sandbox.
DEFAULT_WORKSPACE = os.environ.get(
    "WORKSPACE_DIR", os.path.join(os.path.expanduser("~"), "coding-guy-workspace")
)

STATUS_COMPLETE = "complete"
STATUS_MAX_ROUNDS = "max_rounds"
STATUS_ERROR = "error"


def _get_commit_hash() -> str:
    """Return the short git commit hash, or 'unknown' if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


COMMIT_HASH = _get_commit_hash()

SYSTEM_PROMPT = """\
You are an expert coding agent. All file operations execute inside a Docker sandbox \
with the project directory mounted at /workspace. File paths are relative to the \
project root.

Available tools: read_file, write_file, patch_file, grep_file, ls_file, \
execute_command, multi_read_file, multi_write_file, read_dockerfile, \
write_dockerfile, rebuild_container, web, ask_ollama, \
browser_navigate, browser_action, browser_get_content, browser_close.

Suno Music Tools (when SUNO_API_KEY is configured):
- suno_generate_song: Generate AI music with custom lyrics and style
- suno_get_job_status: Check generation progress
- suno_get_song_data: Get song metadata and download URLs
- suno_list_songs: Browse generated songs
- suno_delete_song: Remove a generated song

When given a task:
1. Use ls_file and grep_file to explore the codebase.
2. Read relevant files to understand the current state.
3. Plan your changes.
4. Use patch_file for targeted edits or write_file for new files.
5. Use execute_command to run builds, tests, or scripts (e.g. "go run main.go", "python3 app.py", "npm test").
6. Verify your work by reading the result.

For web browsing and data collection:
1. Use browser_navigate to go to a website.
2. Use browser_action (click, type, press, wait) to interact with the page.
3. Use browser_get_content to retrieve page text. Set include_images=True if you need image URLs and alt text.
4. Use browser_get_elements to extract granular data like phone numbers, aria-labels, or specific attributes (e.g. src for images, href for links) from matching selectors.
5. If you need to extract specific JSON data (amenities, business hours, reviews), use your own reasoning over the tool outputs to format it as requested.
6. Always call browser_close when finished to release resources.

If a command or build fails because of a missing OS package, library, or runtime:
1. Call read_dockerfile to get the current Dockerfile content.
2. Modify the content to add the missing package (e.g. to the apt-get install line).
3. Call write_dockerfile with the updated content.
4. Call rebuild_container to rebuild the sandbox with the updated Dockerfile.
5. Retry the failed operation.

You have plenty of tool rounds available. Work through the entire task methodically — \
explore, implement, verify, and fix issues until the task is truly complete. \
If you encounter errors, debug and retry rather than giving up.

When cloning repositories:
- Both SSH URLs (git@host:owner/repo.git) and HTTPS URLs work automatically.
- If SSH is not configured, SSH URLs are transparently rewritten to HTTPS.
- If cloning fails with an authentication error, suggest the user check their \
GIT_TOKEN or SSH key configuration.

Use the tools provided to complete the user's request. Be precise with file paths \
and edits. Prefer patch_file over write_file when modifying existing files.\
"""


def get_api_key():
    """Get an API key from the pool or environment."""
    # First try the pool
    pool = get_global_pool()
    if pool:
        key_obj = pool.select_key()
        if key_obj:
            return key_obj.key

    # Fall back to single key
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        print("Error: No API keys configured. Set NVIDIA_API_KEY or NVIDIA_API_KEYS in .env.", file=sys.stderr)
        print("Copy .env.example to .env and add your key(s).", file=sys.stderr)
        sys.exit(1)
    return key


def get_pool_key():
    """Get a key object from the pool for tracking."""
    pool = get_global_pool()
    if pool:
        return pool.select_key()
    return None


def build_messages(conversation_history, user_input, docker_manager=None):
    system = SYSTEM_PROMPT
    if docker_manager and docker_manager.startup_warnings:
        warnings = "\n".join(docker_manager.startup_warnings)
        system += (
            "\n\nIMPORTANT — the following issues were detected when "
            "setting up the Docker sandbox:\n" + warnings
            + "\nYou MUST fix these before proceeding. The most likely cause "
            "is a missing package (e.g. git). Update the Dockerfile and call "
            "rebuild_container, then retry the failed configuration."
        )
    if docker_manager:
        mode = docker_manager.ssh_mode
        if mode == "agent":
            system += "\n\nSSH mode: agent. SSH agent is forwarded from the host — SSH cloning works natively."
        elif mode == "keys":
            system += "\n\nSSH mode: keys. SSH keys are mounted from the host — SSH cloning works natively."
        else:
            system += "\n\nSSH mode: none. SSH URLs are automatically converted to HTTPS with token auth."
    messages = [{"role": "system", "content": system}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_input})
    return messages


def call_llm_api(messages, api_key, invoke_url, model, stream=True):
    """Call the Nvidia API. Returns the full response JSON (non-streamed) or
    the assembled message dict (streamed) including any tool_calls."""
    # Get pool key for tracking
    pool = get_global_pool()
    pool_key = pool.select_key() if pool else None
    actual_key = pool_key.key if pool_key else api_key

    # Apply rate limiting before making the request
    limiter = get_global_limiter()
    if limiter:
        waited = limiter.wait_if_needed()
        if waited > 0:
            print(f"[Rate limit] Waiting {waited:.2f}s before next request", file=sys.stderr)

    headers = {
        "Authorization": f"Bearer {actual_key}",
        "Accept": "text/event-stream" if stream else "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 32768,
        "temperature": 1.00,
        "top_p": 1.00,
        "stream": stream,
        "tools": TOOL_DEFINITIONS,
        "chat_template_kwargs": {"thinking": True},
    }

    try:
        response = requests.post(invoke_url, headers=headers, json=payload, stream=stream, timeout=300)
        response.raise_for_status()

        # Record successful request
        tokens_used = 0  # We don't get token count from streaming
        if pool_key:
            pool_key.record_usage(tokens_used=tokens_used, success=True)

        # Record successful request for adaptive rate limiting
        if limiter:
            limiter.record_success()

    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError) as e:
        # Record failure for pool key
        is_http_error = isinstance(e, requests.exceptions.HTTPError)
        status_code = e.response.status_code if is_http_error and e.response is not None else 0

        if pool_key:
            if status_code == 429:
                pool_key.record_rate_limit_hit(status_code=status_code)
                if limiter:
                    limiter.record_rate_limit_hit()
                print(f"[Pool] Key {pool_key.name} hit rate limit, recorded cooldown", file=sys.stderr)
            else:
                pool_key.record_usage(tokens_used=0, success=False)

        raise

    if not stream:
        data = response.json()
        return data["choices"][0]["message"]

    # Stream and reassemble the full message (content + tool_calls)
    content_parts = []
    tool_calls_by_index = {}

    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if not decoded.startswith("data: "):
            continue
        data_str = decoded[len("data: "):]
        if data_str.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
            delta = chunk["choices"][0].get("delta", {})

            # Text content
            if delta.get("content"):
                print(delta["content"], end="", flush=True, file=sys.stderr)
                content_parts.append(delta["content"])

            # Tool call deltas
            for tc_delta in delta.get("tool_calls", []):
                idx = tc_delta["index"]
                if idx not in tool_calls_by_index:
                    tool_calls_by_index[idx] = {
                        "id": tc_delta.get("id", ""),
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                tc = tool_calls_by_index[idx]
                if tc_delta.get("id"):
                    tc["id"] = tc_delta["id"]
                fn = tc_delta.get("function", {})
                if fn.get("name"):
                    tc["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    tc["function"]["arguments"] += fn["arguments"]

        except (json.JSONDecodeError, KeyError, IndexError):
            continue

    content = "".join(content_parts)
    if content:
        print(file=sys.stderr)  # newline after streamed text

    # Build the assembled message
    message = {"role": "assistant"}
    if content:
        message["content"] = content
    if tool_calls_by_index:
        message["tool_calls"] = [
            tool_calls_by_index[i] for i in sorted(tool_calls_by_index)
        ]

    # Warn if message appears incomplete (stream interrupted)
    if content and not (content.endswith('.') or content.endswith('!') or content.endswith('?') or content.endswith('```')):
        print("[Warning] Response may be incomplete due to connection interruption", file=sys.stderr)

    return message


def execute_tool(name, arguments_str):
    """Parse arguments and execute a tool, returning the result string."""
    try:
        # Use raw_decode with strict=False to gracefully handle cases where the model
        # might include trailing text or use literal control characters (like newlines)
        # in JSON strings.
        decoder = json.JSONDecoder(strict=False)
        args, _ = decoder.raw_decode(arguments_str.strip())
    except json.JSONDecodeError as e:
        print(f"\n[Error] Failed to parse tool arguments for '{name}': {e}", file=sys.stderr)
        print(f"[Raw arguments] {arguments_str}", file=sys.stderr)
        return json.dumps({"error": f"Invalid JSON arguments for {name}: {e}"})

    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return json.dumps({"error": f"Unknown tool: {name}"})

    print(f" -> {name}({', '.join(f'{k}={repr(v)[:60]}' for k, v in args.items())})", file=sys.stderr)
    return handler(args)


def agent_loop(user_input, conversation_history, api_key, invoke_url, model, docker_manager=None,
               max_rounds=None, progress_callback=None):
    """Run the agent loop: call the model, execute tools, repeat until done.

    Returns (reply_text, status) where status is one of
    STATUS_COMPLETE, STATUS_MAX_ROUNDS, or STATUS_ERROR.
    """
    messages = build_messages(conversation_history, user_input, docker_manager)
    effective_max = MAX_TOOL_ROUNDS if max_rounds is None else max_rounds
    assistant_msg = {}
    limiter = get_global_limiter()

    for round_num in range(effective_max):
        print("\nAssistant: " if round_num == 0 else "", end="", flush=True, file=sys.stderr)

        max_retries = 5
        for attempt in range(max_retries + 1):
            try:
                assistant_msg = call_llm_api(messages, api_key, invoke_url, model, stream=True)
                break
            except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError) as e:
                is_http_error = isinstance(e, requests.exceptions.HTTPError)
                status_code = e.response.status_code if is_http_error and e.response is not None else 0

                is_retryable = (
                    (is_http_error and (status_code == 429 or status_code >= 500)) or
                    isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError))
                )

                if is_retryable and attempt < max_retries:
                    wait = 10 * (attempt + 1)
                    error_type = status_code if is_http_error else "Connection"
                    print(f"\nAPI error ({error_type}), retrying in {wait}s... (attempt {attempt + 1}/{max_retries})", file=sys.stderr)

                    # Record rate limit hit for adaptive rate limiting
                    if status_code == 429 and limiter:
                        limiter.record_rate_limit_hit()
                        print(f"[Rate limit] Recorded 429 error. Adaptive delay may increase.", file=sys.stderr)

                    time.sleep(wait)
                elif not is_retryable:
                    print(f"\nAPI error: {e}", file=sys.stderr)
                    return None, STATUS_ERROR
                # else: is_retryable and it's the last attempt. Loop will finish.
        else:
            print(f"\nAPI error: max retries exceeded", file=sys.stderr)
            return None, STATUS_ERROR

        messages.append(assistant_msg)

        # If no tool calls, the agent is done
        tool_calls = assistant_msg.get("tool_calls")
        if not tool_calls:
            return assistant_msg.get("content", ""), STATUS_COMPLETE

        # Execute each tool call and add results
        print("\n[Tool calls]", file=sys.stderr)
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_args = tc["function"]["arguments"]
            result = execute_tool(fn_name, fn_args)

            # Show a preview of the result
            result_preview = result[:200] + ("..." if len(result) > 200 else "")
            print(f" <- {result_preview}", file=sys.stderr)

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

        # Report progress after tool execution
        if progress_callback:
            tool_names = [tc["function"]["name"] for tc in tool_calls]
            progress_callback(round_num + 1, effective_max, tool_names)

        print(file=sys.stderr)  # spacer before next model response

    print("\n[Reached maximum tool rounds]", file=sys.stderr)
    return assistant_msg.get("content", ""), STATUS_MAX_ROUNDS


def main():
    parser = argparse.ArgumentParser(description="Coding agent powered by Nvidia API (Kimi K2.5)")
    parser.add_argument("--serve", action="store_true", help="Start Telegram bot webhook server")
    parser.add_argument("--slack", action="store_true", help="Start Slack bot server")
    parser.add_argument(
        "--reload", action="store_true",
        help="Auto-restart the server when watched files change (use with --serve)",
    )
    parser.add_argument(
        "--watch-path",
        default=None,
        help="Directory to watch for hot-reload (default: .git in current directory)",
    )
    parser.add_argument(
        "--workspace",
        default=DEFAULT_WORKSPACE,
        help="Workspace directory for the agent sandbox (default: %(default)s)",
    )
    parser.add_argument("--ollama", action="store_true", help="Use local Ollama instead of Nvidia API")
    parser.add_argument("--openrouter", action="store_true", help="Use OpenRouter API instead of Nvidia API")
    parser.add_argument("--model", type=str, help="Override the model to use (default: gemma4:e4b for Ollama)")
    parser.add_argument("--api-base", type=str, help="Override the API base URL")
    # Rate limiting arguments
    parser.add_argument(
        "--rate-limit-strategy",
        type=str,
        choices=['adaptive', 'fixed', 'token_bucket', 'none'],
        default=None,
        help="Rate limiting strategy (overrides RATE_LIMIT_STRATEGY env var)"
    )
    parser.add_argument(
        "--rate-limit-initial-delay",
        type=float,
        default=None,
        help="Initial delay between requests in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--rate-limit-min-delay",
        type=float,
        default=None,
        help="Minimum delay for adaptive rate limiting (default: 0.1)"
    )
    parser.add_argument(
        "--rate-limit-max-delay",
        type=float,
        default=None,
        help="Maximum delay for adaptive rate limiting (default: 60.0)"
    )
    args = parser.parse_args()

    # Hot-reload mode: delegate to watcher which spawns the server as a child.
    if args.serve and args.reload:
        from hot_reload import run_with_reload

        watch_path = args.watch_path or os.path.join(os.getcwd(), ".git")
        extra_args = ["--workspace", args.workspace]
        if args.openrouter:
            extra_args.append("--openrouter")
            if not get_openrouter_api_key():
                print("Error: OPENROUTER_API_KEY not found. Please set it in .env.", file=sys.stderr)
                sys.exit(1)
        elif args.ollama:
            extra_args.append("--ollama")
        if args.model:
            extra_args.extend(["--model", args.model])
        if args.api_base:
            extra_args.extend(["--api-base", args.api_base])
        sys.exit(run_with_reload(watch_path, extra_args))

    # Initialize rate limiter based on configuration
    # Command line args take precedence over env vars
    strategy = args.rate_limit_strategy
    if strategy is None:
        strategy = DEFAULT_RATE_LIMIT_STRATEGY

    if strategy != 'none':
        initial_delay = args.rate_limit_initial_delay
        if initial_delay is None:
            initial_delay = DEFAULT_RATE_LIMIT_INITIAL_DELAY

        min_delay = args.rate_limit_min_delay
        if min_delay is None:
            min_delay = DEFAULT_RATE_LIMIT_MIN_DELAY

        max_delay = args.rate_limit_max_delay
        if max_delay is None:
            max_delay = DEFAULT_RATE_LIMIT_MAX_DELAY

        limiter = init_global_limiter(
            strategy=strategy,
            initial_delay=initial_delay,
            min_delay=min_delay,
            max_delay=max_delay
        )
        print(f"Rate limiting enabled: {strategy} strategy", file=sys.stderr)
        if isinstance(limiter._limiter, AdaptiveRateLimiter):
            print(f" Initial delay: {initial_delay}s, Min: {min_delay}s, Max: {max_delay}s", file=sys.stderr)
    else:
        print("Rate limiting disabled", file=sys.stderr)

    # Initialize API key pool if multiple keys configured
    api_keys = parse_api_keys_from_env()
    if len(api_keys) > 1:
        try:
            key_pool = init_key_pool(
                keys=api_keys,
                cooldown_duration=float(os.getenv("API_KEY_COOLDOWN", "60.0"))
            )
            print(f"API key pool initialized with {len(api_keys)} keys", file=sys.stderr)
        except ValueError as e:
            print(f"Warning: Failed to initialize key pool: {e}", file=sys.stderr)

    if args.openrouter:
        invoke_url = args.api_base or OPENROUTER_URL
        model_name = args.model or get_openrouter_model()
        api_key = get_openrouter_api_key()
        if not api_key:
            print("Error: OPENROUTER_API_KEY not found. Please set it in .env.", file=sys.stderr)
            sys.exit(1)
    elif args.ollama:
        invoke_url = args.api_base or "http://127.0.0.1:11434/v1/chat/completions"
        model_name = args.model or "gemma4:e4b"
        api_key = "ollama"
    else:
        invoke_url = args.api_base or INVOKE_URL
        model_name = args.model or MODEL
        api_key = get_api_key()

    conversation_history = []

    # Ensure the dedicated workspace directory exists.
    os.makedirs(args.workspace, exist_ok=True)

    # Initialize Docker sandbox with the dedicated workspace.
    docker = DockerManager(work_dir=args.workspace)
    set_docker_manager(docker)

    # Initialize MCP servers
    mcp_client = init_mcp()
    if mcp_client:
        set_mcp_client(mcp_client)
        # Refresh MCP tools into TOOL_DEFINITIONS
        from tools import refresh_mcp_tools
        refresh_mcp_tools(mcp_client)
        print(f"MCP support: {len(mcp_client.servers)} server(s) connected, {len(TOOL_DEFINITIONS)} total tools available", file=sys.stderr)

    if args.serve:
        from telegram_bot import run_telegram_bot
        try:
            run_telegram_bot(api_key, invoke_url, model_name)
        finally:
            docker.cleanup()
        return

    if args.slack:
        from slack_bot import run_slack_bot
        try:
            run_slack_bot(api_key, invoke_url, model_name)
        finally:
            docker.cleanup()
        return

    # Print status message for the chosen API
    if args.openrouter:
        print(f"OpenRouter Coding Agent (Model: {model_name})", file=sys.stderr)
    elif args.ollama:
        print(f"Ollama Coding Agent (Model: {model_name})", file=sys.stderr)
    else:
        print(f"Nvidia Coding Agent (Model: {model_name})", file=sys.stderr)
    print("Tools: read_file, write_file, patch_file, grep_file, ls_file, execute_command, multi_read_file, multi_write_file, read_dockerfile, write_dockerfile, rebuild_container, web, ask_ollama", file=sys.stderr)
    print("Docker sandbox: files are isolated in a container.", file=sys.stderr)
    print("Type 'quit' to exit, 'clear' to reset conversation.\n", file=sys.stderr)

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!", file=sys.stderr)
                break

            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("Goodbye!", file=sys.stderr)
                break
            if user_input.lower() == "clear":
                conversation_history.clear()
                print("Conversation cleared.\n", file=sys.stderr)
                continue

            reply, status = agent_loop(user_input, conversation_history, api_key, invoke_url, model_name, docker)

            if status == STATUS_MAX_ROUNDS:
                print("[Note: reached maximum tool rounds, response may be incomplete]", file=sys.stderr)

            if reply is not None:
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": reply})
                print()
    finally:
        docker.cleanup()


if __name__ == "__main__":
    main()
