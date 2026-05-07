#!/usr/bin/env python3
"""Coding agent powered by Nvidia API (Kimi K2.5 model) with tool use."""

import argparse
import json
import os
import re
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
from tools import TOOL_DEFINITIONS, TOOL_HANDLERS, set_docker_manager, set_mcp_client, set_task_session_key
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
from error_tracker import (
    get_error_tracker,
    ErrorTracker,
    SEVERITY_HIGH,
    SEVERITY_CRITICAL,
    SEVERITY_MEDIUM,
)

load_dotenv()

# Semantic tool search — lazy imports to avoid hard dependency
_tool_search_engine = None  # Set during main() init if --semantic-search

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "z-ai/glm-5.1"

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
STATUS_BLOCKED = "blocked"


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

SYSTEM_PROMPT_TEMPLATE = """\
You are an expert coding agent. All file operations execute inside a Docker sandbox \
with the project directory mounted at /workspace. File paths are relative to the \
project root.

{tool_list_section}

When given a task:
1. Use create_task to plan your work with concrete steps.
2. Use ls_file and grep_file to explore the codebase.
3. Read relevant files to understand the current state.
4. Update task steps as you progress (mark in_progress → completed/failed).
5. Use patch_file for targeted edits or write_file for new files.
6. Use execute_command to run builds, tests, or scripts (e.g. "go run main.go", "python3 app.py", "npm test").
7. Verify your work by reading the result.
8. Call complete_task when done.

If you encounter errors:
- Mark the step as failed with the error details.
- Try to work around the issue: look for alternative approaches, search for solutions, or try a different method.
- If you've exhausted your workarounds, use ask_human to request guidance.
- When resuming after an error, pick up from the failed step and try a different approach.

When you need human input:
- Use ask_human with a specific question about what you need.
- The task will pause until the human responds.
- After receiving a response, continue from where you left off.

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

# Static tool list section — used when semantic search is disabled
_STATIC_TOOL_LIST_SECTION = """\
Available tools: read_file, write_file, patch_file, grep_file, ls_file, \
execute_command, multi_read_file, multi_write_file, read_dockerfile, \
write_dockerfile, rebuild_container, web, ask_ollama, \
browser_navigate, browser_action, browser_get_content, browser_get_elements, browser_close.

Task Tracking Tools:
- create_task: Plan your work by creating a task with ordered steps before starting.
- update_task_step: Mark steps in_progress, completed, failed, or skipped as you work.
- complete_task: Mark the overall task as done when finished.
- ask_human: Pause and ask the human a question when you need their input.
- list_tasks: View current tasks and their status.

Suno Music Tools (when SUNO_API_KEY is configured):
- suno_generate_song: Generate AI music with custom lyrics and style
- suno_get_job_status: Check generation progress
- suno_get_song_data: Get song metadata and download URLs
- suno_list_songs: Browse generated songs
- suno_delete_song: Remove a generated song

Error Tracking Tools (for self-healing and debugging):
- list_errors: List tracked errors from the error database.
- get_error_details: Get full details of a specific error (stack trace, context).
- resolve_error: Mark an error as resolved after fixing it.
- get_error_summary: Get error statistics and top recurring errors."""


def build_tool_list_section(user_input: str = "") -> str:
    """Build the tool list section for the system prompt.

    If semantic search is enabled and the search engine is initialized:
    - Pre-computes relevant tools for the task description
    - Returns a focused subset of tools ranked by relevance
    Otherwise, returns the full static tool list.
    """
    global _tool_search_engine

    if _tool_search_engine is None or not user_input:
        return _STATIC_TOOL_LIST_SECTION

    try:
        from tool_search_integration import select_tools_for_task
        results = select_tools_for_task(user_input, top_k=15, search_engine=_tool_search_engine)

        if not results:
            return _STATIC_TOOL_LIST_SECTION

        # Build focused tool list from search results
        lines = ["Available tools (ranked by relevance to this task):"]
        for r in results:
            name = r["name"]
            desc = r.get("description", "").split(".")[0] + "."  # First sentence only
            score = r.get("score", 0)
            source = r.get("source", "")
            lines.append(f"- {name}: {desc} [relevance: {score:.2f}]")

        # Add all other tools as a secondary list
        all_tool_names = {tdef["function"]["name"] for tdef in TOOL_DEFINITIONS}
        ranked_names = {r["name"] for r in results}
        remaining = sorted(all_tool_names - ranked_names)
        if remaining:
            lines.append("")
            lines.append("Other available tools: " + ", ".join(remaining))

        return "\n".join(lines)

    except Exception as e:
        print(f"[ToolSearch] Warning: semantic tool selection failed: {e}", file=sys.stderr)
        return _STATIC_TOOL_LIST_SECTION


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


def build_messages(conversation_history, user_input, docker_manager=None, session_key=None):
    system = SYSTEM_PROMPT_TEMPLATE.format(tool_list_section=build_tool_list_section(user_input))
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

    # Inject task resume context if there's an active/blocked/failed task
    if session_key:
        from task_manager import get_task_manager
        tm = get_task_manager()
        resume_ctx = tm.get_resume_context(session_key)
        if resume_ctx:
            system += "\n\n" + resume_ctx

    messages = [{"role": "system", "content": system}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_input})
    return messages


def call_llm_api(messages, api_key, invoke_url, model, stream=True,
                session_key=None, conversation_round=-1):
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

    # Build a summary of the request for error tracking (don't log full messages)
    payload_summary = json.dumps({
        "model": model, "stream": stream,
        "num_messages": len(messages),
        "num_tools": len(TOOL_DEFINITIONS),
    })[:500]

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

        # Track successful agent call
        tracker = get_error_tracker()
        tracker.track_agent_call(
            url=invoke_url, method="POST", model=model,
            request_payload_summary=payload_summary,
            response_status_code=response.status_code,
            session_key=session_key or "",
            conversation_round=conversation_round,
        )

    except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError) as e:
        # Record failure for pool key
        is_http_error = isinstance(e, requests.exceptions.HTTPError)
        status_code = e.response.status_code if is_http_error and e.response is not None else 0

        # Track the API failure in the error database
        tracker = get_error_tracker()
        resp_body = ""
        try:
            resp_body = e.response.text[:2000] if is_http_error and e.response is not None else ""
        except Exception:
            pass
        tracker.record_api_failure(
            url=invoke_url, method="POST", status_code=status_code,
            error_message=str(e),
            request_payload_summary=payload_summary,
            response_body_summary=resp_body,
            source_module="coding_agent", source_function="call_llm_api",
            session_key=session_key or "",
            conversation_round=conversation_round,
            severity=SEVERITY_CRITICAL if status_code >= 500 or status_code == 429 else SEVERITY_HIGH,
        )

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
        # Filter out incomplete tool calls (missing id or function name)
        complete_tool_calls = {}
        for idx, tc in tool_calls_by_index.items():
            if tc.get("id") and tc["function"].get("name"):
                complete_tool_calls[idx] = tc
            else:
                missing = []
                if not tc.get("id"):
                    missing.append("id")
                if not tc["function"].get("name"):
                    missing.append("function name")
                print(f"[Warning] Dropping incomplete tool call (index {idx}): missing {', '.join(missing)}", file=sys.stderr)
        
        if complete_tool_calls:
            message["tool_calls"] = [
                complete_tool_calls[i] for i in sorted(complete_tool_calls)
            ]
        
        # Warn if message appears incomplete (stream interrupted)
    if content and not (content.endswith('.') or content.endswith('!') or content.endswith('?') or content.endswith('```')):
        print("[Warning] Response may be incomplete due to connection interruption", file=sys.stderr)

    return message


def _parse_tool_args(arguments_str: str) -> dict:
    """Parse tool arguments with fallbacks for malformed LLM JSON."""
    text = arguments_str.strip()
    if not text:
        return {}
    try:
        decoder = json.JSONDecoder(strict=False)
        args, _ = decoder.raw_decode(text)
        if isinstance(args, dict):
            return args
    except json.JSONDecodeError:
        pass
    for suffix in ("}", '"}', '""}', '"'):
        try:
            args, _ = decoder.raw_decode(text + suffix)
            if isinstance(args, dict):
                return args
        except json.JSONDecodeError:
            pass
    args = {}
    # Handles escaped quotes inside string values (e.g. "val with \"quote\"")
    pattern = re.compile(r'"(\w+)"\s*:\s*(?:"((?:[^"\\]|\\.)*)"|(\d+(?:\.\d+)?)|true|false|null)')
    for m in pattern.finditer(text):
        key = m.group(1)
        if m.group(2) is not None:
            val = m.group(2)
        elif m.group(3) is not None:
            val = float(m.group(3)) if '.' in m.group(3) else int(m.group(3))
        else:
            snippet = text[m.start():m.end()].split(':')[1].strip().lower()
            val = snippet == 'true'
        args[key] = val
    if args:
        return args
    raise json.JSONDecodeError("Unable to parse tool arguments", text, 0)


def execute_tool(name, arguments_str):
    """Parse arguments and execute a tool, returning the result string."""
    try:
        args = _parse_tool_args(arguments_str)
    except json.JSONDecodeError as e:
        print(f"\n[Error] Failed to parse tool arguments for '{name}': {e}", file=sys.stderr)
        print(f"[Raw arguments] {arguments_str}", file=sys.stderr)
        return json.dumps({"error": f"Invalid JSON arguments for {name}: {e}"})

    # Guard against empty or whitespace-only tool names
    if not name or not name.strip():
        print("\n[Warning] LLM returned a tool call with empty function name, skipping", file=sys.stderr)
        return json.dumps({"error": "Empty tool name received from LLM - this is likely a streaming assembly issue"})
    
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return json.dumps({"error": f"Unknown tool: {name}"})

    print(f" -> {name}({', '.join(f'{k}={repr(v)[:60]}' for k, v in args.items())})", file=sys.stderr)
    try:
        result = handler(args)
    except Exception as exc:
        # Track tool execution failure
        tracker = get_error_tracker()
        tracker.record_exception(
            exc,
            source_module="tools",
            source_function=name,
            context={"tool_name": name, "tool_args": arguments_str[:500]},
            session_key=getattr(execute_tool, '_session_key', ''),
            conversation_round=getattr(execute_tool, '_conversation_round', -1),
            severity=SEVERITY_MEDIUM,
        )
        result = json.dumps({"error": f"Tool '{name}' raised {type(exc).__name__}: {str(exc)}"})
    return result


def agent_loop(user_input, conversation_history, api_key, invoke_url, model, docker_manager=None,
               max_rounds=None, progress_callback=None, session_key=None):
    """Run the agent loop: call the model, execute tools, repeat until done.

    Returns (reply_text, status) where status is one of
    STATUS_COMPLETE, STATUS_MAX_ROUNDS, STATUS_ERROR, or STATUS_BLOCKED.
    """
    # Set task session key so task tools know which conversation they belong to
    if session_key:
        set_task_session_key(session_key)

    messages = build_messages(conversation_history, user_input, docker_manager, session_key=session_key)
    effective_max = MAX_TOOL_ROUNDS if max_rounds is None else max_rounds
    assistant_msg = {}
    limiter = get_global_limiter()

    for round_num in range(effective_max):
        print("\nAssistant: " if round_num == 0 else "", end="", flush=True, file=sys.stderr)

        max_retries = 5
        for attempt in range(max_retries + 1):
            try:
                assistant_msg = call_llm_api(messages, api_key, invoke_url, model, stream=True,
                                             session_key=session_key, conversation_round=round_num)
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
                    error_type = status_code if is_http_error else type(e).__name__
                    print(f"\nAPI error ({error_type}): {e}, retrying in {wait}s... (attempt {attempt + 1}/{max_retries})", file=sys.stderr)

                    # Record rate limit hit for adaptive rate limiting
                    if status_code == 429 and limiter:
                        limiter.record_rate_limit_hit()
                        print(f"[Rate limit] Recorded 429 error. Adaptive delay may increase.", file=sys.stderr)

                    time.sleep(wait)
                elif not is_retryable:
                    if status_code == 410:
                        print(f"\nAPI error: Model '{model}' has reached end-of-life (410 Gone).", file=sys.stderr)
                        print(f"Update the model name or use --openrouter / --model to switch.", file=sys.stderr)
                    else:
                        print(f"\nAPI error: {e}", file=sys.stderr)
                    return None, STATUS_ERROR
                # else: is_retryable and it's the last attempt. Loop will finish.
                else:
                    print(f"\nAPI error: max retries exceeded", file=sys.stderr)
                    # Track the max-retries-exceeded failure
                    tracker = get_error_tracker()
                    tracker.record_api_failure(
                        url=invoke_url, method="POST",
                        error_message=f"Max retries ({max_retries}) exceeded for API call",
                        source_module="coding_agent", source_function="agent_loop",
                        session_key=session_key or "",
                        conversation_round=round_num,
                        severity=SEVERITY_CRITICAL,
                    )
                    return None, STATUS_ERROR

        messages.append(assistant_msg)

        # If no tool calls, the agent is done
        tool_calls = assistant_msg.get("tool_calls")
        if not tool_calls:
            return assistant_msg.get("content", ""), STATUS_COMPLETE

        # Execute each tool call and add results
        print("\n[Tool calls]", file=sys.stderr)
        asked_human = False
        human_question = None
        # Make session_key and round_num available to execute_tool for error tracking
        execute_tool._session_key = session_key or ""
        execute_tool._conversation_round = round_num

        for tc in tool_calls:
            try:
                fn_name = tc["function"]["name"]
                fn_args = tc["function"]["arguments"]
            except (KeyError, TypeError) as e:
                print(f"\n[Warning] Malformed tool_call in response, skipping: {e}", file=sys.stderr)
                continue

        # Validate arguments look complete — detect truncated streaming assembly.
        # Common patterns: empty args "", partial JSON like "{" or '{"path":'
        if fn_args is None:
            fn_args = ""
        fn_args_stripped = fn_args.strip()
        if fn_args_stripped and fn_args_stripped.startswith("{") and not fn_args_stripped.endswith("}"):
            # Looks like truncated JSON — try to complete it
            print(f"\n[Warning] Tool '{fn_name}' has truncated arguments, attempting repair", file=sys.stderr)
            # Try adding closing braces — simplest repair
            for closing in ("}", "}}", "]}"):
                test = fn_args_stripped + closing
                try:
                    json.loads(test)
                    fn_args = test
                    print(f"[Repair] Completed arguments by adding '{closing}'", file=sys.stderr)
                    break
                except json.JSONDecodeError:
                    continue
            else:
                # Could not repair — return error so LLM self-corrects
                result = json.dumps({
                    "error": f"Tool '{fn_name}' received malformed/truncated arguments that could not be repaired",
                    "raw_args_preview": fn_args_stripped[:200],
                    "hint": "The arguments JSON was incomplete, likely due to a stream interruption. Please retry with complete arguments.",
                })
                print(f" <- {result[:200]}", file=sys.stderr)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                continue
        
            result = execute_tool(fn_name, fn_args)

            # Show a preview of the result
            result_preview = result[:200] + ("..." if len(result) > 200 else "")
            print(f" <- {result_preview}", file=sys.stderr)

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

            # Detect ask_human — pause the loop and return blocked status
            if fn_name == "ask_human":
                asked_human = True
                try:
                    args = _parse_tool_args(fn_args)
                    human_question = args.get("question", "Human input needed")
                except Exception:
                    human_question = "Human input needed"
                break

        # If ask_human was called, stop the loop and return blocked
        if asked_human:
            print(f"\n[Blocked] Waiting for human: {human_question}", file=sys.stderr)
            return assistant_msg.get("content", "") or f"I need your input: {human_question}", STATUS_BLOCKED

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
        help="Directory to watch for hot-reload (default: current directory)",
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
    # Semantic tool search arguments
    parser.add_argument(
        "--semantic-search", action="store_true",
        help="Enable semantic tool search to rank tools by relevance to each task",
    )
    parser.add_argument(
        "--search-model", type=str, default=None,
        help="Embedding model for tool search (default: auto-select best available)",
    )
    parser.add_argument(
        "--search-verbose", action="store_true", default=False,
        help="Enable verbose color-coded logging of semantic search scores",
    )
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

        watch_path = args.watch_path or os.getcwd()
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

    # Initialize semantic tool search if requested
    global _tool_search_engine
    if args.semantic_search:
        try:
            from tool_search import init_tool_search
            from tool_registry import reset_name_index
            reset_name_index()  # Rebuild index after MCP tools loaded

            emb_api_key = ""
            emb_base_url = ""
            if args.openrouter:
                emb_api_key = get_openrouter_api_key()
                emb_base_url = "https://openrouter.ai/api/v1"

            _tool_search_engine = init_tool_search(
                api_key=emb_api_key,
                api_base_url=emb_base_url,
                embedding_model=args.search_model or "",
                verbose=args.search_verbose,
                use_cache=True,
            )
            if _tool_search_engine:
                print(f"Semantic tool search: enabled (backend={_tool_search_engine.backend_name}, "
                      f"{_tool_search_engine.tool_count} tools indexed)", file=sys.stderr)
            else:
                print("Semantic tool search: initialization failed, using static tool list", file=sys.stderr)
        except Exception as e:
            print(f"Semantic tool search: init error: {e}", file=sys.stderr)
            _tool_search_engine = None

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
    if _tool_search_engine:
        print(f"Semantic search: enabled (backend={_tool_search_engine.backend_name})", file=sys.stderr)
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

            # Unblock any blocked task with the user's response
            from task_manager import get_task_manager
            tm = get_task_manager()
            active_task = tm.get_active_task("cli")
            if active_task and active_task.status == "blocked" and active_task.blocker:
                tm.unblock_task(active_task.uuid, user_input)

            reply, status = agent_loop(user_input, conversation_history, api_key, invoke_url, model_name, docker,
                                       session_key="cli")

            if status == STATUS_MAX_ROUNDS:
                print("[Note: reached maximum tool rounds, response may be incomplete]", file=sys.stderr)
            elif status == STATUS_BLOCKED:
                print("[Task paused — reply to continue]", file=sys.stderr)

            if reply is not None:
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": reply})
                print()
    finally:
        # Save outcome logger data before exit
        try:
            from tool_search_integration import get_outcome_logger
            get_outcome_logger().save()
        except Exception:
            pass
        docker.cleanup()


if __name__ == "__main__":
    main()
