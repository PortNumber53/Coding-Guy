#!/usr/bin/env python3
"""Coding agent powered by Nvidia API (Kimi K2.5 model) with tool use."""

import json
import os
import sys

import requests
from dotenv import load_dotenv

from docker_manager import DockerManager
from tools import TOOL_DEFINITIONS, TOOL_HANDLERS, set_docker_manager

load_dotenv()

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "moonshotai/kimi-k2.5"
MAX_TOOL_ROUNDS = 15

SYSTEM_PROMPT = """\
You are an expert coding agent. All file operations execute inside a Docker sandbox \
with the project directory mounted at /workspace. File paths are relative to the \
project root.

Available tools: read_file, write_file, patch_file, grep_file, ls_file, \
execute_command, rebuild_container, web.

When given a task:
1. Use ls_file and grep_file to explore the codebase.
2. Read relevant files to understand the current state.
3. Plan your changes.
4. Use patch_file for targeted edits or write_file for new files.
5. Use execute_command to run builds, tests, or scripts (e.g. "go run main.go", "python3 app.py", "npm test").
6. Verify your work by reading the result.

If a command or build fails because of a missing OS package, library, or runtime:
1. Read the current Dockerfile at .coding-guy/Dockerfile (or create it).
2. Add the missing package to the apt-get install line (or add new install commands).
3. Call rebuild_container to rebuild the sandbox with the updated Dockerfile.
4. Retry the failed operation.

Use the tools provided to complete the user's request. Be precise with file paths \
and edits. Prefer patch_file over write_file when modifying existing files.\
"""


def get_api_key():
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        print("Error: NVIDIA_API_KEY not found in environment or .env file.", file=sys.stderr)
        print("Copy .env.example to .env and add your key.", file=sys.stderr)
        sys.exit(1)
    return key


def build_messages(conversation_history, user_input):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_input})
    return messages


def call_nvidia_api(messages, api_key, stream=True):
    """Call the Nvidia API. Returns the full response JSON (non-streamed) or
    the assembled message dict (streamed) including any tool_calls."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream" if stream else "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 16384,
        "temperature": 1.00,
        "top_p": 1.00,
        "stream": stream,
        "tools": TOOL_DEFINITIONS,
        "chat_template_kwargs": {"thinking": True},
    }

    response = requests.post(INVOKE_URL, headers=headers, json=payload, stream=stream, timeout=300)
    response.raise_for_status()

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
    return message


def execute_tool(name, arguments_str):
    """Parse arguments and execute a tool, returning the result string."""
    try:
        args = json.loads(arguments_str)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON arguments: {e}"})

    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return json.dumps({"error": f"Unknown tool: {name}"})

    print(f"  -> {name}({', '.join(f'{k}={repr(v)[:60]}' for k, v in args.items())})", file=sys.stderr)
    return handler(args)


def agent_loop(user_input, conversation_history, api_key):
    """Run the agent loop: call the model, execute tools, repeat until done."""
    messages = build_messages(conversation_history, user_input)

    for round_num in range(MAX_TOOL_ROUNDS):
        print("\nAssistant: " if round_num == 0 else "", end="", flush=True, file=sys.stderr)

        try:
            assistant_msg = call_nvidia_api(messages, api_key, stream=True)
        except requests.exceptions.HTTPError as e:
            print(f"\nAPI error: {e}", file=sys.stderr)
            return None

        messages.append(assistant_msg)

        # If no tool calls, the agent is done
        tool_calls = assistant_msg.get("tool_calls")
        if not tool_calls:
            return assistant_msg.get("content", "")

        # Execute each tool call and add results
        print("\n[Tool calls]", file=sys.stderr)
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_args = tc["function"]["arguments"]
            result = execute_tool(fn_name, fn_args)

            # Show a preview of the result
            result_preview = result[:200] + ("..." if len(result) > 200 else "")
            print(f"  <- {result_preview}", file=sys.stderr)

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

        print(file=sys.stderr)  # spacer before next model response

    print("\n[Reached maximum tool rounds]", file=sys.stderr)
    return assistant_msg.get("content", "")


def main():
    api_key = get_api_key()
    conversation_history = []

    # Initialize Docker sandbox
    docker = DockerManager(work_dir=os.getcwd())
    set_docker_manager(docker)

    print("Nvidia Coding Agent (Kimi K2.5)", file=sys.stderr)
    print("Tools: read_file, write_file, patch_file, grep_file, ls_file, execute_command, rebuild_container, web", file=sys.stderr)
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

            reply = agent_loop(user_input, conversation_history, api_key)

            if reply is not None:
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": reply})
            print()
    finally:
        docker.cleanup()


if __name__ == "__main__":
    main()
