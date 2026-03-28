#!/usr/bin/env python3
"""Coding agent powered by Nvidia API (Kimi K2.5 model)."""

import json
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "moonshotai/kimi-k2.5"
SYSTEM_PROMPT = (
    "You are an expert coding assistant. When given a task, produce clean, "
    "correct, and well-structured code. Explain your reasoning briefly, then "
    "provide the implementation. If the task is ambiguous, state your assumptions."
)


def get_api_key():
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        print("Error: NVIDIA_API_KEY not found in environment or .env file.")
        print("Copy .env.example to .env and add your key.")
        sys.exit(1)
    return key


def build_messages(conversation_history, user_input):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_input})
    return messages


def call_nvidia_api(messages, api_key, stream=True):
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
        "chat_template_kwargs": {"thinking": True},
    }

    response = requests.post(INVOKE_URL, headers=headers, json=payload, stream=stream)
    response.raise_for_status()

    if not stream:
        data = response.json()
        return data["choices"][0]["message"]["content"]

    # Stream response and collect full text
    full_content = []
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
            content = delta.get("content", "")
            if content:
                print(content, end="", flush=True)
                full_content.append(content)
        except (json.JSONDecodeError, KeyError, IndexError):
            continue

    print()  # newline after stream
    return "".join(full_content)


def main():
    api_key = get_api_key()
    conversation_history = []

    print("Nvidia Coding Agent (Kimi K2.5)")
    print("Type 'quit' to exit, 'clear' to reset conversation.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            conversation_history.clear()
            print("Conversation cleared.\n")
            continue

        messages = build_messages(conversation_history, user_input)

        print("\nAssistant: ", end="", flush=True)
        try:
            assistant_reply = call_nvidia_api(messages, api_key, stream=True)
        except requests.exceptions.HTTPError as e:
            print(f"\nAPI error: {e}")
            continue

        # Save to history
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": assistant_reply})
        print()


if __name__ == "__main__":
    main()
