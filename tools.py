"""Tools the coding agent can call to interact with files and the web."""

import difflib
import json
import os

import requests


def read_file(path: str) -> str:
    """Read and return the contents of a file."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        return json.dumps({"error": f"File not found: {path}"})
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return json.dumps({"path": path, "content": content, "size": len(content)})
    except Exception as e:
        return json.dumps({"error": str(e)})


def write_file(path: str, content: str) -> str:
    """Write content to a file, creating directories if needed."""
    path = os.path.abspath(path)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return json.dumps({"path": path, "status": "written", "size": len(content)})
    except Exception as e:
        return json.dumps({"error": str(e)})


def patch_file(path: str, patches: list[dict]) -> str:
    """Apply search-and-replace patches to a file.

    Each patch is {"old": "text to find", "new": "replacement text"}.
    Patches are applied in order.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        return json.dumps({"error": f"File not found: {path}"})
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        original = content
        applied = []
        failed = []

        for i, patch in enumerate(patches):
            old_text = patch.get("old", "")
            new_text = patch.get("new", "")
            if old_text not in content:
                failed.append({"index": i, "old": old_text[:80], "reason": "not found"})
            else:
                content = content.replace(old_text, new_text, 1)
                applied.append(i)

        if applied:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

        # Generate unified diff for visibility
        diff = "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=f"a/{os.path.basename(path)}",
                tofile=f"b/{os.path.basename(path)}",
            )
        )

        result = {"path": path, "applied": applied, "diff": diff}
        if failed:
            result["failed"] = failed
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def web(url: str, method: str = "GET", headers: dict | None = None,
        body: str | None = None) -> str:
    """Make an HTTP request and return the response."""
    method = method.upper()
    try:
        kwargs = {"headers": headers or {}, "timeout": 30}
        if body and method in ("POST", "PUT", "PATCH"):
            kwargs["data"] = body

        resp = requests.request(method, url, **kwargs)

        # Limit response body to avoid flooding context
        body_text = resp.text[:50000]
        truncated = len(resp.text) > 50000

        return json.dumps({
            "status_code": resp.status_code,
            "headers": dict(resp.headers),
            "body": body_text,
            "truncated": truncated,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# --- Tool definitions for the OpenAI-compatible tool-calling API ---

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read."
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating it and parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write."
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file."
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "patch_file",
            "description": (
                "Apply search-and-replace patches to an existing file. "
                "Each patch replaces only the *first* occurrence of an 'old' string with a 'new' string. "
                "Patches are applied in order. Use this for targeted edits instead of rewriting entire files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to patch."
                    },
                    "patches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old": {
                                    "type": "string",
                                    "description": "The exact text to find in the file."
                                },
                                "new": {
                                    "type": "string",
                                    "description": "The replacement text."
                                }
                            },
                            "required": ["old", "new"]
                        },
                        "description": "List of search-and-replace patches to apply."
                    }
                },
                "required": ["path", "patches"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web",
            "description": (
                "Make an HTTP request. Use this to fetch web pages, call APIs, "
                "download documentation, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to request."
                    },
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"],
                        "description": "HTTP method (default: GET)."
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional HTTP headers as key-value pairs."
                    },
                    "body": {
                        "type": "string",
                        "description": "Optional request body (for POST/PUT/PATCH)."
                    }
                },
                "required": ["url"]
            }
        }
    },
]

# Map function names to callables
TOOL_HANDLERS = {
    "read_file": lambda args: read_file(**args),
    "write_file": lambda args: write_file(**args),
    "patch_file": lambda args: patch_file(**args),
    "web": lambda args: web(**args),
}
