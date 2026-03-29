"""Tools the coding agent can call, executing inside a Docker sandbox."""

import difflib
import inspect
import json
import os
import shlex

import requests

# ---------------------------------------------------------------------------
# Docker manager reference (set from coding_agent.py at startup)
# ---------------------------------------------------------------------------

_docker_manager = None


def set_docker_manager(dm):
    """Set the module-level DockerManager instance."""
    global _docker_manager
    _docker_manager = dm


def _get_docker_manager():
    if _docker_manager is None:
        raise RuntimeError("DockerManager not initialized")
    return _docker_manager


# ---------------------------------------------------------------------------
# Tool implementations — file tools execute inside the Docker container
# ---------------------------------------------------------------------------


def read_file(path: str) -> str:
    """Read a file inside the Docker container."""
    dm = _get_docker_manager()
    rc, stdout, stderr = dm.exec(["cat", path])
    if rc != 0:
        return json.dumps({"error": f"Failed to read {path}: {stderr.strip()}"})
    return json.dumps({"path": path, "content": stdout, "size": len(stdout)})


def write_file(path: str, content: str) -> str:
    """Write content to a file inside the Docker container."""
    dm = _get_docker_manager()
    dir_path = os.path.dirname(path)
    if dir_path:
        rc, _, stderr = dm.exec(["mkdir", "-p", dir_path])
        if rc != 0:
            return json.dumps({"error": f"Failed to create directory '{dir_path}': {stderr.strip()}"})
    rc, _, stderr = dm.exec(["tee", path], stdin_data=content)
    if rc != 0:
        return json.dumps({"error": f"Failed to write {path}: {stderr.strip()}"})
    return json.dumps({"path": path, "status": "written", "size": len(content)})


def patch_file(path: str, patches: list[dict]) -> str:
    """Apply search-and-replace patches to a file inside the Docker container.

    Each patch is {"old": "text to find", "new": "replacement text"}.
    Patches are applied in order.
    """
    dm = _get_docker_manager()
    rc, content, stderr = dm.exec(["cat", path])
    if rc != 0:
        return json.dumps({"error": f"Failed to read file for patching {path}: {stderr.strip()}"})

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
        rc, _, stderr = dm.exec(["tee", path], stdin_data=content)
        if rc != 0:
            return json.dumps({"error": f"Failed to write patched file: {stderr.strip()}"})

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


def grep_file(pattern: str, path: str = ".", recursive: bool = True) -> str:
    """Search for a pattern in files inside the Docker container."""
    dm = _get_docker_manager()
    cmd = ["grep", "-n", "--color=never"]
    if recursive:
        cmd.append("-r")
    cmd.extend(["--", pattern, path])
    rc, stdout, stderr = dm.exec(cmd)
    if rc == 1:
        return json.dumps({"pattern": pattern, "path": path, "matches": [], "count": 0})
    if rc > 1:
        return json.dumps({"error": f"grep failed: {stderr.strip()}"})
    matches = [line for line in stdout.strip().split("\n") if line]
    # Limit output to avoid flooding context
    truncated = len(matches) > 200
    matches = matches[:200]
    return json.dumps({
        "pattern": pattern,
        "path": path,
        "matches": matches,
        "count": len(matches),
        "truncated": truncated,
    })


def ls_file(path: str = ".") -> str:
    """List directory contents inside the Docker container."""
    dm = _get_docker_manager()
    rc, stdout, stderr = dm.exec(["ls", "-la", path])
    if rc != 0:
        return json.dumps({"error": f"Failed to list {path}: {stderr.strip()}"})
    return json.dumps({"path": path, "entries": stdout.strip()})


def execute_command(command: str, working_dir: str | None = None) -> str:
    """Execute a shell command inside the Docker container."""
    dm = _get_docker_manager()
    if working_dir:
        command = f"cd {shlex.quote(working_dir)} && {command}"
    cmd = ["bash", "-c", command]
    rc, stdout, stderr = dm.exec(cmd)
    # Limit output to avoid flooding context
    max_output = 50000
    stdout_truncated = len(stdout) > max_output
    stderr_truncated = len(stderr) > max_output
    return json.dumps({
        "exit_code": rc,
        "stdout": stdout[:max_output],
        "stderr": stderr[:max_output],
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    })


def multi_read_file(paths: list[str]) -> str:
    """Read multiple files inside the Docker container in one call."""
    dm = _get_docker_manager()
    results = []
    for path in paths:
        rc, stdout, stderr = dm.exec(["cat", path])
        if rc != 0:
            results.append({"path": path, "error": f"Failed to read {path}: {stderr.strip()}"})
        else:
            results.append({"path": path, "content": stdout, "size": len(stdout)})
    return json.dumps({"results": results})


def multi_write_file(files: list[dict]) -> str:
    """Write multiple files inside the Docker container in one call."""
    dm = _get_docker_manager()
    results = []
    for entry in files:
        path = entry.get("path")
        content = entry.get("content")
        if path is None or content is None:
            results.append({"error": "Each file entry must have 'path' and 'content' keys.", "entry_keys": list(entry.keys())})
            continue
        dir_path = os.path.dirname(path)
        if dir_path:
            rc, _, stderr = dm.exec(["mkdir", "-p", dir_path])
            if rc != 0:
                results.append({"path": path, "error": f"Failed to create directory '{dir_path}': {stderr.strip()}"})
                continue
        rc, _, stderr = dm.exec(["tee", path], stdin_data=content)
        if rc != 0:
            results.append({"path": path, "error": f"Failed to write {path}: {stderr.strip()}"})
        else:
            results.append({"path": path, "status": "written", "size": len(content)})
    return json.dumps({"results": results})


def rebuild_container() -> str:
    """Rebuild the Docker sandbox after Dockerfile changes."""
    dm = _get_docker_manager()
    try:
        info = dm.rebuild()
        return json.dumps(info)
    except RuntimeError as e:
        return json.dumps({"error": str(e)})


def read_dockerfile() -> str:
    """Read the current Dockerfile content from the host filesystem."""
    dm = _get_docker_manager()
    path = dm.find_dockerfile()
    if path:
        try:
            with open(path) as f:
                content = f.read()
            return json.dumps({"path": path, "content": content, "source": "custom"})
        except OSError as e:
            return json.dumps({"error": str(e)})
    else:
        from docker_manager import DEFAULT_DOCKERFILE
        return json.dumps({"content": DEFAULT_DOCKERFILE, "source": "default (embedded)"})


def write_dockerfile(content: str) -> str:
    """Write or update the Dockerfile on the host filesystem."""
    dm = _get_docker_manager()
    path = dm.get_dockerfile_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return json.dumps({"path": path, "status": "written", "size": len(content)})
    except OSError as e:
        return json.dumps({"error": str(e)})


def web(url: str, method: str = "GET", headers: dict | None = None,
        body: str | None = None) -> str:
    """Make an HTTP request and return the response."""
    method = method.upper()
    try:
        kwargs = {"headers": headers or {}, "timeout": 60}
        if body and method in ("POST", "PUT", "PATCH"):
            kwargs["data"] = body

        resp = requests.request(method, url, **kwargs)

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
            "name": "grep_file",
            "description": (
                "Search for a regex pattern in files. Returns matching lines with "
                "line numbers. Useful for finding code, functions, variables, or "
                "any text pattern across the project."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The regex pattern to search for."
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in (default: current directory)."
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Search recursively in subdirectories (default: true)."
                    }
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ls_file",
            "description": "List the contents of a directory with details (permissions, size, dates).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list (default: current directory)."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": (
                "Execute a shell command inside the Docker sandbox. Use this to "
                "run builds, tests, scripts, install packages, or any shell command. "
                "Returns stdout, stderr, and exit code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute (run via bash -c)."
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Optional working directory (default: /workspace)."
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "multi_read_file",
            "description": "Read multiple files at once. Returns an array of results, each with path and content (or error).",
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths to read."
                    }
                },
                "required": ["paths"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "multi_write_file",
            "description": "Write multiple files at once. Each entry needs a path and content. Creates parent directories as needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
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
                        },
                        "description": "List of files to write, each with path and content."
                    }
                },
                "required": ["files"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rebuild_container",
            "description": (
                "Rebuild and restart the Docker sandbox. Use this after calling "
                "write_dockerfile to apply Dockerfile changes. The workspace "
                "files are preserved across rebuilds."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_dockerfile",
            "description": (
                "Read the current Dockerfile used to build the sandbox. "
                "Returns the content whether it's a custom Dockerfile or the "
                "embedded default. This reads from the host, not inside the container."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_dockerfile",
            "description": (
                "Write or update the Dockerfile used to build the sandbox. "
                "This writes to the host filesystem. After writing, call "
                "rebuild_container to apply the changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The full Dockerfile content."
                    }
                },
                "required": ["content"]
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

def _make_handler(func):
    """Create a handler that filters out unexpected keyword arguments."""
    valid_params = set(inspect.signature(func).parameters.keys())

    def handler(args):
        filtered = {k: v for k, v in args.items() if k in valid_params}
        return func(**filtered)

    return handler


# Map function names to callables (handlers filter unexpected kwargs from LLM)
TOOL_HANDLERS = {
    "read_file": _make_handler(read_file),
    "write_file": _make_handler(write_file),
    "patch_file": _make_handler(patch_file),
    "grep_file": _make_handler(grep_file),
    "ls_file": _make_handler(ls_file),
    "execute_command": _make_handler(execute_command),
    "multi_read_file": _make_handler(multi_read_file),
    "multi_write_file": _make_handler(multi_write_file),
    "rebuild_container": lambda args: rebuild_container(),
    "read_dockerfile": lambda args: read_dockerfile(),
    "write_dockerfile": _make_handler(write_dockerfile),
    "web": _make_handler(web),
}
