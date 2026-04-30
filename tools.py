"""Tools the coding agent can call, executing inside a Docker sandbox."""

import inspect
import json
import os
import shlex
import difflib

import requests
from bs4 import BeautifulSoup
try:
 from playwright.sync_api import sync_playwright
except ImportError:
 sync_playwright = None

# Import Suno client tools
from suno_client import (
 suno_generate_song,
 suno_get_job_status,
 suno_get_song_data,
 suno_list_songs,
 suno_delete_song,
)

from task_manager import get_task_manager

# MCP support
_mcp_client = None
_mcp_tools_loaded = False

def set_mcp_client(client):
    """Set the MCP client from coding_agent.py."""
    global _mcp_client, _mcp_tools_loaded
    _mcp_client = client
    _mcp_tools_loaded = True


def _init_mcp_tools():
    """Initialize MCP tools if config exists."""
    global _mcp_client, _mcp_tools_loaded
    if _mcp_tools_loaded:
        return

    try:
        from mcp_client import get_mcp_client
        _mcp_client = get_mcp_client()
    except Exception:
        pass

    _mcp_tools_loaded = True


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
# Tool implementations - file tools execute inside the Docker container
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


def ask_ollama(prompt: str, model: str = "gemma4:e4b") -> str:
    """Ask a local Ollama instance a question or to perform a task."""
    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        resp = requests.post(url, json=payload, headers={'Content-Type': 'application/json'}, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return json.dumps({"response": data.get("response", ""), "model": model})
    except Exception as e:
        return json.dumps({"error": f"Failed to call Ollama: {str(e)}"})


# --- Browser tools using Playwright ---

_playwright_ctx = None
_browser = None
_page = None


def _get_browser_page():
    global _playwright_ctx, _browser, _page
    if sync_playwright is None:
        raise RuntimeError("playwright is not installed. Add it to requirements.txt and install.")
    if _playwright_ctx is None:
        _playwright_ctx = sync_playwright().start()
        _browser = _playwright_ctx.chromium.launch(headless=True)
        _page = _browser.new_page()
    return _page


def browser_navigate(url: str, wait_until: str = "load", timeout: int = 60000) -> str:
    """Navigate the browser to a URL.
    wait_until: "load", "domcontentloaded", "networkidle", "commit".
    """
    try:
        page = _get_browser_page()
        page.goto(url, wait_until=wait_until, timeout=timeout)
        return json.dumps({
            "url": page.url,
            "title": page.title(),
            "status": "navigated"
        })
    except Exception as e:
        return json.dumps({
            "url": page.url if 'page' in locals() else url,
            "title": page.title() if 'page' in locals() else "Unknown",
            "status": "timeout",
            "error": str(e)
        })


def browser_action(action: str, selector: str | None = None, text: str | None = None,
                   key: str | None = None) -> str:
    """Perform an action on the current page.
    Supported actions: click, type, press, wait_for_selector, wait_for_timeout.
    """
    try:
        page = _get_browser_page()
        if action == "click":
            page.click(selector, timeout=60000)
        elif action == "type":
            page.fill(selector, text, timeout=60000)
        elif action == "press":
            page.press(selector or "body", key, timeout=60000)
        elif action == "wait_for_selector":
            page.wait_for_selector(selector, timeout=60000)
        elif action == "wait_for_timeout":
            page.wait_for_timeout(int(text or 1000))
        else:
            return json.dumps({"error": f"Unsupported action: {action}"})
        
        return json.dumps({"status": "success", "action": action, "selector": selector})
    except Exception as e:
        return json.dumps({"error": str(e)})


def browser_get_content(include_images: bool = False) -> str:
    """Get the simplified content of the current page.
    If include_images is True, it will include image src and alt tags.
    """
    try:
        page = _get_browser_page()
        content = page.content()
        soup = BeautifulSoup(content, "html.parser")
        
        for script_or_style in soup(["script", "style", "meta", "link", "noscript"]):
            script_or_style.decompose()
        
        if include_images:
            for img in soup.find_all("img"):
                src = img.get("src", "")
                alt = img.get("alt", "")
                if src:
                    img.replace_with(f"\n[IMAGE: {src} | ALT: {alt}]\n")

        text = soup.get_text(separator="\n")
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split(" "))
        clean_text = "\n".join(chunk for chunk in chunks if chunk)
        
        max_len = 30000
        truncated = len(clean_text) > max_len
        
        return json.dumps({
            "url": page.url,
            "title": page.title(),
            "content": clean_text[:max_len],
            "truncated": truncated
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def browser_get_elements(selector: str, attributes: list[str] | None = None) -> str:
    """Get details of elements matching a CSS selector."""
    try:
        page = _get_browser_page()
        elements = page.query_selector_all(selector)
        results = []
        for el in elements:
            data = {"text": el.inner_text()}
            if attributes:
                for attr in attributes:
                    data[attr] = el.get_attribute(attr)
            results.append(data)
        
        return json.dumps({
            "selector": selector,
            "count": len(results),
            "elements": results[:100]
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def browser_close() -> str:
    """Close the browser and cleanup."""
    global _playwright_ctx, _browser, _page
    try:
        if _page:
            _page.close()
        if _browser:
            _browser.close()
        if _playwright_ctx:
            _playwright_ctx.stop()
        _page = _browser = _playwright_ctx = None
        return json.dumps({"status": "closed"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# --- Task management tools ---

# Session key for task tracking (set by coding_agent.py per conversation)
_task_session_key: str | None = None


def set_task_session_key(key: str):
    """Set the session key used to associate tasks with a conversation."""
    global _task_session_key
    _task_session_key = key


def create_task(description: str, steps: list[str] | None = None) -> str:
    """Create a tracked task with optional ordered steps."""
    tm = get_task_manager()
    session_key = _task_session_key or "default"
    task = tm.create_task(description, steps=steps, session_key=session_key)
    return json.dumps({
        "task_id": task.uuid,
        "display_id": task.display_id,
        "description": task.description,
        "status": task.status,
        "steps": len(task.steps),
    })


def update_task_step(task_id: str, step_index: int, status: str,
                     result: str | None = None, error: str | None = None) -> str:
    """Update a task step's status. Status: in_progress, completed, failed, skipped."""
    tm = get_task_manager()
    task = tm.update_step(task_id, step_index, status, result=result, error=error)
    if not task:
        return json.dumps({"error": f"Task or step not found: {task_id} step {step_index}"})
    return json.dumps({
        "task_id": task.uuid,
        "status": task.status,
        "step_index": step_index,
        "step_status": status,
    })


def complete_task(task_id: str, result: str | None = None) -> str:
    """Mark a task as completed with an optional result summary."""
    tm = get_task_manager()
    task = tm.complete_task(task_id, result=result)
    if not task:
        return json.dumps({"error": f"Task not found: {task_id}"})
    return json.dumps({
        "task_id": task.uuid,
        "status": "completed",
        "result": result,
    })


def ask_human(question: str) -> str:
    """Ask the human a question when you need their input to proceed.
    
    Use this when:
    - You need a decision that requires human judgment
    - You need credentials, tokens, or access that only the human can provide
    - You've tried workarounds and need guidance on how to proceed
    - The task requires approval before making a potentially destructive change
    
    The task will be paused until the human responds.
    """
    tm = get_task_manager()
    session_key = _task_session_key or "default"
    task = tm.get_active_task(session_key)
    if task:
        tm.block_task(task.uuid, question)
    return json.dumps({
        "status": "blocked",
        "question": question,
        "message": "Task is paused. Waiting for human response.",
    })


def list_tasks(status: str | None = None) -> str:
    """List tracked tasks, optionally filtered by status."""
    tm = get_task_manager()
    tasks = tm.list_tasks(status=status)
    result = []
    for t in tasks:
        result.append({
            "task_id": t.uuid,
            "display_id": t.display_id,
            "description": t.description,
            "status": t.status,
            "steps_total": len(t.steps),
            "steps_completed": sum(1 for s in t.get_step_objects() if s.status == "completed"),
        })
    return json.dumps({"tasks": result, "count": len(result)})


# --- Tool definitions for the OpenAI-compatible tool-calling API ---

_BASE_TOOL_DEFINITIONS = [
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
 {
  "type": "function",
  "function": {
   "name": "ask_ollama",
   "description": (
    "Delegate a sub-task or ask a question to a local Ollama model. "
    "Useful for text generation, summarizing, or analyzing data using a local open-source model."
   ),
   "parameters": {
    "type": "object",
    "properties": {
     "prompt": {
      "type": "string",
      "description": "The prompt or task instruction to send to Ollama."
     },
     "model": {
      "type": "string",
      "description": "The Ollama model to use (default: gemma4:e4b)."
     }
    },
    "required": ["prompt"]
   }
  }
 },
 {
  "type": "function",
  "function": {
   "name": "browser_navigate",
   "description": "Navigate to a URL using a headless browser.",
   "parameters": {
    "type": "object",
    "properties": {
     "url": {
      "type": "string",
      "description": "The URL to navigate to."
     },
     "wait_until": {
      "type": "string",
      "enum": ["load", "domcontentloaded", "networkidle", "commit"],
      "description": "When to consider navigation finished (default: load)."
     },
     "timeout": {
      "type": "integer",
      "description": "Maximum time in milliseconds (default: 60000)."
     }
    },
    "required": ["url"]
   }
  }
 },
 {
  "type": "function",
  "function": {
   "name": "browser_action",
   "description": "Perform an action like click, type, or press on the current page.",
   "parameters": {
    "type": "object",
    "properties": {
     "action": {
      "type": "string",
      "enum": ["click", "type", "press", "wait_for_selector", "wait_for_timeout"],
      "description": "The action to perform."
     },
     "selector": {
      "type": "string",
      "description": "The CSS or Playwright selector for the element."
     },
     "text": {
      "type": "string",
      "description": "The text to type (for 'type' or 'wait_for_timeout')."
     },
     "key": {
      "type": "string",
      "description": "The key to press (for 'press')."
     }
    },
    "required": ["action"]
   }
  }
 },
 {
  "type": "function",
  "function": {
   "name": "browser_get_content",
   "description": "Get the text content of the current page, cleaned of HTML tags and scripts.",
   "parameters": {
    "type": "object",
    "properties": {
     "include_images": {
      "type": "boolean",
      "description": "Whether to include image URLs and alt text in the output (default: false)."
     }
    },
    "required": []
   }
  }
 },
 {
  "type": "function",
  "function": {
   "name": "browser_get_elements",
   "description": "Extract text and attributes from elements matching a CSS selector.",
   "parameters": {
    "type": "object",
    "properties": {
     "selector": {
      "type": "string",
      "description": "The CSS or Playwright selector to find elements."
     },
     "attributes": {
      "type": "array",
      "items": {"type": "string"},
      "description": "List of attribute names to extract from each matching element."
     }
    },
    "required": ["selector"]
   }
  }
 },
 {
  "type": "function",
  "function": {
   "name": "browser_close",
   "description": "Close the browser and release resources.",
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
   "name": "suno_generate_song",
   "description": (
    "Generate an AI song using Suno. Creates a musical composition with vocals based on provided lyrics and style. "
    "Returns a job ID that can be used to check generation status. "
    "Use wait_for_completion=True to poll until the song is ready."
   ),
   "parameters": {
    "type": "object",
    "properties": {
     "lyrics": {
      "type": "string",
      "description": "The lyrics for the song. Can be multiple lines with verses and chorus."
     },
     "style": {
      "type": "string",
      "description": "Musical style/genre description (e.g., 'Upbeat electronic pop with female vocals', 'Acoustic folk ballad', 'Rap with heavy beat')."
     },
     "title": {
      "type": "string",
      "description": "Optional title for the song."
     },
     "wait_for_completion": {
      "type": "boolean",
      "description": "If true, wait for generation to complete and return full song data. Default: false (returns job_id only)."
     },
     "timeout": {
      "type": "integer",
      "description": "Maximum seconds to wait for completion if wait_for_completion=True. Default: 300."
     }
    },
    "required": ["lyrics", "style"]
   }
  }
 },
 {
  "type": "function",
  "function": {
   "name": "suno_get_job_status",
   "description": (
    "Check the status of a song generation job submitted to Suno. "
    "Returns current status (pending, processing, completed, failed) and progress information."
   ),
   "parameters": {
    "type": "object",
    "properties": {
     "job_id": {
      "type": "string",
      "description": "The job ID returned from suno_generate_song."
     }
    },
    "required": ["job_id"]
   }
  }
 },
 {
  "type": "function",
  "function": {
   "name": "suno_get_song_data",
   "description": (
    "Retrieve complete metadata and URLs for a generated song from Suno. "
    "Includes title, artist, duration, audio download URLs, and cover image. "
    "Use this after a job status shows 'completed'."
   ),
   "parameters": {
    "type": "object",
    "properties": {
     "song_id": {
      "type": "string",
      "description": "The song ID from a completed job, obtained via suno_get_job_status."
     }
    },
    "required": ["song_id"]
   }
  }
 },
 {
  "type": "function",
  "function": {
   "name": "suno_list_songs",
   "description": (
    "List all songs generated by the user in Suno. "
    "Supports pagination and status filtering."
   ),
   "parameters": {
    "type": "object",
    "properties": {
     "limit": {
      "type": "integer",
      "description": "Number of results to return (default: 20, max: 100)."
     },
     "offset": {
      "type": "integer",
      "description": "Pagination offset for fetching more results. Default: 0."
     },
     "status": {
      "type": "string",
      "enum": ["pending", "processing", "completed", "failed"],
      "description": "Optional filter by song status."
     }
    },
    "required": []
   }
  }
 },
 {
  "type": "function",
  "function": {
   "name": "suno_delete_song",
   "description": (
    "Delete a generated song from Suno. "
    "This permanently removes the song and all associated data."
   ),
   "parameters": {
    "type": "object",
    "properties": {
     "song_id": {
      "type": "string",
      "description": "The song ID to delete."
     }
    },
    "required": ["song_id"]
   }
  }
 },
 {
  "type": "function",
  "function": {
   "name": "create_task",
   "description": (
    "Create a tracked task with ordered steps. Use this at the start of any "
    "non-trivial task to plan your work. Each step should be a concrete action. "
    "Update steps as you progress. This helps you resume if errors occur."
   ),
   "parameters": {
    "type": "object",
    "properties": {
     "description": {
      "type": "string",
      "description": "A clear description of the overall task."
     },
     "steps": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Ordered list of step descriptions. Break the task into concrete, verifiable actions."
     }
    },
    "required": ["description"]
   }
  }
 },
 {
  "type": "function",
  "function": {
   "name": "update_task_step",
   "description": (
    "Update the status of a task step. Mark steps in_progress when you start them, "
    "completed when done, failed if an error occurs, or skipped if no longer needed."
   ),
   "parameters": {
    "type": "object",
    "properties": {
     "task_id": {
      "type": "string",
      "description": "The task ID returned by create_task."
     },
     "step_index": {
      "type": "integer",
      "description": "0-based index of the step to update."
     },
     "status": {
      "type": "string",
      "enum": ["in_progress", "completed", "failed", "skipped"],
      "description": "New status for the step."
     },
     "result": {
      "type": "string",
      "description": "Optional result summary when completing a step."
     },
     "error": {
      "type": "string",
      "description": "Error description when marking a step as failed."
     }
    },
    "required": ["task_id", "step_index", "status"]
   }
  }
 },
 {
  "type": "function",
  "function": {
   "name": "complete_task",
   "description": (
    "Mark a task as completed. Call this when all work is done and verified. "
    "Include a brief result summary."
   ),
   "parameters": {
    "type": "object",
    "properties": {
     "task_id": {
      "type": "string",
      "description": "The task ID to complete."
     },
     "result": {
      "type": "string",
      "description": "Optional summary of what was accomplished."
     }
    },
    "required": ["task_id"]
   }
  }
 },
 {
  "type": "function",
  "function": {
   "name": "ask_human",
   "description": (
    "Ask the human a question and pause the task until they respond. "
    "Use this when you need: a decision requiring human judgment, "
    "credentials or access only the human can provide, guidance after "
    "failed workarounds, or approval before destructive changes. "
    "The task will be blocked until the human responds."
   ),
   "parameters": {
    "type": "object",
    "properties": {
     "question": {
      "type": "string",
      "description": "The question to ask the human. Be specific about what you need."
     }
    },
    "required": ["question"]
   }
  }
 },
 {
  "type": "function",
  "function": {
   "name": "list_tasks",
   "description": "List tracked tasks, optionally filtered by status (pending, in_progress, completed, failed, blocked).",
   "parameters": {
    "type": "object",
    "properties": {
     "status": {
      "type": "string",
      "description": "Optional status filter.",
      "enum": ["pending", "in_progress", "completed", "failed", "blocked"]
     }
    },
    "required": []
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
_BASE_TOOL_HANDLERS = {
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
 "ask_ollama": _make_handler(ask_ollama),
 "browser_navigate": _make_handler(browser_navigate),
 "browser_action": _make_handler(browser_action),
 "browser_get_content": _make_handler(browser_get_content),
 "browser_get_elements": _make_handler(browser_get_elements),
 "browser_close": _make_handler(browser_close),
 "suno_generate_song": _make_handler(suno_generate_song),
 "suno_get_job_status": _make_handler(suno_get_job_status),
 "suno_get_song_data": _make_handler(suno_get_song_data),
 "suno_list_songs": _make_handler(suno_list_songs),
 "suno_delete_song": _make_handler(suno_delete_song),
 "create_task": _make_handler(create_task),
 "update_task_step": _make_handler(update_task_step),
 "complete_task": _make_handler(complete_task),
 "ask_human": _make_handler(ask_human),
 "list_tasks": _make_handler(list_tasks),
}


# Export combined definitions and handlers
TOOL_DEFINITIONS = _BASE_TOOL_DEFINITIONS.copy()
TOOL_HANDLERS = _BASE_TOOL_HANDLERS.copy()


def refresh_mcp_tools(mcp_client=None):
    """Refresh MCP tools - called by coding_agent.py after MCP initialization.
    
    This function merges MCP tools into TOOL_DEFINITIONS and TOOL_HANDLERS.
    """
    global TOOL_DEFINITIONS, TOOL_HANDLERS, _mcp_client, _mcp_tools_loaded
    
    if mcp_client:
        _mcp_client = mcp_client
        _mcp_tools_loaded = True
    
    if not _mcp_client:
        return [], {}
    
    # Get MCP tools and convert to OpenAI format
    tools = _mcp_client.get_all_tools()
    definitions = []
    for tool in tools:
        definitions.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", f"MCP tool from {tool.get('_mcp_server', 'unknown')}"),
                "parameters": tool.get("parameters", {
                    "type": "object",
                    "properties": {},
                    "required": []
                })
            }
        })
    
    # Create handlers for MCP tools
    handlers = {}
    for tool in tools:
        name = tool["name"]
        handlers[name] = lambda args, tn=name, client=_mcp_client: json.dumps(client.call_tool(tn, args or {}))
    
    # Merge into global tools
    TOOL_DEFINITIONS = _BASE_TOOL_DEFINITIONS + definitions
    TOOL_HANDLERS = {**_BASE_TOOL_HANDLERS, **handlers}
    
    return definitions, handlers
