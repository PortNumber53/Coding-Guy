"""Hot-reload watcher for the Coding-Guy server mode.

Monitors a directory (default: .git) for file changes and restarts the server
subprocess automatically when changes are detected.
"""

import os
import signal
import subprocess
import sys
import time

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


# Debounce: ignore rapid successive changes within this window (seconds).
DEBOUNCE_SECONDS = 2


class _ReloadHandler(FileSystemEventHandler):
    """Sets a flag when file system activity has settled."""

    def __init__(self):
        self.triggered = False
        self._last_activity = 0
        self._last_event_path: str | None = None
        self._mtimes_ns: dict[str, int] = {}

    def on_any_event(self, event):
        src_path = getattr(event, "src_path", "") or ""

        # Ignore directory-level events.
        if getattr(event, "is_directory", False):
            return

        # Only treat actual content/metadata changes as restart triggers.
        # Some backends emit opened/closed events for reads which would otherwise
        # cause an infinite restart loop on startup.
        event_type = getattr(event, "event_type", None)
        if event_type not in {"modified", "created", "moved", "deleted"}:
            return

        # watchdog may report "modified" for metadata-only changes (e.g. atime updates
        # when a file is merely read). Only trigger reload if the file's mtime changed.
        if event_type == "modified" and src_path:
            try:
                mtime_ns = os.stat(src_path).st_mtime_ns
            except OSError:
                # If the file disappeared between event and stat, treat it as a real change.
                mtime_ns = -1
            prev = self._mtimes_ns.get(src_path)
            if prev is not None and prev == mtime_ns:
                return
            self._mtimes_ns[src_path] = mtime_ns

        # Ignore noisy paths/files that commonly change and should not restart the server.
        ignored_parts = (
            f"{os.sep}.git{os.sep}",
            f"{os.sep}__pycache__{os.sep}",
            f"{os.sep}.pytest_cache{os.sep}",
            f"{os.sep}.mypy_cache{os.sep}",
            f"{os.sep}.ruff_cache{os.sep}",
            f"{os.sep}.venv{os.sep}",
            f"{os.sep}node_modules{os.sep}",
        )
        if any(part in src_path for part in ignored_parts):
            return

        ignored_substrings = (".lock", ".tmp", ".swp", ".swo")
        if any(x in src_path for x in ignored_substrings):
            return

        # Ignore environment files which can be touched by external tooling and
        # can cause restart loops.
        base = os.path.basename(src_path)
        if base == ".env" or base.startswith(".env."):
            return

        # Only restart on changes that are likely to affect the running server.
        allowed_exts = (".py", ".json", ".toml", ".yaml", ".yml")
        _, ext = os.path.splitext(src_path)
        if ext and ext.lower() not in allowed_exts:
            return

        self._last_event_path = src_path
        self._last_activity = time.monotonic()

    def check_settled(self):
        """Returns True if activity has settled for the debounce period."""
        if not self._last_activity:
            return False
        if time.monotonic() - self._last_activity > DEBOUNCE_SECONDS:
            self.triggered = True
            return True
        return False


def run_with_reload(watch_path: str, extra_args: list[str] | None = None) -> int:
    """Run the server as a subprocess, restarting on file changes.

    Args:
        watch_path: Directory to watch for changes.
        extra_args: Additional CLI arguments forwarded to the server subprocess.

    Returns:
        Exit code of the last server process.
    """
    if not os.path.isdir(watch_path):
        print(f"[hot-reload] Watch path does not exist: {watch_path}", file=sys.stderr)
        return 1

    # Build the command: re-run coding_agent.py --serve (without --reload to
    # avoid infinite recursion).
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "coding_agent.py"), "--serve"]
    if extra_args:
        cmd.extend(extra_args)

    print(f"[hot-reload] Watching {watch_path} for changes", file=sys.stderr)

    while True:
        print("[hot-reload] Starting server…", file=sys.stderr)
        proc = subprocess.Popen(cmd)

        handler = _ReloadHandler()
        observer = Observer()
        observer.schedule(handler, watch_path, recursive=True)
        observer.start()

        try:
            while not handler.check_settled():
                # Check if the child exited on its own (crash / clean shutdown).
                ret = proc.poll()
                if ret is not None:
                    observer.stop()
                    observer.join()
                    return ret
                time.sleep(0.5)
        except KeyboardInterrupt:
            # Ctrl-C: shut down cleanly.
            proc.send_signal(signal.SIGINT)
            proc.wait()
            observer.stop()
            observer.join()
            return 0
        finally:
            observer.stop()
            observer.join()

        # Change detected - terminate the running server and restart.
        reason = f" (trigger: {handler._last_event_path})" if handler._last_event_path else ""
        print(f"[hot-reload] Change detected, restarting server…{reason}", file=sys.stderr)
        proc.terminate()
        try:
            # Give more time for graceful shutdown - allow Telegram messages to complete
            proc.wait(timeout=35)
        except subprocess.TimeoutExpired:
            print("[hot-reload] Graceful shutdown timed out, force killing...", file=sys.stderr)
            proc.kill()
            proc.wait()
