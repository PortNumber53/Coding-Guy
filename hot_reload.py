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

    def on_any_event(self, event):
        # Ignore common temporary files that don't warrant a restart
        if any(x in event.src_path for x in (".lock", ".tmp", "__pycache__")):
            return
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
        print("[hot-reload] Change detected, restarting server…", file=sys.stderr)
        proc.terminate()
        try:
            # Give more time for graceful shutdown - allow Telegram messages to complete
            proc.wait(timeout=35)
        except subprocess.TimeoutExpired:
            print("[hot-reload] Graceful shutdown timed out, force killing...", file=sys.stderr)
            proc.kill()
            proc.wait()
