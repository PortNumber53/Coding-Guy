"""Docker container manager for sandboxed agent execution."""

import atexit
import os
import subprocess
import sys
import uuid

# Dockerfile search paths, in priority order
DOCKERFILE_SEARCH_PATHS = [
    ".coding-guy/Dockerfile",  # project-local
]

DEFAULT_DOCKERFILE = """\
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \\
    python3 python3-pip python3-venv \\
    nodejs npm \\
    golang-go \\
    git curl wget grep findutils openssh-client \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
"""

IMAGE_NAME = "coding-guy-sandbox"
CONTAINER_PREFIX = "coding-guy-session"
MOUNT_TARGET = "/workspace"

# Optional environment variables forwarded into the container.
_ENV_FORWARD = ["GIT_TOKEN", "GIT_USER_NAME", "GIT_USER_EMAIL"]


class DockerManager:
    """Manages a persistent Docker container for sandboxed tool execution."""

    def __init__(self, work_dir: str, subprocess_timeout: int = 300):
        self.work_dir = os.path.abspath(work_dir)
        self.container_id: str | None = None
        self.image_tag: str = IMAGE_NAME
        self.subprocess_timeout = subprocess_timeout
        self.startup_warnings: list[str] = []
        self.ssh_mode: str = "none"  # "none", "agent", or "keys"
        atexit.register(self.cleanup)

    def _run(self, cmd: list[str], timeout: int | None = None, **kwargs) -> subprocess.CompletedProcess:
        """Run a subprocess command."""
        return subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout or self.subprocess_timeout, **kwargs
        )

    def find_dockerfile(self) -> str | None:
        """Return path to a custom Dockerfile if one exists, else None."""
        for rel in DOCKERFILE_SEARCH_PATHS:
            full = os.path.join(self.work_dir, rel)
            if os.path.isfile(full):
                return full
        # Also check user-global config
        global_path = os.path.expanduser("~/.config/coding-guy/Dockerfile")
        if os.path.isfile(global_path):
            return global_path
        return None

    def build_image(self) -> dict:
        """Build the Docker image. Returns dict with status info."""
        dockerfile_path = self.find_dockerfile()

        if dockerfile_path:
            print(
                f"  Building Docker image from {dockerfile_path}...",
                file=sys.stderr,
            )
            ctx_dir = os.path.dirname(dockerfile_path)
            result = self._run(
                ["docker", "build", "-t", self.image_tag, "-f", dockerfile_path, ctx_dir]
            )
        else:
            print(
                "  Building Docker image from default Dockerfile...",
                file=sys.stderr,
            )
            result = self._run(
                ["docker", "build", "-t", self.image_tag, "-"],
                input=DEFAULT_DOCKERFILE,
            )

        if result.returncode != 0:
            raise RuntimeError(f"Docker build failed:\n{result.stderr}")

        source = dockerfile_path or "default (embedded)"
        print(f"  Image '{self.image_tag}' built successfully.", file=sys.stderr)
        return {"status": "built", "image": self.image_tag, "dockerfile": source}

    def _detect_ssh(self) -> dict:
        """Detect SSH availability on the host. Returns dict with mode and paths."""
        ssh_auth_sock = os.getenv("SSH_AUTH_SOCK")
        if ssh_auth_sock and os.path.exists(ssh_auth_sock):
            return {"mode": "agent", "ssh_auth_sock": ssh_auth_sock}

        ssh_dir = os.path.expanduser("~/.ssh")
        if os.path.isdir(ssh_dir):
            for name in ("id_ed25519", "id_rsa", "id_ecdsa"):
                key_path = os.path.join(ssh_dir, name)
                if os.path.isfile(key_path):
                    return {"mode": "keys", "ssh_dir": ssh_dir}

        return {"mode": "none"}

    def start_container(self) -> None:
        """Start a persistent container with work_dir mounted."""
        ssh_info = self._detect_ssh()
        self.ssh_mode = ssh_info["mode"]

        name = f"{CONTAINER_PREFIX}-{uuid.uuid4().hex[:8]}"
        cmd = [
            "docker", "run", "-d",
            "--name", name,
            "-v", f"{self.work_dir}:{MOUNT_TARGET}",
            "-w", MOUNT_TARGET,
        ]

        # Mount SSH agent socket or SSH keys directory when available.
        if self.ssh_mode == "agent":
            host_sock = ssh_info["ssh_auth_sock"]
            cmd.extend([
                "-v", f"{host_sock}:/run/ssh-agent.sock:ro",
                "-e", "SSH_AUTH_SOCK=/run/ssh-agent.sock",
            ])
        elif self.ssh_mode == "keys":
            cmd.extend(["-v", f"{ssh_info['ssh_dir']}:/root/.ssh:ro"])

        # Forward optional env vars (e.g. GIT_TOKEN) into the container.
        for var in _ENV_FORWARD:
            val = os.getenv(var)
            if val:
                cmd.extend(["-e", f"{var}={val}"])
        cmd.extend([self.image_tag, "tail", "-f", "/dev/null"])
        result = self._run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"Container start failed:\n{result.stderr}")
        self.container_id = result.stdout.strip()
        print(f"  Container started: {name} (ssh: {self.ssh_mode})", file=sys.stderr)
        self._configure_git()
        self._configure_ssh()

    def _configure_git(self) -> None:
        """Set git identity and credential helpers inside the container."""
        configs = {
            "user.name": os.getenv("GIT_USER_NAME"),
            "user.email": os.getenv("GIT_USER_EMAIL"),
        }
        for key, value in configs.items():
            if value:
                result = self._run([
                    "docker", "exec", self.container_id,
                    "git", "config", "--global", key, value
                ])
                if result.returncode != 0:
                    msg = f"Failed to set git {key}: {result.stderr.strip()}"
                    print(f"  Warning: {msg}", file=sys.stderr)
                    self.startup_warnings.append(msg)

        # When SSH is not available, rewrite SSH URLs to HTTPS using GIT_TOKEN.
        if self.ssh_mode == "none":
            git_token = os.getenv("GIT_TOKEN")
            if git_token:
                self._configure_https_fallback(git_token)

    def _configure_https_fallback(self, git_token: str) -> None:
        """Set up git to rewrite SSH URLs to HTTPS and authenticate with a token."""
        for host in ("github.com", "gitlab.com", "bitbucket.org"):
            # Rewrite git@host: → https://host/
            self._run([
                "docker", "exec", self.container_id,
                "git", "config", "--global",
                f"url.https://{host}/.insteadOf", f"git@{host}:",
            ])
        # Store credentials so the token is not exposed in remote URLs.
        cred_lines = "\n".join(
            f"https://{git_token}@{h}" for h in ("github.com", "gitlab.com", "bitbucket.org")
        ) + "\n"
        self._run([
            "docker", "exec", "-i", self.container_id,
            "bash", "-c", "cat > /root/.git-credentials",
        ], input=cred_lines)
        self._run([
            "docker", "exec", self.container_id,
            "git", "config", "--global",
            "credential.helper", "store --file=/root/.git-credentials",
        ])

    def _configure_ssh(self) -> None:
        """Configure SSH inside the container based on detected mode."""
        if self.ssh_mode == "none":
            return

        # Accept new host keys automatically (fresh container has no known_hosts).
        self._run([
            "docker", "exec", self.container_id,
            "bash", "-c",
            "mkdir -p /root/.ssh && printf '%s\\n' "
            "'Host *' '    StrictHostKeyChecking accept-new' "
            ">> /root/.ssh/config && chmod 600 /root/.ssh/config",
        ])

        if self.ssh_mode == "agent":
            # Verify the forwarded agent is usable.
            result = self._run([
                "docker", "exec", self.container_id,
                "ssh-add", "-l",
            ])
            if result.returncode != 0:
                msg = "SSH agent forwarded but no keys accessible — falling back to HTTPS."
                print(f"  Warning: {msg}", file=sys.stderr)
                self.startup_warnings.append(msg)
                self.ssh_mode = "none"
                git_token = os.getenv("GIT_TOKEN")
                if git_token:
                    self._configure_https_fallback(git_token)

    def exec(self, cmd: list[str], stdin_data: str | None = None) -> tuple[int, str, str]:
        """Execute a command inside the container.

        Returns (returncode, stdout, stderr).
        """
        self.ensure_running()
        docker_cmd = ["docker", "exec"]
        if stdin_data is not None:
            docker_cmd.append("-i")
        docker_cmd.extend([self.container_id] + cmd)
        result = self._run(docker_cmd, input=stdin_data)
        return result.returncode, result.stdout, result.stderr

    def is_running(self) -> bool:
        """Check if the container is still running."""
        if not self.container_id:
            return False
        result = self._run(
            ["docker", "inspect", "-f", "{{.State.Running}}", self.container_id]
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    def ensure_running(self) -> None:
        """Build image and start container if not already running."""
        if self.container_id and self.is_running():
            return
        if self.container_id:
            # Container died, clean up and restart
            self._run(["docker", "rm", "-f", self.container_id])
            self.container_id = None
        self.build_image()
        self.start_container()

    def rebuild(self) -> dict:
        """Rebuild image and restart container (used after Dockerfile changes)."""
        self.cleanup()
        self.startup_warnings.clear()
        info = self.build_image()
        self.start_container()
        info["status"] = "rebuilt"
        return info

    def cleanup(self) -> None:
        """Stop and remove the container."""
        if self.container_id:
            self._run(["docker", "rm", "-f", self.container_id])
            self.container_id = None

    def get_dockerfile_path(self) -> str:
        """Return the path to the Dockerfile in use.

        If no custom one exists, returns the default project-local path
        where one should be created.
        """
        existing = self.find_dockerfile()
        if existing:
            return existing
        return os.path.join(self.work_dir, ".coding-guy", "Dockerfile")
