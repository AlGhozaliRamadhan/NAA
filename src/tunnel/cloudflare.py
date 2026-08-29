"""Cloudflare and localhost.run tunnel management."""

import os
import queue
import re
import time
import platform
import subprocess
import threading
import tempfile
import urllib.request
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

CLOUDFLARED_URLS = {
    "linux_amd64":  "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
    "linux_arm64":  "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64",
    "darwin_amd64": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64",
    "windows":      "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe",
}

_TUNNEL_LOGS: Deque[str] = deque(maxlen=200)
_HTTPS_URL_RE = re.compile(r"https://[^\s<>\[\]()]+", re.IGNORECASE)


def tunnel_log_tail(limit: int = 20) -> list[str]:
    """Return recent cloudflared output without exposing it through a pipe."""

    return list(_TUNNEL_LOGS)[-max(1, limit):]


def _drain_output(
    proc: subprocess.Popen,
    lines: queue.Queue,
    startup_done: threading.Event,
) -> None:
    """Continuously drain cloudflared stdout so its pipe can never deadlock."""

    stdout = proc.stdout
    if stdout is None:
        return
    try:
        for line in iter(stdout.readline, ""):
            clean = line.rstrip("\r\n")
            if clean:
                _TUNNEL_LOGS.append(clean)
                # The queue is consumed only during startup.  Continue draining
                # into the bounded log tail afterwards without growing memory.
                if not startup_done.is_set():
                    lines.put(clean)
    finally:
        try:
            stdout.close()
        except Exception:
            pass


def _localhost_run_url(line: str) -> Optional[str]:
    """Extract the anonymous public URL printed by localhost.run."""

    for match in _HTTPS_URL_RE.findall(line):
        candidate = match.rstrip(".,;:'\"")
        lowered = candidate.lower()
        if ".lhr.life" in lowered or (
            ".localhost.run" in lowered
            and "admin.localhost.run" not in lowered
        ):
            return candidate.rstrip("/")
    return None


def _ensure_localhost_run_identity() -> Optional[Path]:
    """Create a dedicated key so free reconnects can retain their hostname."""

    configured = os.environ.get("NAA_SSH_IDENTITY_FILE", "").strip()
    if configured:
        identity = Path(configured).expanduser()
    else:
        # Keep the key for the lifetime of the notebook runtime.  localhost.run
        # maps free tunnels to SSH identities, whereas the `nokey` account gets
        # a fresh hostname whenever the connection is recreated.
        runtime_dir = Path("/content") if Path("/content").is_dir() else Path(tempfile.gettempdir())
        identity = runtime_dir / ".naa_localhost_run_ed25519"

    if identity.exists() and identity.with_suffix(identity.suffix + ".pub").exists():
        return identity

    try:
        identity.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                os.environ.get("NAA_SSH_KEYGEN_BINARY", "ssh-keygen"),
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-f",
                str(identity),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if result.returncode == 0 and identity.exists():
            return identity
        detail = (result.stderr or result.stdout or "unknown ssh-keygen error").strip()
        _TUNNEL_LOGS.append(f"Could not create localhost.run SSH identity: {detail}")
    except Exception as exc:
        _TUNNEL_LOGS.append(f"Could not create localhost.run SSH identity: {exc}")
    return None


def _start_localhost_run_tunnel(
    port: int,
) -> Tuple[Optional[subprocess.Popen], Optional[str]]:
    """Start a free anonymous HTTPS reverse tunnel over SSH."""

    ssh_binary = os.environ.get("NAA_SSH_BINARY", "ssh").strip() or "ssh"
    known_hosts = os.environ.get(
        "NAA_SSH_KNOWN_HOSTS",
        "/tmp/naa_localhost_run_known_hosts",
    )
    identity = _ensure_localhost_run_identity()
    command = [
        ssh_binary,
        "-T",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "ExitOnForwardFailure=yes",
    ]
    if identity is not None:
        command.extend([
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-i",
            str(identity),
        ])
        destination = "localhost.run"
    else:
        # Remain usable on minimal images without ssh-keygen, but make it clear
        # in the logs that a reconnect will receive a different public URL.
        destination = "nokey@localhost.run"
        _TUNNEL_LOGS.append(
            "Using localhost.run without an SSH identity; reconnects will change the URL."
        )
    command.extend(["-R", f"80:127.0.0.1:{port}", destination])
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: queue.Queue = queue.Queue()
    startup_done = threading.Event()
    threading.Thread(
        target=_drain_output,
        args=(proc, lines, startup_done),
        daemon=True,
        name="naa-localhost-run-log-drain",
    ).start()

    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        try:
            line = lines.get(timeout=0.5)
        except queue.Empty:
            continue
        public_url = _localhost_run_url(line)
        if public_url:
            startup_done.set()
            return proc, public_url

    startup_done.set()
    try:
        proc.terminate()
    except Exception:
        pass
    return None, None

def cf_binary_path() -> Path:
    suffix = ".exe" if platform.system().lower() == "windows" else ""
    return Path("/tmp") / f"cloudflared{suffix}"

def download_cloudflared() -> Path:
    dest = cf_binary_path()
    if dest.exists():
        return dest
    system = platform.system().lower()
    arch = platform.machine().lower()
    if system == "windows":
        key = "windows"
    elif "arm" in arch or "aarch" in arch:
        key = "linux_arm64"
    elif system == "darwin":
        key = "darwin_amd64"
    else:
        key = "linux_amd64"

    url = CLOUDFLARED_URLS[key]
    urllib.request.urlretrieve(url, str(dest))
    if system != "windows":
        os.chmod(str(dest), 0o755)
    return dest

def start_tunnel(port: int) -> Tuple[Optional[subprocess.Popen], Optional[str]]:
    """Start a named Cloudflare Tunnel or a limited TryCloudflare fallback.

    Agent clients require SSE, which Cloudflare does not support on Quick
    Tunnels.  Set both NAA_CF_TUNNEL_TOKEN and NAA_PUBLIC_URL to use a regular
    remotely-managed tunnel with a stable hostname.
    """

    try:
        provider = os.environ.get("NAA_TUNNEL_PROVIDER", "cloudflare").strip().lower()
        if provider in {"localhost-run", "localhost.run", "localhostrun", "ssh"}:
            return _start_localhost_run_tunnel(port)
        if provider in {"none", "off", "disabled"}:
            return None, None
        if provider not in {"cloudflare", "quick", "trycloudflare"}:
            _TUNNEL_LOGS.append(f"Unknown NAA_TUNNEL_PROVIDER: {provider}")
            return None, None

        cf = download_cloudflared()
        tunnel_token = os.environ.get("NAA_CF_TUNNEL_TOKEN", "").strip()
        configured_url = os.environ.get("NAA_PUBLIC_URL", "").strip().rstrip("/")

        if tunnel_token and not configured_url:
            _TUNNEL_LOGS.append(
                "NAA_CF_TUNNEL_TOKEN is set but NAA_PUBLIC_URL is missing."
            )
            return None, None

        if tunnel_token:
            command = [
                str(cf),
                "tunnel",
                "--no-autoupdate",
                "run",
                "--token",
                tunnel_token,
            ]
        else:
            command = [
                str(cf),
                "tunnel",
                "--url",
                f"http://localhost:{port}",
                "--no-autoupdate",
            ]

        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines: queue.Queue = queue.Queue()
        startup_done = threading.Event()
        threading.Thread(
            target=_drain_output,
            args=(proc, lines, startup_done),
            daemon=True,
            name="naa-cloudflared-log-drain",
        ).start()

        deadline = time.time() + 60
        named_fallback_deadline = time.time() + 10
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                line = lines.get(timeout=0.5)
            except queue.Empty:
                if tunnel_token and time.time() >= named_fallback_deadline:
                    startup_done.set()
                    return proc, configured_url
                continue

            if tunnel_token and (
                "registered tunnel connection" in line.lower()
                or "connection registered" in line.lower()
            ):
                startup_done.set()
                return proc, configured_url

            if "trycloudflare.com" in line or ".cfargotunnel.com" in line:
                for part in line.split():
                    clean = part.strip().rstrip(".,;)")
                    if clean.startswith("https://") and ("trycloudflare" in clean or "cfargotunnel" in clean):
                        startup_done.set()
                        return proc, clean

        # Some cloudflared versions do not emit a stable readiness phrase for
        # token-based tunnels.  A live process plus the configured public URL
        # is enough for the caller's public health probe to verify it.
        if tunnel_token and proc.poll() is None:
            startup_done.set()
            return proc, configured_url

        startup_done.set()
        try:
            proc.terminate()
        except Exception:
            pass
        return None, None
    except Exception as exc:
        _TUNNEL_LOGS.append(f"cloudflared startup failed: {exc}")
        return None, None
