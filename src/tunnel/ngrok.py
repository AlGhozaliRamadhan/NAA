"""Ngrok tunnel management for NAA."""

from __future__ import annotations

import os
import subprocess
from collections import deque
from typing import Deque, Optional, Tuple

_NGROK_LOGS: Deque[str] = deque(maxlen=200)


def ngrok_log_tail(limit: int = 20) -> list[str]:
    """Return recent ngrok log output."""
    return list(_NGROK_LOGS)[-max(1, limit):]


def start_ngrok_tunnel(
    port: int,
) -> Tuple[Optional[subprocess.Popen], Optional[str]]:
    """Start an ngrok HTTPS tunnel using pyngrok.

    Supports full bidirectional SSE streaming required for agentic workflows
    (such as Claude Code and OpenCode).
    """
    token = (
        os.environ.get("NAA_NGROK_AUTHTOKEN", "")
        or os.environ.get("NGROK_AUTHTOKEN", "")
        or os.environ.get("NAA_NGROK_TOKEN", "")
        or os.environ.get("NGROK_TOKEN", "")
    ).strip()

    if not token:
        _NGROK_LOGS.append(
            "NAA_NGROK_AUTHTOKEN (or NGROK_AUTHTOKEN) is required for the ngrok tunnel provider."
        )
        return None, None

    try:
        from pyngrok import ngrok

        ngrok.set_auth_token(token)

        # Disconnect any lingering tunnels to avoid port collision
        try:
            for active in ngrok.get_tunnels():
                ngrok.disconnect(active.public_url)
        except Exception:
            pass

        tunnel = ngrok.connect(port, "http")
        public_url = tunnel.public_url.rstrip("/")
        _NGROK_LOGS.append(f"ngrok tunnel established: {public_url}")

        ngrok_proc = ngrok.get_ngrok_process()
        proc = getattr(ngrok_proc, "proc", None)
        return proc, public_url
    except ImportError:
        _NGROK_LOGS.append(
            "pyngrok package is not installed. Please install it with: pip install pyngrok"
        )
        return None, None
    except Exception as exc:
        _NGROK_LOGS.append(f"ngrok tunnel startup failed: {exc}")
        return None, None
