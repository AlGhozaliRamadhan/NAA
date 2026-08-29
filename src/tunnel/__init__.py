"""NAA Tunnel Management Dispatcher (Cloudflare, Ngrok, localhost.run)."""

from __future__ import annotations

import os
import subprocess
from typing import Optional, Tuple

from src.tunnel.cloudflare import (
    CLOUDFLARED_URLS,
    _start_localhost_run_tunnel,
    cf_binary_path,
    download_cloudflared,
    start_tunnel as start_cloudflare_tunnel,
    tunnel_log_tail as cloudflare_tunnel_log_tail,
)
from src.tunnel.ngrok import ngrok_log_tail, start_ngrok_tunnel


def tunnel_log_tail(limit: int = 20) -> list[str]:
    """Return recent tunnel log output across active providers."""
    provider = os.environ.get("NAA_TUNNEL_PROVIDER", "cloudflare").strip().lower()
    if provider in {"ngrok", "pyngrok"}:
        return ngrok_log_tail(limit)
    return cloudflare_tunnel_log_tail(limit)


def start_tunnel(port: int) -> Tuple[Optional[subprocess.Popen], Optional[str]]:
    """Dispatch tunnel startup to the configured provider."""
    provider = os.environ.get("NAA_TUNNEL_PROVIDER", "cloudflare").strip().lower()

    if provider in {"ngrok", "pyngrok"}:
        return start_ngrok_tunnel(port)

    if provider in {"localhost-run", "localhost.run", "localhostrun", "ssh"}:
        return _start_localhost_run_tunnel(port)

    if provider in {"none", "off", "disabled", "external"}:
        configured_url = os.environ.get("NAA_PUBLIC_URL", "").strip().rstrip("/")
        if configured_url:
            return None, configured_url
        return None, None

    return start_cloudflare_tunnel(port)


__all__ = [
    "CLOUDFLARED_URLS",
    "cf_binary_path",
    "download_cloudflared",
    "ngrok_log_tail",
    "start_cloudflare_tunnel",
    "start_ngrok_tunnel",
    "start_tunnel",
    "tunnel_log_tail",
]
