"""
Keepalive and Process Supervision for NAA
"""

import time
import socket
import logging
import threading
import urllib.request
import json
from typing import Optional, Any

logger = logging.getLogger("naa-supervisor")


def get_server_health(port: int) -> Optional[dict]:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5) as r:
            if r.status != 200:
                return None
            return json.loads(r.read())
    except Exception:
        return None

def start_keepalive(port: int):
    def _loop():
        while True:
            try:
                time.sleep(50)
                _ = sum(i * i for i in range(300_000))
                urllib.request.urlopen(f"http://localhost:{port}/ping", timeout=5)
            except Exception:
                pass
    t = threading.Thread(target=_loop, daemon=True)
    t.start()

def is_server_healthy(port: int) -> bool:
    data = get_server_health(port)
    return bool(data and data.get("model_loaded"))


def is_server_loading(port: int) -> bool:
    data = get_server_health(port)
    if not data or data.get("load_error"):
        return False
    return bool(
        data.get("model_loading")
        or data.get("load_stage") == "loading"
    )

def public_health_ok(url: str, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=3) as r:
                if r.status == 200:
                    data = json.loads(r.read())
                    if data.get("model_loaded"):
                        return True
        except Exception:
            pass
        time.sleep(1)
    return False

def wait_for_port(port: int, timeout: float = 180.0, proc: Optional[Any] = None) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and hasattr(proc, "poll") and proc.poll() is not None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    return False
