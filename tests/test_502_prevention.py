"""
Automated pytest test for 502 Bad Gateway prevention logic, tunnel startup sequencing,
and supervisor watchdog process recovery in NAA.
"""

import sys
import time
import threading
from pathlib import Path
import pytest

import naa

class FakeProc:
    def __init__(self, calls):
        self._alive = True
        self.calls = calls
        self.pid = len([c for c in calls if c[0] == "Popen"]) + 100
        self.calls.append(("Popen",))

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self._alive = False
        self.calls.append(("terminate",))

    def kill(self):
        self._alive = False

    def wait(self, timeout=None):
        self._alive = False


def test_502_prevention_startup_sequencing_and_watchdog(tmp_path: Path, monkeypatch):
    calls = []
    fake_model_file = tmp_path / "fake.gguf"
    fake_model_file.write_bytes(b"\x00")

    monkeypatch.setattr(naa, "_download_cloudflared", lambda: "/tmp/cloudflared")
    monkeypatch.setattr(naa, "download_model", lambda model: fake_model_file)

    def fake_start_server(model_path, admin_key, model_cfg, preset="default", system_prompt=None, **_extra):
        proc = FakeProc(calls)
        naa._server_proc = proc
        calls.append(("start_server",))
        return True

    monkeypatch.setattr(naa, "start_server", fake_start_server)

    _state = {"model_loaded": False}

    def fake_is_server_healthy(port):
        calls.append(("is_server_healthy", _state["model_loaded"]))
        return _state["model_loaded"]

    monkeypatch.setattr(naa, "_is_server_healthy", fake_is_server_healthy)

    def fake_public_health_ok(url, timeout=10.0):
        calls.append(("public_health_ok", url))
        return True

    monkeypatch.setattr(naa, "_public_health_ok", fake_public_health_ok)

    def fake_start_tunnel(port):
        proc = FakeProc(calls)
        naa._tunnel_proc = proc
        calls.append(("start_tunnel",))
        return proc, "https://example.trycloudflare.com"

    monkeypatch.setattr(naa, "start_tunnel", fake_start_tunnel)
    monkeypatch.setattr(naa, "start_keepalive", lambda port: calls.append(("start_keepalive",)))
    monkeypatch.setattr(naa, "load_state", lambda: {"model_key": "auto", "model_path": str(fake_model_file), "admin_key": "naa-test"})
    monkeypatch.setattr(naa, "save_state", lambda d: calls.append(("save_state", d)))

    for fn in ("info", "ok", "warn", "err", "step", "header", "rule", "print_banner"):
        monkeypatch.setattr(naa, fn, lambda *a, **k: None)

    # Launch cmd_start in background thread
    t = threading.Thread(target=naa.cmd_start, args=([],), daemon=True)
    t.start()

    # Wait for server start probe then flip model_loaded=True
    time.sleep(0.3)
    _state["model_loaded"] = True

    # Wait for the public smoke test to be recorded
    deadline = time.time() + 20
    while time.time() < deadline:
        if any(c[0] == "public_health_ok" for c in calls):
            break
        time.sleep(0.1)

    seq = [c[0] for c in calls]
    assert "start_server" in seq, "server never started"
    assert "start_keepalive" in seq, "keepalive never started"
    assert "start_tunnel" in seq, "tunnel never started"

    tunnel_idx = seq.index("start_tunnel")
    true_health_idxs = [i for i, c in enumerate(calls) if c[0] == "is_server_healthy" and c[1] is True]
    assert true_health_idxs, "model was never reported loaded"
    first_true = true_health_idxs[0]

    # Verify tunnel was started AFTER model was confirmed loaded
    assert tunnel_idx > first_true, "Tunnel started before model was loaded"

    ph_idx = seq.index("public_health_ok")
    assert ph_idx > tunnel_idx, "Smoke test ran before tunnel was up"

    url_saves = [c for c in calls if c[0] == "save_state" and "public_url" in c[1]]
    assert url_saves, "Public URL was never saved to state"

    # Test Watchdog recovery
    calls_before = list(calls)
    naa._server_proc._alive = False

    # Simulate watchdog tick
    deadline = time.time() + 35
    while time.time() < deadline:
        new_starts = [c for c in calls[len(calls_before):] if c[0] == "start_server"]
        if new_starts:
            break
        time.sleep(0.2)

    new_calls = calls[len(calls_before):]
    assert any(c[0] == "start_server" for c in new_calls), "watchdog did not restart server"
    assert any(c[0] == "terminate" for c in new_calls), "watchdog did not terminate dead proc"
