"""Readiness checks must distinguish a live loader from a ready model."""

import json

from src.supervisor import watchdog


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_local_health_distinguishes_loading_from_ready(monkeypatch):
    state = {
        "model_loaded": False,
        "model_loading": True,
        "load_stage": "loading",
        "load_error": None,
    }
    monkeypatch.setattr(
        watchdog.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(state),
    )

    assert watchdog.is_server_healthy(8000) is False
    assert watchdog.is_server_loading(8000) is True

    state.update(model_loaded=True, model_loading=False, load_stage="ready")
    assert watchdog.is_server_healthy(8000) is True
    assert watchdog.is_server_loading(8000) is False


def test_public_health_requires_loaded_model(monkeypatch):
    state = {"model_loaded": False, "model_loading": True}
    monkeypatch.setattr(
        watchdog.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(state),
    )
    monkeypatch.setattr(watchdog.time, "sleep", lambda *_: None)

    assert watchdog.public_health_ok("https://example.test", timeout=0.01) is False

    state["model_loaded"] = True
    assert watchdog.public_health_ok("https://example.test", timeout=0.01) is True
