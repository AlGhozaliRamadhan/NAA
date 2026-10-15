"""Loader failures must reach the local supervisor but never the public tunnel."""

from fastapi.testclient import TestClient

from src.server.app import create_app


SECRET = "failed to load /kaggle/working/models/HauhauCS/Gemma4-12B.gguf: unknown model architecture"


def _client(app_instance, peer):
    return TestClient(app_instance, raise_server_exceptions=False, client=peer)


def test_loopback_supervisor_sees_the_loader_error(app_instance, mock_engine):
    mock_engine.model_loaded = False
    mock_engine.model_loading = False
    mock_engine.load_error = SECRET

    resp = _client(app_instance, ("127.0.0.1", 51000)).get("/health")

    assert resp.status_code == 200
    # Without this the CLI cannot tell a failed load from a slow one and only
    # ever reports "timed out after 600s" while the VRAM bar falls back to the
    # bare CUDA context.
    assert resp.json()["load_error"] == SECRET


def test_remote_caller_never_receives_the_loader_error(app_instance, mock_engine):
    mock_engine.model_loaded = False
    mock_engine.model_loading = False
    mock_engine.load_error = SECRET

    resp = _client(app_instance, ("198.51.100.7", 443)).get("/health")

    assert resp.status_code == 200
    assert not resp.json().get("load_error")


def test_failed_load_still_reports_terminal_stage(app_instance, mock_engine):
    mock_engine.model_loaded = False
    mock_engine.model_loading = False
    mock_engine.load_stage = "error"
    mock_engine.load_error = SECRET

    resp = _client(app_instance, ("127.0.0.1", 51000)).get("/health")
    body = resp.json()

    assert body["load_stage"] == "error"
    assert body["model_ready"] is False