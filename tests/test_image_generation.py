"""
Tests for the OpenAI-compatible image shim (POST /v1/images/generations).

The shim renders a single still frame via the Wan 2.2 visual backend so
image clients (e.g. clauoff) work without a separate text-to-image
checkpoint download. In mock mode (no GPU/diffusers) it returns a
deterministic placeholder PNG.
"""

import base64

from fastapi.testclient import TestClient

from src.core.video_engine import VideoEngine, _mock_still_png
from src.server.app import create_app


def _visual_client(client: TestClient, tmp_path) -> TestClient:
    app = create_app(
        engine=client.app.state.engine,
        key_manager=client.app.state.key_manager,
        auto_load_model=False,
        video_engine=VideoEngine(output_dir=str(tmp_path)),
        llm_enabled=False,
        visual_enabled=True,
    )
    return TestClient(app, raise_server_exceptions=False)


def test_mock_still_png_is_valid_png():
    png = _mock_still_png("a cat", 256, 256)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    # Prompt-derived colour: different prompts differ.
    assert _mock_still_png("a cat") != _mock_still_png("a dog")


def test_images_generations_openai_shape(client: TestClient, user_headers, tmp_path):
    c2 = _visual_client(client, tmp_path)
    resp = c2.post(
        "/v1/images/generations",
        headers=user_headers,
        json={"prompt": "A cat surfing at the beach", "size": "1024x1024",
              "response_format": "b64_json"},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert "data" in payload and len(payload["data"]) == 1
    item = payload["data"][0]
    raw = base64.b64decode(item["b64_json"])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert item["revised_prompt"] == "A cat surfing at the beach"


def test_images_auth_required(client: TestClient):
    resp = client.post("/v1/images/generations", json={"prompt": "hello"})
    assert resp.status_code in (401, 403)


def test_models_lists_visual_in_visual_only(client: TestClient, user_headers, tmp_path):
    c2 = _visual_client(client, tmp_path)
    resp = c2.get("/v1/models", headers=user_headers)
    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert "Wan-AI/Wan2.2-TI2V-5B-Diffusers" in ids
