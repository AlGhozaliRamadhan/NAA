"""
Tests for Wan 2.2 video generation (Text-to-Video + Image-to-Video) in NAA.
"""

import base64
import time

from fastapi.testclient import TestClient

from src.core.video_engine import (
    DEFAULT_LORA_STRENGTH,
    DEFAULT_MOTION_BUCKET_ID,
    DEFAULT_VIDEO_ATTENTION,
    DEFAULT_VIDEO_LORA_URL,
    DEFAULT_VIDEO_MODEL_ID,
    DEFAULT_VIDEO_PROFILE,
    DEFAULT_VIDEO_STEPS,
    VideoEngine,
    decode_image_payload,
    parse_lora_repo,
    resolve_video_model_id,
)
from src.server.app import create_app


def _wait_for_job(client: TestClient, headers: dict, job_id: str, timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        resp = client.get(f"/v1/videos/{job_id}", headers=headers)
        assert resp.status_code == 200
        last = resp.json()
        if last["status"] in ("succeeded", "failed"):
            return last
        time.sleep(0.2)
    raise AssertionError(f"video job {job_id} did not finish: {last}")


def test_backend_testing_defaults():
    """Goal-mandated LoRA + generation parameters must stay exact."""
    assert DEFAULT_VIDEO_LORA_URL == "https://huggingface.co/lkzd7/WAN2.2_LoraSet_NSFW"
    assert DEFAULT_LORA_STRENGTH == 0.8
    assert DEFAULT_VIDEO_STEPS == 4
    assert DEFAULT_VIDEO_PROFILE == 2
    assert DEFAULT_VIDEO_ATTENTION == "sage"
    assert DEFAULT_MOTION_BUCKET_ID == 150


def test_resolve_video_model_aliases():
    assert resolve_video_model_id("wan2.2") == DEFAULT_VIDEO_MODEL_ID
    assert resolve_video_model_id("Wan2.2-TI2V-5B") == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert resolve_video_model_id(None) == DEFAULT_VIDEO_MODEL_ID
    assert resolve_video_model_id("Wan-AI/Wan2.2-T2V-A14B-Diffusers") == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"


def test_parse_lora_repo():
    assert parse_lora_repo("https://huggingface.co/lkzd7/WAN2.2_LoraSet_NSFW") == "lkzd7/WAN2.2_LoraSet_NSFW"
    assert parse_lora_repo(None) is None


def test_video_config_endpoint(client: TestClient, user_headers, tmp_path, monkeypatch):
    import src.core.video_engine as ve

    monkeypatch.setattr(ve, "get_video_engine", lambda **kw: VideoEngine(output_dir=str(tmp_path)))
    app = create_app(
        engine=client.app.state.engine,
        key_manager=client.app.state.key_manager,
        auto_load_model=False,
        video_engine=VideoEngine(output_dir=str(tmp_path)),
    )
    c2 = TestClient(app, raise_server_exceptions=False)
    resp = c2.get("/v1/videos/config", headers=user_headers)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["lora_url"] == "https://huggingface.co/lkzd7/WAN2.2_LoraSet_NSFW"
    assert data["lora_strength"] == 0.8
    assert data["steps"] == 4
    assert data["profile"] == 2
    assert data["attention"] == "sage"
    assert data["motion_bucket_id"] == 150


def test_text_to_video_flow(client: TestClient, user_headers, tmp_path):
    app = create_app(
        engine=client.app.state.engine,
        key_manager=client.app.state.key_manager,
        auto_load_model=False,
        video_engine=VideoEngine(output_dir=str(tmp_path)),
    )
    c2 = TestClient(app, raise_server_exceptions=False)
    resp = c2.post(
        "/v1/videos/generations",
        headers=user_headers,
        json={"prompt": "Two cats boxing on a spotlighted stage"},
    )
    assert resp.status_code == 202, resp.text
    job = resp.json()
    assert job["mode"] == "text-to-video"
    assert job["status"] in ("queued", "running", "succeeded")
    assert job["params"]["lora_strength"] == 0.8
    assert job["params"]["num_inference_steps"] == 4

    done = _wait_for_job(c2, user_headers, job["id"])
    assert done["status"] == "succeeded"
    assert done["result_url"].endswith("/download")

    dl = c2.get(f"/v1/videos/{job['id']}/download", headers=user_headers)
    assert dl.status_code == 200
    assert len(dl.content) > 0


def test_image_to_video_flow(client: TestClient, user_headers, tmp_path):
    app = create_app(
        engine=client.app.state.engine,
        key_manager=client.app.state.key_manager,
        auto_load_model=False,
        video_engine=VideoEngine(output_dir=str(tmp_path)),
    )
    c2 = TestClient(app, raise_server_exceptions=False)
    fake_image = base64.b64encode(b"fakepng-bytes").decode()
    resp = c2.post(
        "/v1/videos/generations",
        headers=user_headers,
        json={"prompt": "A cat surfing at the beach", "image": fake_image},
    )
    assert resp.status_code == 202, resp.text
    job = resp.json()
    assert job["mode"] == "image-to-video"

    done = _wait_for_job(c2, user_headers, job["id"])
    assert done["status"] == "succeeded"


def test_video_auth_required(client: TestClient):
    resp = client.post("/v1/videos/generations", json={"prompt": "hello"})
    assert resp.status_code in (401, 403)


def test_video_job_not_found(client: TestClient, user_headers, tmp_path):
    app = create_app(
        engine=client.app.state.engine,
        key_manager=client.app.state.key_manager,
        auto_load_model=False,
        video_engine=VideoEngine(output_dir=str(tmp_path)),
    )
    c2 = TestClient(app, raise_server_exceptions=False)
    resp = c2.get("/v1/videos/does-not-exist", headers=user_headers)
    assert resp.status_code == 404


def test_decode_image_payload_helpers():
    assert decode_image_payload(None) is None
    assert decode_image_payload("https://example.com/a.jpg") is None
    raw = base64.b64encode(b"abc123").decode()
    assert decode_image_payload(raw) == b"abc123"
