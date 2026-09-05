"""Runtime media-capability discovery for the Cogito image backend.

Capabilities come from the live runner — its ``__call__`` signature and
declared adapter attributes — never from model ids, filenames,
quantization (FP8), or brand strings.
"""

import base64

from fastapi.testclient import TestClient

from src.core.image_engine import (
    IMAGE_MODEL_ALIASES,
    ImageCapabilities,
    ImageEngine,
    _mock_png,
    discover_image_capabilities,
)
from src.server.app import create_app


def _visual_client(client: TestClient, tmp_path, image_engine=None) -> TestClient:
    app = create_app(
        engine=client.app.state.engine,
        key_manager=client.app.state.key_manager,
        auto_load_model=False,
        image_engine=image_engine or ImageEngine(output_dir=str(tmp_path)),
        llm_enabled=False,
        visual_enabled=True,
    )
    return TestClient(app, raise_server_exceptions=False)


def _b64_ref(prompt: str = "ref") -> dict:
    return {
        "mime_type": "image/png",
        "b64_json": base64.b64encode(_mock_png(prompt)).decode(),
    }


# -- unit: discovery -------------------------------------------------------


def test_text_only_runner_reports_no_edit_or_video():
    class TextOnly:
        def __call__(self, prompt, width=1024):
            return b"png"

    caps = discover_image_capabilities(TextOnly())
    assert caps.image_generation is True
    assert caps.image_edit is False
    assert caps.video_generation is False
    assert caps.max_reference_images == 0


def test_edit_workflow_reports_image_edit_and_limit():
    class EditWorkflow:
        image_input_param = "image_prompt"
        max_reference_images = 4

        def __call__(self, prompt, image_prompt=None):
            return b"png"

    caps = discover_image_capabilities(EditWorkflow())
    assert caps.image_generation is True
    assert caps.image_edit is True
    assert caps.max_reference_images == 4


def test_no_video_without_explicit_video_pipeline():
    class SuspiciousIdRunner:
        # A model id/brand can never grant video: only video_available=True.
        def __call__(self, prompt, video=None):
            return b"png"

    assert discover_image_capabilities(SuspiciousIdRunner()).video_generation is False
    assert (
        discover_image_capabilities(
            SuspiciousIdRunner(), video_available=True
        ).video_generation
        is True
    )


def test_fp8_style_id_does_not_grant_capabilities():
    engine = ImageEngine(model_id="black-forest-labs/FLUX.1-dev-FP8")
    caps = engine.refresh_capabilities()
    # Mock-mode ImageEngine.__call__ exposes prompt only: generation yes,
    # edit/video no — the "FLUX"/"FP8" substrings grant nothing.
    assert caps.image_generation is True
    assert caps.image_edit is False
    assert caps.video_generation is False
    assert caps.max_reference_images == 0


# -- integration: routes ----------------------------------------------------


def test_models_reports_cogito_shape(client, user_headers, tmp_path):
    c2 = _visual_client(client, tmp_path)
    resp = c2.get("/v1/models", headers=user_headers)
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert len(data) == 1
    entry = data[0]
    assert entry["id"] == "RunDiffusion/Juggernaut-XL-v9"
    assert entry["label"] == "Juggernaut-XL-v9"
    assert entry["capabilities"] == {
        "image_generation": True,
        "image_edit": False,
        "video_generation": False,
        "max_reference_images": 0,
    }


def test_hot_swap_recalculates_capabilities(tmp_path):
    engine = ImageEngine(output_dir=str(tmp_path))
    before = engine.refresh_capabilities()
    assert before.image_edit is False

    class EditWorkflow:
        image_input_param = "image"
        max_reference_images = 2

        def __call__(self, prompt, image=None):
            return b"png"

    after = engine.set_pipeline(EditWorkflow())
    assert after.image_edit is True
    assert after.max_reference_images == 2
    assert isinstance(engine.capabilities, ImageCapabilities)


def test_unsupported_edit_returns_400(client, user_headers, tmp_path):
    c2 = _visual_client(client, tmp_path)
    resp = c2.post(
        "/v1/images/generations",
        headers=user_headers,
        json={
            "prompt": "make the cat wear a hat",
            "model": "RunDiffusion/Juggernaut-XL-v9",
            "generation_mode": "edit",
            "reference_images": [_b64_ref()],
        },
    )
    assert resp.status_code == 400, resp.text
    assert "image_edit=false" in resp.json()["error"]["message"]


def test_edit_roundtrip_through_edit_workflow(client, user_headers, tmp_path):
    class EditWorkflow:
        image_input_param = "image"
        max_reference_images = 2

        def __call__(self, prompt, image=None, **kwargs):
            assert image is not None  # reference mapped to the runner input
            from PIL import Image
            import io

            assert isinstance(image, Image.Image)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            return type("R", (), {"images": [Image.open(io.BytesIO(buf.getvalue()))]})()

    engine = ImageEngine(output_dir=str(tmp_path))
    engine.set_pipeline(EditWorkflow())
    c2 = _visual_client(client, tmp_path, image_engine=engine)
    resp = c2.post(
        "/v1/images/generations",
        headers=user_headers,
        json={
            "prompt": "make the cat wear a hat",
            "generation_mode": "edit",
            "reference_images": [_b64_ref("a cat")],
        },
    )
    assert resp.status_code == 200, resp.text
    item = resp.json()["data"][0]
    raw = base64.b64decode(item["b64_json"])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert item["revised_prompt"] == "make the cat wear a hat"


def test_excess_reference_images_rejected(client, user_headers, tmp_path):
    class SingleRefWorkflow:
        image_input_param = "image"
        max_reference_images = 1

        def __call__(self, prompt, image=None):
            return b"png"

    engine = ImageEngine(output_dir=str(tmp_path))
    engine.set_pipeline(SingleRefWorkflow())
    c2 = _visual_client(client, tmp_path, image_engine=engine)
    resp = c2.post(
        "/v1/images/generations",
        headers=user_headers,
        json={
            "prompt": "edit",
            "generation_mode": "edit",
            "reference_images": [_b64_ref("a"), _b64_ref("b")],
        },
    )
    assert resp.status_code == 400, resp.text
    assert "at most 1" in resp.json()["error"]["message"]


def test_aliases_still_resolve():
    assert (
        IMAGE_MODEL_ALIASES["flux.1"]
        == "black-forest-labs/FLUX.1-dev-FP8"
    )
