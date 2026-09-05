"""OpenAI-compatible image endpoint backed by SDXL/Juggernaut-XL.

Image clients (e.g. clauoff) POST to ``/v1/images/generations`` and expect an
immediate ``{ data: [{ b64_json }] }`` payload. Wan 2.2 used to back this as
a single-frame render but is too heavy for stills on a T4; the route now
goes straight to the dedicated ``ImageEngine`` (SDXL) which fits and runs
fast on consumer GPUs.
"""

from __future__ import annotations

import base64
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from src.core.image_engine import (
    ImageEditUnsupportedError,
    decode_reference_image,
    get_image_engine,
    is_flux_model,
    resolve_image_model_id,
)
from src.server.auth import get_api_key
from src.server.schemas import ImageGenerationRequest

router = APIRouter(prefix="/v1", tags=["Images"])

# Per-request engine cache keyed by resolved model id, so the default
# Juggernaut pipeline stays loaded while FLUX requests use their own.
_flux_engines: dict = {}


def _parse_size(size: Optional[str]) -> Tuple[int, int]:
    if not size:
        return 1024, 1024
    try:
        w_raw, h_raw = size.lower().replace(" ", "").split("x", 1)
        w, h = int(w_raw), int(h_raw)
        return (
            max(256, min(2048, w)),
            max(256, min(1536, h)),
        )
    except Exception:
        return 1024, 1024


def _engine(request: Request, model: str | None = None):
    # Explicit per-request model ("flux", "flux.1", full repo id, ...) selects
    # that backend via a cached side engine; otherwise fall back to the
    # server-configured default (Juggernaut unless --image-model was passed).
    if model:
        from src.config import settings as _settings
        from src.core.image_engine import ImageEngine

        resolved = resolve_image_model_id(model)
        # An unknown id that resolves to neither backend keeps the default:
        # only honour FLUX ids and the configured Juggernaut id.
        default = getattr(request.app.state, "image_engine", None)
        default_id = getattr(default, "model_id", None) or _settings.image_model_id
        if is_flux_model(resolved) or resolved == default_id:
            engine = _flux_engines.get(resolved)
            if engine is None:
                engine = ImageEngine(model_id=resolved)
                _flux_engines[resolved] = engine
            return engine
    engine = getattr(request.app.state, "image_engine", None)
    if engine is None:
        engine = get_image_engine()
        request.app.state.image_engine = engine
    return engine


@router.post("/images/generations")
async def create_image_generation(
    body: ImageGenerationRequest,
    request: Request,
    kd: Dict[str, Any] = Depends(get_api_key),
):
    """Render a still image synchronously. Returns OpenAI ``{ data }`` shape."""
    # `model` selects the image backend per request: "flux"/"flux.1"/full
    # FLUX repo id -> FLUX.1-dev-FP8; Juggernaut id -> SDXL default.
    engine = _engine(request, body.model)
    km = request.app.state.key_manager
    width, height = _parse_size(body.size)
    n = max(1, min(4, body.n or 1))
    default_guidance = 3.5 if getattr(engine, "is_flux", False) else 5.0
    guidance = body.guidance_scale if body.guidance_scale is not None else default_guidance
    # Cogito edit shape: generation_mode="edit" + reference_images. Any
    # reference images mean an edit request; decode to bytes now so the 400
    # for excess/undecodable refs fires before rendering.
    refs: List[bytes] = []
    if body.reference_images:
        # Any reference images mean an edit request, whether or not the
        # client also set generation_mode="edit". Fail fast on the
        # discovered capabilities rather than after a render.
        caps = engine.refresh_capabilities()
        if not caps.image_edit:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Model '{engine.model_id}' does not support image editing: "
                    "/v1/models reports image_edit=false for this backend."
                ),
            )
        limit = max(0, caps.max_reference_images)
        if len(body.reference_images) > limit:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Got {len(body.reference_images)} reference image(s) but model "
                    f"'{engine.model_id}' supports at most {limit}."
                ),
            )
        try:
            refs = [decode_reference_image(r.model_dump()) for r in body.reference_images]
        except ImageEditUnsupportedError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            )
    data: List[Dict[str, Any]] = []
    for _ in range(n):
        # SDXL render blocks for seconds — keep it off the event loop.
        # Engine.generate maps refs to the runner's image input and raises
        # ImageEditUnsupportedError (→ 400) when the runner has no image path.
        try:
            png = await run_in_threadpool(
                engine.generate,
                body.prompt,
                body.negative_prompt,
                width,
                height,
                body.num_inference_steps,
                guidance,
                body.seed,
                refs or None,
            )
        except ImageEditUnsupportedError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            )
        data.append(
            {
                "b64_json": base64.b64encode(png).decode(),
                "revised_prompt": body.prompt,
            }
        )
    km.record_usage(kd["key"], 1)
    return {"created": int(time.time()), "model": engine.model_id, "data": data}
