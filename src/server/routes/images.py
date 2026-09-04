"""OpenAI-compatible image endpoints backed by the Wan 2.2 visual engine.

Image clients (e.g. clauoff) POST to ``/v1/images/generations`` and expect an
immediate ``{ data: [{ b64_json }] }`` payload — no separate text-to-image
checkpoint download needed: the visual backend renders a single still frame.
"""

from __future__ import annotations

import base64
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool

from src.core.video_engine import get_video_engine, resolve_video_model_id
from src.server.auth import get_api_key
from src.server.schemas import ImageGenerationRequest

router = APIRouter(prefix="/v1", tags=["Images"])


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


def _engine(request: Request):
    engine = getattr(request.app.state, "video_engine", None)
    if engine is None:
        engine = get_video_engine()
        request.app.state.video_engine = engine
    return engine


@router.post("/images/generations")
async def create_image_generation(
    body: ImageGenerationRequest,
    request: Request,
    kd: Dict[str, Any] = Depends(get_api_key),
):
    """Render a still image synchronously. Returns OpenAI ``{ data }`` shape."""
    engine = _engine(request)
    km = request.app.state.key_manager
    width, height = _parse_size(body.size)
    n = max(1, min(4, body.n or 1))
    data: List[Dict[str, Any]] = []
    # Only honor model values that resolve to a real checkpoint (alias or
    # owner/repo). Image clients often send placeholder ids like "flux-1" —
    # those must NOT trigger a bogus HuggingFace download; fall back to the
    # engine default instead.
    resolved = resolve_video_model_id(body.model) if body.model else None
    model = resolved if resolved and "/" in resolved else None
    guidance = body.guidance_scale if body.guidance_scale is not None else 5.0
    for _ in range(n):
        # Diffusion render blocks for seconds/minutes — keep it off the loop.
        png = await run_in_threadpool(
            engine.generate_still,
            body.prompt,
            model,
            body.negative_prompt,
            width,
            height,
            body.num_inference_steps,
            guidance,
            body.seed,
        )
        data.append(
            {
                "b64_json": base64.b64encode(png).decode(),
                "revised_prompt": body.prompt,
            }
        )
    km.record_usage(kd["key"], 1)
    return {"created": int(time.time()), "model": model or engine.model_id, "data": data}
