"""
Model Listing & Embedding Routes for NAA
"""

from typing import Dict, Any, List
from fastapi import APIRouter, Request, Depends, HTTPException, status
from src.config import settings
from src.server.schemas import EmbeddingRequest
from src.server.auth import get_api_key

router = APIRouter(prefix="/v1", tags=["Models"])

# ---------------------------------------------------------------------------
# Universal model aliases
# ---------------------------------------------------------------------------
# Many clients hard-code a specific model ID or filter the /v1/models list by
# provider prefix (Claude Code requires "claude"/"anthropic"; some OpenAI tools
# require "gpt-").  By advertising all common aliases as pointing to the one
# loaded NAA model, any client works out-of-the-box without custom config.
# ---------------------------------------------------------------------------

_CLAUDE_ALIASES: List[str] = [
    "claude-naa",
    # Claude Code defaults
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    # Older Claude Code / SDK defaults
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]

_OPENAI_ALIASES: List[str] = [
    # OpenCode / Cursor / Copilot defaults
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
]


def _make_alias(alias_id: str, root: str, display: str) -> Dict[str, Any]:
    return {
        "id": alias_id,
        "object": "model",
        "created": 1700000000,
        "owned_by": "naa",
        "permission": [],
        "root": root,
        "parent": root,
        "display_name": display,
    }


@router.get("/models")
async def list_models(request: Request, kd: Dict[str, Any] = Depends(get_api_key)):
    if not getattr(request.app.state, "llm_enabled", True):
        # Visual-only: Cogito media discovery. Capabilities come from the
        # live runner (discover_image_capabilities), never from model ids —
        # an id containing "flux"/"fp8" says nothing about edit or video.
        from src.core.video_engine import get_video_engine
        from src.core.image_engine import get_image_engine

        video = getattr(request.app.state, "video_engine", None) or get_video_engine()
        image = getattr(request.app.state, "image_engine", None) or get_image_engine()
        video_available = video is not None and not getattr(video, "is_mock", True)
        caps = image.refresh_capabilities(video_available=video_available)
        seen: List[Dict[str, Any]] = [
            {
                "id": image.model_id,
                "object": "model",
                "created": 1700000000,
                "owned_by": "naa",
                "label": image.label,
                "capabilities": {
                    "image_generation": caps.image_generation,
                    "image_edit": caps.image_edit,
                    "video_generation": caps.video_generation,
                    "max_reference_images": caps.max_reference_images,
                },
            }
        ]
        return {"object": "list", "data": seen}

    engine = getattr(request.app.state, "engine", None)
    active_model = (
        getattr(engine, "model_name", settings.model_name) if engine else settings.model_name
    )
    display_name = f"NAA ({active_model})"

    models: List[Dict[str, Any]] = [
        {
            "id": active_model,
            "object": "model",
            "created": 1700000000,
            "owned_by": "naa",
            "permission": [],
            "root": active_model,
            "parent": None,
            "display_name": active_model,
        }
    ]

    active_lower = active_model.lower()

    # Add Claude aliases unless the real model is already a Claude model.
    if "claude" not in active_lower and "anthropic" not in active_lower:
        for alias in _CLAUDE_ALIASES:
            models.append(_make_alias(alias, active_model, display_name))

    # Add OpenAI aliases unless the real model is already an OpenAI model.
    if not any(active_lower.startswith(p) for p in ("gpt-", "o1", "o3", "o4")):
        for alias in _OPENAI_ALIASES:
            models.append(_make_alias(alias, active_model, display_name))

    return {"object": "list", "data": models}


@router.post("/embeddings")
async def create_embeddings(body: EmbeddingRequest, kd: Dict[str, Any] = Depends(get_api_key)):
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Embeddings are not supported for this generative CausalLM model.",
    )
