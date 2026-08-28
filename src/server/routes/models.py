"""
Model Listing & Embedding Routes for NAA
"""

from typing import Dict, Any
from fastapi import APIRouter, Request, Depends, HTTPException, status
from src.config import settings
from src.server.schemas import EmbeddingRequest
from src.server.auth import get_api_key

router = APIRouter(prefix="/v1", tags=["Models"])

@router.get("/models")
async def list_models(request: Request, kd: Dict[str, Any] = Depends(get_api_key)):
    engine = getattr(request.app.state, "engine", None)
    active_model = getattr(engine, "model_name", settings.model_name) if engine else settings.model_name
    models = [
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
    # Claude Code's optional gateway discovery intentionally filters out IDs
    # that contain neither "claude" nor "anthropic".  This stable alias maps to
    # the one loaded NAA model without changing OpenAI-compatible discovery.
    if "claude" not in active_model.lower() and "anthropic" not in active_model.lower():
        models.append(
            {
                "id": "claude-naa",
                "object": "model",
                "created": 1700000000,
                "owned_by": "naa",
                "permission": [],
                "root": active_model,
                "parent": active_model,
                "display_name": f"NAA ({active_model})",
            }
        )
    return {
        "object": "list",
        "data": models,
    }

@router.post("/embeddings")
async def create_embeddings(body: EmbeddingRequest, kd: Dict[str, Any] = Depends(get_api_key)):
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Embeddings are not supported for this generative CausalLM model.",
    )
