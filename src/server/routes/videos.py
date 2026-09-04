"""Wan 2.2 Video Generation endpoints (Text-to-Video + Image-to-Video)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse

from src.core.video_engine import get_video_engine, resolve_video_model_id
from src.server.auth import get_api_key
from src.server.schemas import VideoGenerationRequest

router = APIRouter(prefix="/v1", tags=["Videos"])


def _engine(request: Request):
    engine = getattr(request.app.state, "video_engine", None)
    if engine is None:
        engine = get_video_engine()
        request.app.state.video_engine = engine
    return engine


@router.get("/videos/config")
async def video_config(request: Request, kd: Dict[str, Any] = Depends(get_api_key)):
    """Show the active Wan 2.2 backend defaults (LoRA, steps, profile, ...)."""
    return {"object": "video.config", "data": _engine(request).default_config()}


@router.get("/videos/models")
async def video_models(request: Request, kd: Dict[str, Any] = Depends(get_api_key)):
    engine = _engine(request)
    ids = [
        engine.model_id,
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    ]
    seen: List[Dict[str, Any]] = []
    for mid in ids:
        if mid and all(m["id"] != mid for m in seen):
            seen.append(
                {
                    "id": mid,
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "naa-video",
                    "capabilities": ["text-to-video", "image-to-video"],
                }
            )
    return {"object": "list", "data": seen}


@router.post("/videos/generations", status_code=status.HTTP_202_ACCEPTED)
async def create_video_generation(
    body: VideoGenerationRequest,
    request: Request,
    kd: Dict[str, Any] = Depends(get_api_key),
):
    """Submit a T2V (no ``image``) or I2V (with ``image``) job. Returns 202 + job."""
    engine = _engine(request)
    km = request.app.state.key_manager
    job = engine.submit(
        prompt=body.prompt,
        image=body.image,
        model=resolve_video_model_id(body.model) if body.model else None,
        negative_prompt=body.negative_prompt,
        num_frames=body.num_frames or 81,
        height=body.height or 704,
        width=body.width or 1280,
        num_inference_steps=body.num_inference_steps,
        guidance_scale=body.guidance_scale if body.guidance_scale is not None else 5.0,
        fps=body.fps or 24,
        seed=body.seed,
        lora_url=body.lora_url,
        lora_strength=body.lora_strength,
        profile=body.profile,
        attention=body.attention,
        motion_bucket_id=body.motion_bucket_id,
    )
    km.record_usage(kd["key"], 1)
    return job.to_dict()


@router.get("/videos/{job_id}")
async def get_video_job(job_id: str, request: Request, kd: Dict[str, Any] = Depends(get_api_key)):
    job = _engine(request).get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Video job '{job_id}' not found.")
    return job.to_dict()


@router.get("/videos")
async def list_video_jobs(request: Request, kd: Dict[str, Any] = Depends(get_api_key)):
    jobs = _engine(request).list_jobs()
    return {"object": "list", "data": [j.to_dict() for j in jobs]}


@router.get("/videos/{job_id}/download")
async def download_video(job_id: str, request: Request, kd: Dict[str, Any] = Depends(get_api_key)):
    job = _engine(request).get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Video job '{job_id}' not found.")
    if job.status != "succeeded" or not job.output_path or not Path(job.output_path).is_file():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Video job '{job_id}' is not ready (status={job.status}).",
        )
    return FileResponse(job.output_path, media_type="video/mp4", filename=Path(job.output_path).name)


# OpenAI-style alias: POST /v1/videos mirrors /v1/videos/generations.
@router.post("/videos", status_code=status.HTTP_202_ACCEPTED)
async def create_video_alias(
    body: VideoGenerationRequest,
    request: Request,
    kd: Dict[str, Any] = Depends(get_api_key),
):
    return await create_video_generation(body, request, kd)
