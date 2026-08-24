"""
Health, Ping, and Dashboard Routes for NAA
"""

import time
from pathlib import Path
from datetime import datetime, timezone
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from src.config import settings, ENV

router = APIRouter(tags=["Health & Status"])

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
DASHBOARD_FILE = STATIC_DIR / "dashboard.html"

@router.get("/", include_in_schema=False)
async def root():
    if DASHBOARD_FILE.exists():
        return HTMLResponse(DASHBOARD_FILE.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>NAA - Notebooks AI API Server Online</h1>")

@router.get("/health")
async def health(request: Request):
    engine = request.app.state.engine
    start_time = request.app.state.start_time
    uptime = (datetime.now(timezone.utc) - start_time).total_seconds()
    active_model = getattr(engine, "model_name", settings.model_name)
    load_stage = getattr(engine, "load_stage", "ready" if getattr(engine, "model_loaded", False) else "loading")
    load_error = getattr(engine, "load_error", None)

    gpu_vram_used_mb = 0
    gpu_vram_total_mb = ENV.get("gpu_vram_mb", 0)
    try:
        import torch
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            gpu_vram_total_mb = int(total_b / (1024 * 1024))
            gpu_vram_used_mb = int((total_b - free_b) / (1024 * 1024))
    except Exception:
        pass

    return {
        "ok": True,
        "status": "ok",
        "service": "NAA (Notebooks AI API)",
        "model_loaded": engine.model_loaded,
        "model_loading": engine.model_loading,
        "load_stage": load_stage,
        "load_error": load_error,
        "model": active_model,
        "environment": ENV["name"],
        "gpu": ENV["gpu_name"] or "CPU",
        "gpu_count": ENV["gpu_count"],
        "gpu_vram_used_mb": gpu_vram_used_mb,
        "gpu_vram_total_mb": gpu_vram_total_mb,
        "uptime": uptime,
        "uptime_seconds": uptime,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

@router.get("/ping")
async def ping():
    return {"pong": True, "ts": int(time.time())}
