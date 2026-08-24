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
    
    return {
        "ok": True,
        "status": "ok",
        "service": "NAA (Notebooks AI API)",
        "model_loaded": engine.model_loaded,
        "model_loading": engine.model_loading,
        "model": active_model,
        "environment": ENV["name"],
        "gpu": ENV["gpu_name"] or "CPU",
        "gpu_count": ENV["gpu_count"],
        "uptime": uptime,
        "uptime_seconds": uptime,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

@router.get("/ping")
async def ping():
    return {"pong": True, "ts": int(time.time())}
