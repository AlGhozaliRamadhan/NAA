"""
Admin Key Management and Server Statistics Endpoints for NAA
"""

from datetime import datetime, timezone
from typing import Dict, Any
from fastapi import APIRouter, Request, Depends
from src.server.schemas import CreateKeyRequest, RevokeKeyRequest
from src.server.auth import require_admin
from src.config import settings

router = APIRouter(prefix="/v1/admin", tags=["Admin"])

@router.post("/keys/create")
async def admin_create_key(
    body: CreateKeyRequest,
    request: Request,
    adm: Dict[str, Any] = Depends(require_admin)
):
    km = request.app.state.key_manager
    record = km.create_key(name=body.name, role=body.role, rate_limit_rpm=body.rpm)
    return {"success": True, "key": record}

@router.get("/keys/list")
async def admin_list_keys(
    request: Request,
    adm: Dict[str, Any] = Depends(require_admin)
):
    km = request.app.state.key_manager
    keys = km.list_keys(include_key_value=True)
    return {"keys": keys, "count": len(keys)}

@router.post("/keys/revoke")
async def admin_revoke_key(
    body: RevokeKeyRequest,
    request: Request,
    adm: Dict[str, Any] = Depends(require_admin)
):
    km = request.app.state.key_manager
    success = km.revoke_key(body.key)
    return {"success": success}

@router.delete("/keys/{key}")
async def admin_delete_key(
    key: str,
    request: Request,
    adm: Dict[str, Any] = Depends(require_admin)
):
    km = request.app.state.key_manager
    success = km.delete_key(key)
    return {"success": success}

@router.get("/stats")
async def admin_stats(
    request: Request,
    adm: Dict[str, Any] = Depends(require_admin)
):
    engine = request.app.state.engine
    km = request.app.state.key_manager
    start_time = request.app.state.start_time

    uptime = (datetime.now(timezone.utc) - start_time).total_seconds()
    keys = km.list_keys(include_key_value=False)
    total_requests = sum(k.get("total_requests", k.get("reqs", 0)) for k in keys)
    total_tokens = sum(k.get("total_tokens", k.get("tokens", 0)) for k in keys)
    active_model = getattr(engine, "model_name", settings.model_name)

    return {
        "uptime": uptime,
        "uptime_seconds": uptime,
        "model_loaded": engine.model_loaded,
        "model": active_model,
        "model_path": str(engine.model_path),
        "total_keys": len(keys),
        "active_keys": sum(1 for k in keys if k.get("active")),
        "total_requests": total_requests,
        "total_tokens": total_tokens,
    }
