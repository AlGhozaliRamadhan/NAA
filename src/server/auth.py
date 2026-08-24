"""
Authentication & Authorization Dependencies for NAA
"""

from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from src.core.key_manager import APIKeyManager
from src.config import settings

security = HTTPBearer(auto_error=False)

def get_key_manager(request: Request) -> APIKeyManager:
    return request.app.state.key_manager

async def get_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    km: APIKeyManager = Depends(get_key_manager),
) -> Dict[str, Any]:
    token = credentials.credentials if credentials else None
    if not token:
        token = request.headers.get("x-api-key")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide Authorization: Bearer <key> or x-api-key header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    record = km.validate_key(token)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key.",
        )

    if not km.check_rate_limit(token, default_rpm=settings.default_rpm):
        limit = record.get("rate_limit_rpm", record.get("rpm", settings.default_rpm))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {limit} requests per minute.",
        )

    return record

async def require_admin(key_data: Dict[str, Any] = Depends(get_api_key)) -> Dict[str, Any]:
    if key_data.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return key_data
