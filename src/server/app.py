"""
FastAPI Application Factory & Lifecycle for NAA (Notebooks AI API)
"""

import logging
import threading
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn

from src.config import settings
from src.core.engine import InferenceEngine
from src.core.key_manager import APIKeyManager
from src.server.routes.health import router as health_router
from src.server.routes.models import router as models_router
from src.server.routes.chat import router as chat_router
from src.server.routes.anthropic import router as anthropic_router
from src.server.routes.completions import router as completions_router
from src.server.routes.admin import router as admin_router
from src.server.routes.videos import router as videos_router

logger = logging.getLogger("naa-app")

def create_app(
    engine: InferenceEngine = None,
    key_manager: APIKeyManager = None,
    auto_load_model: bool = True,
    video_engine=None,
    llm_enabled: bool = None,
    visual_enabled: bool = None,
) -> FastAPI:
    if llm_enabled is None:
        llm_enabled = settings.llm_enabled
    if visual_enabled is None:
        visual_enabled = settings.visual_enabled
    # Visual-only: LLM disabled (e.g. `start --visual` without `--llm`).
    visual_only = not llm_enabled and visual_enabled
    if not llm_enabled and not visual_enabled:
        raise ValueError("Nothing to serve: both llm_enabled and visual_enabled are False.")
    if engine is None:
        engine = InferenceEngine(
            model_path=settings.model_path,
            model_name=settings.model_name,
            quant_mode=settings.quant_mode,
            preset=settings.preset,
            system_prompt=settings.system_prompt,
            n_ctx=settings.max_context,
            n_gpu_layers=settings.n_gpu_layers,
            flash_attn=settings.flash_attn,
            cache_type_k=settings.cache_type_k,
            cache_type_v=settings.cache_type_v,
            trust_remote_code=settings.trust_remote_code
        )

    if key_manager is None:
        key_manager = APIKeyManager(
            keys_file=settings.keys_file,
            admin_key=settings.admin_key
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if visual_only:
            from src.core.video_engine import get_video_engine

            if app.state.video_engine is None:
                app.state.video_engine = get_video_engine(
                    model_id=settings.video_model_id,
                    lora_url=settings.video_lora_url,
                    lora_strength=settings.video_lora_strength,
                    steps=settings.video_steps,
                    profile=settings.video_profile,
                    attention=settings.video_attention,
                    motion_bucket_id=settings.video_motion_bucket_id,
                )
            logger.info("Visual-only mode: LLM engine disabled; Wan 2.2 video backend active.")
        elif auto_load_model and Path(settings.model_path).exists():
            threading.Thread(target=engine.load, daemon=True).start()
        else:
            logger.warning(f"Model path {settings.model_path} does not exist. Awaiting download or configuration.")
        yield

    app = FastAPI(
        title="NAA (Notebooks AI API)",
        description="Production OpenAI/Anthropic-compatible Universal REST API for agentic local models",
        version="2.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.state.engine = engine
    app.state.key_manager = key_manager
    app.state.video_engine = video_engine
    app.state.llm_enabled = llm_enabled
    app.state.visual_enabled = visual_enabled
    app.state.visual_only = visual_only
    app.state.start_time = datetime.now(timezone.utc)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        error_type = "invalid_request_error"
        if exc.status_code == 401:
            error_type = "authentication_error"
        elif exc.status_code == 403:
            error_type = "permission_error"
        elif exc.status_code == 429:
            error_type = "rate_limit_error"
        elif exc.status_code >= 500:
            error_type = "server_error"

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": exc.detail,
                    "type": error_type,
                    "param": None,
                    "code": exc.status_code,
                }
            },
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        errors = exc.errors()
        msg = "; ".join(f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in errors)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "message": f"Invalid request body: {msg}",
                    "type": "invalid_request_error",
                    "param": errors[0]["loc"][-1] if errors and errors[0]["loc"] else None,
                    "code": 422,
                }
            },
        )

    app.include_router(health_router)
    if llm_enabled:
        app.include_router(models_router)
        app.include_router(chat_router)
        app.include_router(anthropic_router)
        app.include_router(completions_router)
    app.include_router(admin_router)
    if visual_enabled:
        app.include_router(videos_router)

    return app

def run():
    app = create_app()
    logger.info(f"Starting NAA API Server on port {settings.port}")
    logger.info(f"Admin Key: {settings.admin_key[:8]}...{settings.admin_key[-4:]}")
    logger.info(f"Model Path: {settings.model_path}")
    uvicorn.run(app, host="0.0.0.0", port=settings.port, log_level="info")

if __name__ == "__main__":
    run()
