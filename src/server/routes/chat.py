"""
Chat Completions Endpoint (/v1/chat/completions) for NAA
"""

import time
import uuid
import json
import logging
import asyncio
from typing import Dict, Any
from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from src.config import settings
from src.server.schemas import ChatCompletionRequest
from src.server.auth import get_api_key

logger = logging.getLogger("naa-chat")
router = APIRouter(prefix="/v1", tags=["Chat"])

def ensure_engine_ready(engine):
    if not engine.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is loading. Please retry in a few moments." if engine.model_loading else "Model is not loaded."
        )

@router.post("/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    kd: Dict[str, Any] = Depends(get_api_key),
):
    engine = request.app.state.engine
    km = request.app.state.key_manager
    ensure_engine_ready(engine)

    custom_stops = [body.stop] if isinstance(body.stop, str) else (body.stop or [])
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_ts = int(time.time())
    model_name = body.model if body.model and body.model != "NAA-AI-Model" else getattr(engine, "model_name", settings.model_name)

    # Streaming Execution
    if body.stream:
        cancel_event = asyncio.Event()

        async def stream_generator():
            tok_count = 0
            last_heartbeat = time.time()

            # Initial role announcement chunk
            yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created_ts, 'model': model_name, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"

            try:
                stream_iter = engine.generate_chat_stream(
                    messages=body.messages,
                    max_tokens=body.max_tokens or settings.default_tokens,
                    temperature=body.temperature if body.temperature is not None else settings.default_temperature,
                    top_p=body.top_p if body.top_p is not None else settings.default_top_p,
                    min_p=body.min_p if body.min_p is not None else settings.default_min_p,
                    top_k=body.top_k or settings.default_top_k,
                    repeat_penalty=body.repeat_penalty or settings.default_repetition_penalty,
                    custom_stops=custom_stops,
                    cancel_event=cancel_event,
                )

                async for chunk in stream_iter:
                    if await request.is_disconnected():
                        logger.info(f"Client disconnected: {request_id}")
                        cancel_event.set()
                        break

                    now = time.time()
                    if now - last_heartbeat >= settings.sse_heartbeat_secs:
                        yield ": heartbeat\n\n"
                        last_heartbeat = now

                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        finish_reason = choices[0].get("finish_reason")

                        if content:
                            tok_count += 1

                        chunk_payload = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": delta,
                                "finish_reason": finish_reason,
                            }]
                        }
                        yield f"data: {json.dumps(chunk_payload)}\n\n"

                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Streaming error in chat: {e}")
                try:
                    yield "data: [DONE]\n\n"
                except Exception:
                    pass
            finally:
                km.record_usage(kd["key"], tok_count)

        resp = StreamingResponse(stream_generator(), media_type="text/event-stream")
        resp.headers["Connection"] = "close"
        resp.headers["X-Accel-Buffering"] = "no"
        return resp

    # Non-Streaming Execution
    raw_res = await engine.generate_chat_non_streaming(
        messages=body.messages,
        max_tokens=body.max_tokens or settings.default_tokens,
        temperature=body.temperature if body.temperature is not None else settings.default_temperature,
        top_p=body.top_p if body.top_p is not None else settings.default_top_p,
        min_p=body.min_p if body.min_p is not None else settings.default_min_p,
        top_k=body.top_k or settings.default_top_k,
        repeat_penalty=body.repeat_penalty or settings.default_repetition_penalty,
        custom_stops=custom_stops,
    )

    choices = raw_res.get("choices", [{}])
    message = choices[0].get("message", {"role": "assistant", "content": ""})
    usage = raw_res.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    total_tokens = usage.get("total_tokens", usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))

    km.record_usage(kd["key"], total_tokens)

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": choices[0].get("finish_reason", "stop"),
        }],
        "usage": usage,
    }
