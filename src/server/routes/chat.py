"""OpenAI Chat Completions endpoint with agentic tool-call support."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from src.config import settings
from src.core.tool_calls import (
    TOOL_RECOVERY_PROMPT,
    ToolTextStreamBuffer,
    build_tool_system_prompt,
    looks_like_abandoned_tool_intent,
    normalize_openai_message,
    tool_choice_requires_call,
)
from src.server.auth import get_api_key
from src.server.schemas import ChatCompletionRequest

logger = logging.getLogger("naa-chat")
router = APIRouter(prefix="/v1", tags=["Chat"])


class EngineUnavailableError(RuntimeError):
    """A model load failed or did not finish within the allowed wait."""


def _engine_is_loading(engine: Any) -> bool:
    return bool(
        getattr(engine, "model_loading", False)
        or getattr(engine, "load_stage", "") == "loading"
    )


def ensure_engine_ready(engine: Any) -> None:
    if not engine.is_ready():
        load_error = getattr(engine, "load_error", None)
        if load_error:
            detail = f"Model failed to load: {load_error}"
        elif _engine_is_loading(engine):
            detail = "Model is loading. Please retry in a few moments."
        else:
            detail = "Model is not loaded."
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            headers={"Retry-After": "5"},
        )


async def _stream_engine_readiness(
    engine: Any,
    request: Request,
) -> AsyncGenerator[None, None]:
    """Keep an SSE response alive while a restarted model loads."""

    timeout = max(0.1, float(settings.model_wait_timeout_secs))
    heartbeat = max(0.05, float(settings.sse_heartbeat_secs))
    deadline = time.monotonic() + timeout

    while not engine.is_ready():
        if await request.is_disconnected():
            return
        load_error = getattr(engine, "load_error", None)
        if load_error:
            raise EngineUnavailableError(f"Model failed to load: {load_error}")
        if not _engine_is_loading(engine):
            raise EngineUnavailableError("Model is not loaded.")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise EngineUnavailableError(
                f"Model did not finish loading within {timeout:g} seconds."
            )
        await asyncio.sleep(min(heartbeat, remaining))
        if not engine.is_ready():
            yield None


def _message_dicts(body: ChatCompletionRequest) -> List[Dict[str, Any]]:
    return [message.model_dump(exclude_none=True) for message in body.messages]


def _tool_choice(body: ChatCompletionRequest) -> Any:
    if not body.tools:
        return "none"
    return body.tool_choice if body.tool_choice is not None else "auto"


def _generation_kwargs(
    body: ChatCompletionRequest,
    *,
    messages: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Any = None,
) -> Dict[str, Any]:
    custom_stops = [body.stop] if isinstance(body.stop, str) else (body.stop or [])
    return {
        "messages": messages if messages is not None else body.messages,
        "max_tokens": body.max_tokens or settings.default_tokens,
        "temperature": (
            body.temperature if body.temperature is not None else settings.default_temperature
        ),
        "top_p": body.top_p if body.top_p is not None else settings.default_top_p,
        "min_p": body.min_p if body.min_p is not None else settings.default_min_p,
        "top_k": body.top_k if body.top_k is not None else settings.default_top_k,
        "repeat_penalty": (
            body.repeat_penalty
            if body.repeat_penalty is not None
            else settings.default_repetition_penalty
        ),
        "custom_stops": custom_stops,
        "tools": body.tools,
        "tool_choice": _tool_choice(body) if tool_choice is None else tool_choice,
    }


def _raw_to_message(
    raw_res: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]],
) -> Tuple[Dict[str, Any], str, Dict[str, int]]:
    choices = raw_res.get("choices") or [{}]
    choice = choices[0]
    message, finish_reason = normalize_openai_message(
        choice.get("message", {"role": "assistant", "content": ""}),
        choice.get("finish_reason"),
        tools,
    )
    usage = raw_res.get("usage") or {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    return message, finish_reason, usage


async def _recover_missing_tool_call(
    engine: Any,
    body: ChatCompletionRequest,
    message: Dict[str, Any],
    finish_reason: str,
    usage: Dict[str, int],
) -> Tuple[Dict[str, Any], str, Dict[str, int]]:
    """Retry once when a model promises an action but emits no call."""

    choice = _tool_choice(body)
    should_retry = bool(body.tools) and finish_reason != "tool_calls" and (
        tool_choice_requires_call(choice)
        or looks_like_abandoned_tool_intent(message.get("content"))
    )
    if not should_retry:
        return message, finish_reason, usage

    retry_messages = _message_dicts(body)
    if message.get("content"):
        retry_messages.append({"role": "assistant", "content": message["content"]})
    recovery_prompt = (
        f"{TOOL_RECOVERY_PROMPT}\n\n"
        f"{build_tool_system_prompt(body.tools, 'required')}"
    )
    retry_messages.append({"role": "user", "content": recovery_prompt})
    logger.info("Retrying an incomplete agent turn that emitted no structured tool call")
    raw_retry = await engine.generate_chat_non_streaming(
        **_generation_kwargs(body, messages=retry_messages, tool_choice="required")
    )
    retry_message, retry_finish, retry_usage = _raw_to_message(raw_retry, body.tools)
    if retry_finish == "tool_calls":
        combined_usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0) + retry_usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0)
            + retry_usage.get("completion_tokens", 0),
        }
        combined_usage["total_tokens"] = (
            combined_usage["prompt_tokens"] + combined_usage["completion_tokens"]
        )
        return retry_message, retry_finish, combined_usage
    return message, finish_reason, usage


def _accumulate_tool_deltas(
    calls: Dict[int, Dict[str, Any]], delta_calls: List[Dict[str, Any]]
) -> None:
    for fallback_index, delta_call in enumerate(delta_calls):
        index = int(delta_call.get("index", fallback_index))
        target = calls.setdefault(
            index,
            {
                "index": index,
                "id": None,
                "type": "function",
                "function": {"name": "", "arguments": ""},
            },
        )
        if delta_call.get("id"):
            target["id"] = delta_call["id"]
        if delta_call.get("type"):
            target["type"] = delta_call["type"]
        function = delta_call.get("function") or {}
        if function.get("name"):
            target["function"]["name"] += function["name"]
        arguments = function.get("arguments")
        if arguments is not None:
            if isinstance(arguments, str):
                target["function"]["arguments"] += arguments
            else:
                target["function"]["arguments"] += json.dumps(
                    arguments, ensure_ascii=False, separators=(",", ":")
                )


async def _collect_stream(
    engine: Any,
    kwargs: Dict[str, Any],
    cancel_event: asyncio.Event,
    queue: asyncio.Queue,
) -> None:
    try:
        async for chunk in engine.generate_chat_stream(
            **kwargs,
            cancel_event=cancel_event,
        ):
            await queue.put(("chunk", chunk))
    except Exception as exc:
        await queue.put(("error", exc))
    finally:
        await queue.put(("done", None))


@router.post("/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    kd: Dict[str, Any] = Depends(get_api_key),
):
    engine = request.app.state.engine
    km = request.app.state.key_manager
    if not body.stream or not _engine_is_loading(engine):
        ensure_engine_ready(engine)

    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_ts = int(time.time())
    model_name = (
        body.model
        if body.model and body.model != "NAA-AI-Model"
        else getattr(engine, "model_name", settings.model_name)
    )

    if body.stream:
        cancel_event = asyncio.Event()

        async def stream_generator():
            completion_units = 0
            queue: asyncio.Queue = asyncio.Queue()
            producer: Optional[asyncio.Task] = None

            initial = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(initial)}\n\n"

            content_parts: List[str] = []
            emitted_text_parts: List[str] = []
            native_calls: Dict[int, Dict[str, Any]] = {}
            raw_finish: Optional[str] = None
            text_buffer = (
                ToolTextStreamBuffer(
                    hold_all=tool_choice_requires_call(_tool_choice(body))
                )
                if body.tools
                else None
            )
            try:
                async for _ in _stream_engine_readiness(engine, request):
                    yield ": model-loading\n\n"
                if not engine.is_ready():
                    return

                producer = asyncio.create_task(
                    _collect_stream(engine, _generation_kwargs(body), cancel_event, queue)
                )
                while True:
                    if await request.is_disconnected():
                        cancel_event.set()
                        break
                    try:
                        kind, payload = await asyncio.wait_for(
                            queue.get(), timeout=settings.sse_heartbeat_secs
                        )
                    except asyncio.TimeoutError:
                        yield ": heartbeat\n\n"
                        continue
                    if kind == "done":
                        break
                    if kind == "error":
                        raise payload

                    choices = payload.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if content:
                        content_parts.append(content)
                        safe_text = text_buffer.feed(content) if text_buffer else content
                        if safe_text:
                            emitted_text_parts.append(safe_text)
                            live_payload = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": safe_text},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(live_payload)}\n\n"
                    if delta.get("tool_calls"):
                        safe_prefix = text_buffer.begin_native_tool() if text_buffer else ""
                        if safe_prefix:
                            emitted_text_parts.append(safe_prefix)
                            live_payload = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": safe_prefix},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(live_payload)}\n\n"
                        _accumulate_tool_deltas(native_calls, delta["tool_calls"])
                    if choice.get("finish_reason"):
                        raw_finish = choice["finish_reason"]

                if cancel_event.is_set():
                    return

                tail = (text_buffer.finish() if text_buffer else "") or ""
                if tail:
                    emitted_text_parts.append(tail)
                    live_payload = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": tail},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(live_payload)}\n\n"

                raw_content = "".join(content_parts)
                message: Dict[str, Any] = {
                    "role": "assistant",
                    "content": raw_content,
                }
                if native_calls:
                    message["tool_calls"] = [native_calls[i] for i in sorted(native_calls)]
                message, finish_reason = normalize_openai_message(
                    message, raw_finish, body.tools
                )
                usage = {
                    "prompt_tokens": 0,
                    "completion_tokens": max(1, len(raw_content) // 4),
                    "total_tokens": max(1, len(raw_content) // 4),
                }

                recovery = asyncio.create_task(
                    _recover_missing_tool_call(
                        engine, body, message, finish_reason, usage
                    )
                )
                while not recovery.done():
                    done, _ = await asyncio.wait(
                        {recovery}, timeout=settings.sse_heartbeat_secs
                    )
                    if not done:
                        yield ": heartbeat\n\n"
                message, finish_reason, usage = await recovery

                emitted_text = "".join(emitted_text_parts)
                clean_text = str(message.get("content") or "")
                remaining_text = (
                    clean_text[len(emitted_text):]
                    if emitted_text and clean_text.startswith(emitted_text)
                    else (clean_text if not emitted_text else "")
                )
                completion_units += max(1, len(emitted_text + remaining_text) // 4)
                if remaining_text:
                    payload = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": remaining_text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

                if message.get("tool_calls"):
                    completion_units += len(message["tool_calls"])
                    streamed_calls = []
                    for index, call in enumerate(message["tool_calls"]):
                        streamed_call = dict(call)
                        streamed_call["index"] = index
                        streamed_calls.append(streamed_call)
                    payload = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"tool_calls": streamed_calls},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

                final_payload = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model_name,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": finish_reason}
                    ],
                }
                yield f"data: {json.dumps(final_payload)}\n\n"
                if body.stream_options and body.stream_options.get("include_usage"):
                    usage_payload = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_name,
                        "choices": [],
                        "usage": usage,
                    }
                    yield f"data: {json.dumps(usage_payload)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                logger.error("Streaming error in chat: %s", exc, exc_info=True)
                error_code = 503 if isinstance(exc, EngineUnavailableError) else 500
                error = {
                    "error": {
                        "message": str(exc),
                        "type": "server_error",
                        "param": None,
                        "code": error_code,
                    }
                }
                yield f"data: {json.dumps(error)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                cancel_event.set()
                if producer is not None and not producer.done():
                    producer.cancel()
                km.record_usage(kd["key"], completion_units)

        response = StreamingResponse(stream_generator(), media_type="text/event-stream")
        response.headers["Connection"] = "close"
        response.headers["X-Accel-Buffering"] = "no"
        response.headers["Cache-Control"] = "no-cache"
        return response

    raw_res = await engine.generate_chat_non_streaming(**_generation_kwargs(body))
    message, finish_reason, usage = _raw_to_message(raw_res, body.tools)
    message, finish_reason, usage = await _recover_missing_tool_call(
        engine, body, message, finish_reason, usage
    )

    total_tokens = usage.get(
        "total_tokens",
        usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
    )
    km.record_usage(kd["key"], total_tokens)

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }
