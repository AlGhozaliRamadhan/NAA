"""Anthropic Messages compatibility layer for Claude Code and SDK clients."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Body, Depends, Request, Response
from fastapi.responses import StreamingResponse

from src.config import settings
from src.core.tool_calls import (
    ToolTextStreamBuffer,
    normalize_openai_message,
    tool_choice_requires_call,
)
from src.server.auth import get_api_key
from src.server.routes.chat import (
    _accumulate_tool_deltas,
    _collect_stream,
    _generation_kwargs,
    _raw_to_message,
    _recover_missing_tool_call,
    ensure_engine_ready,
)
from src.server.schemas import AnthropicMessageRequest, ChatCompletionRequest

logger = logging.getLogger("naa-anthropic")
router = APIRouter(tags=["Anthropic Messages"])


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif not isinstance(block, dict):
                parts.append(str(block))
            elif block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif block.get("type") == "image":
                parts.append("[Image content supplied to a text-only local model]")
            else:
                nested = block.get("content")
                if nested is not None:
                    parts.append(_content_text(nested))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def anthropic_tools_to_openai(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for tool in tools or []:
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        function: Dict[str, Any] = {
            "name": name,
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object"}),
        }
        if "strict" in tool:
            function["strict"] = bool(tool["strict"])
        converted.append({"type": "function", "function": function})
    return converted


def anthropic_tool_choice_to_openai(choice: Optional[Dict[str, Any]]) -> Any:
    if not choice:
        return "auto"
    choice_type = choice.get("type", "auto")
    if choice_type == "any":
        return "required"
    if choice_type == "none":
        return "none"
    if choice_type == "tool" and choice.get("name"):
        return {
            "type": "function",
            "function": {"name": choice["name"]},
        }
    return "auto"


def anthropic_messages_to_openai(body: AnthropicMessageRequest) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    system_text = _content_text(body.system)
    if system_text:
        messages.append({"role": "system", "content": system_text})

    tool_names_by_id: Dict[str, str] = {}
    for source in body.messages:
        role = source.get("role", "user")
        content = source.get("content", "")
        blocks = content if isinstance(content, list) else None

        if role == "assistant" and blocks is not None:
            text_parts: List[str] = []
            calls: List[Dict[str, Any]] = []
            for block in blocks:
                if not isinstance(block, dict):
                    text_parts.append(str(block))
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(str(block.get("text", "")))
                elif block_type == "thinking" and block.get("thinking"):
                    text_parts.append(str(block["thinking"]))
                elif block_type == "tool_use":
                    call_id = str(block.get("id") or f"toolu_{uuid.uuid4().hex[:24]}")
                    name = str(block.get("name", ""))
                    tool_names_by_id[call_id] = name
                    calls.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(
                                    block.get("input", {}),
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            },
                        }
                    )
            message: Dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(part for part in text_parts if part) or None,
            }
            if calls:
                message["tool_calls"] = calls
            messages.append(message)
            continue

        if role == "user" and blocks is not None:
            text_parts: List[str] = []
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    call_id = str(block.get("tool_use_id", ""))
                    tool_message: Dict[str, Any] = {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": _content_text(block.get("content")),
                    }
                    if tool_names_by_id.get(call_id):
                        tool_message["name"] = tool_names_by_id[call_id]
                    messages.append(tool_message)
                elif isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
                elif isinstance(block, dict) and block.get("type") == "image":
                    text_parts.append("[Image content supplied to a text-only local model]")
                elif isinstance(block, dict) and block.get("type") == "tool_reference":
                    text_parts.append(json.dumps(block, ensure_ascii=False))
                else:
                    text = _content_text(block)
                    if text:
                        text_parts.append(text)
            if text_parts:
                messages.append({"role": "user", "content": "\n".join(text_parts)})
            continue

        messages.append({"role": role, "content": _content_text(content)})

    return messages


def _compat_body(body: AnthropicMessageRequest) -> ChatCompletionRequest:
    tools = anthropic_tools_to_openai(body.tools)
    return ChatCompletionRequest(
        model=body.model,
        messages=anthropic_messages_to_openai(body),
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        top_k=body.top_k,
        min_p=settings.default_min_p,
        repeat_penalty=settings.default_repetition_penalty,
        stop=body.stop_sequences,
        stream=body.stream,
        tools=tools or None,
        tool_choice=anthropic_tool_choice_to_openai(body.tool_choice),
    )


def openai_message_to_anthropic(
    message: Dict[str, Any], finish_reason: str
) -> Tuple[List[Dict[str, Any]], str]:
    blocks: List[Dict[str, Any]] = []
    if message.get("content"):
        blocks.append({"type": "text", "text": str(message["content"])})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": function.get("name", ""),
                "input": arguments,
            }
        )
    if not blocks:
        blocks.append({"type": "text", "text": ""})

    stop_reason = "end_turn"
    if finish_reason in ("tool_calls", "function_call"):
        stop_reason = "tool_use"
    elif finish_reason == "length":
        stop_reason = "max_tokens"
    elif finish_reason == "stop_sequence":
        stop_reason = "stop_sequence"
    return blocks, stop_reason


def _usage_to_anthropic(usage: Dict[str, Any]) -> Dict[str, int]:
    return {
        "input_tokens": int(usage.get("prompt_tokens", 0)),
        "output_tokens": int(usage.get("completion_tokens", 0)),
    }


def _sse(event: str, payload: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.head("/api/hello", include_in_schema=False)
async def anthropic_connection_probe() -> Response:
    return Response(status_code=204)


@router.post("/v1/messages/count_tokens")
async def count_anthropic_tokens(
    request: Request,
    body: Dict[str, Any] = Body(...),
    kd: Dict[str, Any] = Depends(get_api_key),
):
    # The endpoint is optional in Claude Code.  A conservative approximation is
    # more useful than forcing a generation merely to obtain an exact count.
    serialized = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    return {"input_tokens": max(1, (len(serialized) + 3) // 4)}


@router.post("/v1/messages")
async def create_anthropic_message(
    body: AnthropicMessageRequest,
    request: Request,
    kd: Dict[str, Any] = Depends(get_api_key),
):
    engine = request.app.state.engine
    km = request.app.state.key_manager
    ensure_engine_ready(engine)
    compat = _compat_body(body)
    message_id = f"msg_{uuid.uuid4().hex}"

    if body.stream:
        cancel_event = asyncio.Event()

        async def stream_generator():
            queue: asyncio.Queue = asyncio.Queue()
            producer = asyncio.create_task(
                _collect_stream(engine, _generation_kwargs(compat), cancel_event, queue)
            )
            estimated_input = max(
                1,
                len(json.dumps(body.messages, ensure_ascii=False)) // 4,
            )
            start = {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": body.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": estimated_input, "output_tokens": 0},
                },
            }
            yield _sse("message_start", start)

            content_parts: List[str] = []
            emitted_text_parts: List[str] = []
            native_calls: Dict[int, Dict[str, Any]] = {}
            raw_finish: Optional[str] = None
            text_block_started = False
            text_buffer = (
                ToolTextStreamBuffer(
                    hold_all=tool_choice_requires_call(compat.tool_choice)
                )
                if compat.tools
                else None
            )
            try:
                while True:
                    if await request.is_disconnected():
                        cancel_event.set()
                        return
                    try:
                        kind, payload = await asyncio.wait_for(
                            queue.get(), timeout=settings.sse_heartbeat_secs
                        )
                    except asyncio.TimeoutError:
                        yield _sse("ping", {"type": "ping"})
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
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                        safe_text = (
                            text_buffer.feed(delta["content"])
                            if text_buffer
                            else delta["content"]
                        )
                        if safe_text:
                            if not text_block_started:
                                yield _sse(
                                    "content_block_start",
                                    {
                                        "type": "content_block_start",
                                        "index": 0,
                                        "content_block": {"type": "text", "text": ""},
                                    },
                                )
                                text_block_started = True
                            emitted_text_parts.append(safe_text)
                            yield _sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": {"type": "text_delta", "text": safe_text},
                                },
                            )
                    if delta.get("tool_calls"):
                        safe_prefix = text_buffer.begin_native_tool() if text_buffer else ""
                        if safe_prefix:
                            if not text_block_started:
                                yield _sse(
                                    "content_block_start",
                                    {
                                        "type": "content_block_start",
                                        "index": 0,
                                        "content_block": {"type": "text", "text": ""},
                                    },
                                )
                                text_block_started = True
                            emitted_text_parts.append(safe_prefix)
                            yield _sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": {"type": "text_delta", "text": safe_prefix},
                                },
                            )
                        _accumulate_tool_deltas(native_calls, delta["tool_calls"])
                    if choice.get("finish_reason"):
                        raw_finish = choice["finish_reason"]

                tail = text_buffer.finish() if text_buffer else ""
                if tail:
                    if not text_block_started:
                        yield _sse(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": 0,
                                "content_block": {"type": "text", "text": ""},
                            },
                        )
                        text_block_started = True
                    emitted_text_parts.append(tail)
                    yield _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": tail},
                        },
                    )

                message: Dict[str, Any] = {
                    "role": "assistant",
                    "content": "".join(content_parts),
                }
                if native_calls:
                    message["tool_calls"] = [native_calls[i] for i in sorted(native_calls)]
                message, finish_reason = normalize_openai_message(
                    message, raw_finish, compat.tools
                )
                usage = {
                    "prompt_tokens": estimated_input,
                    "completion_tokens": max(1, len("".join(content_parts)) // 4),
                    "total_tokens": estimated_input
                    + max(1, len("".join(content_parts)) // 4),
                }

                recovery = asyncio.create_task(
                    _recover_missing_tool_call(
                        engine, compat, message, finish_reason, usage
                    )
                )
                while not recovery.done():
                    done, _ = await asyncio.wait(
                        {recovery}, timeout=settings.sse_heartbeat_secs
                    )
                    if not done:
                        yield _sse("ping", {"type": "ping"})
                message, finish_reason, usage = await recovery
                blocks, stop_reason = openai_message_to_anthropic(message, finish_reason)

                emitted_text = "".join(emitted_text_parts)
                if emitted_text:
                    if blocks and blocks[0].get("type") == "text":
                        clean_text = str(blocks[0].get("text", ""))
                        blocks[0]["text"] = (
                            clean_text[len(emitted_text):]
                            if clean_text.startswith(emitted_text)
                            else ""
                        )
                    else:
                        # The already-streamed preamble remains part of the
                        # assistant message even if recovery produced only a call.
                        blocks.insert(0, {"type": "text", "text": ""})

                for index, block in enumerate(blocks):
                    if block["type"] == "text":
                        if not (text_block_started and index == 0):
                            yield _sse(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": index,
                                    "content_block": {"type": "text", "text": ""},
                                },
                            )
                        if block["text"]:
                            yield _sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": index,
                                    "delta": {"type": "text_delta", "text": block["text"]},
                                },
                            )
                    else:
                        yield _sse(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": block["id"],
                                    "name": block["name"],
                                    "input": {},
                                },
                            },
                        )
                        yield _sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": json.dumps(
                                        block["input"],
                                        ensure_ascii=False,
                                        separators=(",", ":"),
                                    ),
                                },
                            },
                        )
                    yield _sse(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": index},
                    )

                output_tokens = int(usage.get("completion_tokens", 0))
                yield _sse(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                        "usage": {"output_tokens": output_tokens},
                    },
                )
                yield _sse("message_stop", {"type": "message_stop"})
                km.record_usage(kd["key"], int(usage.get("total_tokens", output_tokens)))
            except Exception as exc:
                logger.error("Anthropic streaming error: %s", exc, exc_info=True)
                yield _sse(
                    "error",
                    {
                        "type": "error",
                        "error": {"type": "api_error", "message": str(exc)},
                    },
                )
            finally:
                cancel_event.set()
                if not producer.done():
                    producer.cancel()

        response = StreamingResponse(stream_generator(), media_type="text/event-stream")
        response.headers["Connection"] = "close"
        response.headers["X-Accel-Buffering"] = "no"
        response.headers["Cache-Control"] = "no-cache"
        return response

    raw_res = await engine.generate_chat_non_streaming(**_generation_kwargs(compat))
    message, finish_reason, usage = _raw_to_message(raw_res, compat.tools)
    message, finish_reason, usage = await _recover_missing_tool_call(
        engine, compat, message, finish_reason, usage
    )
    blocks, stop_reason = openai_message_to_anthropic(message, finish_reason)
    total_tokens = int(
        usage.get(
            "total_tokens",
            usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
        )
    )
    km.record_usage(kd["key"], total_tokens)
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": body.model,
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": _usage_to_anthropic(usage),
    }
