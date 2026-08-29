"""End-to-end wire compatibility tests for coding-agent tool loops."""

import json
import threading
import time

from fastapi.testclient import TestClient


OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Glob",
            "description": "Find files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "pathStyle": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    }
]

ANTHROPIC_TOOLS = [
    {
        "name": "Glob",
        "description": "Find files matching a glob pattern.",
        "input_schema": OPENAI_TOOLS[0]["function"]["parameters"],
    }
]

XML_CALL = """<tool_call>
<function=Glob>
<parameter=pattern>
*.ts
</parameter>
<parameter=pathStyle>
flat
</parameter>
</function>
</tool_call>"""


def _completion(content, finish_reason="stop"):
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    }


def _sse_data(response_text):
    return [
        json.loads(line[6:])
        for line in response_text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def _finish_model_load(engine, delay=0.08):
    def finish():
        time.sleep(delay)
        engine.model_loaded = True
        engine.model_loading = False
        engine.load_stage = "ready"
        engine.ready_event.set()

    thread = threading.Thread(target=finish, daemon=True)
    thread.start()
    return thread


def test_openai_xml_tool_call_becomes_structured(
    client: TestClient, user_headers, mock_engine, monkeypatch
):
    seen = {}

    def create_chat_completion(**kwargs):
        seen.update(kwargs)
        return _completion(XML_CALL)

    monkeypatch.setattr(mock_engine.model, "create_chat_completion", create_chat_completion)
    response = client.post(
        "/v1/chat/completions",
        headers=user_headers,
        json={
            "model": "NAA-AI-Model",
            "messages": [{"role": "user", "content": "Find TypeScript files"}],
            "tools": OPENAI_TOOLS,
            "tool_choice": "auto",
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "Glob"
    assert json.loads(call["function"]["arguments"]) == {
        "pattern": "*.ts",
        "pathStyle": "flat",
    }
    assert "index" not in call
    assert seen["tools"] == OPENAI_TOOLS
    assert seen["tool_choice"] == "auto"


def test_openai_stream_converts_split_qwen_call(
    client: TestClient, user_headers, mock_engine, monkeypatch
):
    def create_chat_completion(stream=False, **kwargs):
        assert stream is True

        def chunks():
            yield {
                "choices": [
                    {"delta": {"role": "assistant", "content": "<tool_"}, "finish_reason": None}
                ]
            }
            yield {
                "choices": [
                    {
                        "delta": {
                            "content": 'call>\n{"name":"Glob","arguments":{"pattern":"*.py"}}\n'
                        },
                        "finish_reason": None,
                    }
                ]
            }
            yield {
                "choices": [
                    {"delta": {"content": "</tool_call>"}, "finish_reason": None}
                ]
            }
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

        return chunks()

    monkeypatch.setattr(mock_engine.model, "create_chat_completion", create_chat_completion)
    response = client.post(
        "/v1/chat/completions",
        headers=user_headers,
        json={
            "messages": [{"role": "user", "content": "Find Python files"}],
            "tools": OPENAI_TOOLS,
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert "<tool_call>" not in response.text
    chunks = _sse_data(response.text)
    tool_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("choices")
        and chunk["choices"][0].get("delta", {}).get("tool_calls")
    ]
    assert len(tool_chunks) == 1
    call = tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
    assert call["index"] == 0
    assert call["function"]["name"] == "Glob"
    assert json.loads(call["function"]["arguments"]) == {"pattern": "*.py"}
    assert any(
        chunk.get("choices")
        and chunk["choices"][0].get("finish_reason") == "tool_calls"
        for chunk in chunks
    )


def test_openai_stream_waits_through_model_reload(
    client: TestClient, user_headers, mock_engine, monkeypatch
):
    from src.config import settings

    monkeypatch.setattr(settings, "sse_heartbeat_secs", 0.01)
    monkeypatch.setattr(settings, "model_wait_timeout_secs", 1.0)
    mock_engine.model_loaded = False
    mock_engine.model_loading = True
    mock_engine.load_stage = "loading"
    mock_engine.ready_event.clear()
    loader = _finish_model_load(mock_engine)

    response = client.post(
        "/v1/chat/completions",
        headers=user_headers,
        json={
            "messages": [{"role": "user", "content": "Continue after reload"}],
            "stream": True,
        },
    )
    loader.join(timeout=1)

    assert response.status_code == 200
    assert ": model-loading" in response.text
    assert "data: [DONE]" in response.text
    assert '"finish_reason": "stop"' in response.text


def test_openai_preserves_tool_result_history(
    client: TestClient, user_headers, mock_engine, monkeypatch
):
    seen = {}

    def create_chat_completion(**kwargs):
        seen.update(kwargs)
        return _completion("I found one file.")

    monkeypatch.setattr(mock_engine.model, "create_chat_completion", create_chat_completion)
    response = client.post(
        "/v1/chat/completions",
        headers=user_headers,
        json={
            "messages": [
                {"role": "user", "content": "Find TypeScript files"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "Glob",
                                "arguments": '{"pattern":"*.ts"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_123", "content": "src/app.ts"},
            ],
            "tools": OPENAI_TOOLS,
        },
    )

    assert response.status_code == 200
    assert seen["messages"][1]["tool_calls"][0]["id"] == "call_123"
    assert seen["messages"][2]["role"] == "tool"
    assert seen["messages"][2]["tool_call_id"] == "call_123"


def test_abandoned_action_is_retried_once_as_required_tool(
    client: TestClient, user_headers, mock_engine, monkeypatch
):
    calls = []

    def create_chat_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _completion("Yeah, I'll check the project files.")
        return _completion(
            '<tool_call>{"name":"Glob","arguments":{"pattern":"*"}}</tool_call>'
        )

    monkeypatch.setattr(mock_engine.model, "create_chat_completion", create_chat_completion)
    response = client.post(
        "/v1/chat/completions",
        headers=user_headers,
        json={
            "messages": [{"role": "user", "content": "Inspect this project"}],
            "tools": OPENAI_TOOLS,
        },
    )

    assert response.status_code == 200
    assert len(calls) == 2
    assert calls[1]["tool_choice"] == "required"
    assert response.json()["choices"][0]["finish_reason"] == "tool_calls"


def test_streaming_abandoned_action_continues_with_tool_call(
    client: TestClient, user_headers, mock_engine, monkeypatch
):
    calls = []

    def create_chat_completion(stream=False, **kwargs):
        calls.append({"stream": stream, **kwargs})
        if stream:
            def chunks():
                yield {
                    "choices": [
                        {
                            "delta": {"content": "Yeah, I'll check the project."},
                            "finish_reason": None,
                        }
                    ]
                }
                yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

            return chunks()
        return _completion(
            '<tool_call>{"name":"Glob","arguments":{"pattern":"*"}}</tool_call>'
        )

    monkeypatch.setattr(mock_engine.model, "create_chat_completion", create_chat_completion)
    response = client.post(
        "/v1/chat/completions",
        headers=user_headers,
        json={
            "messages": [{"role": "user", "content": "Inspect this project"}],
            "tools": OPENAI_TOOLS,
            "stream": True,
        },
    )

    chunks = _sse_data(response.text)
    assert len(calls) == 2
    assert calls[1]["stream"] is False
    assert calls[1]["tool_choice"] == "required"
    assert any(
        chunk.get("choices")
        and chunk["choices"][0].get("delta", {}).get("tool_calls")
        for chunk in chunks
    )
    assert any(
        chunk.get("choices")
        and chunk["choices"][0].get("finish_reason") == "tool_calls"
        for chunk in chunks
    )


def test_native_openai_tool_call_is_preserved(
    client: TestClient, user_headers, mock_engine, monkeypatch
):
    def create_chat_completion(**kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_native",
                                "type": "function",
                                "function": {
                                    "name": "Glob",
                                    "arguments": {"pattern": "*.md"},
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    monkeypatch.setattr(mock_engine.model, "create_chat_completion", create_chat_completion)
    response = client.post(
        "/v1/chat/completions",
        headers=user_headers,
        json={
            "messages": [{"role": "user", "content": "Find Markdown"}],
            "tools": OPENAI_TOOLS,
        },
    )

    call = response.json()["choices"][0]["message"]["tool_calls"][0]
    assert call["id"] == "call_native"
    assert json.loads(call["function"]["arguments"]) == {"pattern": "*.md"}


def test_anthropic_messages_returns_tool_use(
    client: TestClient, user_headers, mock_engine, monkeypatch
):
    seen = {}

    def create_chat_completion(**kwargs):
        seen.update(kwargs)
        return _completion(XML_CALL)

    monkeypatch.setattr(mock_engine.model, "create_chat_completion", create_chat_completion)
    response = client.post(
        "/v1/messages?beta=true",
        headers={**user_headers, "anthropic-version": "2023-06-01"},
        json={
            "model": "claude-naa",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "Find TypeScript files"}],
            "tools": ANTHROPIC_TOOLS,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "message"
    assert data["stop_reason"] == "tool_use"
    assert data["content"][0]["type"] == "tool_use"
    assert data["content"][0]["name"] == "Glob"
    assert data["content"][0]["input"]["pattern"] == "*.ts"
    assert seen["tools"][0]["function"]["parameters"] == ANTHROPIC_TOOLS[0]["input_schema"]


def test_anthropic_stream_emits_tool_events(
    client: TestClient, user_headers, mock_engine, monkeypatch
):
    def create_chat_completion(stream=False, **kwargs):
        assert stream is True

        def chunks():
            yield {"choices": [{"delta": {"content": XML_CALL}, "finish_reason": None}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

        return chunks()

    monkeypatch.setattr(mock_engine.model, "create_chat_completion", create_chat_completion)
    response = client.post(
        "/v1/messages",
        headers=user_headers,
        json={
            "model": "claude-naa",
            "max_tokens": 256,
            "stream": True,
            "messages": [{"role": "user", "content": "Find TypeScript files"}],
            "tools": ANTHROPIC_TOOLS,
        },
    )

    assert response.status_code == 200
    assert "event: message_start" in response.text
    assert "event: content_block_start" in response.text
    assert '"type": "tool_use"' in response.text
    assert '"stop_reason": "tool_use"' in response.text
    assert "event: message_stop" in response.text
    assert "<function=Glob>" not in response.text


def test_anthropic_stream_waits_through_model_reload(
    client: TestClient, user_headers, mock_engine, monkeypatch
):
    from src.config import settings

    monkeypatch.setattr(settings, "sse_heartbeat_secs", 0.01)
    monkeypatch.setattr(settings, "model_wait_timeout_secs", 1.0)
    mock_engine.model_loaded = False
    mock_engine.model_loading = True
    mock_engine.load_stage = "loading"
    mock_engine.ready_event.clear()
    loader = _finish_model_load(mock_engine)

    response = client.post(
        "/v1/messages",
        headers=user_headers,
        json={
            "model": "claude-naa",
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": "Continue after reload"}],
        },
    )
    loader.join(timeout=1)

    assert response.status_code == 200
    assert "event: message_start" in response.text
    assert "event: ping" in response.text
    assert "event: message_stop" in response.text


def test_anthropic_tool_result_round_trip_and_optional_endpoints(
    client: TestClient, user_headers, mock_engine, monkeypatch
):
    seen = {}

    def create_chat_completion(**kwargs):
        seen.update(kwargs)
        return _completion("Found src/app.ts")

    monkeypatch.setattr(mock_engine.model, "create_chat_completion", create_chat_completion)
    response = client.post(
        "/v1/messages",
        headers=user_headers,
        json={
            "model": "claude-naa",
            "max_tokens": 256,
            "messages": [
                {"role": "user", "content": "Find files"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_123",
                            "name": "Glob",
                            "input": {"pattern": "*.ts"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_123",
                            "content": "src/app.ts",
                        }
                    ],
                },
            ],
            "tools": ANTHROPIC_TOOLS,
        },
    )

    assert response.status_code == 200
    assert seen["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "toolu_123",
        "content": "src/app.ts",
        "name": "Glob",
    }

    count = client.post(
        "/v1/messages/count_tokens",
        headers=user_headers,
        json={"model": "claude-naa", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert count.status_code == 200
    assert count.json()["input_tokens"] > 0
    assert client.head("/api/hello").status_code == 204
