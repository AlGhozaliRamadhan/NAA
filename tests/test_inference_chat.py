"""
Tests for /v1/chat/completions endpoint (streaming and non-streaming inference) in NAA.
"""

import json
from fastapi.testclient import TestClient

def test_chat_completions_non_streaming(client: TestClient, user_headers, mock_engine):
    payload = {
        "model": "NAA-AI-Model",
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": False,
    }
    response = client.post("/v1/chat/completions", headers=user_headers, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "NAA-AI-Model"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "<think>\n" in data["choices"][0]["message"]["content"]
    assert "usage" in data
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["completion_tokens"] > 0
    assert data["usage"]["total_tokens"] == data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]

def test_chat_completions_streaming(client: TestClient, user_headers, mock_engine):
    payload = {
        "model": "NAA-AI-Model",
        "messages": [
            {"role": "user", "content": "Tell me a joke."}
        ],
        "stream": True,
    }
    response = client.post("/v1/chat/completions", headers=user_headers, json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert response.headers["connection"] == "close"
    assert response.headers["x-accel-buffering"] == "no"

    lines = [line for line in response.text.split("\n") if line.strip()]
    assert lines[-1] == "data: [DONE]"

    # Parse JSON SSE chunks
    deltas = []
    for line in lines:
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            assert chunk["object"] == "chat.completion.chunk"
            content = chunk["choices"][0]["delta"].get("content", "")
            if content:
                deltas.append(content)

    assert any("<think>" in d for d in deltas)
