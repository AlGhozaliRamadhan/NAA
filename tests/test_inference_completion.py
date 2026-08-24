"""
Tests for /v1/completions (text completions) endpoint (streaming and non-streaming) in NAA.
"""

from fastapi.testclient import TestClient

def test_text_completions_non_streaming(client: TestClient, user_headers, mock_engine):
    payload = {
        "model": "NAA-AI-Model",
        "prompt": "Once upon a time",
        "max_tokens": 50,
        "stream": False,
    }
    response = client.post("/v1/completions", headers=user_headers, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "text_completion"
    assert len(data["choices"]) == 1
    assert "text" in data["choices"][0]
    assert data["usage"]["total_tokens"] > 0

def test_text_completions_streaming(client: TestClient, user_headers, mock_engine):
    payload = {
        "model": "NAA-AI-Model",
        "prompt": "Explain gravity",
        "stream": True,
    }
    response = client.post("/v1/completions", headers=user_headers, json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    
    lines = [line for line in response.text.split("\n") if line.strip()]
    assert any("text_completion" in l for l in lines)
    assert lines[-1] == "data: [DONE]"
