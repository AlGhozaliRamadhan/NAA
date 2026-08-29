"""
Tests for Pydantic input validation, bounds checks, and OpenAI error format conformity.
"""

from fastapi.testclient import TestClient

def test_chat_invalid_max_tokens_bounds(client: TestClient, user_headers):
    res = client.post("/v1/chat/completions", headers=user_headers, json={
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 0
    })
    assert res.status_code == 422
    data = res.json()
    assert "error" in data
    assert data["error"]["type"] == "invalid_request_error"
    assert data["error"]["code"] == 422

def test_chat_invalid_temperature_bounds(client: TestClient, user_headers):
    res = client.post("/v1/chat/completions", headers=user_headers, json={
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": -0.5
    })
    assert res.status_code == 422
    data = res.json()
    assert "error" in data

def test_chat_empty_messages_validation(client: TestClient, user_headers):
    res = client.post("/v1/chat/completions", headers=user_headers, json={
        "messages": "not-a-list"
    })
    assert res.status_code == 422

def test_completions_missing_prompt(client: TestClient, user_headers):
    res = client.post("/v1/completions", headers=user_headers, json={
        "max_tokens": 100
    })
    assert res.status_code == 422
    data = res.json()
    assert "error" in data

def test_model_not_ready_returns_503(client: TestClient, user_headers, mock_engine):
    mock_engine.model_loaded = False
    res = client.post("/v1/chat/completions", headers=user_headers, json={
        "messages": [{"role": "user", "content": "hello"}]
    })
    assert res.status_code == 503
    assert res.headers["retry-after"] == "5"
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == 503
