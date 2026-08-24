"""
Tests for authentication, authorization, token headers, and security boundaries in NAA.
"""

from fastapi.testclient import TestClient
from src.core.key_manager import APIKeyManager

def test_missing_auth_header(client: TestClient):
    response = client.get("/v1/models")
    assert response.status_code == 401
    data = response.json()
    assert "error" in data
    assert data["error"]["type"] == "authentication_error"
    assert data["error"]["code"] == 401

def test_invalid_bearer_token(client: TestClient):
    response = client.get("/v1/models", headers={"Authorization": "Bearer naa-invalid-fake-key-12345"})
    assert response.status_code == 401
    data = response.json()
    assert "error" in data
    assert data["error"]["type"] == "authentication_error"

def test_x_api_key_header_support(client: TestClient, isolated_key_manager: APIKeyManager):
    key_record = isolated_key_manager.create_key(name="header-test", role="user")
    response = client.get("/v1/models", headers={"x-api-key": key_record["key"]})
    assert response.status_code == 200
    assert response.json()["object"] == "list"

def test_non_admin_forbidden_on_admin_routes(client: TestClient, user_headers):
    response = client.get("/v1/admin/keys/list", headers=user_headers)
    assert response.status_code == 403
    data = response.json()
    assert "error" in data
    assert data["error"]["type"] == "permission_error"
    assert data["error"]["code"] == 403

def test_admin_allowed_on_admin_routes(client: TestClient, admin_headers):
    response = client.get("/v1/admin/keys/list", headers=admin_headers)
    assert response.status_code == 200
    data = response.json()
    assert "keys" in data
    assert "count" in data

def test_revoked_key_rejected(client: TestClient, isolated_key_manager: APIKeyManager):
    key_record = isolated_key_manager.create_key(name="to-revoke", role="user")
    valid_headers = {"Authorization": f"Bearer {key_record['key']}"}
    
    res1 = client.get("/v1/models", headers=valid_headers)
    assert res1.status_code == 200

    isolated_key_manager.revoke_key(key_record["key"])

    res2 = client.get("/v1/models", headers=valid_headers)
    assert res2.status_code == 401
