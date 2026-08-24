"""
Tests for Admin endpoints: key creation, listing, revoking, deletion, and usage stats in NAA.
"""

from fastapi.testclient import TestClient

def test_admin_create_key(client: TestClient, admin_headers):
    payload = {"name": "app-backend", "role": "user", "rpm": 45}
    response = client.post("/v1/admin/keys/create", headers=admin_headers, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "key" in data
    assert data["key"]["name"] == "app-backend"
    assert data["key"]["rate_limit_rpm"] == 45
    assert data["key"]["key"].startswith("naa-")

def test_admin_list_keys(client: TestClient, admin_headers):
    # Create key
    client.post("/v1/admin/keys/create", headers=admin_headers, json={"name": "key-1", "role": "user"})
    
    response = client.get("/v1/admin/keys/list", headers=admin_headers)
    assert response.status_code == 200
    data = response.json()
    assert "keys" in data
    assert data["count"] >= 2  # Admin + newly created key

def test_admin_revoke_key(client: TestClient, admin_headers):
    create_res = client.post("/v1/admin/keys/create", headers=admin_headers, json={"name": "to-kill", "role": "user"})
    raw_key = create_res.json()["key"]["key"]

    revoke_res = client.post("/v1/admin/keys/revoke", headers=admin_headers, json={"key": raw_key})
    assert revoke_res.status_code == 200
    assert revoke_res.json()["success"] is True

def test_admin_delete_key(client: TestClient, admin_headers):
    create_res = client.post("/v1/admin/keys/create", headers=admin_headers, json={"name": "to-delete", "role": "user"})
    raw_key = create_res.json()["key"]["key"]

    del_res = client.delete(f"/v1/admin/keys/{raw_key}", headers=admin_headers)
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

def test_admin_stats(client: TestClient, admin_headers):
    response = client.get("/v1/admin/stats", headers=admin_headers)
    assert response.status_code == 200
    data = response.json()
    assert "uptime" in data
    assert "total_keys" in data
    assert "active_keys" in data
    assert "total_requests" in data
    assert "total_tokens" in data
