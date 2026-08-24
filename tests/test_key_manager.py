"""
Tests for APIKeyManager class: thread safety, atomic file persistence, and recovery from corrupted data in NAA.
"""

import json
import concurrent.futures
from pathlib import Path
from src.core.key_manager import APIKeyManager

def test_key_manager_atomic_save(tmp_path: Path):
    keys_file = tmp_path / "keys_atomic.json"
    km = APIKeyManager(str(keys_file), admin_key="naa-test-admin")
    
    k1 = km.create_key(name="k1", role="user")
    assert keys_file.exists()
    
    km2 = APIKeyManager(str(keys_file), admin_key="naa-test-admin")
    assert km2.validate_key(k1["key"]) is not None

def test_key_manager_concurrency(tmp_path: Path):
    keys_file = tmp_path / "keys_concurrent.json"
    km = APIKeyManager(str(keys_file), admin_key="naa-test-admin")

    def create_batch(i):
        return km.create_key(name=f"worker-{i}", role="user")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(create_batch, i) for i in range(20)]
        results = [f.result() for f in futures]

    assert len(results) == 20
    keys_on_disk = json.loads(keys_file.read_text(encoding="utf-8"))
    assert len(keys_on_disk) >= 21

def test_key_manager_corrupted_file_recovery(tmp_path: Path):
    keys_file = tmp_path / "corrupted_keys.json"
    keys_file.write_text("INVALID JSON CONTENT {[[[", encoding="utf-8")

    km = APIKeyManager(str(keys_file), admin_key="naa-test-admin")
    assert isinstance(km.keys, dict)
    admin_keys = [k for k, v in km.keys.items() if v.get("role") == "admin"]
    assert len(admin_keys) == 1
