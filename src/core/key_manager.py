"""
Thread-safe API Key Manager with Atomic Disk Persistence and Sliding-Window Rate Limiting for NAA
"""

import os
import json
import time
import atexit
import secrets
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

logger = logging.getLogger("naa-keys")

class APIKeyManager:
    """
    Thread-safe, high-performance API key manager with in-memory usage tracking,
    asynchronous periodic disk syncing, and atomic persistence.
    """

    def __init__(self, keys_file: str, admin_key: Optional[str] = None, sync_interval: float = 30.0):
        self.keys_file = Path(keys_file)
        self.admin_key = admin_key
        self.sync_interval = sync_interval
        self.keys: Dict[str, Dict[str, Any]] = {}
        self.rate_tracker: Dict[str, List[float]] = {}
        self.lock = threading.Lock()
        self._dirty = False
        self._running = True

        self._load()
        if self.admin_key:
            self._ensure_admin_key(self.admin_key)

        # Background flusher thread
        self._flusher_thread = threading.Thread(target=self._background_flusher, daemon=True)
        self._flusher_thread.start()
        atexit.register(self.flush)

    def _ensure_admin_key(self, admin_key: str):
        """Ensure the specified admin key is active in storage."""
        if not admin_key.startswith("naa-"):
            admin_key = f"naa-{admin_key}"
        
        admin_keys = [k for k, v in self.keys.items() if v.get("role") == "admin"]
        if not admin_keys:
            self.create_key(name="admin-master", role="admin", rate_limit_rpm=999999, key_override=admin_key)
            logger.info(f"Admin key initialized: {admin_key}")
        else:
            current_admin_key = admin_keys[0]
            if current_admin_key != admin_key:
                record = self.keys.pop(current_admin_key)
                record["key"] = admin_key
                record["active"] = True
                self.keys[admin_key] = record
                self._save()
                logger.info("Migrated existing admin key to configured admin_key")
            else:
                logger.info(f"Admin key loaded: {admin_key[:8]}...{admin_key[-4:]}")

    def _load(self):
        """Safely load keys from disk."""
        try:
            if self.keys_file.exists():
                content = self.keys_file.read_text(encoding="utf-8").strip()
                if content:
                    self.keys = json.loads(content)
                    logger.info(f"Loaded {len(self.keys)} API keys from {self.keys_file}")
                    return
        except Exception as e:
            logger.warning(f"Could not load API keys from {self.keys_file} ({e}), initializing empty store.")
        self.keys = {}

    def _save(self):
        """Atomically write keys to disk via temporary file rename."""
        try:
            self.keys_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file = self.keys_file.with_suffix(f".tmp.{secrets.token_hex(4)}")
            temp_file.write_text(json.dumps(self.keys, indent=2, default=str), encoding="utf-8")
            
            if os.name != "nt":
                try:
                    os.chmod(temp_file, 0o600)
                except Exception:
                    pass

            temp_file.replace(self.keys_file)
            self._dirty = False
        except Exception as e:
            logger.error(f"Failed to persist API keys to {self.keys_file}: {e}")

    def flush(self):
        """Explicitly flush any dirty in-memory state to disk."""
        with self.lock:
            if self._dirty:
                self._save()

    def _background_flusher(self):
        """Periodically sync dirty state to disk in background."""
        while self._running:
            try:
                time.sleep(self.sync_interval)
                with self.lock:
                    if self._dirty:
                        self._save()
            except Exception as e:
                logger.error(f"Error in background key sync: {e}")

    def create_key(
        self,
        name: str,
        role: str = "user",
        rate_limit_rpm: int = 30,
        key_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create and store a new API key."""
        with self.lock:
            key = key_override or f"naa-{secrets.token_urlsafe(32)}"
            now = datetime.now(timezone.utc).isoformat()
            record = {
                "key": key,
                "name": name,
                "role": role,
                "rate_limit_rpm": rate_limit_rpm,
                "rpm": rate_limit_rpm,
                "created_at": now,
                "last_used": None,
                "total_requests": 0,
                "reqs": 0,
                "total_tokens": 0,
                "tokens": 0,
                "active": True,
            }
            self.keys[key] = record
            self._save()
            return record

    def validate_key(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate an API key and return record if active."""
        with self.lock:
            record = self.keys.get(token)
            if record and record.get("active", False):
                return record
            return None

    def check_rate_limit(self, key: str, default_rpm: int = 30) -> bool:
        """Sliding-window (60s) rate check."""
        now = time.time()
        window = 60.0
        with self.lock:
            record = self.keys.get(key)
            if not record:
                return False
            rpm_limit = record.get("rate_limit_rpm", record.get("rpm", default_rpm))
            if rpm_limit <= 0:
                return True

            timestamps = self.rate_tracker.get(key, [])
            valid_timestamps = [t for t in timestamps if now - t < window]
            if len(valid_timestamps) >= rpm_limit:
                self.rate_tracker[key] = valid_timestamps
                return False

            valid_timestamps.append(now)
            self.rate_tracker[key] = valid_timestamps
            return True

    def record_usage(self, key: str, tokens_used: int = 0):
        """Update request and token counters asynchronously in memory."""
        with self.lock:
            if key in self.keys:
                now_str = datetime.now(timezone.utc).isoformat()
                self.keys[key]["last_used"] = now_str
                self.keys[key]["total_requests"] = self.keys[key].get("total_requests", 0) + 1
                self.keys[key]["reqs"] = self.keys[key]["total_requests"]
                self.keys[key]["total_tokens"] = self.keys[key].get("total_tokens", 0) + tokens_used
                self.keys[key]["tokens"] = self.keys[key]["total_tokens"]
                self._dirty = True  # Flag dirty for background sync without blocking caller

    def revoke_key(self, key: str) -> bool:
        """Deactivate a key."""
        with self.lock:
            if key in self.keys:
                self.keys[key]["active"] = False
                self._save()
                return True
            return False

    def list_keys(self, include_key_value: bool = False) -> List[Dict[str, Any]]:
        """List keys with optional masking."""
        with self.lock:
            result = []
            for k, v in self.keys.items():
                entry = v.copy()
                if not include_key_value:
                    entry["key"] = k[:8] + "..." + k[-4:] if len(k) > 12 else "naa-***"
                result.append(entry)
            return result

    def delete_key(self, key: str) -> bool:
        """Delete key permanently."""
        with self.lock:
            if key in self.keys:
                del self.keys[key]
                if key in self.rate_tracker:
                    del self.rate_tracker[key]
                self._save()
                return True
            return False
