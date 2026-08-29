"""Client and process wrapper for a standalone llama.cpp server backend."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


logger = logging.getLogger("naa-llama-server")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class LlamaServerBackend:
    """Expose llama-cpp-python's small completion surface over HTTP."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        process: Optional[subprocess.Popen] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.process = process

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _request(self, path: str, payload: Dict[str, Any], *, stream: bool):
        body = json.dumps(_jsonable(payload), ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=self._headers,
            method="POST",
        )
        try:
            response = urllib.request.urlopen(request, timeout=3600)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"llama-server returned HTTP {exc.code}: {detail[:1000]}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"llama-server request failed: {exc}") from exc

        if stream:
            return self._iter_sse(response)
        try:
            return json.loads(response.read().decode("utf-8"))
        finally:
            response.close()

    @staticmethod
    def _iter_sse(response) -> Iterator[Dict[str, Any]]:
        try:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    if data == "[DONE]":
                        break
                    continue
                payload = json.loads(data)
                if payload.get("error"):
                    raise RuntimeError(
                        f"llama-server stream failed: {payload['error']}"
                    )
                yield payload
        finally:
            response.close()

    def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        min_p: float = 0.05,
        top_k: int = 40,
        repeat_penalty: float = 1.08,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = None,
        **_: Any,
    ):
        payload: Dict[str, Any] = {
            "messages": _jsonable(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "min_p": min_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "stop": stop or [],
            "stream": stream,
        }
        if tools:
            payload["tools"] = _jsonable(tools)
            payload["tool_choice"] = _jsonable(tool_choice or "auto")
        return self._request("/v1/chat/completions", payload, stream=stream)

    def create_completion(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        min_p: float = 0.05,
        top_k: int = 40,
        repeat_penalty: float = 1.08,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **_: Any,
    ):
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "min_p": min_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "stop": stop or [],
            "stream": stream,
        }
        return self._request("/v1/completions", payload, stream=stream)

    @classmethod
    def launch(
        cls,
        binary: str,
        model_path: str,
        *,
        n_ctx: int,
        n_gpu_layers: int,
        flash_attn: bool,
        cache_type_k: Optional[str] = None,
        cache_type_v: Optional[str] = None,
    ) -> "LlamaServerBackend":
        port = int(os.environ.get("NAA_LLAMA_SERVER_PORT", "8001"))
        api_key = os.environ.get(
            "NAA_LLAMA_SERVER_API_KEY", "naa-internal-backend"
        )
        base_url = f"http://127.0.0.1:{port}"

        existing = cls(base_url, api_key)
        if existing.is_healthy(timeout=1.0):
            logger.info("Reusing the existing llama-server backend at %s", base_url)
            return existing

        command = [
            binary,
            "--model",
            model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--ctx-size",
            str(n_ctx),
            "-ngl",
            str(n_gpu_layers),
            "--parallel",
            "1",
            "--jinja",
            "--flash-attn",
            "on" if flash_attn else "off",
            "--api-key",
            api_key,
        ]
        if cache_type_k:
            command.extend(["--cache-type-k", str(cache_type_k)])
        if cache_type_v:
            command.extend(["--cache-type-v", str(cache_type_v)])

        log_path = Path(
            os.environ.get("NAA_LLAMA_SERVER_LOG", "/content/naa_llama_server.log")
        )
        if not Path("/content").exists():
            log_path = Path(os.environ.get("NAA_LLAMA_SERVER_LOG", "naa_llama_server.log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
        log_handle.write(f"========== start @ {time.strftime('%Y-%m-%d %H:%M:%S')} ==========\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        log_handle.close()

        timeout = float(os.environ.get("NAA_LLAMA_SERVER_LOAD_TIMEOUT", "600"))
        deadline = time.monotonic() + max(1.0, timeout)
        backend = cls(base_url, api_key, process=process)
        while time.monotonic() < deadline:
            return_code = process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"llama-server exited with code {return_code}. "
                    f"Last log output:\n{_log_tail(log_path)}"
                )
            if backend.is_healthy(timeout=1.0):
                logger.info("Standalone llama-server is ready at %s", base_url)
                return backend
            time.sleep(1.0)

        process.terminate()
        raise RuntimeError(
            f"llama-server did not become ready within {timeout:g}s. "
            f"Last log output:\n{_log_tail(log_path)}"
        )

    def is_healthy(self, timeout: float = 2.0) -> bool:
        request = urllib.request.Request(
            f"{self.base_url}/health", headers=self._headers
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status == 200
        except Exception:
            return False


def _log_tail(path: Path, limit: int = 60) -> str:
    try:
        return "\n".join(
            path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
        )
    except Exception:
        return "(llama-server log is unavailable)"
