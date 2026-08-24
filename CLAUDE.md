# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

NAA (Notebooks AI API) is a self-hosted, universal, OpenAI-compatible REST API server tailored for running any LLM on Kaggle, Google Colab, or local systems with GPU/CPU hardware. It supports:
- Hugging Face Safetensors / PyTorch models (via `AutoModelForCausalLM` and `AutoTokenizer` with 4-bit NF4 double quantization, 8-bit Int8, and FP16/BF16).
- GGUF models (via `llama_cpp.Llama` with GPU offloading and FlashAttention).
- Full OpenAI wire-compatibility (endpoints, streaming SSE with heartbeats, response schemas, error shapes).
- Native chat templates and uncensored/abliterated presets with epistemic deliberation tags (`<confidence>`, `<thought>`, `<action>`).
- Zero-config Cloudflare Quick Tunnel and self-healing watchdog supervision.

## Common commands

```bash
# Run the automated test suite
pytest -v

# Run the primary CLI manager
python naa.py setup             # install dependencies + download model
python naa.py setup 4bit        # setup with 4-bit NF4 quantization
python naa.py setup 8bit        # setup with 8-bit Int8 quantization
python naa.py setup 16bit       # setup with full precision
python naa.py start             # start server + Cloudflare tunnel + watchdog supervisor
python naa.py keys              # interactive API key manager
python naa.py status            # show public URL, admin key, loaded model, GPU, and health
```

```bash
# Standalone server launch
pip install -r requirements.txt
python -m src.server.app
```

## Architecture

The codebase is organized into modular packages under `src/`:

- `src/config.py`: Platform and GPU detection (`detect_env`), model profiles, typed settings.
- `src/core/prompt.py`: Canonical uncensored/abliterated directives, preset resolution (`prepare_chat_messages`), stop tokens, and ChatML formatting.
- `src/core/stop_criteria.py`: Native stop list resolution and fallback token window matching.
- `src/core/key_manager.py`: Thread-safe `APIKeyManager` with in-memory dirty tracking, periodic 30s background flusher, and atomic disk persistence.
- `src/core/engine.py`: `InferenceEngine` handling model lifecycle (Safetensors / GGUF), GPU offloading, FlashAttention, and non-blocking streaming.
- `src/server/app.py`: FastAPI application factory (`create_app`), CORS, GZip, and OpenAI-compatible error formatting.
- `src/server/schemas.py`: OpenAI-compatible Pydantic request/response schemas.
- `src/server/routes/`: Modular route handlers (`health.py`, `models.py`, `chat.py`, `completions.py`, `admin.py`).
- `src/tunnel/cloudflare.py`: Cloudflare Quick Tunnel binary manager and public URL resolution.
- `src/supervisor/watchdog.py`: Keepalive thread and supervisor watchdog.
- `src/cli.py`: Unified CLI management interface.
- `naa.py`: Primary notebook and CLI runner.

## Output Handling & Streaming Safety

- SSE heartbeats (`: heartbeat\n\n`) and headers (`Connection: close`, `X-Accel-Buffering: no`) prevent Cloudflare 502 Bad Gateway timeouts during long generations.
- Active disconnection cancellation terminates background generation immediately if a client disconnects.
- `<think>...</think>` reasoning tokens are preserved without alteration for reasoning models (DeepSeek-R1, QwQ, etc.).

## Auth and Keys

- `APIKeyManager` tracks usage in-memory with zero disk latency on the request path, syncing to disk every 30 seconds.
- All non-health endpoints require an API key via Bearer token or `x-api-key` header (`naa-...`); admin endpoints require `role == "admin"`.
