# NAA (Notebooks AI API)

Universal, OpenAI-compatible REST API server engineered specifically for Kaggle, Google Colab, and local GPU/CPU environments.

NAA turns any Jupyter, Kaggle, or Google Colab notebook into a private, high-performance OpenAI-compatible inference service. Load any open-weights model from Hugging Face, GGUF binary, or local Safetensors repository (including Llama 3.1, Qwen 2.5, DeepSeek-R1, Mistral, Gemma, Dolphin, uncensored, and abliterated models) with automatic VRAM optimization and built-in Cloudflare tunneling.

---

## Key Features

- **Universal AI Model Support**: Compatible with any open-weights model format (Hugging Face Safetensors, PyTorch, or GGUF).
- **Uncensored and Abliterated Friendly**: No forced refusal guardrails, native chat template passthrough, full preservation of `<think>...</think>` internal reasoning tags, and optional unconstrained epistemic reasoning presets.
- **Hardware-Aware Quantization**: Automatically detects GPU VRAM and applies optimal 4-bit NF4 double-quantization (via bitsandbytes) for 12-16 GB GPUs (such as Kaggle/Colab T4 or P100), 8-bit Int8 for 24 GB GPUs, or full precision (bfloat16/float16).
- **Zero-Configuration Cloudflare Tunnel**: Automatic binary download and HTTPS public URL generation with no account, token, or port forwarding required.
- **Supervisor Watchdog and Keepalive**: Background heartbeat loop prevents notebook kernel idle timeouts and automatically recovers the server process in the event of memory pressure or crashes.
- **API Key and Sliding-Window Rate Limiting**: In-memory sliding-window request throttling (RPM), multi-user key generation, and administrative usage analytics.
- **OpenAI Wire Compatibility**: Drop-in compatible with the OpenAI Python SDK, LangChain, LiteLLM, OpenWebUI, LibreChat, and cURL.

---

## Quickstart Notebook Setup

Paste the corresponding cell into your Kaggle or Colab notebook and run it.

### For Kaggle

```python
import os

# 1. Start from base Kaggle working directory
%cd /kaggle/working

# 2. Clone or pull latest NAA repository
if not os.path.exists("NAA"):
    print("Repository not found. Cloning...")
    !git clone https://github.com/AlGhozaliRamadhan/NAA.git NAA
    %cd /kaggle/working/NAA
else:
    print("Repository found at NAA. Checking for updates...")
    %cd /kaggle/working/NAA
    !git stash
    !git pull origin main
    !git stash drop

# 3. Install dependencies and launch NAA server
!pip install -q -r requirements.txt
!python naa.py start
```

### For Google Colab

```python
import os

# 1. Start from base Colab working directory
%cd /content

# 2. Clone or pull latest NAA repository
if not os.path.exists("NAA"):
    print("Repository not found. Cloning...")
    !git clone https://github.com/AlGhozaliRamadhan/NAA.git NAA
    %cd /content/NAA
else:
    print("Repository found at NAA. Checking for updates...")
    %cd /content/NAA
    !git stash
    !git pull origin main
    !git stash drop

# 3. Install dependencies and launch NAA server
!pip install -q -r requirements.txt
!python naa.py start
```

When the server initializes, the terminal displays:

```text
  +----------------------------------------------------------+
  |  NAA (Notebooks AI API) is LIVE                          |
  |  Model:     Universal LLM                                |
  |  URL:       https://xxxx.trycloudflare.com               |
  |  API Base:  https://xxxx.trycloudflare.com/v1            |
  |  Admin key: naa-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx         |
  |  Docs:      https://xxxx.trycloudflare.com/docs          |
  +----------------------------------------------------------+
```

---

## CLI Usage and Model Configuration

NAA allows dynamic model specification via CLI arguments or environment variables:

```bash
# Start server with default or auto-configured model
python naa.py start

# Load any Hugging Face repository directly
python naa.py start --model Qwen/Qwen2.5-7B-Instruct

# Load an uncensored model with unconstrained reasoning preset
python naa.py start --model cognitivecomputations/dolphin-2.9.4-llama3.1-8b --preset uncensored

# Load DeepSeek reasoning model (preserves <think> deliberation tags)
python naa.py start --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

# Setup and download model weights before starting
python naa.py setup
python naa.py setup 4bit
python naa.py setup 8bit
python naa.py setup 16bit
python naa.py setup meta-llama/Llama-3.1-8B-Instruct

# Manage API keys interactively
python naa.py keys

# Check active server health, loaded model, and uptime
python naa.py status
```

---

## Uncensored and Abliterated Model Support

Standard API servers often hinder uncensored or abliterated models by forcing synthetic alignment prompts, overriding user system messages, or mangling chat templates. NAA eliminates these restrictions:

1. **Clean Passthrough (Default)**: By default, NAA injects no artificial guardrails or moralizing prompts. Incoming conversation history is formatted according to the model's native chat template.
2. **Preservation of System Instructions**: Any system message provided by the caller is preserved verbatim.
3. **Reasoning and Thinking Tag Passthrough**: Internal deliberation tags (such as `<think>...</think>`) are streamed and returned without alteration, ensuring full compatibility with reasoning architectures like DeepSeek-R1.
4. **Uncensored Deliberation Preset (`--preset uncensored`)**: When enabled, NAA provides structured reasoning directives for uninhibited problem-solving:

```text
<confidence>0.XX</confidence>
<thought>
[Internal analysis, premise verification, and response planning]
</thought>
<action>[answer | generate_code | verify | admit_ignorance]</action>
[Finalized comprehensive response]
```

---

## API Usage Examples

NAA implements standard OpenAI wire endpoints. You can route any OpenAI SDK or HTTP client directly to the Cloudflare public URL.

### 1. cURL

```bash
curl -X POST "https://YOUR-URL.trycloudflare.com/v1/chat/completions" \
  -H "Authorization: Bearer naa-YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Explain quantum entanglement."}],
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

### 2. Python (Requests)

```python
import requests

response = requests.post(
    "https://YOUR-URL.trycloudflare.com/v1/chat/completions",
    headers={"Authorization": "Bearer naa-YOUR_KEY"},
    json={
        "model": "default",
        "messages": [{"role": "user", "content": "Write a binary search function in Python."}],
        "temperature": 0.2,
        "max_tokens": 400,
    },
)

print(response.json()["choices"][0]["message"]["content"])
```

### 3. OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://YOUR-URL.trycloudflare.com/v1",
    api_key="naa-YOUR_KEY",
)

# Streaming Chat Completion
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Explain distributed systems architecture."}],
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### 4. LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="https://YOUR-URL.trycloudflare.com/v1",
    api_key="naa-YOUR_KEY",
    model="default",
)

response = llm.invoke("What are the primary tradeoffs of microservices?")
print(response.content)
```

---

## Admin API and Key Management

All administrative routes require the admin key header: `Authorization: Bearer YOUR_ADMIN_KEY`.

### Create API Key

```bash
curl -X POST "https://YOUR-URL.trycloudflare.com/v1/admin/keys/create" \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "application-client", "role": "user", "rpm": 30}'
```

### List API Keys

```bash
curl "https://YOUR-URL.trycloudflare.com/v1/admin/keys/list" \
  -H "Authorization: Bearer YOUR_ADMIN_KEY"
```

### Revoke API Key

```bash
curl -X POST "https://YOUR-URL.trycloudflare.com/v1/admin/keys/revoke" \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"key": "naa-xxxxxxxxxxxx"}'
```

### Delete API Key

```bash
curl -X DELETE "https://YOUR-URL.trycloudflare.com/v1/admin/keys/naa-xxxxxxxxxxxx" \
  -H "Authorization: Bearer YOUR_ADMIN_KEY"
```

### Server Statistics

```bash
curl "https://YOUR-URL.trycloudflare.com/v1/admin/stats" \
  -H "Authorization: Bearer YOUR_ADMIN_KEY"
```

---

## API Endpoints Reference

| Method | Path | Authentication | Description |
|---|---|---|---|
| `GET` | `/` | None | Web Status Dashboard |
| `GET` | `/health` | None | Server health, active model, and GPU status |
| `GET` | `/ping` | None | Liveness probe endpoint |
| `GET` | `/v1/models` | API Key | List active and available models |
| `POST` | `/v1/chat/completions` | API Key | OpenAI Chat Completion (supports SSE streaming) |
| `POST` | `/v1/completions` | API Key | OpenAI Text Completion (supports SSE streaming) |
| `POST` | `/v1/admin/keys/create` | Admin Key | Create a new API key with RPM limit |
| `GET` | `/v1/admin/keys/list` | Admin Key | List all active and revoked API keys |
| `POST` | `/v1/admin/keys/revoke` | Admin Key | Deactivate an API key |
| `DELETE` | `/v1/admin/keys/{key}` | Admin Key | Permanently remove an API key |
| `GET` | `/v1/admin/stats` | Admin Key | Retrieve server uptime, total requests, and token counts |
| `GET` | `/docs` | None | Interactive Swagger UI API documentation |

---

## Hardware Optimization and Quantization Profiles

| Profile | Format | Target VRAM | Description |
|---|---|---|---|
| `auto` (default) | Safetensors / GGUF | Dynamic | Auto-detects VRAM and applies 4-bit NF4 if VRAM <= 16 GB |
| `4bit` | Safetensors (NF4) | ~8.85 GB VRAM | Fits single 12-16 GB GPU (Kaggle/Colab T4 or P100) |
| `8bit` | Safetensors (Int8) | ~16.10 GB VRAM | High-precision quantization for 24 GB GPUs |
| `16bit` | Safetensors (FP16/BF16) | ~30.80 GB VRAM | Full precision for multi-GPU or high-memory instances |
| `q4_k_m` | GGUF | ~8.85 GB VRAM | 4-bit medium quantization via llama-cpp-python |
| `q8_0` | GGUF | ~16.10 GB VRAM | 8-bit high-precision quantization via llama-cpp-python |

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `NAA_MODEL_PATH` | `./models/model` | Local path or directory of model weights |
| `NAA_MODEL_NAME` | `NAA-AI-Model` | Display name of the active model |
| `NAA_HF_REPO` | `Qwen/Qwen2.5-7B-Instruct` | Default Hugging Face repository |
| `NAA_ADMIN_KEY` | Auto-generated | Master admin key for managing API keys and viewing statistics |
| `NAA_KEYS_FILE` | `naa_keys.json` | Path to persistent API keys JSON file |
| `NAA_QUANT` | `auto` | Quantization profile (`auto`, `4bit`, `8bit`, `16bit`, `q4_k_m`) |
| `NAA_PRESET` | `default` | Prompt preset (`default`, `uncensored`, `abliterated`) |
| `NAA_SYSTEM_PROMPT` | `None` | Global fallback system prompt |
| `NAA_PORT` / `PORT` | `8000` | Local port for FastAPI server |
| `NAA_RPM` | `30` | Default rate limit in requests per minute per key |
| `NAA_SSE_HEARTBEAT` | `5.0` | Interval in seconds for SSE streaming keepalive comments |

---

## Repository Structure

```text
NAA/
├── src/
│   ├── config.py              # Environment detection and typed settings
│   ├── cli.py                 # CLI commands and management interface
│   ├── core/
│   │   ├── engine.py          # Universal Transformers and GGUF inference engine
│   │   ├── prompt.py          # ChatML formatting, presets, and directives
│   │   ├── stop_criteria.py   # Stop token resolution and matching
│   │   └── key_manager.py     # Thread-safe API key manager and rate limiter
│   ├── server/
│   │   ├── app.py             # FastAPI application factory, CORS, GZip, and error handling
│   │   ├── schemas.py         # OpenAI-compatible Pydantic request/response schemas
│   │   ├── auth.py            # Bearer and x-api-key authentication dependencies
│   │   ├── routes/            # Route handlers (chat, completions, models, health, admin)
│   │   └── static/            # Web dashboard interface
│   ├── tunnel/
│   │   └── cloudflare.py      # Cloudflare Quick Tunnel binary manager
│   └── supervisor/
│       └── watchdog.py        # Keepalive loop and watchdog supervisor
├── tests/                     # Comprehensive pytest test suite
├── requirements.txt           # Python dependencies
├── naa.py                     # Primary CLI and notebook runner
└── README.md
```

---

## Automated Test Suite

Run the full pytest suite:

```bash
pytest -v
```
