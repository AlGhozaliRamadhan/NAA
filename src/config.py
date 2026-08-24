"""
Configuration and Environment Detection for NAA (Notebooks AI API)
"""

import os
import sys
import secrets
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional

def detect_env() -> Dict[str, Any]:
    """Detect runtime platform (Kaggle, Colab, Local) and GPU availability."""
    env = {
        "name": "local",
        "is_kaggle": False,
        "is_colab": False,
        "is_gpu": False,
        "gpu_name": None,
        "gpu_count": 0,
        "gpu_vram_mb": 0,
        "work_dir": str(Path.cwd()),
        "model_dir": str(Path.cwd() / "models"),
    }

    if os.path.exists("/kaggle"):
        env["name"] = "kaggle"
        env["is_kaggle"] = True
        env["work_dir"] = "/kaggle/working"
        env["model_dir"] = "/kaggle/working/models"
    elif os.path.exists("/content") and ("COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ or os.path.exists("/env/python")):
        env["name"] = "colab"
        env["is_colab"] = True
        env["work_dir"] = "/content"
        env["model_dir"] = "/content/models"

    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            lines = [line.strip() for line in r.stdout.strip().split("\n") if line.strip()]
            env["is_gpu"] = True
            env["gpu_count"] = len(lines)
            parts = lines[0].split(",")
            env["gpu_name"] = parts[0].strip()
            if len(parts) > 1:
                try:
                    env["gpu_vram_mb"] = int(parts[1].strip())
                except ValueError:
                    pass
    except Exception:
        pass

    return env


ENV = detect_env()
WORK_DIR = Path(ENV["work_dir"])
MODEL_DIR = Path(ENV["model_dir"])

KEYS_FILE = WORK_DIR / "naa_keys.json"
SERVER_LOG = WORK_DIR / "naa_server.log"
STATE_FILE = WORK_DIR / ".naa_state.json"

PORT = int(os.environ.get("NAA_PORT", os.environ.get("PORT", "8000")))
QUIET = os.environ.get("NAA_QUIET", "").lower() in ("1", "true", "yes")

# Default model configuration (can be overridden via CLI or env var)
HF_REPO = os.environ.get("NAA_HF_REPO", "Qwen/Qwen2.5-7B-Instruct")
MODEL_NAME = os.environ.get("NAA_MODEL_NAME", os.environ.get("MODEL_NAME", "NAA-AI-Model"))
MODEL_ID = MODEL_NAME

MODELS: Dict[str, Dict[str, str]] = {
    "auto": {
        "name": "auto",
        "file": "model.safetensors.index.json",
        "dir": "model",
        "size": "Auto-VRAM Optimized (4-bit NF4 / 8-bit / 16-bit)",
        "description": "Auto-detects GPU VRAM and applies optimal quantization (4-bit NF4 for <=16GB, 8-bit for <=24GB)",
        "quant": "auto",
    },
    "4bit": {
        "name": "4bit-nf4",
        "file": "model.safetensors.index.json",
        "dir": "model",
        "size": "4-bit NF4 (Fits single 12-16GB GPU like Kaggle/Colab T4/P100)",
        "description": "4-bit NF4 double-quantization (Fast & lowest VRAM footprint)",
        "quant": "4bit",
    },
    "8bit": {
        "name": "8bit-int8",
        "file": "model.safetensors.index.json",
        "dir": "model",
        "size": "8-bit Int8 (Higher precision, ~16GB+ VRAM)",
        "description": "8-bit Int8 quantization (Balanced quality and speed)",
        "quant": "8bit",
    },
    "16bit": {
        "name": "16bit-fp16",
        "file": "model.safetensors.index.json",
        "dir": "model",
        "size": "Full Precision (bfloat16/float16, Multi-GPU or A100)",
        "description": "Full Precision bfloat16/float16",
        "quant": "16bit",
    },
    "q4_k_m": {
        "name": "q4_k_m-gguf",
        "file": "model.q4_k_m.gguf",
        "dir": "model.q4_k_m.gguf",
        "size": "GGUF Q4_K_M (llama.cpp backend)",
        "description": "GGUF Q4_K_M quantization for llama-cpp-python engine",
        "quant": "q4_k_m",
    },
    "q5_k_m": {
        "name": "q5_k_m-gguf",
        "file": "model.q5_k_m.gguf",
        "dir": "model.q5_k_m.gguf",
        "size": "GGUF Q5_K_M (llama.cpp backend)",
        "description": "GGUF Q5_K_M medium quantization",
        "quant": "q5_k_m",
    },
    "q8_0": {
        "name": "q8_0-gguf",
        "file": "model.q8_0.gguf",
        "dir": "model.q8_0.gguf",
        "size": "GGUF Q8_0 (llama.cpp backend, near lossless)",
        "description": "GGUF Q8_0 high precision quantization",
        "quant": "q8_0",
    },
}


@dataclass
class Settings:
    """Typed runtime settings parsed from environment variables."""
    model_path: str = os.environ.get(
        "NAA_MODEL_PATH",
        os.environ.get("MODEL_PATH", str(MODEL_DIR / "model"))
    )
    model_name: str = os.environ.get(
        "NAA_MODEL_NAME",
        os.environ.get("MODEL_NAME", MODEL_NAME)
    )
    admin_key: str = os.environ.get(
        "NAA_ADMIN_KEY",
        os.environ.get("ADMIN_KEY", secrets.token_urlsafe(32))
    )
    keys_file: str = os.environ.get(
        "NAA_KEYS_FILE",
        os.environ.get("API_KEYS_FILE", str(KEYS_FILE))
    )
    quant_mode: str = os.environ.get(
        "NAA_QUANT",
        os.environ.get("QUANT_MODE", "auto")
    ).lower()
    preset: str = os.environ.get(
        "NAA_PRESET",
        os.environ.get("PRESET", "default")
    ).lower()
    system_prompt: Optional[str] = os.environ.get("NAA_SYSTEM_PROMPT", os.environ.get("SYSTEM_PROMPT", None))
    max_context: int = int(os.environ.get("NAA_CTX", os.environ.get("MAX_CONTEXT", "8192")))
    default_tokens: int = int(os.environ.get("NAA_MAX_TOKENS", os.environ.get("MAX_TOKENS_DEFAULT", "2048")))
    default_temperature: float = float(os.environ.get("NAA_TEMPERATURE", "0.70"))
    default_top_p: float = float(os.environ.get("NAA_TOP_P", "0.90"))
    default_min_p: float = float(os.environ.get("NAA_MIN_P", "0.05"))
    default_top_k: int = int(os.environ.get("NAA_TOP_K", "40"))
    default_repetition_penalty: float = float(os.environ.get("NAA_REPEAT_PENALTY", "1.08"))
    n_gpu_layers: int = int(os.environ.get("NAA_N_GPU_LAYERS", "-1"))
    flash_attn: bool = os.environ.get("NAA_FLASH_ATTN", "1").lower() in ("1", "true", "yes")
    default_rpm: int = int(os.environ.get("NAA_RPM", os.environ.get("RATE_LIMIT_RPM", "30")))
    sse_heartbeat_secs: float = float(os.environ.get("NAA_SSE_HEARTBEAT", os.environ.get("SSE_HEARTBEAT_SECS", "5.0")))
    trust_remote_code: bool = os.environ.get("NAA_TRUST_REMOTE_CODE", "1").lower() in ("1", "true", "yes")
    port: int = PORT
    quiet: bool = QUIET

    def __post_init__(self):
        if not self.admin_key.startswith("naa-"):
            self.admin_key = f"naa-{self.admin_key}"

settings = Settings()
