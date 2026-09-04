"""
NAA (Notebooks AI API) CLI Management Interface
"""

import os
import sys
import json
import time
import secrets
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any

import src.config as config
from src.config import (
    ENV,
    WORK_DIR,
    MODEL_DIR,
    KEYS_FILE,
    SERVER_LOG,
    PORT,
    QUIET,
    HF_REPO,
    MODEL_NAME,
    MODELS,
    settings,
)
from src.tunnel import start_tunnel, download_cloudflared, tunnel_log_tail
from src.supervisor.watchdog import (
    start_keepalive,
    is_server_healthy,
    is_server_loading,
    public_health_ok,
    wait_for_port,
)

_server_proc: Optional[subprocess.Popen] = None
_tunnel_proc: Optional[subprocess.Popen] = None
_gguf_backend_repair_attempted = False

RESET = "\033[0m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
BOLD = "\033[1m"

def _get_attr(name: str, fallback: Any) -> Any:
    mod = sys.modules.get("naa")
    if mod and hasattr(mod, name):
        return getattr(mod, name)
    return getattr(sys.modules.get(__name__), name, fallback)

def header(title: str):      print(f"{RESET}\n--- {title.upper()} ---")
def info(msg: str):         print(f"{RESET}[INFO] {msg}")
def ok(msg: str):           print(f"{RESET}{GREEN}[OK]{RESET}   {msg}")
def warn(msg: str):         print(f"{RESET}{YELLOW}[WARN]{RESET} {msg}")
def err(msg: str):          print(f"{RESET}{RED}[ERR]{RESET}  {msg}")
def step(n: int, msg: str):   print(f"{RESET}\n[{n}] {msg}")
def rule():                 print(f"{RESET}" + "-" * 60)

try:
    from tqdm.auto import tqdm as _tqdm_base
except ImportError:
    from tqdm import tqdm as _tqdm_base

class NotebookProgressBar(_tqdm_base):
    """
    Clean single-line progress bar for Jupyter, Colab, Kaggle, and terminals.
    Dynamically shifts color based on download progress (Red -> Yellow -> Cyan -> Green)
    and overwrites a single line using carriage returns without multi-line spam.
    """
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mininterval", 0.2)
        kwargs.setdefault("file", sys.stdout)
        super().__init__(*args, **kwargs)
        self._closed_flushed = False

    def display(self, msg=None, pos=None):
        if self.total and self.total > 0:
            pct = (self.n / self.total) * 100
            cur_gb = self.n / (1024 ** 3)
            tot_gb = self.total / (1024 ** 3)
            rate = self.format_dict.get("rate") or 0
            speed_mb = rate / (1024 ** 2)
            remaining = (self.total - self.n) / rate if rate > 0 else 0

            # Dynamic color transition based on percentage:
            if pct < 25:
                col = "\033[31m"  # Red
            elif pct < 60:
                col = "\033[33m"  # Yellow
            elif pct < 90:
                col = "\033[36m"  # Cyan
            else:
                col = "\033[32m"  # Green

            desc = (self.desc or "Downloading").split(":")[-1].strip()
            if len(desc) > 32:
                desc = "..." + desc[-29:]

            bar_len = 24
            filled = int(bar_len * (self.n / self.total))
            bar = "=" * filled + (">" if filled < bar_len else "=") + " " * max(0, bar_len - filled - 1)
            if filled >= bar_len:
                bar = "=" * bar_len

            mins, secs = divmod(int(remaining), 60)
            eta_str = f"{mins:02d}:{secs:02d}"

            line = f"\r\033[0m{col}[{bar}] {pct:5.1f}%\033[0m | {desc} | {cur_gb:.2f}/{tot_gb:.2f} GB | {speed_mb:.1f} MB/s | ETA: {eta_str}\033[K"
            try:
                self.fp.write(line)
                self.fp.flush()
            except Exception:
                pass
        else:
            super().display(msg=msg, pos=pos)

    def close(self):
        super().close()
        if not self._closed_flushed:
            try:
                self.fp.write("\033[0m\n")
                self.fp.flush()
            except Exception:
                pass
            self._closed_flushed = True

def print_banner(model_name: Optional[str] = None):
    active_model = model_name or MODEL_NAME
    gpu_info = f"GPU ({ENV['gpu_count']}x {ENV['gpu_name']})" if ENV['is_gpu'] else "CPU"
    env_label = f"{ENV['name'].upper()} - {gpu_info}"
    print(f"{RESET}\n============================================================")
    print(f" NAA - Notebooks AI API")
    print(f" Target Model: {active_model}")
    print(f" Environment:  {env_label}")
    print("============================================================\n")

def save_state(data: Dict[str, Any]):
    try:
        existing = load_state()
        existing.update(data)
        state_file = Path(config.STATE_FILE)
        state_file.parent.mkdir(parents=True, exist_ok=True)
        temp_state = state_file.with_suffix(f".tmp.{secrets.token_hex(4)}")
        temp_state.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        temp_state.replace(state_file)
    except Exception as e:
        warn(f"Failed to save state: {e}")

def load_state() -> Dict[str, Any]:
    try:
        state_file = Path(config.STATE_FILE)
        if state_file.exists():
            content = state_file.read_text(encoding="utf-8").strip()
            if content:
                return json.loads(content)
    except Exception:
        pass
    return {}

def run_pip(packages: list, extra_args: list = None, env: dict = None) -> bool:
    cmd = [sys.executable, "-m", "pip", "install"] + (extra_args or []) + packages
    r = subprocess.run(cmd, env=env)
    return r.returncode == 0

def install_deps(video: bool = False):
    header("Installing Dependencies")
    step(1, "Core server packages (fastapi, uvicorn, huggingface_hub, pydantic, etc.)")
    run_pip(["fastapi>=0.111.0", "uvicorn[standard]>=0.29.0", "python-multipart>=0.0.9", "huggingface_hub>=0.23.0", "pydantic>=2.0.0", "requests>=2.31.0"])

    step(2, "Inference engine (transformers, torch, accelerate, bitsandbytes, jinja2)")
    packages = ["transformers>=4.45.0", "accelerate>=0.28.0", "bitsandbytes>=0.43.0", "jinja2>=3.1.4"]
    run_pip(packages)

    if video or os.environ.get("NAA_VIDEO", "").lower() in ("1", "true", "yes"):
        step(3, "Video engine (diffusers from source for Wan 2.2, safetensors, Pillow, imageio)")
        run_pip(["git+https://github.com/huggingface/diffusers", "safetensors>=0.4.0", "Pillow>=10.0.0", "imageio>=2.31.0", "imageio-ffmpeg>=0.4.9"])
        info("Optional: pip install sageattention  # SageAttention kernel (NAA_VIDEO_ATTENTION=sage)")
    ok("Dependencies ready")


def install_video_deps() -> bool:
    """Install the Wan 2.2 video stack (diffusers from source + helpers)."""
    header("Installing Video Dependencies (Wan 2.2)")
    ok1 = run_pip(["git+https://github.com/huggingface/diffusers", "safetensors>=0.4.0", "Pillow>=10.0.0", "imageio>=2.31.0", "imageio-ffmpeg>=0.4.9"])
    if ok1:
        ok("Video dependencies ready (diffusers from source for Wan 2.2)")
    else:
        err("Video dependency install failed.")
    return ok1


def download_video_model(model_id: Optional[str] = None) -> Path:
    """Pre-download a Wan 2.2 checkpoint + the default NSFW LoRA set."""
    from src.core.video_engine import (
        DEFAULT_VIDEO_LORA_URL,
        parse_lora_repo,
        resolve_video_model_id,
    )

    target = resolve_video_model_id(model_id or os.environ.get("NAA_VIDEO_MODEL_ID", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"))
    header(f"Downloading Video Model ({target})")
    try:
        from huggingface_hub import snapshot_download

        kwargs: Dict[str, Any] = {
            "repo_id": target,
            "local_dir": str(MODEL_DIR / target.split("/")[-1]),
            "local_dir_use_symlinks": False,
            "tqdm_class": NotebookProgressBar,
        }
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            kwargs["token"] = hf_token
        path = Path(snapshot_download(**kwargs))
        ok(f"Video model ready: {path}")
    except Exception as e:
        err(f"Video model download failed: {e}")
        sys.exit(1)

    lora_repo = parse_lora_repo(os.environ.get("NAA_VIDEO_LORA_URL", DEFAULT_VIDEO_LORA_URL))
    if lora_repo:
        info(f"Pre-fetching video LoRA: {lora_repo}")
        try:
            from huggingface_hub import snapshot_download as _snap

            _snap(repo_id=lora_repo, local_dir=str(MODEL_DIR / lora_repo.split("/")[-1]),
                  local_dir_use_symlinks=False, tqdm_class=NotebookProgressBar)
            ok(f"Video LoRA ready: {lora_repo}")
        except Exception as e:
            warn(f"Video LoRA prefetch failed (will retry at generation time): {e}")
    return path

def _version_tuple(value: str) -> tuple[int, ...]:
    parts = []
    for part in value.split("."):
        digits = "".join(char for char in part if char.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _cuda_compute_arch() -> Optional[str]:
    """Return the first NVIDIA GPU compute capability as a CMake arch."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        capability = result.stdout.strip().splitlines()[0].strip()
        major, minor = capability.split(".", 1)
        if major.isdigit() and minor.isdigit():
            return f"{major}{minor}"
    except Exception:
        pass
    return None


def _install_llama_cpp_cuda(force_source: bool = False) -> bool:
    """Install a recent CUDA backend, optionally compiled for this exact GPU."""

    requirement = os.environ.get(
        "NAA_LLAMA_CPP_REQUIREMENT",
        "llama-cpp-python>=0.3.25",
    )
    if force_source:
        arch = _cuda_compute_arch()
        if not arch:
            warn("Cannot detect the GPU compute capability for a CUDA source build.")
            return False
        info(f"Rebuilding llama-cpp-python for CUDA compute capability sm_{arch}...")
        build_env = os.environ.copy()
        build_env["CMAKE_ARGS"] = (
            f"-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES={arch} "
            "-DGGML_NATIVE=OFF"
        )
        build_env["FORCE_CMAKE"] = "1"
        build_env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "2")
        success = run_pip(
            [requirement],
            extra_args=[
                "--upgrade",
                "--force-reinstall",
                "--no-cache-dir",
                "--no-binary=llama-cpp-python",
            ],
            env=build_env,
        )
        if success:
            try:
                marker = Path(WORK_DIR) / f".naa_llama_cpp_cuda_sm{arch}.ok"
                marker.write_text(requirement, encoding="utf-8")
            except Exception:
                pass
        return success

    info("Installing a current CUDA prebuilt wheel...")
    return run_pip(
        [requirement],
        extra_args=[
            "--upgrade",
            "--force-reinstall",
            "--no-cache-dir",
            "--only-binary=llama-cpp-python",
            "--extra-index-url",
            "https://abetlen.github.io/llama-cpp-python/whl/cu121",
        ],
    )


def _ensure_standalone_llama_server() -> bool:
    """Build only the current standalone llama-server for new GGUF architectures."""

    explicit = os.environ.get("NAA_LLAMA_SERVER_BIN")
    if explicit and Path(explicit).is_file():
        info(f"Using standalone llama-server: {explicit}")
        return True

    if not ENV.get("is_gpu", False):
        err("The standalone llama-server setup currently requires an NVIDIA GPU.")
        return False

    arch = _cuda_compute_arch()
    if not arch:
        err("Could not detect the GPU compute capability for llama-server.")
        return False

    ref = os.environ.get("NAA_LLAMA_CPP_REF", "b10621")
    safe_ref = "".join(char if char.isalnum() else "_" for char in ref)
    source_dir = Path(WORK_DIR) / f".naa_llama_cpp_{safe_ref}"
    binary = source_dir / "build" / "bin" / (
        "llama-server.exe" if os.name == "nt" else "llama-server"
    )
    if binary.is_file():
        os.environ["NAA_LLAMA_SERVER_BIN"] = str(binary)
        info(f"Using the existing llama-server {ref} build for sm_{arch}.")
        return True

    header(f"Building standalone llama-server {ref}")
    info(
        "This compiles only the current llama.cpp server. It avoids the older "
        "CUDA engine embedded in llama-cpp-python."
    )
    if not source_dir.exists():
        clone = subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                ref,
                "https://github.com/ggml-org/llama.cpp.git",
                str(source_dir),
            ],
            check=False,
        )
        if clone.returncode != 0:
            err("Could not download the requested llama.cpp source revision.")
            return False

    configure = subprocess.run(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(source_dir / "build"),
            "-DGGML_CUDA=ON",
            f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
            "-DGGML_NATIVE=OFF",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DLLAMA_CURL=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=False,
    )
    if configure.returncode != 0:
        err("CMake could not configure the standalone llama-server build.")
        return False

    parallel = os.environ.get("NAA_LLAMA_BUILD_JOBS", "2")
    build = subprocess.run(
        [
            "cmake",
            "--build",
            str(source_dir / "build"),
            "--config",
            "Release",
            "--target",
            "llama-server",
            "--parallel",
            parallel,
        ],
        check=False,
    )
    if build.returncode != 0 or not binary.is_file():
        err("The standalone llama-server CUDA build failed.")
        return False

    os.environ["NAA_LLAMA_SERVER_BIN"] = str(binary)
    ok(f"Standalone llama-server is ready: {binary}")
    return True


def ensure_gguf_deps() -> bool:
    gguf_backend = os.environ.get("NAA_GGUF_BACKEND", "python").lower()
    if gguf_backend in ("llama-server", "server", "external"):
        return _ensure_standalone_llama_server()

    minimum = (0, 3, 25)
    force_gpu_build = os.environ.get("NAA_FORCE_LLAMA_CPP_REBUILD", "0").lower() in (
        "1", "true", "yes",
    )
    if force_gpu_build and ENV.get("is_gpu", False):
        arch = _cuda_compute_arch()
        marker = Path(WORK_DIR) / f".naa_llama_cpp_cuda_sm{arch}.ok" if arch else None
        if marker is not None and marker.exists():
            info(f"Using the existing GPU-specific llama.cpp build for sm_{arch}.")
            return True
        header("Building GPU-specific llama-cpp-python")
        return _install_llama_cpp_cuda(force_source=True)

    try:
        import llama_cpp
        installed = _version_tuple(getattr(llama_cpp, "__version__", "0"))
        if installed >= minimum:
            return True
        warn(
            f"llama-cpp-python {getattr(llama_cpp, '__version__', 'unknown')} "
            "is too old for current Qwen3.5/Qwen3.8 GGUF support."
        )
    except ImportError:
        pass

    header("Installing llama-cpp-python for GGUF support")
    if ENV.get("is_gpu", False):
        success = _install_llama_cpp_cuda(force_source=False)
        if not success:
            warn("Prebuilt CUDA wheel failed; compiling for the detected GPU.")
            success = _install_llama_cpp_cuda(force_source=True)
    else:
        info("Installing current CPU llama-cpp-python...")
        success = run_pip(
            ["llama-cpp-python>=0.3.25"],
            extra_args=["--upgrade", "--no-cache-dir"],
        )

    if success:
        ok("llama-cpp-python installed successfully!")
    else:
        err("Could not install a compatible llama-cpp-python backend.")
    return success


def _select_gguf_backend(model_cfg: Dict[str, Any]) -> None:
    """Prefer a current standalone engine for Qwen3.8's new CUDA path."""

    if os.environ.get("NAA_GGUF_BACKEND"):
        return
    filename = str(model_cfg.get("file", ""))
    name = str(model_cfg.get("name", ""))
    if filename.lower().endswith(".gguf") and "qwen3.8" in f"{name} {filename}".lower():
        os.environ["NAA_GGUF_BACKEND"] = "llama-server"
        info(
            "Qwen3.8 GGUF detected; selecting the current standalone "
            "llama-server CUDA backend."
        )


def _repair_gguf_cuda_backend() -> bool:
    """Rebuild once after a native CUDA abort such as SIGILL or SIGABRT."""

    global _gguf_backend_repair_attempted
    if _gguf_backend_repair_attempted:
        return False
    if not ENV.get("is_gpu", False):
        return False
    enabled = os.environ.get("NAA_AUTO_BACKEND_REPAIR", "1").lower() in (
        "1", "true", "yes",
    )
    if not enabled:
        return False
    _gguf_backend_repair_attempted = True
    warn("Native CUDA backend aborted; rebuilding it for this notebook GPU.")
    success = _install_llama_cpp_cuda(force_source=True)
    if success:
        ok("GPU-specific llama-cpp-python rebuild completed.")
    else:
        err("GPU-specific llama-cpp-python rebuild failed; see pip/CMake output above.")
    return success

def parse_model_target(target: Optional[str]) -> Dict[str, str]:
    """
    Parses any model input string:
    - Direct HuggingFace URL: https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED/blob/main/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf
    - GGUF with repo: OBLITERATUS/Qwen3.8-27B-OBLITERATED:Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf
    - GGUF 3-part path: OBLITERATUS/Qwen3.8-27B-OBLITERATED/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf
    - Standard HuggingFace repo: Qwen/Qwen2.5-7B-Instruct
    - Built-in quant profile: auto, 4bit, 8bit, 16bit, q4_k_m, etc.
    """
    if not target:
        return dict(MODELS["auto"])

    target = target.strip()

    # 1. HuggingFace URL
    if "huggingface.co/" in target:
        cleaned = target.split("huggingface.co/")[-1].strip("/")
        parts = cleaned.split("/")
        if len(parts) >= 2:
            repo_id = f"{parts[0]}/{parts[1]}"
            filename = "model.safetensors.index.json"
            if len(parts) >= 5 and parts[2] in ("blob", "resolve", "raw"):
                filename = "/".join(parts[4:])
            elif len(parts) == 3 and parts[2].endswith(".gguf"):
                filename = parts[2]

            is_gguf = filename.endswith(".gguf")
            model_name = parts[1]
            return {
                "name": model_name,
                "repo": repo_id,
                "repo_gguf": repo_id if is_gguf else None,
                "file": filename,
                "dir": model_name,
                "description": f"HuggingFace Model: {repo_id} ({filename if is_gguf else 'Safetensors'})",
                "quant": "q4_k_m" if is_gguf else "auto",
            }

    # 2. repo:filename format
    if ":" in target and not target.startswith("http"):
        repo_id, filename = target.split(":", 1)
        model_name = repo_id.split("/")[-1]
        is_gguf = filename.endswith(".gguf")
        return {
            "name": model_name,
            "repo": repo_id,
            "repo_gguf": repo_id if is_gguf else None,
            "file": filename,
            "dir": model_name,
            "description": f"HuggingFace Model: {repo_id} ({filename})",
            "quant": "q4_k_m" if is_gguf else "auto",
        }

    # 3. repo/filename.gguf format
    if target.count("/") == 2 and target.endswith(".gguf"):
        parts = target.split("/")
        repo_id = f"{parts[0]}/{parts[1]}"
        filename = parts[2]
        model_name = parts[1]
        return {
            "name": model_name,
            "repo": repo_id,
            "repo_gguf": repo_id,
            "file": filename,
            "dir": model_name,
            "description": f"HuggingFace GGUF Model: {repo_id} ({filename})",
            "quant": "q4_k_m",
        }

    # 4. Standard HuggingFace repo (owner/model)
    if "/" in target:
        repo_name = target.split("/")[-1]
        return {
            "name": repo_name,
            "repo": target,
            "dir": repo_name,
            "file": "model.safetensors.index.json",
            "description": f"Custom HuggingFace Model: {target}",
            "quant": "auto",
        }

    # 5. Built-in profile key
    if target in MODELS:
        return dict(MODELS[target])

    return dict(MODELS["auto"])

def choose_model(auto: Optional[str] = None) -> Dict[str, str]:
    cfg = parse_model_target(auto)
    info(f"Selected profile: {cfg['description']}")
    save_state({
        "model_key": auto or "auto",
        "model_repo": cfg.get("repo", HF_REPO),
        "model_name": cfg.get("name", "NAA-AI-Model"),
        "model_file": cfg.get("file", "model.safetensors.index.json"),
    })
    return cfg

def is_model_complete(model_path: Path, model_cfg: Dict[str, str] = None) -> bool:
    if not model_path.exists():
        return False
    if model_path.is_file():
        return model_path.stat().st_size > 1e6
    if (model_path / "model.safetensors.index.json").exists():
        try:
            with open(model_path / "model.safetensors.index.json", "r", encoding="utf-8") as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            shards = set(weight_map.values())
            if shards and all((model_path / s).exists() for s in shards):
                return True
        except Exception:
            pass
    if (model_path / "model.safetensors").exists():
        return True
    if list(model_path.glob("*.safetensors")) or list(model_path.glob("*.bin")) or list(model_path.glob("*.gguf")):
        return True
    return False

def download_model(model_cfg: Dict[str, str]) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    filename = model_cfg.get("file", "model.safetensors.index.json")
    repo_id = model_cfg.get("repo_gguf") or model_cfg.get("repo", HF_REPO)
    model_dir_name = model_cfg.get("dir", repo_id.split("/")[-1])

    if filename.endswith(".gguf"):
        dest_file = MODEL_DIR / filename
        if dest_file.exists() and dest_file.stat().st_size > 1e6:
            size_gb = dest_file.stat().st_size / 1e9
            ok(f"Model already present: {dest_file.name} ({size_gb:.2f} GB)")
            return dest_file

        header(f"Downloading GGUF Model ({filename}) from {repo_id}")
        try:
            from huggingface_hub import hf_hub_download
            kwargs = {
                "repo_id": repo_id,
                "filename": filename,
                "local_dir": str(MODEL_DIR),
                "local_dir_use_symlinks": False,
                "tqdm_class": NotebookProgressBar,
            }
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                kwargs["token"] = hf_token

            path = hf_hub_download(**kwargs)
            size_gb = Path(path).stat().st_size / 1e9
            ok(f"Downloaded model to: {Path(path).name} ({size_gb:.2f} GB)")
            return Path(path)
        except Exception as e:
            err(f"huggingface hf_hub_download failed: {e}")
            sys.exit(1)

    dest_dir = MODEL_DIR / model_dir_name
    if is_model_complete(dest_dir, model_cfg):
        ok(f"Model already present and complete at: {dest_dir}")
        return dest_dir

    header(f"Downloading Model from HuggingFace ({repo_id})")
    try:
        from huggingface_hub import snapshot_download
        kwargs = {
            "repo_id": repo_id,
            "local_dir": str(dest_dir),
            "local_dir_use_symlinks": False,
            "tqdm_class": NotebookProgressBar,
        }
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            kwargs["token"] = hf_token

        path = snapshot_download(**kwargs)
        ok(f"Downloaded model to: {dest_dir}")
        return Path(dest_dir)
    except Exception as e:
        err(f"huggingface snapshot_download failed: {e}")
        sys.exit(1)

def start_server(
    model_path: Path,
    admin_key: str,
    model_cfg: Dict[str, str],
    preset: str = "default",
    system_prompt: Optional[str] = None,
    llm_enabled: bool = True,
    visual_enabled: bool = True,
    **_extra: Any,
) -> bool:
    global _server_proc
    
    mod = sys.modules.get("naa")
    if mod and hasattr(mod, "_server_proc"):
        current_proc = getattr(mod, "_server_proc")
        if current_proc is not None:
            try:
                if current_proc.poll() is None:
                    current_proc.terminate()
                    try: current_proc.wait(timeout=5)
                    except Exception: current_proc.kill()
            except Exception: pass

    if _server_proc is not None:
        try:
            if _server_proc.poll() is None:
                _server_proc.terminate()
                try: _server_proc.wait(timeout=5)
                except Exception: _server_proc.kill()
        except Exception: pass

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    pythonpath = str(PROJECT_ROOT)
    if "PYTHONPATH" in env:
        pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"

    model_name = model_cfg.get("name", model_path.name)

    env.update({
        "NAA_MODEL_PATH": str(model_path),
        "NAA_MODEL_NAME": model_name,
        "NAA_ADMIN_KEY": admin_key,
        "NAA_KEYS_FILE": str(KEYS_FILE),
        "PORT": str(PORT),
        "NAA_PORT": str(PORT),
        "NAA_QUANT": str(model_cfg.get("quant", "auto")),
        "NAA_PRESET": preset,
        "NAA_LLM": "1" if llm_enabled else "0",
        "NAA_VISUAL": "1" if visual_enabled else "0",
        "PYTHONPATH": pythonpath,
    })
    if system_prompt:
        env["NAA_SYSTEM_PROMPT"] = system_prompt
    # cmd_start already resolved NAA_LLM / NAA_VISUAL explicitly (including the
    # legacy NAA_VIDEO_ONLY alias), so drop the legacy var here — otherwise a
    # lingering NAA_VIDEO_ONLY=1 in the parent env would override explicit
    # flags inside the child's Settings (see src/config.py).
    env.pop("NAA_VIDEO_ONLY", None)

    # Keep only the current child run so a native crash's root error is not
    # pushed out of the diagnostic tail by several restart banners.
    log_handle = open(SERVER_LOG, "w", encoding="utf-8")
    log_handle.write(f"========== start @ {time.strftime('%Y-%m-%d %H:%M:%S')} ==========\n")
    log_handle.flush()

    proc = subprocess.Popen(
        [sys.executable, "-m", "src.server.app"],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    _server_proc = proc
    if mod:
        setattr(mod, "_server_proc", proc)

    info(f"Server starting (PID {proc.pid})...")
    wait_fn = _get_attr("wait_for_port", wait_for_port)
    return wait_fn(PORT, timeout=180, proc=proc)


def _server_log_tail(limit: int = 30) -> list[str]:
    """Read a bounded diagnostic tail from the child server log."""

    try:
        if Path(SERVER_LOG).exists():
            return Path(SERVER_LOG).read_text(
                encoding="utf-8",
                errors="replace",
            ).strip().splitlines()[-max(1, limit):]
    except Exception:
        pass
    return []


def _print_server_log_tail(limit: int = 30) -> None:
    lines = _server_log_tail(limit)
    if not lines:
        return
    print("\n--- LAST SERVER LOG OUTPUT ---")
    for line in lines:
        print(f"  {line}")
    print("------------------------------\n")


def _apply_gguf_memory_recovery(
    model_path: Path,
    attempt: int,
    return_code: Optional[int] = None,
) -> bool:
    """Stage a safer llama.cpp profile after a hard GGUF server crash."""

    enabled = os.environ.get("NAA_AUTO_MEMORY_RECOVERY", "1").lower() in (
        "1", "true", "yes",
    )
    if not enabled or Path(model_path).suffix.lower() != ".gguf":
        return False

    changes = []
    native_abort = return_code in (-4, -6, -11)
    if native_abort:
        # SIGILL/SIGABRT/SIGSEGV inside libggml-cuda indicates a native kernel
        # or backend compatibility fault, not ordinary VRAM exhaustion.  Keep
        # the next boot conservative while the supervisor rebuilds the backend.
        if os.environ.get("NAA_FLASH_ATTN", "1").lower() not in ("0", "false", "no"):
            os.environ["NAA_FLASH_ATTN"] = "0"
            changes.append("Flash Attention disabled")
        for key in ("NAA_CACHE_TYPE_K", "NAA_CACHE_TYPE_V"):
            if os.environ.pop(key, None) is not None:
                changes.append(f"{key}=f16(default)")
        if changes:
            warn("Applying native-backend recovery: " + ", ".join(changes))
        # Fall through to also apply GPU-layer / context reductions so the
        # next restart uses a genuinely smaller memory footprint.

    try:
        context = int(os.environ.get("NAA_CTX", str(settings.max_context)))
    except ValueError:
        context = settings.max_context
    if context > 8192:
        os.environ["NAA_CTX"] = "8192"
        changes.append(f"context {context}->8192")

    try:
        gpu_layers = int(
            os.environ.get("NAA_N_GPU_LAYERS", str(settings.n_gpu_layers))
        )
    except ValueError:
        gpu_layers = settings.n_gpu_layers

    # A 13+ GB GGUF with every layer offloaded leaves too little space for
    # context and compute buffers on a 15 GB T4.  Each subsequent hard crash
    # moves another eight layers to system RAM.  For native aborts, be more
    # aggressive: cap at 32 on the first crash, then reduce further each time.
    if native_abort:
        if gpu_layers < 0 or gpu_layers > 32:
            safer_layers = 32
        elif gpu_layers > 16:
            safer_layers = max(16, gpu_layers - 8)
        else:
            safer_layers = gpu_layers
    elif gpu_layers < 0:
        safer_layers = 48
    elif attempt > 1 and gpu_layers > 16:
        safer_layers = max(16, gpu_layers - 8)
    else:
        safer_layers = gpu_layers
    if safer_layers != gpu_layers:
        os.environ["NAA_N_GPU_LAYERS"] = str(safer_layers)
        changes.append(f"GPU layers {gpu_layers}->{safer_layers}")

    if changes:
        warn("Applying GGUF memory recovery: " + ", ".join(changes))
        return True
    return False

def api_call(method: str, path: str, admin_key: str, data: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    url = f"http://localhost:{PORT}{path}"
    req = urllib.request.Request(url, method=method.upper())
    req.add_header("Authorization", f"Bearer {admin_key}")
    req.add_header("Content-Type", "application/json")
    if data:
        req.data = json.dumps(data).encode()
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        err(str(e))
        return None

def _parse_cli_args(args: Optional[list]) -> Dict[str, Any]:
    res: Dict[str, Any] = {
        "model": None,
        "preset": None,
        "system_prompt": None,
        "llm": None,       # None = default on; True/False from --llm / --no-llm
        "visual": None,    # None = default on; True/False from --visual / --no-visual
        "visual_model": None,
    }
    if not args:
        return res

    i = 0
    while i < len(args):
        arg = args[i]
        low = arg.lower()
        if low == "--llm":
            res["llm"] = True
            i += 1
        elif low in ("--no-llm", "--no_llm"):
            res["llm"] = False
            i += 1
        elif low in ("--visual", "--video", "--wan"):
            res["visual"] = True
            i += 1
        elif low in ("--no-visual", "--no_visual", "--no-video", "--no_video"):
            res["visual"] = False
            i += 1
        elif low in ("--visual-model", "--visual_model", "--video-model", "--video_model") and i + 1 < len(args):
            res["visual_model"] = args[i + 1]
            i += 2
        elif low.startswith("--visual-model="):
            res["visual_model"] = arg.split("=", 1)[1]
            i += 1
        elif low.startswith("--visual_model="):
            res["visual_model"] = arg.split("=", 1)[1]
            i += 1
        elif low.startswith("--video-model="):
            res["visual_model"] = arg.split("=", 1)[1]
            i += 1
        elif low.startswith("--video_model="):
            res["visual_model"] = arg.split("=", 1)[1]
            i += 1
        elif low in ("--model", "-m") and i + 1 < len(args):
            res["model"] = args[i + 1]
            i += 2
        elif low.startswith("--model="):
            res["model"] = arg.split("=", 1)[1]
            i += 1
        elif low in ("--preset", "-p") and i + 1 < len(args):
            res["preset"] = args[i + 1]
            i += 2
        elif low.startswith("--preset="):
            res["preset"] = arg.split("=", 1)[1]
            i += 1
        elif low in ("--system-prompt", "-s") and i + 1 < len(args):
            res["system_prompt"] = args[i + 1]
            i += 2
        elif low.startswith("--system-prompt="):
            res["system_prompt"] = arg.split("=", 1)[1]
            i += 1
        elif low in ("uncensored", "default") and res["preset"] is None:
            res["preset"] = low
            i += 1
        elif not arg.startswith("-") and res["model"] is None:
            res["model"] = arg
            i += 1
        else:
            i += 1

    return res

def cmd_setup(args: list = None):
    parsed = _parse_cli_args(args)
    target_arg = parsed.get("model")
    print_banner(target_arg)
    header("Setup")
    choose_model_fn = _get_attr("choose_model", choose_model)
    download_model_fn = _get_attr("download_model", download_model)
    save_state_fn = _get_attr("save_state", save_state)
    ensure_gguf_deps_fn = _get_attr("ensure_gguf_deps", ensure_gguf_deps)

    model_cfg = choose_model_fn(auto=target_arg)
    _select_gguf_backend(model_cfg)
    install_deps()
    if model_cfg.get("file", "").endswith(".gguf") or model_cfg.get("quant", "").startswith("q"):
        if not ensure_gguf_deps_fn():
            err("GGUF backend setup failed; model startup was cancelled.")
            return

    model_path = download_model_fn(model_cfg)
    model_key = target_arg or "auto"
    save_state_fn({
        "model_path": str(model_path),
        "model_key": model_key,
        "model_name": model_cfg.get("name", model_path.name)
    })
    ok("Setup complete! Run: python naa.py start")

def cmd_start(args: list = None):
    global _tunnel_proc, _server_proc

    load_state_fn = _get_attr("load_state", load_state)
    save_state_fn = _get_attr("save_state", save_state)
    choose_model_fn = _get_attr("choose_model", choose_model)
    download_model_fn = _get_attr("download_model", download_model)
    start_server_fn = _get_attr("start_server", start_server)
    start_tunnel_fn = _get_attr("start_tunnel", start_tunnel)
    start_keepalive_fn = _get_attr("start_keepalive", start_keepalive)
    is_server_healthy_fn = _get_attr("_is_server_healthy", is_server_healthy)
    is_server_loading_fn = _get_attr("_is_server_loading", is_server_loading)
    public_health_ok_fn = _get_attr("_public_health_ok", public_health_ok)
    ensure_gguf_deps_fn = _get_attr("ensure_gguf_deps", ensure_gguf_deps)

    state = load_state_fn()
    parsed = _parse_cli_args(args)

    def _env_flag(name: str, default: bool) -> bool:
        raw = os.environ.get(name, "")
        if not raw:
            return default
        return raw.lower() not in ("0", "false", "no")

    visual_model_arg = parsed.get("visual_model")
    if visual_model_arg:
        from src.core.video_engine import resolve_video_model_id as _resolve_vid

        os.environ["NAA_VIDEO_MODEL_ID"] = _resolve_vid(visual_model_arg)
        settings.video_model_id = os.environ["NAA_VIDEO_MODEL_ID"]

    # Backend selection (`start` args):
    #   plain `start`                -> both backends (backward compatible)
    #   `start --llm`                -> LLM-only (chat/completions)
    #   `start --visual`             -> visual-only (Wan video, skips LLM download/load)
    #   `start --llm --visual`       -> both backends
    #   `start <model> --llm`        -> LLM-only with that model
    #   `start <model> --visual`     -> both (named LLM + visual)
    #   `--no-llm` / `--no-visual`   -> disable one side explicitly
    # Naming an LLM model alongside an explicit `--visual` counts as wanting
    # both; on its own it just configures the LLM and keeps the default.
    llm_flag = parsed.get("llm")
    visual_flag = parsed.get("visual")
    llm_explicit_model = bool(parsed.get("model"))
    has_visual_model = bool(visual_model_arg)

    if llm_flag is True or visual_flag is True or has_visual_model:
        # Closed world: only the backends explicitly asked for come up.
        # `--visual-model X` implies `--visual`. Naming an LLM model next
        # to an explicit `--visual` counts as wanting both.
        visual_enabled = (visual_flag is True) or has_visual_model
        if visual_flag is False:
            visual_enabled = False
        llm_enabled = (llm_flag is True) or (llm_explicit_model and visual_enabled)
        if llm_flag is False:
            llm_enabled = False
    else:
        legacy_visual_only = os.environ.get("NAA_VIDEO_ONLY", "").lower() in ("1", "true", "yes")
        llm_enabled = _env_flag("NAA_LLM", True) and llm_flag is not False and not legacy_visual_only
        visual_enabled = _env_flag("NAA_VISUAL", True) and visual_flag is not False

    if not llm_enabled and not visual_enabled:
        err("Nothing to start: both --llm and --visual are disabled.")
        return

    os.environ["NAA_LLM"] = "1" if llm_enabled else "0"
    os.environ["NAA_VISUAL"] = "1" if visual_enabled else "0"
    settings.llm_enabled = llm_enabled
    settings.visual_enabled = visual_enabled
    visual_only = visual_enabled and not llm_enabled

    model_arg = parsed.get("model")
    preset_arg = parsed.get("preset") or state.get("preset", settings.preset)
    system_prompt_arg = parsed.get("system_prompt") or state.get("system_prompt", settings.system_prompt)

    model_cfg: Dict[str, str] = {}
    model_path = Path(state.get("model_path") or settings.model_path)
    model_key = state.get("model_key") or "auto"
    active_name = "Wan2.2-Visual (LLM off)"

    if llm_enabled:
        model_key = model_arg or state.get("model_key") or "auto"
        model_path_str = state.get("model_path") if not model_arg else None

        model_cfg = choose_model_fn(auto=model_key)
        _select_gguf_backend(model_cfg)
        if model_cfg.get("file", "").endswith(".gguf") or model_cfg.get("quant", "").startswith("q"):
            if not ensure_gguf_deps_fn():
                err("GGUF backend setup failed; model startup was cancelled.")
                return

        if model_path_str and Path(model_path_str).exists() and is_model_complete(Path(model_path_str), model_cfg):
            model_path = Path(model_path_str)
        else:
            filename = model_cfg.get("file", "model.safetensors.index.json")
            if filename.endswith(".gguf"):
                expected_path = MODEL_DIR / filename
            else:
                model_dir_name = model_cfg.get("dir", model_cfg.get("name", "model"))
                expected_path = MODEL_DIR / model_dir_name

            if is_model_complete(expected_path, model_cfg):
                model_path = expected_path
            else:
                model_path = download_model_fn(model_cfg)

        active_name = model_cfg.get("name", model_path.name)
        print_banner(active_name)
    else:
        print_banner("Wan2.2-Visual")

    admin_key = state.get("admin_key") or f"naa-{secrets.token_urlsafe(32)}"
    if not admin_key.startswith("naa-"):
        admin_key = f"naa-{admin_key}"
    save_state_fn({
        "admin_key": admin_key,
        "model_name": active_name,
        "model_key": model_key,
        "model_path": str(model_path),
        "preset": preset_arg,
        "llm_enabled": llm_enabled,
        "visual_enabled": visual_enabled,
        "visual_only": visual_only,
        "visual_model": os.environ.get("NAA_VIDEO_MODEL_ID", settings.video_model_id),
    })

    header("Starting NAA API Server")
    if llm_enabled:
        info(f"LLM:    {active_name} ({model_path.name})")
        info(f"Preset: {preset_arg}")
    else:
        info("LLM:    off (--visual)")
    if visual_enabled:
        info(f"Visual: Wan 2.2 ({os.environ.get('NAA_VIDEO_MODEL_ID', settings.video_model_id)})")
    else:
        info("Visual: off (--llm)")
    info(f"Port:   {PORT}")

    step(1, "Starting FastAPI server...")
    started = start_server_fn(model_path, admin_key, model_cfg, preset=preset_arg, system_prompt=system_prompt_arg, llm_enabled=llm_enabled, visual_enabled=visual_enabled)
    if not started:
        warn("Retrying server start...")
        _print_server_log_tail(30)
        if llm_enabled:
            _apply_gguf_memory_recovery(model_path, 1)
        time.sleep(3)
        started = start_server_fn(model_path, admin_key, model_cfg, preset=preset_arg, system_prompt=system_prompt_arg, llm_enabled=llm_enabled, visual_enabled=visual_enabled)
    
    if started:
        ok(f"Server listening on port {PORT}")
    else:
        err("Server failed to start.")
        _print_server_log_tail(60)
        return

    start_keepalive_fn(PORT)

    step(2, "Waiting for visual backend..." if visual_only else "Waiting for model to load into memory/VRAM...")
    start_wait = time.time()
    spinners = ["|", "/", "-", "\\"]
    spin_idx = 0
    loaded_ok = False

    while time.time() - start_wait < 600:
        elapsed = time.time() - start_wait
        spin = spinners[spin_idx % len(spinners)]
        spin_idx += 1

        # Check if server process exited unexpectedly
        current_server_proc = _get_attr("_server_proc", _server_proc)
        if current_server_proc is not None and hasattr(current_server_proc, "poll") and current_server_proc.poll() is not None:
            sys.stdout.write("\033[0m\n")
            sys.stdout.flush()
            err(f"Server process exited unexpectedly (code {current_server_proc.returncode}).")
            _print_server_log_tail(30)
            return

        # Check health using is_server_healthy_fn
        if is_server_healthy_fn(PORT):
            loaded_ok = True
            vram_info = ""
            try:
                import torch
                if torch.cuda.is_available():
                    free_b, total_b = torch.cuda.mem_get_info()
                    used_gb = (total_b - free_b) / (1024 ** 3)
                    vram_info = f" ({used_gb:.2f} GB VRAM allocated in {int(elapsed)}s)"
            except Exception:
                pass
            if not vram_info:
                vram_info = f" (in {int(elapsed)}s)"

            ready_msg = (
                "Visual backend ready for video generation!"
                if visual_only
                else "Model loaded and ready for inference!"
            )
            sys.stdout.write(f"\r\033[0m\033[32m[OK]   {ready_msg}{vram_info}\033[K\n")
            sys.stdout.flush()
            break

        # Query /health for detailed live VRAM stats if available
        health_data = None
        try:
            req = urllib.request.Request(f"http://localhost:{PORT}/health")
            with urllib.request.urlopen(req, timeout=1.0) as r:
                if r.status == 200:
                    health_data = json.loads(r.read())
        except Exception:
            pass

        if health_data:
            if health_data.get("load_error"):
                sys.stdout.write("\033[0m\n")
                sys.stdout.flush()
                err(f"Model load failed: {health_data['load_error']}")
                return

            gpu_name = health_data.get("gpu", "CPU")
            vram_used_mb = health_data.get("gpu_vram_used_mb", 0)
            vram_total_mb = health_data.get("gpu_vram_total_mb", 0)

            if vram_total_mb > 0:
                pct = min(100.0, (vram_used_mb / vram_total_mb) * 100)
                used_gb = vram_used_mb / 1024
                tot_gb = vram_total_mb / 1024
                col = "\033[33m" if pct < 50 else ("\033[36m" if pct < 85 else "\033[32m")
                bar_len = 16
                filled = int(bar_len * (pct / 100))
                bar = "=" * filled + (">" if filled < bar_len else "=") + " " * max(0, bar_len - filled - 1)
                if filled >= bar_len: bar = "=" * bar_len

                line = f"\r\033[0m{col}[LOADING {spin}] [{bar}] {pct:5.1f}%\033[0m | VRAM: {used_gb:4.1f}/{tot_gb:4.1f} GB ({gpu_name}) | Model: {active_name} | Elapsed: {int(elapsed)}s\033[K"
            else:
                line = f"\r\033[0m\033[33m[LOADING {spin}]\033[0m Loading weights into system memory... | Model: {active_name} | Elapsed: {int(elapsed)}s\033[K"

            sys.stdout.write(line)
            sys.stdout.flush()
        else:
            line = f"\r\033[0m\033[33m[LOADING {spin}]\033[0m Initializing inference process... | Elapsed: {int(elapsed)}s\033[K"
            sys.stdout.write(line)
            sys.stdout.flush()

        time.sleep(0.3)

    if not loaded_ok:
        sys.stdout.write("\033[0m\n")
        sys.stdout.flush()
        err("Model loading timed out after 600s.")
        return

    step(3, "Starting public HTTPS tunnel...")
    public_url = None
    for _ in range(3):
        proc, candidate = start_tunnel_fn(PORT)
        if candidate:
            public_url = candidate
            _tunnel_proc = proc
            mod = sys.modules.get("naa")
            if mod:
                setattr(mod, "_tunnel_proc", proc)
            break
        time.sleep(5)

    if public_url:
        save_state_fn({"public_url": public_url})
        ok(f"Tunnel URL: {public_url}")
        if "trycloudflare.com" in public_url:
            warn(
                "Quick Tunnels do not support SSE. Basic API calls may work, "
                "but OpenCode and Claude Code require a named Cloudflare Tunnel."
            )
        elif "lhr.life" in public_url or "localhost.run" in public_url:
            warn(
                "localhost.run is free and supports the agent stream, but its "
                "anonymous URL may change after a reconnect."
            )
        elif "ngrok" in public_url:
            ok("Ngrok tunnel active (full SSE streaming & agentic support).")
    else:
        err("Public HTTPS tunnel failed. The API is LOCAL ONLY inside this notebook.")
        tunnel_lines = tunnel_log_tail(20)
        if tunnel_lines:
            print("\n--- LAST TUNNEL LOG OUTPUT ---")
            for line in tunnel_lines:
                print(f"  {line}")
            print("------------------------------\n")
        warn("Do not put http://localhost:8000 into OpenCode on another computer.")
        public_url = f"http://localhost:{PORT}"

    if public_url.startswith("http") and not public_url.startswith("http://localhost"):
        step(4, "Smoke-testing public tunnel URL...")
        if public_health_ok_fn(public_url, timeout=15):
            ok("Public URL is reachable through the tunnel.")
        else:
            warn("Public URL is not yet reachable through the tunnel.")

    docs_url = f"{public_url}/docs"
    api_base = f"{public_url}/v1"
    inner_w = max(58, max(len(public_url), len(admin_key), len(docs_url), len(api_base)) + 14)
    print("\n  +" + "-" * inner_w + "+")
    print(f"  |  {'NAA (Notebooks AI API) is LIVE':<{inner_w-3}} |")
    if llm_enabled:
        print(f"  |  LLM:       {active_name:<{inner_w-14}} |")
    if visual_enabled:
        print(f"  |  Visual:    {os.environ.get('NAA_VIDEO_MODEL_ID', settings.video_model_id):<{inner_w-14}} |")
    print(f"  |  URL:       {public_url:<{inner_w-14}} |")
    print(f"  |  API Base:  {api_base:<{inner_w-14}} |")
    print(f"  |  Admin key: {admin_key:<{inner_w-14}} |")
    print(f"  |  Docs:      {docs_url:<{inner_w-14}} |")
    print("  +" + "-" * inner_w + "+\n")

    # Watchdog loop
    proc_restart_attempts = 0
    health_miss_streak = 0
    try:
        while True:
            time.sleep(30)
            current_server_proc = _get_attr("_server_proc", _server_proc)
            
            server_alive = current_server_proc is not None and current_server_proc.poll() is None
            if not server_alive:
                return_code = getattr(current_server_proc, "returncode", None)
                warn(f"Server died (exit code {return_code})! Restarting...")
                _print_server_log_tail(40)
                if current_server_proc:
                    try:
                        current_server_proc.terminate()
                        current_server_proc.wait(timeout=5)
                    except Exception: pass
                proc_restart_attempts += 1
                if llm_enabled and return_code in (-4, -6, -11):
                    _repair_gguf_cuda_backend()
                if llm_enabled:
                    _apply_gguf_memory_recovery(
                        model_path,
                        proc_restart_attempts,
                        return_code=return_code,
                    )
                if proc_restart_attempts > 5:
                    err("Server crashed repeatedly; stopping instead of hiding an endless restart loop.")
                    current_tunnel_proc = _get_attr("_tunnel_proc", _tunnel_proc)
                    if current_tunnel_proc:
                        try: current_tunnel_proc.terminate()
                        except Exception: pass
                    _print_server_log_tail(80)
                    return
                time.sleep(min(30, 2 ** proc_restart_attempts))
                start_server_fn(
                    model_path,
                    admin_key,
                    model_cfg,
                    preset=preset_arg,
                    system_prompt=system_prompt_arg,
                    llm_enabled=llm_enabled,
                    visual_enabled=visual_enabled,
                )

            if is_server_healthy_fn(PORT):
                health_miss_streak = 0
                if proc_restart_attempts:
                    ok("Server recovered and the model is ready again.")
                    proc_restart_attempts = 0
            elif is_server_loading_fn(PORT):
                # A cold model load can take several minutes.  Treat an alive
                # origin that reports "loading" as progress, not as a crash,
                # or the 90-second miss streak becomes an endless restart loop.
                health_miss_streak = 0
            else:
                health_miss_streak += 1
                if health_miss_streak >= 3:
                    warn("Sustained health failure — restarting server.")
                    proc_restart_attempts += 1
                    _print_server_log_tail(40)
                    if llm_enabled:
                        _apply_gguf_memory_recovery(model_path, proc_restart_attempts)
                    # Re-fetch the proc reference; it may have changed if the
                    # server was already restarted in this loop iteration.
                    stale_proc = _get_attr("_server_proc", _server_proc)
                    if stale_proc:
                        try:
                            stale_proc.terminate()
                            stale_proc.wait(5)
                        except Exception: pass
                    start_server_fn(
                        model_path,
                        admin_key,
                        model_cfg,
                        preset=preset_arg,
                        system_prompt=system_prompt_arg,
                        llm_enabled=llm_enabled,
                        visual_enabled=visual_enabled,
                    )
                    health_miss_streak = 0

            current_tunnel_proc = _get_attr("_tunnel_proc", _tunnel_proc)
            tunnel_alive = current_tunnel_proc is not None and current_tunnel_proc.poll() is None
            if not tunnel_alive:
                warn("Tunnel died! Restarting...")
                proc, new_url = start_tunnel_fn(PORT)
                if new_url:
                    previous_url = public_url
                    public_url = new_url
                    _tunnel_proc = proc
                    mod = sys.modules.get("naa")
                    if mod:
                        setattr(mod, "_tunnel_proc", proc)
                    save_state_fn({"public_url": new_url})
                    ok(f"Tunnel restarted: {new_url}")
                    if new_url == previous_url:
                        ok("The public API URL stayed the same.")
                    elif "trycloudflare.com" in new_url:
                        warn("Quick Tunnel restart changed the public URL; update the client.")
                    elif "lhr.life" in new_url or "localhost.run" in new_url:
                        warn(f"Free tunnel URL changed. Update the client API base to: {new_url}/v1")
                        info(f"Admin key is unchanged: {admin_key}")
    except KeyboardInterrupt:
        info("Shutting down NAA...")
        current_server_proc = _get_attr("_server_proc", _server_proc)
        current_tunnel_proc = _get_attr("_tunnel_proc", _tunnel_proc)
        if current_server_proc:
            try: current_server_proc.terminate()
            except Exception: pass
        if current_tunnel_proc:
            try: current_tunnel_proc.terminate()
            except Exception: pass
        ok("Stopped.")

def cmd_keys(args: list = None):
    print_banner()
    state = load_state()
    admin_key = state.get("admin_key")
    if not admin_key:
        err("No admin key found. Run start first.")
        return

    if not sys.stdin.isatty():
        err("Interactive key management requires a TTY.")
        return

    header("API Key Manager")
    def do_list():
        data = api_call("GET", "/v1/admin/keys/list", admin_key)
        if not data: return
        print(f"  {'NAME':<20} {'ROLE':<8} {'RPM':<6} {'KEY'}")
        rule()
        for k in data.get("keys", []):
            print(f"  {k.get('name',''):<20} {k.get('role',''):<8} {k.get('rpm',30):<6} {k.get('key','')}")

    def do_create():
        name = input("  Key name: ").strip()
        if not name: return
        data = api_call("POST", "/v1/admin/keys/create", admin_key, {"name": name, "role": "user", "rpm": 30})
        if data and data.get("success"):
            ok(f"Key created: {data['key']['key']}")

    while True:
        print("\n  [1] List keys\n  [2] Create key\n  [0] Exit\n")
        try: choice = input("  Choice: ").strip()
        except EOFError: break
        if choice == "1": do_list()
        elif choice == "2": do_create()
        elif choice == "0": break

def cmd_status(args: list = None):
    print_banner()
    state = load_state()
    admin_key = state.get("admin_key")
    header("Status")
    print(f"  Service:   NAA (Notebooks AI API)")
    print(f"  URL:       {state.get('public_url', 'Not started')}")
    print(f"  Admin Key: {admin_key or 'Not set'}")
    llm_on = state.get("llm_enabled", True)
    visual_on = state.get("visual_enabled", True)
    if llm_on:
        print(f"  LLM:       {state.get('model_name', MODEL_NAME)} ({state.get('model_key', 'auto')})")
    else:
        print(f"  LLM:       off (visual-only)")
    if visual_on:
        print(f"  Visual:    {state.get('visual_model', settings.video_model_id)}")
    else:
        print(f"  Visual:    off (LLM-only)")
    if admin_key:
        data = api_call("GET", "/health", admin_key)
        if data:
            print(f"  Server:    Running (Model: {'Loaded' if data.get('model_loaded') else 'Loading'})")
            print(f"  GPU:       {data.get('gpu', 'N/A')}")
        else:
            print(f"  Server:    Not running")

def cmd_video(args: list = None):
    """Manage Wan 2.2 video generation: setup deps/model or submit a job."""
    print_banner("Wan2.2-Video")
    sub = (args or ["help"])[0].lower() if args else "help"
    rest = (args or [])[1:]

    if sub in ("setup", "install"):
        install_video_deps()
        model = None
        for i, tok in enumerate(rest):
            if tok in ("--model", "-m") and i + 1 < len(rest):
                model = rest[i + 1]
            elif tok.startswith("--model="):
                model = tok.split("=", 1)[1]
        download_video_model(model)
        ok("Video setup complete! The server exposes POST /v1/videos/generations (T2V + I2V).")
    elif sub in ("generate", "gen", "create"):
        import json as _json

        state = load_state()
        admin_key = state.get("admin_key")
        if not admin_key:
            err("No admin key found. Run start first.")
            return
        prompt = None
        image = None
        i = 0
        while i < len(rest):
            tok = rest[i]
            if tok in ("--prompt", "-p") and i + 1 < len(rest):
                prompt = rest[i + 1]
                i += 2
            elif tok.startswith("--prompt="):
                prompt = tok.split("=", 1)[1]
                i += 1
            elif tok in ("--image", "-i") and i + 1 < len(rest):
                image = rest[i + 1]
                i += 2
            elif tok.startswith("--image="):
                image = tok.split("=", 1)[1]
                i += 1
            else:
                i += 1
        if not prompt:
            err("Usage: python naa.py video generate --prompt \"...\" [--image path|url|base64]")
            return
        payload: Dict[str, Any] = {"prompt": prompt}
        if image:
            p = Path(image)
            if p.is_file():
                import base64 as _b64

                payload["image"] = _b64.b64encode(p.read_bytes()).decode()
            else:
                payload["image"] = image
        data = api_call("POST", "/v1/videos/generations", admin_key, payload)
        if data:
            print(_json.dumps(data, indent=2))
            mode = "image-to-video" if image else "text-to-video"
            ok(f"Submitted {mode} job {data.get('id')}. Poll: GET /v1/videos/{data.get('id')}")
    elif sub in ("start", "serve", "up"):
        # Boot the server with the visual backend (`start --visual`, plus any
        # extra flags the user passed, e.g. `video start --visual-model ...`).
        # Forward the full rest (not just dash-tokens) so `--visual-model X`
        # keeps its value; ensure an explicit --visual is present.
        target = list(rest) if rest else ["--visual"]
        lowered = [t.lower() for t in target]
        if not any(t in ("--visual", "--video", "--wan") for t in lowered):
            target = ["--visual"] + target
        info(f"Starting server (equivalent to: python naa.py start {' '.join(target)})...")
        cmd_start(target)
    else:
        print("Usage:")
        print("  python naa.py video setup [--model Wan-AI/Wan2.2-TI2V-5B-Diffusers]")
        print("  python naa.py video start                       # visual-only server (no LLM)")
        print("  python naa.py video generate --prompt \"...\" [--image path|url|base64]")
        print("")
        print("  Tip: `python naa.py start --visual` boots visual-only;")
        print("       `python naa.py start --llm` boots LLM-only (add --model ... for the LLM).")


COMMANDS = {
    "setup": cmd_setup,
    "start": cmd_start,
    "keys": cmd_keys,
    "status": cmd_status,
    "video": cmd_video,
}

def main():
    args = sys.argv[1:]
    if not args:
        cmd_start()
    elif args[0] in COMMANDS:
        COMMANDS[args[0]](args[1:])
    else:
        print("Usage: python naa.py [setup|start|keys|status|video]")

if __name__ == "__main__":
    main()
