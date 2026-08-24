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
from src.tunnel.cloudflare import start_tunnel, download_cloudflared
from src.supervisor.watchdog import (
    start_keepalive,
    is_server_healthy,
    public_health_ok,
    wait_for_port,
)

_server_proc: Optional[subprocess.Popen] = None
_tunnel_proc: Optional[subprocess.Popen] = None

def _get_attr(name: str, fallback: Any) -> Any:
    mod = sys.modules.get("naa")
    if mod and hasattr(mod, name):
        return getattr(mod, name)
    return getattr(sys.modules.get(__name__), name, fallback)

def header(title: str):      print(f"\n--- {title.upper()} ---")
def info(msg: str):         print(f"[INFO] {msg}")
def ok(msg: str):           print(f"[OK]   {msg}")
def warn(msg: str):         print(f"[WARN] {msg}")
def err(msg: str):          print(f"[ERR]  {msg}")
def step(n: int, msg: str):   print(f"\n[{n}] {msg}")
def rule():                 print("-" * 60)

def print_banner(model_name: Optional[str] = None):
    active_model = model_name or MODEL_NAME
    gpu_info = f"GPU ({ENV['gpu_count']}x {ENV['gpu_name']})" if ENV['is_gpu'] else "CPU"
    env_label = f"{ENV['name'].upper()} - {gpu_info}"
    print("\n============================================================")
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

def run_pip(packages: list, extra_args: list = None) -> bool:
    cmd = [sys.executable, "-m", "pip", "install"] + (extra_args or []) + packages
    r = subprocess.run(cmd)
    return r.returncode == 0

def install_deps():
    header("Installing Dependencies")
    step(1, "Core server packages (fastapi, uvicorn, huggingface_hub, pydantic, etc.)")
    run_pip(["fastapi>=0.111.0", "uvicorn[standard]>=0.29.0", "python-multipart>=0.0.9", "huggingface_hub>=0.23.0", "pydantic>=2.0.0", "requests>=2.31.0"])

    step(2, "Inference engine (transformers, torch, accelerate, bitsandbytes, jinja2)")
    packages = ["transformers>=4.45.0", "accelerate>=0.28.0", "bitsandbytes>=0.43.0", "jinja2>=3.1.4"]
    run_pip(packages)
    ok("Dependencies ready")

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
    system_prompt: Optional[str] = None
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
        "PYTHONPATH": pythonpath,
    })
    if system_prompt:
        env["NAA_SYSTEM_PROMPT"] = system_prompt

    log_handle = open(SERVER_LOG, "a", encoding="utf-8")
    log_handle.write(f"\n\n========== restart @ {time.strftime('%Y-%m-%d %H:%M:%S')} ==========\n")
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

def cmd_setup(args: list = None):
    target_arg = args[0] if args else None
    print_banner(target_arg)
    header("Setup")
    install_deps()
    choose_model_fn = _get_attr("choose_model", choose_model)
    download_model_fn = _get_attr("download_model", download_model)
    save_state_fn = _get_attr("save_state", save_state)

    model_cfg = choose_model_fn(auto=target_arg)
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
    download_model_fn = _get_attr("download_model", download_model)
    start_server_fn = _get_attr("start_server", start_server)
    start_tunnel_fn = _get_attr("start_tunnel", start_tunnel)
    start_keepalive_fn = _get_attr("start_keepalive", start_keepalive)
    is_server_healthy_fn = _get_attr("_is_server_healthy", is_server_healthy)
    public_health_ok_fn = _get_attr("_public_health_ok", public_health_ok)

    state = load_state_fn()

    model_arg = args[0] if args else None
    preset_arg = "uncensored" if (args and "uncensored" in args) else state.get("preset", settings.preset)

    model_key = model_arg or state.get("model_key")
    model_path_str = state.get("model_path")

    if not model_key and not model_path_str:
        warn("No model configured. Running setup first...")
        cmd_setup()
        state = load_state_fn()
        model_key = state.get("model_key")
        model_path_str = state.get("model_path")

    if model_key and model_key in MODELS:
        model_cfg = MODELS[model_key]
    elif model_key and "/" in model_key:
        model_cfg = {"name": model_key.split("/")[-1], "repo": model_key, "dir": model_key.split("/")[-1], "quant": "auto"}
    else:
        model_cfg = list(MODELS.values())[0]

    model_path = Path(model_path_str) if model_path_str else MODEL_DIR / model_cfg.get("dir", "model")

    if not is_model_complete(model_path, model_cfg):
        model_path = download_model_fn(model_cfg)

    active_name = state.get("model_name", model_cfg.get("name", model_path.name))
    print_banner(active_name)

    admin_key = state.get("admin_key") or f"naa-{secrets.token_urlsafe(32)}"
    if not admin_key.startswith("naa-"):
        admin_key = f"naa-{admin_key}"
    save_state_fn({"admin_key": admin_key, "model_name": active_name})

    header("Starting NAA API Server")
    info(f"Model:  {active_name} ({model_path.name})")
    info(f"Preset: {preset_arg}")
    info(f"Port:   {PORT}")

    step(1, "Starting FastAPI server...")
    started = start_server_fn(model_path, admin_key, model_cfg, preset=preset_arg)
    if not started:
        warn("Retrying server start...")
        time.sleep(3)
        started = start_server_fn(model_path, admin_key, model_cfg, preset=preset_arg)
    
    if started:
        ok(f"Server listening on port {PORT}")
    else:
        err("Server failed to start.")
        if Path(SERVER_LOG).exists():
            log_tail = Path(SERVER_LOG).read_text(encoding="utf-8", errors="replace").strip().splitlines()[-20:]
            if log_tail:
                print("\n--- LAST SERVER LOG OUTPUT ---")
                for l in log_tail:
                    print(f"  {l}")
                print("------------------------------\n")
        return

    start_keepalive_fn(PORT)

    step(2, "Waiting for model to load into memory/VRAM...")
    start_wait = time.time()
    while time.time() - start_wait < 600:
        if is_server_healthy_fn(PORT):
            ok("Model loaded and ready for inference!")
            break
        time.sleep(3)

    step(3, "Starting Cloudflare Quick Tunnel...")
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
    else:
        public_url = f"http://localhost:{PORT}"

    if public_url.startswith("http") and not public_url.startswith("http://localhost"):
        step(4, "Smoke-testing public URL through Cloudflare...")
        if public_health_ok_fn(public_url, timeout=15):
            ok("Public URL is reachable through Cloudflare.")
        else:
            warn("Public URL not yet reachable through Cloudflare.")

    docs_url = f"{public_url}/docs"
    api_base = f"{public_url}/v1"
    inner_w = max(58, max(len(public_url), len(admin_key), len(docs_url), len(api_base)) + 14)
    print("\n  +" + "-" * inner_w + "+")
    print(f"  |  {'NAA (Notebooks AI API) is LIVE':<{inner_w-3}} |")
    print(f"  |  Model:     {active_name:<{inner_w-14}} |")
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
                warn("Server died! Restarting...")
                if current_server_proc:
                    try:
                        current_server_proc.terminate()
                        current_server_proc.wait(timeout=5)
                    except Exception: pass
                proc_restart_attempts += 1
                time.sleep(min(30, 2 ** proc_restart_attempts))
                if start_server_fn(model_path, admin_key, model_cfg, preset=preset_arg):
                    proc_restart_attempts = 0

            if is_server_healthy_fn(PORT):
                health_miss_streak = 0
            else:
                health_miss_streak += 1
                if health_miss_streak >= 3:
                    warn("Sustained health failure — restarting server.")
                    if current_server_proc:
                        try:
                            current_server_proc.terminate()
                            current_server_proc.wait(5)
                        except Exception: pass
                    if start_server_fn(model_path, admin_key, model_cfg, preset=preset_arg):
                        health_miss_streak = 0

            current_tunnel_proc = _get_attr("_tunnel_proc", _tunnel_proc)
            tunnel_alive = current_tunnel_proc is not None and current_tunnel_proc.poll() is None
            if not tunnel_alive:
                warn("Tunnel died! Restarting...")
                proc, new_url = start_tunnel_fn(PORT)
                if new_url:
                    _tunnel_proc = proc
                    mod = sys.modules.get("naa")
                    if mod:
                        setattr(mod, "_tunnel_proc", proc)
                    save_state_fn({"public_url": new_url})
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
    print(f"  Model:     {state.get('model_name', MODEL_NAME)} ({state.get('model_key', 'auto')})")
    if admin_key:
        data = api_call("GET", "/health", admin_key)
        if data:
            print(f"  Server:    Running (Model: {'Loaded' if data.get('model_loaded') else 'Loading'})")
            print(f"  GPU:       {data.get('gpu', 'N/A')}")
        else:
            print(f"  Server:    Not running")

COMMANDS = {
    "setup": cmd_setup,
    "start": cmd_start,
    "keys": cmd_keys,
    "status": cmd_status,
}

def main():
    args = sys.argv[1:]
    if not args:
        cmd_start()
    elif args[0] in COMMANDS:
        COMMANDS[args[0]](args[1:])
    else:
        print("Usage: python naa.py [setup|start|keys|status]")

if __name__ == "__main__":
    main()
