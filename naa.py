#!/usr/bin/env python3
"""
NAA (Notebooks AI API) - Universal OpenAI-Compatible REST API for Kaggle & Colab
"""

import sys
from pathlib import Path

# Ensure root directory is in sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.cli as cli
from src.config import (
    ENV,
    WORK_DIR,
    MODEL_DIR,
    KEYS_FILE,
    SERVER_LOG,
    STATE_FILE,
    PORT,
    QUIET,
    HF_REPO,
    MODEL_NAME,
    MODELS,
    detect_env,
)
from src.cli import (
    header,
    info,
    ok,
    warn,
    err,
    step,
    rule,
    print_banner,
    save_state,
    load_state,
    install_deps,
    choose_model,
    download_model,
    start_server,
    cmd_setup,
    cmd_start,
    cmd_keys,
    cmd_status,
    cmd_video,
    download_video_model,
    install_video_deps,
    main,
)
from src.tunnel.cloudflare import (
    start_tunnel,
    download_cloudflared as _download_cloudflared,
)
from src.supervisor.watchdog import (
    start_keepalive,
    is_server_healthy as _is_server_healthy,
    is_server_loading as _is_server_loading,
    public_health_ok as _public_health_ok,
    wait_for_port,
)

# Reference module variables for compatibility with test monkeypatching
_server_proc = None
_tunnel_proc = None

if __name__ == "__main__":
    sys.stdout.write("\033[0m")
    sys.stdout.flush()
    main()
