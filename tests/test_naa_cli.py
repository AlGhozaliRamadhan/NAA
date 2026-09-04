"""
Tests for NAA CLI environment detection, state persistence, model selection, URL parsing, and helper functions.
"""

from pathlib import Path
import src.config as config
import src.cli as cli

def test_detect_env():
    env = config.detect_env()
    assert "name" in env
    assert "is_kaggle" in env
    assert "is_colab" in env
    assert "is_gpu" in env
    assert "work_dir" in env
    assert "model_dir" in env

def test_choose_model():
    m_auto = cli.choose_model("auto")
    assert m_auto["quant"] == "auto"

    m_4bit = cli.choose_model("4bit")
    assert m_4bit["quant"] == "4bit"

    m_8bit = cli.choose_model("8bit")
    assert m_8bit["quant"] == "8bit"

    m_16bit = cli.choose_model("16bit")
    assert m_16bit["quant"] == "16bit"

    # Custom HuggingFace model repo
    m_custom = cli.choose_model("Qwen/Qwen2.5-7B-Instruct")
    assert m_custom["name"] == "Qwen2.5-7B-Instruct"
    assert m_custom["repo"] == "Qwen/Qwen2.5-7B-Instruct"

def test_parse_model_target_huggingface_url():
    url = "https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED/blob/main/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"
    cfg = cli.parse_model_target(url)
    assert cfg["name"] == "Qwen3.8-27B-OBLITERATED"
    assert cfg["repo"] == "OBLITERATUS/Qwen3.8-27B-OBLITERATED"
    assert cfg["file"] == "Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"
    assert cfg["quant"] == "q4_k_m"

def test_parse_model_target_colon_syntax():
    target = "OBLITERATUS/Qwen3.8-27B-OBLITERATED:Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"
    cfg = cli.parse_model_target(target)
    assert cfg["name"] == "Qwen3.8-27B-OBLITERATED"
    assert cfg["repo"] == "OBLITERATUS/Qwen3.8-27B-OBLITERATED"
    assert cfg["file"] == "Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"
    assert cfg["quant"] == "q4_k_m"

def test_parse_model_target_path_syntax():
    target = "OBLITERATUS/Qwen3.8-27B-OBLITERATED/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"
    cfg = cli.parse_model_target(target)
    assert cfg["name"] == "Qwen3.8-27B-OBLITERATED"
    assert cfg["repo"] == "OBLITERATUS/Qwen3.8-27B-OBLITERATED"
    assert cfg["file"] == "Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"

def test_state_persistence(tmp_path: Path, monkeypatch):
    test_state_file = tmp_path / ".test_naa_state.json"
    monkeypatch.setattr(config, "STATE_FILE", test_state_file)

    cli.save_state({"model_key": "4bit", "admin_key": "naa-test-cli-key"})
    assert test_state_file.exists()

    state = cli.load_state()
    assert state.get("model_key") == "4bit"
    assert state.get("admin_key") == "naa-test-cli-key"

def test_parse_cli_args():
    # Flag syntax: --model <url>
    url = "https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED/blob/main/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"
    res1 = cli._parse_cli_args(["--model", url])
    assert res1["model"] == url
    assert res1["preset"] is None

    # Flag syntax: -m <url> and --preset uncensored
    res2 = cli._parse_cli_args(["-m", url, "--preset", "uncensored"])
    assert res2["model"] == url
    assert res2["preset"] == "uncensored"

    # Equals syntax
    res3 = cli._parse_cli_args([f"--model={url}", "--preset=uncensored"])
    assert res3["model"] == url
    assert res3["preset"] == "uncensored"

    # Positional model with positional preset
    res4 = cli._parse_cli_args(["4bit", "uncensored"])
    assert res4["model"] == "4bit"
    assert res4["preset"] == "uncensored"

    # Uncensored flag only
    res5 = cli._parse_cli_args(["uncensored"])
    assert res5["model"] is None
    assert res5["preset"] == "uncensored"


def test_parse_cli_args_backend_flags():
    # Case-insensitive --LLM / --VISUAL (Kaggle cells use uppercase)
    res = cli._parse_cli_args(["--LLM"])
    assert res["llm"] is True
    assert res["visual"] is None

    res = cli._parse_cli_args(["--VISUAL"])
    assert res["visual"] is True

    res = cli._parse_cli_args(["--llm", "--visual"])
    assert res["llm"] is True
    assert res["visual"] is True

    res = cli._parse_cli_args(["--no-llm"])
    assert res["llm"] is False

    res = cli._parse_cli_args(["--no-visual"])
    assert res["visual"] is False

    # --visual-model implies visual backend + keeps its value
    res = cli._parse_cli_args(["--visual-model", "Wan-AI/Wan2.2-T2V-A14B-Diffusers"])
    assert res["visual_model"] == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    res = cli._parse_cli_args(["--video-model=wan2.2"])
    assert res["visual_model"] == "wan2.2"

    # --video / --wan are aliases for --visual
    assert cli._parse_cli_args(["--video"])["visual"] is True
    assert cli._parse_cli_args(["--wan"])["visual"] is True

    # LLM model + backend flag coexist (model configures LLM side)
    res = cli._parse_cli_args(["4bit", "--visual"])
    assert res["model"] == "4bit"
    assert res["visual"] is True

