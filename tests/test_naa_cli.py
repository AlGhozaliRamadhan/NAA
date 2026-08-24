"""
Tests for NAA CLI environment detection, state persistence, model selection, and helper functions.
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

def test_state_persistence(tmp_path: Path, monkeypatch):
    test_state_file = tmp_path / ".test_naa_state.json"
    monkeypatch.setattr(config, "STATE_FILE", test_state_file)

    cli.save_state({"model_key": "4bit", "admin_key": "naa-test-cli-key"})
    assert test_state_file.exists()

    state = cli.load_state()
    assert state.get("model_key") == "4bit"
    assert state.get("admin_key") == "naa-test-cli-key"
