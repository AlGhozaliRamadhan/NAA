"""
Tests for NAA Universal Model support, custom HuggingFace models, and uncensored presets.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

import naa
from src.config import Settings
from src.core.prompt import prepare_chat_messages, CANONICAL_SYSTEM_PROMPT
from src.core.engine import InferenceEngine
from src.server.app import create_app
from src.core.key_manager import APIKeyManager

def test_universal_model_preset_resolution():
    # 1. Clean passthrough when preset is default
    messages = [{"role": "user", "content": "How does photosynthesis work?"}]
    formatted = prepare_chat_messages(messages, preset="default")
    assert len(formatted) == 1
    assert formatted[0]["role"] == "user"

    # 2. Uncensored preset prepends deliberation directives
    formatted_uncensored = prepare_chat_messages(messages, preset="uncensored")
    assert len(formatted_uncensored) == 2
    assert formatted_uncensored[0]["role"] == "system"
    assert "EPISTEMIC RIGOR" in formatted_uncensored[0]["content"]

    # 3. Custom system prompt
    formatted_custom = prepare_chat_messages(messages, custom_system_prompt="You are a helpful coding assistant.")
    assert len(formatted_custom) == 2
    assert formatted_custom[0]["content"] == "You are a helpful coding assistant."

    # 4. User-provided system message is respected verbatim
    user_with_sys = [{"role": "system", "content": "Special Persona"}, {"role": "user", "content": "Hi"}]
    formatted_user = prepare_chat_messages(user_with_sys, preset="uncensored")
    assert len(formatted_user) == 2
    assert formatted_user[0]["content"] == "Special Persona"

def test_engine_dynamic_model_name():
    engine = InferenceEngine(model_path="models/Qwen2.5-7B-Instruct", model_name="Qwen2.5-7B-Instruct")
    assert engine.model_name == "Qwen2.5-7B-Instruct"

def test_naa_entrypoint_functions():
    assert hasattr(naa, "main")
    assert hasattr(naa, "cmd_start")
    assert hasattr(naa, "cmd_setup")
    assert hasattr(naa, "cmd_keys")
    assert hasattr(naa, "cmd_status")
