"""
Pytest configuration and shared fixtures for NAA test suite.
"""

import sys
import pytest
from pathlib import Path
from typing import Generator, Dict, Any
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings, MODEL_NAME
from src.core.key_manager import APIKeyManager
from src.core.engine import InferenceEngine
from src.server.app import create_app

class MockTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
        self._vocab = {
            "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
            "<|im_start|>": 10, "<|im_end|>": 11, "<think>": 12, "</think>": 13,
            "Hello": 20, " world": 21, "!": 22, "NAA": 23, "is": 24, "here": 25,
            "NdrFc": 30, "⊋": 31, "stop": 32
        }
        self._rev_vocab = {v: k for k, v in self._vocab.items()}

    def __call__(self, text: str, return_tensors: str = "pt", **kwargs):
        class MockTensorObj:
            def __init__(self, data):
                self.shape = (1, len(data))
                self._data = data
            def to(self, device):
                return self
            def __getitem__(self, idx):
                return self._data

        tokens = [self._vocab.get(w, 40) for w in text.split()]
        if not tokens:
            tokens = [1]
        return {
            "input_ids": MockTensorObj(tokens),
            "attention_mask": MockTensorObj([1] * len(tokens)),
        }

    def decode(self, token_ids, skip_special_tokens=False) -> str:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        
        words = []
        for tid in token_ids:
            w = self._rev_vocab.get(int(tid), f"w_{tid}")
            if skip_special_tokens and (w.startswith("<|") or w.startswith("<")):
                continue
            words.append(w)
        return " ".join(words)


class MockModel:
    def __init__(self, model_name: str = "NAA-AI-Model"):
        self.device = "cpu"
        self.model_name = model_name

    def create_chat_completion(self, messages, max_tokens=512, stream=False, **kwargs):
        if stream:
            def _gen():
                yield {"choices": [{"delta": {"role": "assistant", "content": "<think>\n"}, "finish_reason": None}]}
                yield {"choices": [{"delta": {"content": "Thinking process and verification.\n</think>\n"}, "finish_reason": None}]}
                yield {"choices": [{"delta": {"content": "Here is the verified answer."}, "finish_reason": None}]}
                yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
            return _gen()
        else:
            return {
                "id": "chatcmpl-mock-12345",
                "object": "chat.completion",
                "created": 1700000000,
                "model": self.model_name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "<think>\nThinking process and verification.\n</think>\nHere is the verified answer."
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35}
            }

    def create_completion(self, prompt, max_tokens=512, stream=False, **kwargs):
        if stream:
            def _gen():
                yield {"choices": [{"text": " Verified", "index": 0, "finish_reason": None}]}
                yield {"choices": [{"text": " completion text.", "index": 0, "finish_reason": None}]}
                yield {"choices": [{"text": "", "index": 0, "finish_reason": "stop"}]}
            return _gen()
        else:
            return {
                "id": "cmpl-mock-12345",
                "object": "text_completion",
                "created": 1700000000,
                "model": self.model_name,
                "choices": [{"text": " Verified completion text.", "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }

    def eval(self):
        return self

    def generate(self, **kwargs):
        input_ids = kwargs.get("input_ids")
        streamer = kwargs.get("streamer", None)
        gen_tokens = [23, 24, 25, 11]
        
        if streamer is not None:
            for tok in gen_tokens:
                text_chunk = f" tok_{tok}"
                if tok == 11:
                    text_chunk = "<|im_end|>"
                streamer.on_finalized_text(text_chunk)
            streamer.end()
            return None

        in_len = input_ids.shape[1] if hasattr(input_ids, "shape") else len(input_ids)
        full_tokens = [1] * in_len + gen_tokens
        class MockOutTensor:
            def __init__(self, t):
                self._t = t
            def __getitem__(self, idx):
                return self._t
        return MockOutTensor(full_tokens)


@pytest.fixture
def mock_engine() -> InferenceEngine:
    engine = InferenceEngine(model_path="models/mock", model_name="NAA-AI-Model", quant_mode="auto")
    engine.model = MockModel("NAA-AI-Model")
    engine.tokenizer = MockTokenizer()
    engine.model_loaded = True
    engine.model_loading = False
    engine.ready_event.set()
    return engine


@pytest.fixture
def isolated_key_manager(tmp_path: Path) -> APIKeyManager:
    keys_file = tmp_path / "test_keys.json"
    return APIKeyManager(str(keys_file), admin_key="naa-test-admin-key-123456789")


@pytest.fixture
def app_instance(mock_engine, isolated_key_manager):
    return create_app(engine=mock_engine, key_manager=isolated_key_manager, auto_load_model=False)


@pytest.fixture
def client(app_instance) -> TestClient:
    return TestClient(app_instance, raise_server_exceptions=False)


@pytest.fixture
def admin_headers() -> Dict[str, str]:
    return {"Authorization": "Bearer naa-test-admin-key-123456789"}


@pytest.fixture
def user_headers(isolated_key_manager) -> Dict[str, str]:
    record = isolated_key_manager.create_key(name="test-user", role="user", rate_limit_rpm=60)
    return {"Authorization": f"Bearer {record['key']}"}
