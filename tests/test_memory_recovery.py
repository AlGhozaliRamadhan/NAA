"""GGUF watchdog memory-recovery regression tests."""

from pathlib import Path

from src import cli


def test_gguf_memory_recovery_reduces_tight_colab_profile(monkeypatch):
    monkeypatch.setenv("NAA_AUTO_MEMORY_RECOVERY", "1")
    monkeypatch.setenv("NAA_CTX", "16384")
    monkeypatch.setenv("NAA_N_GPU_LAYERS", "-1")
    monkeypatch.delenv("NAA_CACHE_TYPE_K", raising=False)
    monkeypatch.delenv("NAA_CACHE_TYPE_V", raising=False)

    changed = cli._apply_gguf_memory_recovery(Path("large-model.gguf"), 1)

    assert changed is True
    assert cli.os.environ["NAA_CTX"] == "8192"
    assert cli.os.environ["NAA_N_GPU_LAYERS"] == "48"
    assert cli.os.environ["NAA_CACHE_TYPE_K"] == "q8_0"
    assert cli.os.environ["NAA_CACHE_TYPE_V"] == "q8_0"

    changed_again = cli._apply_gguf_memory_recovery(Path("large-model.gguf"), 2)

    assert changed_again is True
    assert cli.os.environ["NAA_N_GPU_LAYERS"] == "40"


def test_memory_recovery_does_not_change_non_gguf(monkeypatch):
    monkeypatch.setenv("NAA_AUTO_MEMORY_RECOVERY", "1")
    monkeypatch.setenv("NAA_CTX", "16384")

    assert cli._apply_gguf_memory_recovery(Path("model.safetensors"), 1) is False
    assert cli.os.environ["NAA_CTX"] == "16384"
