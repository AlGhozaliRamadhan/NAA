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
    assert "NAA_CACHE_TYPE_K" not in cli.os.environ
    assert "NAA_CACHE_TYPE_V" not in cli.os.environ

    changed_again = cli._apply_gguf_memory_recovery(Path("large-model.gguf"), 2)

    assert changed_again is True
    assert cli.os.environ["NAA_N_GPU_LAYERS"] == "40"


def test_memory_recovery_does_not_change_non_gguf(monkeypatch):
    monkeypatch.setenv("NAA_AUTO_MEMORY_RECOVERY", "1")
    monkeypatch.setenv("NAA_CTX", "16384")

    assert cli._apply_gguf_memory_recovery(Path("model.safetensors"), 1) is False
    assert cli.os.environ["NAA_CTX"] == "16384"


def test_native_abort_disables_unstable_cuda_options(monkeypatch):
    monkeypatch.setenv("NAA_AUTO_MEMORY_RECOVERY", "1")
    monkeypatch.setenv("NAA_FLASH_ATTN", "1")
    monkeypatch.setenv("NAA_CACHE_TYPE_K", "q8_0")
    monkeypatch.setenv("NAA_CACHE_TYPE_V", "q8_0")
    monkeypatch.setenv("NAA_N_GPU_LAYERS", "48")

    changed = cli._apply_gguf_memory_recovery(
        Path("large-model.gguf"),
        1,
        return_code=-6,
    )

    assert changed is True
    assert cli.os.environ["NAA_FLASH_ATTN"] == "0"
    assert "NAA_CACHE_TYPE_K" not in cli.os.environ
    assert "NAA_CACHE_TYPE_V" not in cli.os.environ
    assert cli.os.environ["NAA_N_GPU_LAYERS"] == "48"


def test_gpu_source_build_targets_detected_compute_capability(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "_cuda_compute_arch", lambda: "75")

    def fake_run_pip(packages, extra_args=None, env=None):
        captured["packages"] = packages
        captured["extra_args"] = extra_args
        captured["env"] = env
        return True

    monkeypatch.setattr(cli, "run_pip", fake_run_pip)
    monkeypatch.setattr(cli, "WORK_DIR", cli.Path("missing-test-work-dir"))

    assert cli._install_llama_cpp_cuda(force_source=True) is True
    assert "-DCMAKE_CUDA_ARCHITECTURES=75" in captured["env"]["CMAKE_ARGS"]
    assert "--no-binary=llama-cpp-python" in captured["extra_args"]
