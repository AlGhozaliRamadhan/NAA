"""SDXL still-image backend for NAA.

Backs the OpenAI-compatible ``POST /v1/images/generations`` route. Loads
RunDiffusion/Juggernaut-XL-v9 (single-file checkpoint) with
``StableDiffusionXLPipeline`` in float16 and ``safety_checker=None``.

Wan 2.2 was a poor still-image substitute — 5B params, BF16 weights, ~10 GB
VRAM just to render one frame, and on T4/Kaggle-T4 the offload path was
broken. Juggernaut-XL v9 is ~7 GB on disk, runs at float16 on a T4, and
generates a 1024×1024 image in ~5 s.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("naa-image")

DEFAULT_IMAGE_MODEL_ID = os.environ.get(
    "NAA_IMAGE_MODEL_ID", "RunDiffusion/Juggernaut-XL-v9"
)
# Where to grab the single-file checkpoint. Defaults to the RunDiffusion
# HuggingFace repo (contains Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors).
DEFAULT_IMAGE_CHECKPOINT_FILE = os.environ.get(
    "NAA_IMAGE_CHECKPOINT_FILE",
    "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
)
DEFAULT_IMAGE_STEPS = int(os.environ.get("NAA_IMAGE_STEPS", "25"))
DEFAULT_IMAGE_GUIDANCE = float(os.environ.get("NAA_IMAGE_GUIDANCE", "5.0"))
DEFAULT_IMAGE_WIDTH = int(os.environ.get("NAA_IMAGE_WIDTH", "1024"))
DEFAULT_IMAGE_HEIGHT = int(os.environ.get("NAA_IMAGE_HEIGHT", "1024"))
DEFAULT_IMAGE_DTYPE = os.environ.get("NAA_IMAGE_DTYPE", "float16")

# Allowed SDXL aspect buckets (divisible by 8, ≤ 1536 on the long side).
SIZE_LIMITS = (256, 256, 2048, 1536)  # min_w, min_h, max_w, max_h


def _mock_png(prompt: str = "", width: int = 256, height: int = 256) -> bytes:
    """Deterministic placeholder PNG used when diffusers/torch is unavailable."""
    import hashlib

    w = max(64, min(512, int(width) // 4 or 64))
    h = max(64, min(512, int(height) // 4 or 64))
    try:
        from PIL import Image

        digest = hashlib.md5(prompt.encode("utf-8")).digest()
        img = Image.new("RGB", (w, h), (digest[0], digest[1], digest[2]))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        # Minimal 1x1 RGB PNG fallback (PIL missing).
        import struct
        import zlib

        def chunk(ctype: bytes, data: bytes) -> bytes:
            c = ctype + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        raw = b"\x00\xff\x00\x00"
        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b"")
        )


@dataclass
class ImageJob:
    id: str
    prompt: str
    status: str = "queued"  # queued | running | succeeded | failed
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


class ImageEngine:
    """Lazy-loaded SDXL/Juggernaut backend.

    The pipeline is constructed on first request, not at server startup, so
    importing this module doesn't force a model download or GPU allocation.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_IMAGE_MODEL_ID,
        checkpoint_file: str = DEFAULT_IMAGE_CHECKPOINT_FILE,
        output_dir: Optional[str] = None,
        steps: int = DEFAULT_IMAGE_STEPS,
        guidance: float = DEFAULT_IMAGE_GUIDANCE,
        width: int = DEFAULT_IMAGE_WIDTH,
        height: int = DEFAULT_IMAGE_HEIGHT,
        dtype: str = DEFAULT_IMAGE_DTYPE,
    ):
        from src.config import WORK_DIR

        self.model_id = model_id
        self.checkpoint_file = checkpoint_file
        self.output_dir = Path(output_dir) if output_dir else (Path(WORK_DIR) / "images")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.steps = int(steps)
        self.guidance = float(guidance)
        self.width = int(width)
        self.height = int(height)
        self.dtype = dtype

        self._pipe: Optional[Any] = None
        self._mock_mode = False
        self._lock = threading.Lock()

    # -- pipeline ----------------------------------------------------------
    def _ensure_pipeline(self) -> Optional[Any]:
        if self._pipe is not None:
            return self._pipe
        if self._mock_mode:
            return None
        try:
            import torch
            from diffusers import StableDiffusionXLPipeline
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            logger.warning("diffusers/torch missing (%s); image backend in mock mode.", exc)
            self._mock_mode = True
            return None

        try:
            logger.info("Downloading SDXL checkpoint %s/%s", self.model_id, self.checkpoint_file)
            ckpt_path = hf_hub_download(
                repo_id=self.model_id,
                filename=self.checkpoint_file,
            )
            dtype = torch.float16 if self.dtype == "float16" else torch.bfloat16
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = StableDiffusionXLPipeline.from_single_file(
                ckpt_path,
                torch_dtype=dtype,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
                use_safetensors=True,
            )
            try:
                pipe.to(device)
            except Exception as exc:
                logger.warning("pipe.to(%s) failed: %s", device, exc)
            # Memory hygiene for T4 (15 GB) and similar.
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass  # xformers optional; SDPA is fine on a T4
            self._pipe = pipe
            logger.info("SDXL pipeline ready on %s (%s)", device, dtype)
            return pipe
        except Exception as exc:
            logger.error("Failed to load SDXL pipeline %s: %s", self.model_id, exc, exc_info=True)
            self._mock_mode = True
            return None

    # -- public API --------------------------------------------------------
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> bytes:
        """Render one still image and return raw PNG bytes."""
        import torch

        w = int(width) if width else self.width
        h = int(height) if height else self.height
        # SDXL requires dims divisible by 8; clamp to supported range.
        w = max(SIZE_LIMITS[0], min(SIZE_LIMITS[2], (w // 8) * 8))
        h = max(SIZE_LIMITS[1], min(SIZE_LIMITS[3], (h // 8) * 8))
        steps = int(num_inference_steps) if num_inference_steps else self.steps
        cfg = float(guidance_scale) if guidance_scale is not None else self.guidance

        pipe = self._ensure_pipeline()
        if pipe is None or self._mock_mode:
            return _mock_png(prompt, w, h)

        generator = None
        if seed is not None:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                generator = torch.Generator(device=device).manual_seed(int(seed))
            except Exception:
                generator = None

        call_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "width": w,
            "height": h,
            "num_inference_steps": steps,
            "guidance_scale": cfg,
            "generator": generator,
        }
        if negative_prompt:
            call_kwargs["negative_prompt"] = negative_prompt

        try:
            result = pipe(**call_kwargs)
            image = result.images[0]
        except Exception as exc:
            logger.error("SDXL generation failed: %s", exc, exc_info=True)
            raise

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    @property
    def is_mock(self) -> bool:
        return self._mock_mode


_engine: Optional[ImageEngine] = None
_engine_lock = threading.Lock()


def get_image_engine(**overrides: Any) -> ImageEngine:
    """Module-level singleton (lazy; safe to import without GPU deps)."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = ImageEngine(**overrides) if overrides else ImageEngine()
        return _engine
