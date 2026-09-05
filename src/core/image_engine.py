"""Still-image backend for NAA with runtime capability discovery.

Backs the OpenAI-compatible ``POST /v1/images/generations`` route. The
default checkpoint is RunDiffusion/Juggernaut-XL-v9 (single-file SDXL), with
an opt-in FLUX.1-dev-FP8 transformer backend — but what the backend can
*do* is never inferred from a model id, filename, quantization, or brand.
Capabilities (text-to-image, image edit, video, max reference images) are
discovered at runtime by inspecting the active runner's call signature and
adapter attributes (see :func:`discover_image_capabilities`).
"""

from __future__ import annotations

import base64
import inspect
import io
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("naa-image")


class ImageEditUnsupportedError(ValueError):
    """Raised when an edit is requested but the active runner has no image input."""

DEFAULT_IMAGE_MODEL_ID = os.environ.get(
    "NAA_IMAGE_MODEL_ID", "RunDiffusion/Juggernaut-XL-v9"
)
# Where to grab the single-file checkpoint. Defaults to the RunDiffusion
# HuggingFace repo (contains Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors).
DEFAULT_IMAGE_CHECKPOINT_FILE = os.environ.get(
    "NAA_IMAGE_CHECKPOINT_FILE",
    "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
)
# FLUX.1-dev-FP8 alternative (kept separate so the Juggernaut default never
# changes unless explicitly requested). The FP8 repo holds a single
# transformer checkpoint (~12 GB) loaded on top of the FLUX.1-dev pipeline.
DEFAULT_FLUX_MODEL_ID = os.environ.get(
    "NAA_FLUX_MODEL_ID", "black-forest-labs/FLUX.1-dev-FP8"
)
DEFAULT_FLUX_CHECKPOINT_FILE = os.environ.get(
    "NAA_FLUX_CHECKPOINT_FILE",
    "flux1-dev-fp8.safetensors",
)
# Base FLUX pipeline repo used for tokenizer/encoders/scheduler when running
# the FP8 transformer. Gated — requires HF_TOKEN with license accepted.
FLUX_BASE_REPO = os.environ.get("NAA_FLUX_BASE_REPO", "black-forest-labs/FLUX.1-dev")

# Friendly aliases so `--image-model flux` (or flux.1, flux-fp8, ...) selects
# the FLUX backend without typing the full HuggingFace repo id.
IMAGE_MODEL_ALIASES: Dict[str, str] = {
    "flux": DEFAULT_FLUX_MODEL_ID,
    "flux.1": DEFAULT_FLUX_MODEL_ID,
    "flux-dev": DEFAULT_FLUX_MODEL_ID,
    "flux.1-dev": DEFAULT_FLUX_MODEL_ID,
    "flux-fp8": DEFAULT_FLUX_MODEL_ID,
    "flux.1-dev-fp8": DEFAULT_FLUX_MODEL_ID,
    "flux-fp8-dev": DEFAULT_FLUX_MODEL_ID,
    "flux-dev-fp8": DEFAULT_FLUX_MODEL_ID,
    "juggernaut": DEFAULT_IMAGE_MODEL_ID,
    "juggernaut-xl": DEFAULT_IMAGE_MODEL_ID,
    "juggernaut-v9": DEFAULT_IMAGE_MODEL_ID,
    "juggernaut-xl-v9": DEFAULT_IMAGE_MODEL_ID,
    "sdxl": DEFAULT_IMAGE_MODEL_ID,
    "sdxl-juggernaut": DEFAULT_IMAGE_MODEL_ID,
}


def resolve_image_model_id(model: Optional[str]) -> str:
    """Resolve a friendly alias (``flux``) to a HuggingFace image model id."""
    if not model:
        return DEFAULT_IMAGE_MODEL_ID
    key = model.strip()
    return IMAGE_MODEL_ALIASES.get(key.lower(), key)


def is_flux_model(model_id: Optional[str]) -> bool:
    """True when the model id selects the FLUX.1 transformer backend."""
    return "flux" in (model_id or "").lower()


DEFAULT_FLUX_STEPS = int(os.environ.get("NAA_FLUX_STEPS", "28"))
DEFAULT_FLUX_GUIDANCE = float(os.environ.get("NAA_FLUX_GUIDANCE", "3.5"))
DEFAULT_IMAGE_STEPS = int(os.environ.get("NAA_IMAGE_STEPS", "25"))
DEFAULT_IMAGE_GUIDANCE = float(os.environ.get("NAA_IMAGE_GUIDANCE", "5.0"))
DEFAULT_IMAGE_WIDTH = int(os.environ.get("NAA_IMAGE_WIDTH", "1024"))
DEFAULT_IMAGE_HEIGHT = int(os.environ.get("NAA_IMAGE_HEIGHT", "1024"))
DEFAULT_IMAGE_DTYPE = os.environ.get("NAA_IMAGE_DTYPE", "float16")

# Allowed SDXL aspect buckets (divisible by 8, ≤ 1536 on the long side).
SIZE_LIMITS = (256, 256, 2048, 1536)  # min_w, min_h, max_w, max_h


def decode_reference_image(item: Any) -> bytes:
    """Decode one Cogito ``reference_images`` entry to raw image bytes.

    Accepts a mapping with ``b64_json`` (raw or data-URI base64) or a plain
    base64 string / http(s) URL passthrough string.
    """
    if isinstance(item, dict):
        payload = item.get("b64_json") or item.get("image") or item.get("data") or ""
    else:
        payload = item
    if not isinstance(payload, str) or not payload.strip():
        raise ImageEditUnsupportedError("Each reference image needs a 'b64_json' value.")
    text = payload.strip()
    if text.startswith("data:"):
        try:
            text = text.split(",", 1)[1]
        except IndexError:
            raise ImageEditUnsupportedError("Reference image data-URI is malformed.")
    if text.startswith("http://") or text.startswith("https://"):
        raise ImageEditUnsupportedError(
            "Reference image URLs are not fetched server-side; send b64_json instead."
        )
    try:
        return base64.b64decode(text, validate=True)
    except Exception:
        raise ImageEditUnsupportedError("Reference image 'b64_json' is not valid base64.")


def _reference_pil_images(pipe: Any, refs: List[bytes]) -> Any:
    """Convert decoded reference bytes to the runner's expected image input."""
    try:
        from PIL import Image
    except ImportError:
        return refs[0] if len(refs) == 1 else refs
    images = []
    for raw in refs:
        try:
            images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception as exc:
            raise ImageEditUnsupportedError(f"Could not decode a reference image: {exc}")
    # Single-input runners (diffusers img2img-style) take one image; adapters
    # declaring several reference slots take the list.
    limit = int(getattr(pipe, "max_reference_images", 1) or 1)
    if len(images) == 1:
        return images[0]
    return images if limit > 1 else images[0]


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
class ImageCapabilities:
    """Runtime-discovered capabilities of the active image runner.

    Built by :func:`discover_image_capabilities` from the live pipeline /
    adapter — never from a model id, filename, or brand string.
    """

    image_generation: bool = False
    image_edit: bool = False
    video_generation: bool = False
    max_reference_images: int = 0
    supported_sizes: List[str] = field(default_factory=list)
    supported_formats: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_generation": self.image_generation,
            "image_edit": self.image_edit,
            "video_generation": self.video_generation,
            "max_reference_images": self.max_reference_images,
            "supported_sizes": list(self.supported_sizes),
            "supported_formats": list(self.supported_formats),
        }


# Accepted runner call-signature parameters for an input image. Checked in
# preference order so adapters exposing several names map deterministically.
IMAGE_INPUT_PARAMS: Tuple[str, ...] = (
    "image_prompt",
    "image",
    "init_image",
    "source_image",
    "control_image",
    "mask_image",
)


def _runner_call_params(runner: Any) -> set:
    """Parameter names of the runner's ``__call__`` (empty set if unknown)."""
    call = getattr(runner, "__call__", None)
    if call is None:
        return set()
    try:
        return set(inspect.signature(call).parameters)
    except (TypeError, ValueError):
        return set()


def _adapter_image_inputs(runner: Any) -> Tuple[Optional[str], int, List[str]]:
    """Return ``(input_param, max_refs, forms)`` declared by the runner.

    Adapters that wrap non-diffusers workflows (ComfyUI, external providers)
    declare their image-input path via attributes instead of a call signature:

    - ``image_input_param`` (str): the request field an input image maps to.
    - ``max_reference_images`` (int): workflow reference-image limit.
    - ``image_input_forms`` (list[str]): accepted input forms, e.g.
      ``["image_prompt", "kontext"]``.
    """
    param = getattr(runner, "image_input_param", None)
    forms = list(getattr(runner, "image_input_forms", []) or [])
    try:
        max_refs = int(getattr(runner, "max_reference_images", 1) or 0)
    except (TypeError, ValueError):
        max_refs = 0
    if isinstance(param, str) and param:
        if param not in forms:
            forms = [param] + forms
        return param, max(0, max_refs), forms
    return None, 0, forms


def discover_image_capabilities(
    runner: Any,
    *,
    video_available: bool = False,
) -> ImageCapabilities:
    """Inspect the active image runner and report what it can actually do.

    - text-to-image: the runner's ``__call__`` accepts a ``prompt``.
    - image edit: ``__call__`` accepts one of :data:`IMAGE_INPUT_PARAMS`
      (plus ``prompt``), or the adapter declares ``image_input_param``.
    - video: only when the caller confirms a real video pipeline/endpoint
      exists (never inferred from the image runner or model id).
    """
    caps = ImageCapabilities(
        supported_sizes=["256x256", "512x512", "1024x1024", "1024x1536", "1536x1024"],
        supported_formats=["png"],
    )
    if runner is None:
        return caps
    params = _runner_call_params(runner)
    adapter_param, adapter_max, _forms = _adapter_image_inputs(runner)
    if "prompt" in params:
        caps.image_generation = True
    image_param: Optional[str] = None
    if "prompt" in params:
        for candidate in IMAGE_INPUT_PARAMS:
            if candidate in params:
                image_param = candidate
                break
        if image_param is None and adapter_param:
            image_param = adapter_param
    if image_param:
        caps.image_edit = True
        if adapter_param == image_param:
            caps.max_reference_images = max(0, adapter_max)
        else:
            try:
                declared = int(getattr(runner, "max_reference_images", 1) or 0)
            except (TypeError, ValueError):
                declared = 1
            caps.max_reference_images = max(1, declared)
    if video_available:
        caps.video_generation = True
    return caps


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
    """Lazy-loaded still-image backend (SDXL/Juggernaut default, FLUX optional).

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

        self.model_id = resolve_image_model_id(model_id)
        self.is_flux = is_flux_model(self.model_id)
        # Pick the matching default checkpoint/steps unless the caller pinned
        # one explicitly (env var or argument differing from the SDXL default).
        if self.is_flux:
            if checkpoint_file == DEFAULT_IMAGE_CHECKPOINT_FILE:
                checkpoint_file = DEFAULT_FLUX_CHECKPOINT_FILE
            if steps == DEFAULT_IMAGE_STEPS:
                steps = DEFAULT_FLUX_STEPS
            if guidance == DEFAULT_IMAGE_GUIDANCE:
                guidance = DEFAULT_FLUX_GUIDANCE
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
        self._caps = ImageCapabilities()
        self._caps_lock = threading.Lock()

    # -- capabilities ------------------------------------------------------
    @property
    def label(self) -> str:
        """Human-readable name of the loaded image backend."""
        return self.model_id.split("/")[-1] if self.model_id else "image"

    def refresh_capabilities(self, *, video_available: Optional[bool] = None) -> ImageCapabilities:
        """Re-discover capabilities from the active runner.

        Call at backend startup, model load, workflow change, or provider
        change — and after swapping the pipeline — so ``/v1/models`` always
        reflects what is actually loaded.
        """
        runner = self._pipe if self._pipe is not None else self
        if video_available is None:
            with self._caps_lock:
                video_available = self._caps.video_generation
        caps = discover_image_capabilities(runner, video_available=video_available)
        with self._caps_lock:
            self._caps = caps
        return caps

    def set_pipeline(self, pipe: Any, *, video_available: Optional[bool] = None) -> Any:
        """Hot-swap the active runner (model/workflow/provider change).

        Stores the new pipeline and recalculates capabilities so ``/v1/models``
        reflects the swap immediately.
        """
        with self._lock:
            self._pipe = pipe
            self._mock_mode = pipe is None
            if isinstance(pipe, ImageEngine):
                model_id = getattr(pipe, "model_id", None)
                if model_id:
                    self.model_id = resolve_image_model_id(model_id)
                    self.is_flux = is_flux_model(self.model_id)
            elif pipe is not None:
                model_id = getattr(pipe, "model_id", None) or getattr(pipe, "_model_id", None)
                if isinstance(model_id, str) and model_id:
                    self.model_id = resolve_image_model_id(model_id)
                    self.is_flux = is_flux_model(self.model_id)
        return self.refresh_capabilities(video_available=video_available)

    @property
    def capabilities(self) -> ImageCapabilities:
        with self._caps_lock:
            return self._caps

    def image_input_param(self) -> Optional[str]:
        """Name of the runner input a reference image maps to, if any."""
        runner = self._pipe if self._pipe is not None else self
        adapter_param, _, _ = _adapter_image_inputs(runner)
        params = _runner_call_params(runner)
        if "prompt" not in params:
            return None
        for candidate in IMAGE_INPUT_PARAMS:
            if candidate in params:
                return candidate
        return adapter_param

    def _store_pipeline(self, pipe: Any) -> Any:
        self._pipe = pipe
        self.refresh_capabilities()
        return pipe

    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> bytes:
        """Text-to-image call signature (no image input).

        Exists so :func:`discover_image_capabilities` sees ``prompt`` (and no
        image param) when no pipeline is loaded yet or the backend is in mock
        mode — i.e. text-to-image yes, edit no, inferred from the live
        object rather than any model id.
        """
        return self.generate(
            prompt,
            negative_prompt,
            width,
            height,
            num_inference_steps,
            guidance_scale,
            seed,
        )

    # -- pipeline ----------------------------------------------------------
    def _ensure_pipeline(self) -> Optional[Any]:
        if self._pipe is not None:
            return self._pipe
        if self._mock_mode:
            return None
        if self.is_flux:
            return self._ensure_flux_pipeline()
        try:
            import torch
            from diffusers import StableDiffusionXLPipeline
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            logger.warning("diffusers/torch missing (%s); image backend in mock mode.", exc)
            self._mock_mode = True
            self.refresh_capabilities()
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
            logger.info("SDXL pipeline ready on %s (%s)", device, dtype)
            return self._store_pipeline(pipe)
        except Exception as exc:
            logger.error("Failed to load SDXL pipeline %s: %s", self.model_id, exc, exc_info=True)
            self._mock_mode = True
            self.refresh_capabilities()
            return None

    def _ensure_flux_pipeline(self) -> Optional[Any]:
        """Load the FLUX.1-dev-FP8 transformer on top of the base pipeline."""
        try:
            import torch
            from diffusers import FluxPipeline, FluxTransformer2DModel
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            logger.warning("diffusers/torch missing (%s); image backend in mock mode.", exc)
            self._mock_mode = True
            self.refresh_capabilities()
            return None

        try:
            logger.info(
                "Loading FLUX.1-dev-FP8 transformer %s/%s",
                self.model_id, self.checkpoint_file,
            )
            ckpt_path = hf_hub_download(
                repo_id=self.model_id,
                filename=self.checkpoint_file,
            )
            # FP8 transformer weights run best in bfloat16 compute precision.
            pipe = FluxPipeline.from_pretrained(
                FLUX_BASE_REPO,
                transformer=FluxTransformer2DModel.from_single_file(
                    ckpt_path,
                    torch_dtype=torch.bfloat16,
                ),
                torch_dtype=torch.bfloat16,
            )
            try:
                # Keeps the 12B transformer usable on a T4 / Kaggle GPU.
                pipe.enable_model_cpu_offload()
            except Exception:
                try:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    pipe.to(device)
                except Exception as exc:
                    logger.warning("flux pipe.to failed: %s", exc)
            logger.info("FLUX.1-dev-FP8 pipeline ready")
            return self._store_pipeline(pipe)
        except Exception as exc:
            logger.error("Failed to load FLUX pipeline %s: %s", self.model_id, exc, exc_info=True)
            logger.warning(
                "FLUX.1-dev is a gated repo — set HF_TOKEN with the license "
                "accepted on HuggingFace, else the image backend stays in mock mode."
            )
            self._mock_mode = True
            self.refresh_capabilities()
            return None

    # -- public API --------------------------------------------------------
    def generate(  # noqa: C901 — branching maps the generic request to the runner
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        reference_images: Optional[List[bytes]] = None,
    ) -> bytes:
        """Render one still image and return raw PNG bytes.

        ``reference_images`` holds decoded input images for edit workflows.
        They are mapped to whichever image input the active runner exposes
        (``image_prompt`` first, then ``image``/``init_image``/``source_image``,
        ...). When the runner has no real image-input path,
        :class:`ImageEditUnsupportedError` is raised so the route can return a
        readable 400 and ``/v1/models`` keeps reporting ``image_edit=false``.
        """
        import torch

        w = int(width) if width else self.width
        h = int(height) if height else self.height
        # SDXL requires dims divisible by 8; clamp to supported range.
        w = max(SIZE_LIMITS[0], min(SIZE_LIMITS[2], (w // 8) * 8))
        h = max(SIZE_LIMITS[1], min(SIZE_LIMITS[3], (h // 8) * 8))
        steps = int(num_inference_steps) if num_inference_steps else self.steps
        cfg = float(guidance_scale) if guidance_scale is not None else self.guidance

        refs = [r for r in (reference_images or []) if r]
        pipe = self._ensure_pipeline()
        if pipe is None or self._mock_mode:
            if refs:
                raise ImageEditUnsupportedError(
                    f"Model '{self.model_id}' does not support image editing: "
                    "the backend is in mock mode with no image-input runner loaded. "
                    "/v1/models reports image_edit=false for this backend."
                )
            return _mock_png(prompt, w, h)

        generator = None
        if seed is not None:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                generator = torch.Generator(device=device).manual_seed(int(seed))
            except Exception:
                generator = None

        if refs:
            limit = self.capabilities.max_reference_images
            if limit <= 0 or self.image_input_param() is None:
                raise ImageEditUnsupportedError(
                    f"Model '{self.model_id}' does not support image editing: "
                    "the active runner has no image input (checked image_prompt, "
                    "image, init_image, source_image, control_image, mask_image). "
                    "/v1/models reports image_edit=false for this backend."
                )
            if len(refs) > limit:
                raise ImageEditUnsupportedError(
                    f"Got {len(refs)} reference image(s) but model '{self.model_id}' "
                    f"supports at most {limit}."
                )

        call_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "width": w,
            "height": h,
            "num_inference_steps": steps,
            "guidance_scale": cfg,
            "generator": generator,
        }
        if refs:
            call_kwargs[self.image_input_param()] = _reference_pil_images(pipe, refs)
        # FLUX has no negative-prompt input and expects dims divisible by 16.
        if self.is_flux:
            call_kwargs["width"] = (w // 16) * 16
            call_kwargs["height"] = (h // 16) * 16
        elif negative_prompt:
            call_kwargs["negative_prompt"] = negative_prompt

        try:
            result = pipe(**call_kwargs)
            image = result.images[0]
        except Exception as exc:
            logger.error(
                "%s generation failed: %s",
                "FLUX" if self.is_flux else "SDXL", exc, exc_info=True,
            )
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
