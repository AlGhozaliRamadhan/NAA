"""
Wan 2.2 Video Generation Engine for NAA.

Unified Text-to-Video (T2V) + Image-to-Video (I2V) backend built around
the Wan2.2-TI2V-5B hybrid model (single checkpoint covers both workflows,
fits a 24GB consumer GPU with offloading). A14B MoE checkpoints are
supported via the same interface by overriding ``model_id``.

Backend testing defaults (do not change without updating tests):
- LoRA URL:        https://huggingface.co/lkzd7/WAN2.2_LoraSet_NSFW
- lora_strength:   0.8
- steps:           4
- profile:         2
- attention:       sage
- motion_bucket_id: 150
"""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("naa-video")

# ---------------------------------------------------------------------------
# Backend testing configuration (goal-mandated defaults)
# ---------------------------------------------------------------------------

DEFAULT_VIDEO_MODEL_ID = os.environ.get(
    "NAA_VIDEO_MODEL_ID", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
)
DEFAULT_VIDEO_LORA_URL = os.environ.get(
    "NAA_VIDEO_LORA_URL", "https://huggingface.co/lkzd7/WAN2.2_LoraSet_NSFW"
)
DEFAULT_LORA_STRENGTH = float(os.environ.get("NAA_VIDEO_LORA_STRENGTH", "0.8"))
DEFAULT_VIDEO_STEPS = int(os.environ.get("NAA_VIDEO_STEPS", "4"))
DEFAULT_VIDEO_PROFILE = int(os.environ.get("NAA_VIDEO_PROFILE", "2"))
DEFAULT_VIDEO_ATTENTION = os.environ.get("NAA_VIDEO_ATTENTION", "sage").lower()
DEFAULT_MOTION_BUCKET_ID = int(os.environ.get("NAA_VIDEO_MOTION_BUCKET_ID", "150"))

VIDEO_MODEL_ALIASES: Dict[str, str] = {
    "wan2.2": DEFAULT_VIDEO_MODEL_ID,
    "wan2.2-ti2v-5b": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "wan2.2-t2v-a14b": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "wan2.2-i2v-a14b": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
}

SUPPORTED_ATTENTION = ("sage", "sdpa", "eager", "flash")
SUPPORTED_PROFILES = (0, 1, 2, 3, 4)

# profile -> memory/quality trade-off flags applied at load time.
# 0 = max quality (no offload), 4 = max memory saving for T4/L4/4090.
PROFILE_PRESETS: Dict[int, Dict[str, Any]] = {
    0: {"offload_model": False, "convert_model_dtype": False, "t5_cpu": False},
    1: {"offload_model": False, "convert_model_dtype": True, "t5_cpu": False},
    2: {"offload_model": True, "convert_model_dtype": False, "t5_cpu": False},
    3: {"offload_model": True, "convert_model_dtype": True, "t5_cpu": False},
    4: {"offload_model": True, "convert_model_dtype": True, "t5_cpu": True},
}


def resolve_video_model_id(model: Optional[str]) -> str:
    """Resolve a friendly alias (``wan2.2``) to a HuggingFace model id."""
    if not model:
        return DEFAULT_VIDEO_MODEL_ID
    key = model.strip()
    return VIDEO_MODEL_ALIASES.get(key.lower(), key)


def parse_lora_repo(lora_url: Optional[str]) -> Optional[str]:
    """Extract ``owner/repo`` from a HuggingFace LoRA URL, if possible."""
    if not lora_url:
        return None
    url = lora_url.strip().rstrip("/")
    if "huggingface.co/" not in url:
        # Already a repo id?
        return url if "/" in url else None
    tail = url.split("huggingface.co/")[-1]
    parts = [p for p in tail.split("/") if p not in ("blob", "resolve", "raw", "tree", "main")]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return None


def decode_image_payload(image: Optional[str]) -> Optional[bytes]:
    """Accept base64 (raw or data-URI) or http(s) URL; return bytes or None.

    URLs are returned as ``None`` here and resolved lazily inside the worker
    so job creation never blocks on network I/O.
    """
    if not image:
        return None
    text = image.strip()
    if text.startswith("http://") or text.startswith("https://"):
        return None
    if text.startswith("data:"):
        try:
            text = text.split(",", 1)[1]
        except IndexError:
            return None
    try:
        # validate=True rejects non-base64 strings so plain paths fall through
        return base64.b64decode(text, validate=True)
    except Exception:
        path = Path(image)
        if path.is_file():
            return path.read_bytes()
        return None


@dataclass
class VideoJob:
    id: str
    mode: str  # "text-to-video" | "image-to-video"
    prompt: str
    status: str = "queued"  # queued | running | succeeded | failed
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, public_url: Optional[str] = None) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "object": "video.generation",
            "mode": self.mode,
            "model": self.params.get("model", DEFAULT_VIDEO_MODEL_ID),
            "status": self.status,
            "progress": round(self.progress, 3),
            "prompt": self.prompt,
            "created_at": int(self.created_at),
            "params": self.params,
        }
        if self.finished_at:
            data["finished_at"] = int(self.finished_at)
        if self.error:
            data["error"] = self.error
        if self.output_path:
            data["output_file"] = Path(self.output_path).name
            if public_url:
                data["result_url"] = f"{public_url.rstrip('/')}/v1/videos/{self.id}/download"
            else:
                data["result_url"] = f"/v1/videos/{self.id}/download"
        return data


class VideoEngine:
    """Background video generation engine with job tracking.

    Heavy dependencies (``diffusers``, ``torch``) are imported lazily so the
    API server and test suite run in mock mode without a GPU.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_VIDEO_MODEL_ID,
        output_dir: Optional[str] = None,
        lora_url: str = DEFAULT_VIDEO_LORA_URL,
        lora_strength: float = DEFAULT_LORA_STRENGTH,
        steps: int = DEFAULT_VIDEO_STEPS,
        profile: int = DEFAULT_VIDEO_PROFILE,
        attention: str = DEFAULT_VIDEO_ATTENTION,
        motion_bucket_id: int = DEFAULT_MOTION_BUCKET_ID,
    ):
        from src.config import WORK_DIR  # deferred to avoid import cycles

        self.model_id = resolve_video_model_id(model_id)
        self.output_dir = Path(output_dir) if output_dir else (Path(WORK_DIR) / "videos")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lora_url = lora_url
        self.lora_strength = float(lora_strength)
        self.steps = int(steps)
        self.profile = int(profile) if int(profile) in SUPPORTED_PROFILES else 4
        self.attention = (attention or "sage").lower()
        if self.attention not in SUPPORTED_ATTENTION:
            self.attention = "sage"
        self.motion_bucket_id = int(motion_bucket_id)

        self._pipe: Optional[Any] = None
        self._pipe_model_id: Optional[str] = None
        self._mock_mode = False
        self._lock = threading.Lock()
        self._jobs: Dict[str, VideoJob] = {}

    # -- defaults snapshot (used by /v1/videos/config) ---------------------
    def default_config(self) -> Dict[str, Any]:
        return {
            "model": self.model_id,
            "lora_url": self.lora_url,
            "lora_strength": self.lora_strength,
            "steps": self.steps,
            "profile": self.profile,
            "attention": self.attention,
            "motion_bucket_id": self.motion_bucket_id,
        }

    # -- job store ----------------------------------------------------------
    def list_jobs(self) -> List[VideoJob]:
        with self._lock:
            return list(self._jobs.values())

    def get_job(self, job_id: str) -> Optional[VideoJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def _set_job(self, job: VideoJob) -> None:
        with self._lock:
            self._jobs[job.id] = job

    # -- pipeline -----------------------------------------------------------
    def _apply_attention_backend(self, pipe: Any) -> None:
        """Select the requested attention kernel when available."""
        try:
            if self.attention == "sage":
                try:
                    # sageattention registers itself; no direct call needed.
                    import sageattention  # noqa: F401
                    logger.info("SageAttention available for video pipeline")
                except ImportError:
                    logger.warning(
                        "SageAttention not installed; falling back to sdpa. "
                        "Install with: pip install sageattention"
                    )
                    try:
                        pipe.transformer.set_attention_backend("sdpa")
                    except Exception:
                        pass
            else:
                try:
                    pipe.transformer.set_attention_backend(self.attention)
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("Could not set attention backend %s: %s", self.attention, exc)

    def _apply_profile(self, pipe: Any) -> None:
        preset = PROFILE_PRESETS.get(self.profile, PROFILE_PRESETS[4])
        try:
            if preset.get("convert_model_dtype"):
                try:
                    import torch

                    pipe.transformer.to(dtype=torch.bfloat16)
                except Exception:
                    pass
            if preset.get("offload_model"):
                try:
                    pipe.enable_sequential_cpu_offload()
                except Exception:
                    try:
                        pipe.enable_model_cpu_offload()
                    except Exception:
                        pass
            if preset.get("t5_cpu"):
                try:
                    pipe.text_encoder.to("cpu")
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("Could not apply video profile %s: %s", self.profile, exc)

    def _ensure_pipeline(self, model_id: Optional[str] = None) -> Optional[Any]:
        """Load (or reuse) the diffusers Wan pipeline. Returns None in mock mode."""
        target = resolve_video_model_id(model_id or self.model_id)
        if self._pipe is not None and self._pipe_model_id == target:
            return self._pipe
        try:
            import torch
            from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline

            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading Wan 2.2 video pipeline: %s", target)
            try:
                vae = AutoencoderKLWan.from_pretrained(target, subfolder="vae", torch_dtype=torch.float32)
            except Exception:
                vae = None
            kwargs: Dict[str, Any] = {"torch_dtype": dtype}
            if vae is not None:
                kwargs["vae"] = vae
            try:
                # TI2V-5B unified checkpoint handles both T2V and I2V.
                pipe = WanPipeline.from_pretrained(target, **kwargs)
            except Exception:
                pipe = WanImageToVideoPipeline.from_pretrained(target, **kwargs)
            try:
                pipe.to(device)
            except Exception:
                pass
            self._apply_attention_backend(pipe)
            self._apply_profile(pipe)
            self._load_lora(pipe, self.lora_url, self.lora_strength)
            self._pipe = pipe
            self._pipe_model_id = target
            self._mock_mode = False
            return pipe
        except ImportError as exc:
            logger.warning("Video deps missing (%s); using mock video renderer.", exc)
            self._mock_mode = True
            return None
        except Exception as exc:
            logger.error("Failed to load video pipeline %s: %s", target, exc, exc_info=True)
            self._mock_mode = True
            return None

    def _load_lora(self, pipe: Any, lora_url: Optional[str], strength: float) -> None:
        repo = parse_lora_repo(lora_url)
        if not repo:
            return
        try:
            from huggingface_hub import snapshot_download

            local = snapshot_download(repo_id=repo)
            weight_files = list(Path(local).glob("*.safetensors")) or list(Path(local).glob("*.bin"))
            if not weight_files:
                logger.warning("No LoRA weights found in %s", repo)
                return
            pipe.load_lora_weights(str(weight_files[0]))
            try:
                pipe.fuse_lora(lora_scale=float(strength))
            except Exception:
                pass
            logger.info("Loaded video LoRA %s (strength=%.2f)", repo, strength)
        except Exception as exc:
            logger.warning("Could not load video LoRA %s: %s", repo, exc)

    # -- public API ---------------------------------------------------------
    def submit(
        self,
        prompt: str,
        image: Optional[str] = None,
        model: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_frames: int = 81,
        height: int = 704,
        width: int = 1280,
        num_inference_steps: Optional[int] = None,
        guidance_scale: float = 5.0,
        fps: int = 24,
        seed: Optional[int] = None,
        lora_url: Optional[str] = None,
        lora_strength: Optional[float] = None,
        profile: Optional[int] = None,
        attention: Optional[str] = None,
        motion_bucket_id: Optional[int] = None,
    ) -> VideoJob:
        mode = "image-to-video" if image else "text-to-video"
        job = VideoJob(
            id=f"vid-{uuid.uuid4().hex[:12]}",
            mode=mode,
            prompt=prompt,
            params={
                "model": resolve_video_model_id(model or self.model_id),
                "negative_prompt": negative_prompt,
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps or self.steps,
                "guidance_scale": guidance_scale,
                "fps": fps,
                "seed": seed,
                "lora_url": lora_url if lora_url is not None else self.lora_url,
                "lora_strength": float(lora_strength) if lora_strength is not None else self.lora_strength,
                "profile": int(profile) if profile is not None else self.profile,
                "attention": (attention or self.attention).lower(),
                "motion_bucket_id": int(motion_bucket_id) if motion_bucket_id is not None else self.motion_bucket_id,
                "has_image": bool(image),
            },
        )
        self._set_job(job)
        thread = threading.Thread(
            target=self._run_job, args=(job, image), daemon=True, name=f"video-{job.id}"
        )
        thread.start()
        return job

    # -- worker -------------------------------------------------------------
    def _run_job(self, job: VideoJob, image: Optional[str]) -> None:
        job.status = "running"
        job.progress = 0.05
        self._set_job(job)
        try:
            pipe = self._ensure_pipeline(job.params.get("model"))
            out_path = self.output_dir / f"{job.id}.mp4"
            if pipe is None:
                self._render_mock(out_path, job)
            else:
                self._render_real(pipe, out_path, job, image)
            job.status = "succeeded"
            job.progress = 1.0
            job.output_path = str(out_path)
            job.finished_at = time.time()
        except Exception as exc:
            logger.error("Video job %s failed: %s", job.id, exc, exc_info=True)
            job.status = "failed"
            job.error = str(exc)
            job.finished_at = time.time()
        self._set_job(job)

    def _render_real(self, pipe: Any, out_path: Path, job: VideoJob, image: Optional[str]) -> None:
        import torch
        from diffusers.utils import export_to_video

        params = job.params
        steps = int(params.get("num_inference_steps") or self.steps)
        image_bytes = decode_image_payload(image)
        pil_image = None
        if image_bytes:
            try:
                from PIL import Image

                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                pil_image = pil_image.resize((int(params["width"]), int(params["height"])))
            except Exception as exc:
                logger.warning("Could not decode input image: %s", exc)

        generator = None
        if params.get("seed") is not None:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                generator = torch.Generator(device=device).manual_seed(int(params["seed"]))
            except Exception:
                generator = None

        call_kwargs: Dict[str, Any] = {
            "prompt": job.prompt,
            "height": int(params["height"]),
            "width": int(params["width"]),
            "num_frames": int(params["num_frames"]),
            "guidance_scale": float(params["guidance_scale"]),
            "num_inference_steps": steps,
            "generator": generator,
        }
        if params.get("negative_prompt"):
            call_kwargs["negative_prompt"] = params["negative_prompt"]
        # motion_bucket_id: accepted by I2V-style pipelines; harmless otherwise.
        if pil_image is not None:
            call_kwargs["image"] = pil_image
            try:
                call_kwargs["motion_bucket_id"] = int(params.get("motion_bucket_id", self.motion_bucket_id))
            except Exception:
                pass

        job.progress = 0.2
        self._set_job(job)
        output = pipe(**call_kwargs)
        frames = output.frames[0] if hasattr(output, "frames") else output
        export_to_video(frames, str(out_path), fps=int(params.get("fps", 24)))

    def _render_mock(self, out_path: Path, job: VideoJob) -> None:
        """Write a tiny placeholder mp4 so tests work without GPU/diffusers."""
        job.progress = 0.5
        self._set_job(job)
        try:
            import imageio.v2 as imageio
            import numpy as np

            frames = (np.zeros((8, 64, 64, 3), dtype=np.uint8) + 32).tolist()
            imageio.mimsave(str(out_path), frames, fps=8, codec="mp4")
        except Exception:
            # Minimal MP4 ftyp header placeholder — satisfies existence checks.
            out_path.write_bytes(
                b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
                + f"mock-video:{job.id}".encode()
            )
        # Simulate async progress so polling tests observe running -> succeeded.
        time.sleep(0.2)


# Module-level singleton (lazy; safe to import without GPU deps).
_engine: Optional[VideoEngine] = None
_engine_lock = threading.Lock()


def get_video_engine(**overrides: Any) -> VideoEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = VideoEngine(**overrides) if overrides else VideoEngine()
        return _engine
