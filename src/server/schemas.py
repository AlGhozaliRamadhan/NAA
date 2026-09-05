"""
OpenAI-Compatible Pydantic Request & Response Schemas for NAA
"""

from typing import Optional, List, Union, Dict, Any
from pydantic import BaseModel, ConfigDict, Field
from src.core.prompt import ChatMessage
from src.config import settings

class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = Field(default="NAA-AI-Model")
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=2048, ge=1, le=65536)
    temperature: Optional[float] = Field(default=0.70, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.90, ge=0.0, le=1.0)
    min_p: Optional[float] = Field(default=0.05, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=40, ge=0)
    repeat_penalty: Optional[float] = Field(default=1.08, ge=0.0, le=2.0)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    n: Optional[int] = Field(default=1, ge=1, le=1)
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    parallel_tool_calls: Optional[bool] = True
    stream_options: Optional[Dict[str, Any]] = None


class AnthropicMessageRequest(BaseModel):
    """Compatible subset of the Anthropic Messages request used by Claude Code.

    Unknown beta fields are accepted so newer Claude Code releases degrade
    gracefully when the loaded local model does not implement that capability.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    messages: List[Dict[str, Any]]
    max_tokens: int = Field(ge=1, le=65536)
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    temperature: Optional[float] = Field(default=0.70, ge=0.0, le=1.0)
    top_p: Optional[float] = Field(default=0.90, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=40, ge=0)
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Dict[str, Any]] = None

class CompletionRequest(BaseModel):
    model: str = Field(default="NAA-AI-Model")
    prompt: str
    max_tokens: Optional[int] = Field(default=2048, ge=1, le=65536)
    temperature: Optional[float] = Field(default=0.70, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.90, ge=0.0, le=1.0)
    min_p: Optional[float] = Field(default=0.05, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=40, ge=0)
    repeat_penalty: Optional[float] = Field(default=1.08, ge=0.0, le=2.0)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

class ImageReference(BaseModel):
    """One Cogito reference image for ``generation_mode="edit"``."""

    model_config = ConfigDict(extra="allow")

    mime_type: Optional[str] = None
    b64_json: str = Field(min_length=1)
    filename: Optional[str] = None


class ImageGenerationRequest(BaseModel):
    """OpenAI-compatible image request with Cogito edit support.

    Text-to-image is the default (``prompt`` only). When the loaded backend
    reports ``image_edit=true`` via ``GET /v1/models``, clients may send
    ``generation_mode="edit"`` with ``reference_images``; the route maps them
    to whichever image input the active runner exposes. Extra keys
    (``response_format``, ``style``, ...) are accepted and ignored.
    """

    model_config = ConfigDict(extra="allow")

    prompt: str = Field(min_length=1)
    model: Optional[str] = Field(default=None)
    size: Optional[str] = Field(default=None)
    quality: Optional[str] = Field(default=None)
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=100)
    n: Optional[int] = Field(default=1, ge=1, le=4)
    generation_mode: Optional[str] = Field(default=None)
    reference_images: Optional[List[ImageReference]] = Field(default=None)


class CreateKeyRequest(BaseModel):
    name: str
    role: str = "user"
    rpm: int = Field(default=30, ge=1)

class RevokeKeyRequest(BaseModel):
    key: str

class EmbeddingRequest(BaseModel):
    model: str = Field(default="NAA-AI-Model")
    input: Union[str, List[str]]


class VideoGenerationRequest(BaseModel):
    """Unified Text-to-Video / Image-to-Video request for Wan 2.2.

    Provide ``image`` (base64, data-URI, file path, or http(s) URL) for
    image-to-video; omit it for pure text-to-video. Unspecified generation
    knobs fall back to the backend testing defaults in
    ``src/core/video_engine.py`` (LoRA lkzd7/WAN2.2_LoraSet_NSFW @ 0.8,
    steps=4, profile=4, attention=sage, motion_bucket_id=150).
    """

    model_config = ConfigDict(extra="allow")

    model: Optional[str] = Field(default=None)
    prompt: str = Field(min_length=1)
    image: Optional[str] = Field(default=None)
    negative_prompt: Optional[str] = None
    num_frames: Optional[int] = Field(default=81, ge=1, le=241)
    height: Optional[int] = Field(default=704, ge=64, le=1536)
    width: Optional[int] = Field(default=1280, ge=64, le=2048)
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=100)
    guidance_scale: Optional[float] = Field(default=5.0, ge=0.0, le=20.0)
    fps: Optional[int] = Field(default=24, ge=1, le=60)
    seed: Optional[int] = None
    lora_url: Optional[str] = None
    lora_strength: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    profile: Optional[int] = Field(default=None, ge=0, le=4)
    attention: Optional[str] = None
    motion_bucket_id: Optional[int] = Field(default=None, ge=1, le=1000)
