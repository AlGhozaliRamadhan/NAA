"""
OpenAI-Compatible Pydantic Request & Response Schemas for NAA
"""

from typing import Optional, List, Union, Dict, Any
from pydantic import BaseModel, Field
from src.core.prompt import ChatMessage
from src.config import settings

class ChatCompletionRequest(BaseModel):
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

class CreateKeyRequest(BaseModel):
    name: str
    role: str = "user"
    rpm: int = Field(default=30, ge=1)

class RevokeKeyRequest(BaseModel):
    key: str

class EmbeddingRequest(BaseModel):
    model: str = Field(default="NAA-AI-Model")
    input: Union[str, List[str]]
