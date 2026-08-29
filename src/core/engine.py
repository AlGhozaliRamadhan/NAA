"""
Universal AI Inference Engine: Model Lifecycle, Hardware Offloading, and Non-blocking Execution for NAA
"""

import os
import time
import queue
import logging
import threading
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List, AsyncGenerator, Tuple, Union

logger = logging.getLogger("naa-engine")

try:
    import llama_cpp
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama_cpp not available in environment.")

from src.core.stop_criteria import get_combined_stop_tokens
from src.core.prompt import DEFAULT_STOP_TOKENS, CANONICAL_SYSTEM_PROMPT, ChatMessage, prepare_chat_messages
from src.core.llama_server import LlamaServerBackend
from src.core.tool_calls import (
    inject_tool_prompt,
    messages_for_transformers,
    tokenizer_supports_tools,
    tools_for_transformers,
)

class InferenceEngine:
    """
    Universal Inference Engine supporting any AI Model:
    - GGUF models via llama.cpp C++ backend
    - HuggingFace Safetensors / PyTorch models via Transformers (AutoModelForCausalLM)
    - 4-bit NF4, 8-bit Int8, and FP16/BF16 GPU quantization
    - FlashAttention, context caching, and non-blocking streaming with cancellation
    """

    def __init__(
        self,
        model_path: str,
        model_name: Optional[str] = None,
        quant_mode: str = "auto",
        preset: str = "default",
        system_prompt: Optional[str] = None,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        flash_attn: bool = True,
        cache_type_k: Optional[Union[int, str]] = None,
        cache_type_v: Optional[Union[int, str]] = None,
        trust_remote_code: bool = True,
    ):
        self.model_path = Path(model_path)
        self.model_name = model_name or self._derive_model_name(model_path)
        self.quant_mode = quant_mode
        self.preset = preset
        self.system_prompt = system_prompt
        self.n_ctx = int(n_ctx) if n_ctx else 8192
        self.n_gpu_layers = int(n_gpu_layers) if n_gpu_layers is not None else -1
        self.flash_attn = bool(flash_attn)
        self.cache_type_k = cache_type_k
        self.cache_type_v = cache_type_v
        self.trust_remote_code = trust_remote_code

        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.model_loaded: bool = False
        self.model_loading: bool = False
        self.load_error: Optional[str] = None
        self.load_stage: str = "idle"
        self.lock = threading.Lock()
        self.ready_event = threading.Event()

    def _derive_model_name(self, path_or_repo: str) -> str:
        p = Path(path_or_repo)
        if p.name and p.name not in (".", ""):
            return p.name.replace(".gguf", "")
        return "NAA-AI-Model"

    def _map_ggml_type(self, val: Any) -> Optional[int]:
        if val is None or isinstance(val, int):
            return val
        if isinstance(val, str) and LLAMA_CPP_AVAILABLE:
            val_lower = val.lower().strip()
            if hasattr(llama_cpp, "GGML_TYPE_Q8_0") and val_lower in ("q8_0", "8bit"):
                return getattr(llama_cpp, "GGML_TYPE_Q8_0")
            elif hasattr(llama_cpp, "GGML_TYPE_Q4_0") and val_lower in ("q4_0", "4bit"):
                return getattr(llama_cpp, "GGML_TYPE_Q4_0")
            elif hasattr(llama_cpp, "GGML_TYPE_F16") and val_lower in ("f16", "fp16", "16bit"):
                return getattr(llama_cpp, "GGML_TYPE_F16")
        return None

    def load(self):
        """Loads the model into GPU VRAM / system memory (Safetensors / GGUF)."""
        if self.model_loaded or self.model_loading:
            return
        self.model_loading = True
        self.load_error = None
        self.load_stage = "loading"
        logger.info(f"Loading model '{self.model_name}' from: {self.model_path} (quant={self.quant_mode})")

        try:
            model_path_str = str(self.model_path)
            is_gguf = (
                model_path_str.endswith(".gguf")
                or (self.model_path.is_dir() and list(self.model_path.glob("*.gguf")))
            )

            if is_gguf:
                model_file = self.model_path
                if model_file.is_dir():
                    model_file = list(model_file.glob("*.gguf"))[0]
                gguf_backend = os.environ.get("NAA_GGUF_BACKEND", "python").lower()
                if gguf_backend in ("llama-server", "server", "external"):
                    binary = os.environ.get("NAA_LLAMA_SERVER_BIN")
                    if not binary or not Path(binary).is_file():
                        raise RuntimeError(
                            "NAA_GGUF_BACKEND=llama-server but NAA_LLAMA_SERVER_BIN "
                            "does not point to a built llama-server executable."
                        )
                    logger.info(
                        "Launching standalone llama-server backend for GGUF weights: %s",
                        model_file,
                    )
                    self.model = LlamaServerBackend.launch(
                        binary,
                        str(model_file),
                        n_ctx=self.n_ctx,
                        n_gpu_layers=self.n_gpu_layers,
                        flash_attn=self.flash_attn,
                        cache_type_k=self.cache_type_k,
                        cache_type_v=self.cache_type_v,
                    )
                else:
                    if not LLAMA_CPP_AVAILABLE:
                        raise RuntimeError(
                            "Model is in GGUF format, but 'llama-cpp-python' is not installed. "
                            "Install it or set NAA_GGUF_BACKEND=llama-server."
                        )
                    llama_kwargs: Dict[str, Any] = {
                        "model_path": str(model_file),
                        "n_ctx": self.n_ctx,
                        "n_gpu_layers": self.n_gpu_layers,
                        "verbose": False,
                    }
                    if self.flash_attn:
                        llama_kwargs["flash_attn"] = True

                    type_k_val = self._map_ggml_type(self.cache_type_k)
                    if type_k_val is not None:
                        llama_kwargs["type_k"] = type_k_val

                    type_v_val = self._map_ggml_type(self.cache_type_v)
                    if type_v_val is not None:
                        llama_kwargs["type_v"] = type_v_val

                    logger.info(f"Initializing llama_cpp.Llama with GGUF weights: {model_file}")
                    self.model = llama_cpp.Llama(**llama_kwargs)
            else:
                # Load Safetensors / Hugging Face model
                try:
                    import torch
                    from transformers import AutoModelForCausalLM, AutoTokenizer

                    logger.info(f"Initializing Transformers from: {model_path_str}")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_path_str,
                        trust_remote_code=self.trust_remote_code,
                    )

                    load_kwargs = {
                        "trust_remote_code": self.trust_remote_code,
                        "device_map": "auto" if torch.cuda.is_available() else None,
                    }

                    if self.quant_mode in ("4bit", "auto") and torch.cuda.is_available():
                        try:
                            from transformers import BitsAndBytesConfig
                            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16,
                            )
                            load_kwargs["dtype"] = torch.bfloat16
                        except ImportError:
                            load_kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    elif self.quant_mode == "8bit" and torch.cuda.is_available():
                        try:
                            from transformers import BitsAndBytesConfig
                            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                        except ImportError:
                            load_kwargs["dtype"] = torch.bfloat16
                    elif torch.cuda.is_available():
                        load_kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path_str,
                        **load_kwargs,
                    )
                except ImportError:
                    logger.warning("torch / transformers not available; operating in mock/compatibility mode.")

            self.model_loaded = True
            self.load_stage = "ready"
            self.ready_event.set()
            logger.info("Model successfully loaded and ready for inference!")
        except Exception as e:
            self.load_error = str(e)
            self.load_stage = "error"
            logger.error(f"Failed to load model: {e}", exc_info=True)
        finally:
            self.model_loading = False

    def is_ready(self) -> bool:
        return self.model_loaded and self.model is not None

    def _render_transformers_chat_prompt(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = "auto",
    ) -> str:
        """Render the model's native tool template, with a portable fallback."""

        template_messages = messages_for_transformers(messages)
        supports_tools = bool(tools) and tokenizer_supports_tools(self.tokenizer)
        if tools and not supports_tools:
            template_messages = inject_tool_prompt(template_messages, tools, tool_choice)

        if hasattr(self.tokenizer, "apply_chat_template"):
            kwargs: Dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            template = getattr(self.tokenizer, "chat_template", None)
            if supports_tools:
                kwargs["tools"] = tools_for_transformers(tools)
                kwargs["tool_choice"] = tool_choice
                if isinstance(template, dict) and "tool_use" in template:
                    kwargs["chat_template"] = "tool_use"
            try:
                return self.tokenizer.apply_chat_template(template_messages, **kwargs)
            except (TypeError, ValueError) as exc:
                if not tools:
                    raise
                logger.warning("Native tool chat template failed; using portable tool prompt: %s", exc)
                fallback = inject_tool_prompt(messages_for_transformers(messages), tools, tool_choice)
                return self.tokenizer.apply_chat_template(
                    fallback,
                    tokenize=False,
                    add_generation_prompt=True,
                )

        from src.core.prompt import build_chatml_prompt

        fallback = inject_tool_prompt(template_messages, tools, tool_choice) if tools else template_messages
        chat_msgs = [ChatMessage(**message) for message in fallback]
        return build_chatml_prompt(chat_msgs)

    async def generate_chat_stream(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        max_tokens: int = 2048,
        temperature: float = 0.70,
        top_p: float = 0.90,
        min_p: float = 0.05,
        top_k: int = 40,
        repeat_penalty: float = 1.08,
        custom_stops: Optional[List[str]] = None,
        cancel_event: Optional[asyncio.Event] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = "auto",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Non-blocking streaming chat completion generator with instant cancellation.
        """
        stop_list = get_combined_stop_tokens(custom_stops)
        formatted_messages = prepare_chat_messages(
            messages,
            preset=self.preset,
            custom_system_prompt=self.system_prompt,
        )

        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        stop_worker = threading.Event()

        def _worker():
            try:
                if hasattr(self.model, "create_chat_completion"):
                    kwargs = {
                        "messages": formatted_messages,
                        "max_tokens": max_tokens,
                        "temperature": max(temperature, 1e-4) if temperature > 0 else 1e-4,
                        "top_p": top_p if (top_p is not None and temperature > 0) else 1.0,
                        "min_p": min_p,
                        "repeat_penalty": repeat_penalty,
                        "top_k": top_k,
                        "stop": stop_list,
                        "stream": True,
                    }
                    if tools:
                        kwargs["tools"] = tools
                        kwargs["tool_choice"] = tool_choice or "auto"
                    for chunk in self.model.create_chat_completion(**kwargs):
                        if stop_worker.is_set():
                            break
                        loop.call_soon_threadsafe(q.put_nowait, chunk)
                elif self.tokenizer is not None and hasattr(self.model, "generate"):
                    import torch
                    from transformers import TextIteratorStreamer

                    prompt_text = self._render_transformers_chat_prompt(
                        formatted_messages,
                        tools=tools,
                        tool_choice=tool_choice,
                    )

                    inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
                    if hasattr(self.model, "device"):
                        inputs = inputs.to(self.model.device)

                    streamer = TextIteratorStreamer(
                        self.tokenizer,
                        skip_prompt=True,
                        skip_special_tokens=True,
                    )

                    gen_kwargs = {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs.get("attention_mask"),
                        "max_new_tokens": max_tokens,
                        "streamer": streamer,
                        "temperature": max(temperature, 1e-4) if temperature > 0 else 1.0,
                        "top_p": top_p if (top_p is not None and temperature > 0) else 1.0,
                        "top_k": top_k if temperature > 0 else 50,
                        "repetition_penalty": repeat_penalty,
                        "do_sample": temperature > 0,
                        "pad_token_id": getattr(self.tokenizer, "eos_token_id", None),
                    }

                    gen_thread = threading.Thread(
                        target=self.model.generate,
                        kwargs=gen_kwargs,
                        daemon=True,
                    )
                    gen_thread.start()

                    first_chunk = True
                    for new_text in streamer:
                        if stop_worker.is_set():
                            break
                        delta = {"content": new_text}
                        if first_chunk:
                            delta["role"] = "assistant"
                            first_chunk = False
                        loop.call_soon_threadsafe(
                            q.put_nowait,
                            {"choices": [{"delta": delta, "finish_reason": None}]}
                        )

                    loop.call_soon_threadsafe(
                        q.put_nowait,
                        {"choices": [{"delta": {}, "finish_reason": "stop"}]}
                    )
                else:
                    # Fallback / Mock compatibility
                    loop.call_soon_threadsafe(
                        q.put_nowait,
                        {"choices": [{"delta": {"role": "assistant", "content": "<think>\nThinking...\n</think>\nResponse content."}, "finish_reason": None}]}
                    )
                    loop.call_soon_threadsafe(
                        q.put_nowait,
                        {"choices": [{"delta": {}, "finish_reason": "stop"}]}
                    )
            except Exception as e:
                logger.error(f"Error in chat stream worker: {e}", exc_info=True)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        worker_thread = threading.Thread(target=_worker, daemon=True)
        worker_thread.start()

        try:
            while True:
                if cancel_event and cancel_event.is_set():
                    stop_worker.set()
                    break
                chunk = await q.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            stop_worker.set()

    async def generate_chat_non_streaming(
        self,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        max_tokens: int = 2048,
        temperature: float = 0.70,
        top_p: float = 0.90,
        min_p: float = 0.05,
        top_k: int = 40,
        repeat_penalty: float = 1.08,
        custom_stops: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = "auto",
    ) -> Dict[str, Any]:
        """
        Non-blocking execution of complete chat completion.
        """
        stop_list = get_combined_stop_tokens(custom_stops)
        formatted_messages = prepare_chat_messages(
            messages,
            preset=self.preset,
            custom_system_prompt=self.system_prompt,
        )

        def _run():
            if hasattr(self.model, "create_chat_completion"):
                kwargs: Dict[str, Any] = dict(
                    messages=formatted_messages,
                    max_tokens=max_tokens,
                    temperature=max(temperature, 1e-4) if temperature > 0 else 1e-4,
                    top_p=top_p if (top_p is not None and temperature > 0) else 1.0,
                    min_p=min_p,
                    repeat_penalty=repeat_penalty,
                    top_k=top_k,
                    stop=stop_list,
                    stream=False,
                )
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = tool_choice or "auto"
                return self.model.create_chat_completion(**kwargs)
            elif self.tokenizer is not None and hasattr(self.model, "generate"):
                import torch
                prompt_text = self._render_transformers_chat_prompt(
                    formatted_messages,
                    tools=tools,
                    tool_choice=tool_choice,
                )

                inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
                if hasattr(self.model, "device"):
                    inputs = inputs.to(self.model.device)

                gen_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs.get("attention_mask"),
                    "max_new_tokens": max_tokens,
                    "temperature": max(temperature, 1e-4) if temperature > 0 else 1.0,
                    "top_p": top_p if (top_p is not None and temperature > 0) else 1.0,
                    "top_k": top_k if temperature > 0 else 50,
                    "repetition_penalty": repeat_penalty,
                    "do_sample": temperature > 0,
                    "pad_token_id": getattr(self.tokenizer, "eos_token_id", None),
                }
                outputs = self.model.generate(**gen_kwargs)
                gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                response_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.model_name,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": int(inputs["input_ids"].shape[1]),
                        "completion_tokens": int(gen_tokens.shape[0]),
                        "total_tokens": int(inputs["input_ids"].shape[1] + gen_tokens.shape[0]),
                    }
                }
            else:
                # Mock fallback
                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.model_name,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "<think>\nVerified reasoning.\n</think>\nComplete output."},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}
                }

        return await asyncio.to_thread(_run)

    async def generate_text_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.70,
        top_p: float = 0.90,
        min_p: float = 0.05,
        top_k: int = 40,
        repeat_penalty: float = 1.08,
        custom_stops: Optional[List[str]] = None,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Non-blocking streaming text completion generator.
        """
        stop_list = get_combined_stop_tokens(custom_stops)
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        stop_worker = threading.Event()

        def _worker():
            try:
                if hasattr(self.model, "create_completion"):
                    kwargs = {
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": max(temperature, 1e-4) if temperature > 0 else 1e-4,
                        "top_p": top_p if (top_p is not None and temperature > 0) else 1.0,
                        "min_p": min_p,
                        "repeat_penalty": repeat_penalty,
                        "top_k": top_k,
                        "stop": stop_list,
                        "stream": True,
                    }
                    for chunk in self.model.create_completion(**kwargs):
                        if stop_worker.is_set():
                            break
                        loop.call_soon_threadsafe(q.put_nowait, chunk)
                elif self.tokenizer is not None and hasattr(self.model, "generate"):
                    from transformers import TextIteratorStreamer

                    inputs = self.tokenizer(prompt, return_tensors="pt")
                    if hasattr(self.model, "device"):
                        inputs = inputs.to(self.model.device)

                    streamer = TextIteratorStreamer(
                        self.tokenizer,
                        skip_prompt=True,
                        skip_special_tokens=True,
                    )

                    gen_kwargs = {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs.get("attention_mask"),
                        "max_new_tokens": max_tokens,
                        "streamer": streamer,
                        "temperature": max(temperature, 1e-4) if temperature > 0 else 1.0,
                        "top_p": top_p if (top_p is not None and temperature > 0) else 1.0,
                        "top_k": top_k if temperature > 0 else 50,
                        "repetition_penalty": repeat_penalty,
                        "do_sample": temperature > 0,
                        "pad_token_id": getattr(self.tokenizer, "eos_token_id", None),
                    }

                    gen_thread = threading.Thread(
                        target=self.model.generate,
                        kwargs=gen_kwargs,
                        daemon=True,
                    )
                    gen_thread.start()

                    for new_text in streamer:
                        if stop_worker.is_set():
                            break
                        loop.call_soon_threadsafe(
                            q.put_nowait,
                            {"choices": [{"text": new_text, "index": 0, "finish_reason": None}]}
                        )

                    loop.call_soon_threadsafe(
                        q.put_nowait,
                        {"choices": [{"text": "", "index": 0, "finish_reason": "stop"}]}
                    )
                else:
                    loop.call_soon_threadsafe(
                        q.put_nowait,
                        {"choices": [{"text": " Generated completion text.", "index": 0, "finish_reason": None}]}
                    )
                    loop.call_soon_threadsafe(
                        q.put_nowait,
                        {"choices": [{"text": "", "index": 0, "finish_reason": "stop"}]}
                    )
            except Exception as e:
                logger.error(f"Error in text stream worker: {e}", exc_info=True)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        worker_thread = threading.Thread(target=_worker, daemon=True)
        worker_thread.start()

        try:
            while True:
                if cancel_event and cancel_event.is_set():
                    stop_worker.set()
                    break
                chunk = await q.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            stop_worker.set()

    async def generate_text_non_streaming(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.70,
        top_p: float = 0.90,
        min_p: float = 0.05,
        top_k: int = 40,
        repeat_penalty: float = 1.08,
        custom_stops: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Non-blocking execution of raw text completion.
        """
        stop_list = get_combined_stop_tokens(custom_stops)

        def _run():
            if hasattr(self.model, "create_completion"):
                return self.model.create_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=max(temperature, 1e-4) if temperature > 0 else 1e-4,
                    top_p=top_p if (top_p is not None and temperature > 0) else 1.0,
                    min_p=min_p,
                    repeat_penalty=repeat_penalty,
                    top_k=top_k,
                    stop=stop_list,
                    stream=False,
                )
            elif self.tokenizer is not None and hasattr(self.model, "generate"):
                inputs = self.tokenizer(prompt, return_tensors="pt")
                if hasattr(self.model, "device"):
                    inputs = inputs.to(self.model.device)

                gen_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs.get("attention_mask"),
                    "max_new_tokens": max_tokens,
                    "temperature": max(temperature, 1e-4) if temperature > 0 else 1.0,
                    "top_p": top_p if (top_p is not None and temperature > 0) else 1.0,
                    "top_k": top_k if temperature > 0 else 50,
                    "repetition_penalty": repeat_penalty,
                    "do_sample": temperature > 0,
                    "pad_token_id": getattr(self.tokenizer, "eos_token_id", None),
                }
                outputs = self.model.generate(**gen_kwargs)
                gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                response_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                return {
                    "id": f"cmpl-{int(time.time())}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": self.model_name,
                    "choices": [{"text": response_text, "index": 0, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": int(inputs["input_ids"].shape[1]),
                        "completion_tokens": int(gen_tokens.shape[0]),
                        "total_tokens": int(inputs["input_ids"].shape[1] + gen_tokens.shape[0]),
                    }
                }
            else:
                return {
                    "id": f"cmpl-{int(time.time())}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": self.model_name,
                    "choices": [{"text": " Generated completion text.", "index": 0, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
                }

        return await asyncio.to_thread(_run)
