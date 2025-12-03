from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from ..core.registry import ClientRegistry
from ..core.types import ChatUsage, Response
from .base import InferenceClient

try:
    from mlx_lm import load, batch_generate, stream_generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


@ClientRegistry.register("mlx")
class MLXClient(InferenceClient):
    """
    Offline MLX client for Apple Silicon.
    """
    client_id = "mlx"
    client_name = "MLX LM"
    DEFAULT_BASE_URL = "offline://mlx"

    def __init__(self, base_url: str | None = None, **config: Any) -> None:
        super().__init__(base_url or self.DEFAULT_BASE_URL, **config)
        self._model = None
        self._tokenizer = None
        self._model_name: str | None = None
        
    def prepare(self, model: str) -> None:
        if not MLX_AVAILABLE:
            raise RuntimeError("mlx-lm is not installed. Please install it to use MLX client.")
            
        if self._model_name == model and self._model is not None:
            return
        
        # Load model
        # We trust remote code as per common mlx usage for some models
        self._model, self._tokenizer = load(model)
        self._model_name = model

    def list_models(self) -> Sequence[str]:
        return [self._model_name] if self._model_name else []

    def health(self) -> bool:
        return MLX_AVAILABLE

    def stream_chat_completion(self, model: str, prompt: str, **params: Any) -> Response:
        self.prepare(model)
        
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            prompt_formatted = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_formatted = prompt

        # Prepare params
        gen_kwargs = self._extract_gen_params(params)
        
        wall_start = time.time()
        perf_start = time.perf_counter()
        ttft_ms = 0.0
        
        text_parts = []
        
        # stream_generate yields a response object with text segment
        gen = stream_generate(
            self._model, 
            self._tokenizer, 
            prompt_formatted, 
            **gen_kwargs
        )
        
        first_token = True
        for response in gen:
            if first_token:
                ttft_ms = (time.perf_counter() - perf_start) * 1000.0
                first_token = False
            
            text_parts.append(response.text)
            
        wall_end = time.time()
        full_text = "".join(text_parts)
        
        p_tokens = len(self._tokenizer.encode(prompt_formatted))
        c_tokens = len(self._tokenizer.encode(full_text))
        
        usage = ChatUsage(
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=p_tokens + c_tokens
        )
        
        return Response(
            content=full_text,
            usage=usage,
            time_to_first_token_ms=ttft_ms,
            request_start_time=wall_start,
            request_end_time=wall_end
        )

    def run_concurrent(
        self,
        model: str,
        prompt_iter: Iterable[tuple[int, str]],
        max_in_flight: int,
        **params: Any,
    ) -> Iterator[tuple[int, Response]]:
        self.prepare(model)
        
        all_items = list(prompt_iter)
        total = len(all_items)
        
        gen_kwargs = self._extract_gen_params(params)
        
        # Process in batches
        batch_size = max(1, max_in_flight)
        
        for i in range(0, total, batch_size):
            batch_items = all_items[i : i + batch_size]
            batch_indices = [x[0] for x in batch_items]
            raw_prompts = [x[1] for x in batch_items]
            
            formatted_prompts = []
            for p in raw_prompts:
                messages = [{"role": "user", "content": p}]
                # mlx_lm.batch_generate expects list of token ids for padding
                if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
                    # tokenize=True is default, returns List[int]
                    formatted_prompts.append(self._tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True
                    ))
                else:
                    # Manually tokenize if no chat template
                    formatted_prompts.append(self._tokenizer.encode(p))
            
            wall_start = time.time()
            
            result = batch_generate(
                self._model, 
                self._tokenizer, 
                formatted_prompts, 
                verbose=False, 
                **gen_kwargs
            )
            
            wall_end = time.time()
            
            for j, text in enumerate(result.texts):
                idx = batch_indices[j]
                
                # formatted_prompts[j] is now List[int], so length is token count
                p_tokens = len(formatted_prompts[j])
                c_tokens = len(self._tokenizer.encode(text))
                
                usage = ChatUsage(
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    total_tokens=p_tokens + c_tokens
                )
                
                yield idx, Response(
                    content=text,
                    usage=usage,
                    time_to_first_token_ms=0.0,
                    request_start_time=wall_start,
                    request_end_time=wall_end
                )

    def _extract_gen_params(self, params: dict[str, Any]) -> dict[str, Any]:
        kwargs = {}
        if "temperature" in params:
            kwargs["temp"] = params["temperature"]
        elif "temp" in params:
            kwargs["temp"] = params["temp"]
            
        if "max_tokens" in params:
            kwargs["max_tokens"] = params["max_tokens"]
        elif "max_output_tokens" in params:
            kwargs["max_tokens"] = params["max_output_tokens"]
            
        if "top_p" in params:
            kwargs["top_p"] = params["top_p"]
            
        if "repetition_penalty" in params:
            kwargs["repetition_penalty"] = params["repetition_penalty"]
            
        if "max_tokens" not in kwargs:
             kwargs["max_tokens"] = 1024
             
        return kwargs
