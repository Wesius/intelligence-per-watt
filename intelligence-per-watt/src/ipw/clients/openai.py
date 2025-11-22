from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Sequence

import requests
from dotenv import load_dotenv

from ..core.registry import ClientRegistry
from ..core.types import ChatUsage, Response
from .base import InferenceClient

load_dotenv()


@ClientRegistry.register("openai")
class OpenAIClient(InferenceClient):
    """
    OpenAI-compatible chat client for evaluation and inference.

    This client is stateless with respect to secrets:
    - ``base_url`` and ``model`` are passed in construction.
    - API key is only read inside ``stream_chat_completion()`` (from ``IPW_EVAL_API_KEY`` or ``OPENAI_API_KEY``)
      and is never stored on the instance.
    """

    client_id, client_name = "openai", "OpenAI"

    def __init__(self, base_url: str | None = None, **config: Any) -> None:
        host = (base_url or "https://api.openai.com/v1").rstrip("/")
        super().__init__(host, **config)
        self.model = config.get("model", "gpt-4-turbo") # Default model
        self.timeout_seconds = float(config.get("timeout_seconds", 60.0))
        self.temperature = float(config.get("temperature", 0.0))
        self.max_output_tokens = int(config.get("max_output_tokens", 1024))

    def stream_chat_completion(
        self, model: str, prompt: str, **params: Any
    ) -> Response:
        """
        Execute a single chat completion and return the response.
        Note: This implementation currently does *not* actually stream in the Python generator sense,
        but it fulfills the interface contract by returning a full Response object.
        It uses standard HTTP requests.
        """
        # Secrets are fetched at call time.
        # Prioritize IPW_EVAL_API_KEY for evaluation context, fall back to OPENAI_API_KEY
        api_key = os.getenv("IPW_EVAL_API_KEY") or os.getenv("OPENAI_API_KEY")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Allow overriding params
        request_model = model or self.model
        temperature = params.get("temperature", self.temperature)
        max_tokens = params.get("max_tokens", self.max_output_tokens)

        # If 'messages' is passed in params, use it directly (bridge for EvaluationClient).
        # Otherwise, construct a simple user message from 'prompt'.
        if "messages" in params:
            messages = params["messages"]
        else:
            messages = [{"role": "user", "content": prompt}]

        payload: Dict[str, Any] = {
            "model": request_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Merge other params, but exclude known keys to avoid conflict/duplication
        exclude_keys = {"model", "messages", "temperature", "max_tokens", "prompt"}
        for k, v in params.items():
            if k not in exclude_keys:
                payload[k] = v

        try:
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"OpenAIClient request failed: {exc}") from exc

        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError("OpenAIClient received non-JSON response") from exc

        # OpenAI-style response extraction
        try:
            choices = data.get("choices") or []
            if not choices:
                return Response(content="", usage=ChatUsage(0,0,0), time_to_first_token_ms=0.0)
            
            first = choices[0]
            message = first.get("message") or {}
            content = message.get("content") or ""
            
            usage_data = data.get("usage") or {}
            usage = ChatUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )
            
            # We don't have TTFT for non-streaming request here
            return Response(content=content, usage=usage, time_to_first_token_ms=0.0)

        except Exception as exc:
            raise RuntimeError(
                f"OpenAIClient could not extract content from response: {exc}"
            ) from exc

    def list_models(self) -> Sequence[str]:
        # Minimal implementation, typically requires a GET /models endpoint
        return []

    def health(self) -> bool:
        # Simple health check (maybe try listing models or just return True if config is sane)
        return True

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> str:
        """
        Helper method to mimic the old EvaluationClient interface for easy migration.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        params = {"messages": messages}
        if temperature is not None:
            params["temperature"] = temperature
        if max_output_tokens is not None:
            params["max_tokens"] = max_output_tokens

        response = self.stream_chat_completion(self.model, "", **params)
        return response.content


__all__ = ["OpenAIClient"]
