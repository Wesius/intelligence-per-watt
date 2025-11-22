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
    OpenAI-compatible chat client for evaluation-only use (judge models).

    This client is stateless with respect to secrets:
    - ``base_url`` and ``model`` are passed in construction.
    - API key is read on each request (``IPW_EVAL_API_KEY`` or ``OPENAI_API_KEY``) and never stored.
    """

    client_id, client_name = "openai", "OpenAI"

    def __init__(self, base_url: str | None = None, **config: Any) -> None:
        host = (base_url or "https://api.openai.com/v1").rstrip("/")
        super().__init__(host, **config)
        self.model = config.get("model", "gpt-4-turbo")  # Default model
        self.timeout_seconds = float(config.get("timeout_seconds", 60.0))
        self.temperature = float(config.get("temperature", 0.0))
        self.max_output_tokens = int(config.get("max_output_tokens", 1024))

    def stream_chat_completion(
        self, model: str, prompt: str, **params: Any
    ) -> Response:
        """Profiling is not supported with this client."""
        raise RuntimeError(
            "OpenAIClient is evaluation-only; do not use it for profiling. "
            "Use a profiling client like ollama or vllm instead."
        )

    def list_models(self) -> Sequence[str]:
        raise RuntimeError("OpenAIClient is evaluation-only and cannot list models.")

    def health(self) -> bool:
        raise RuntimeError("OpenAIClient is evaluation-only and does not expose health checks.")

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

        response = self._chat_completion(self.model, "", **params)
        return response.content

    def _chat_completion(
        self, model: str, prompt: str, **params: Any
    ) -> Response:
        """Internal helper used by evaluation-only chat."""
        api_key = os.getenv("IPW_EVAL_API_KEY") or os.getenv("OPENAI_API_KEY")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request_model = model or self.model
        temperature = params.get("temperature", self.temperature)
        max_tokens = params.get("max_tokens", self.max_output_tokens)

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

        choices = data.get("choices") or []
        if not choices:
            return Response(
                content="",
                usage=ChatUsage(0, 0, 0),
                time_to_first_token_ms=0.0,
            )

        first = choices[0]
        message = first.get("message") or {}
        content = message.get("content") or ""

        return Response(
            content=content,
            usage=ChatUsage(0, 0, 0),
            time_to_first_token_ms=0.0,
        )


__all__ = ["OpenAIClient"]
