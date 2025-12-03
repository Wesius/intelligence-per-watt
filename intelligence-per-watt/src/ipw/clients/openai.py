from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional, Sequence

import requests
from requests import exceptions as req_exc
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

from ..core.registry import ClientRegistry
from ..core.types import ChatUsage, Response
from .base import InferenceClient

DEFAULT_MODEL = "gpt-5-nano-2025-08-07"

load_dotenv()


@ClientRegistry.register("openai")
class OpenAIClient(InferenceClient):
    """
    OpenAI-compatible chat client for evaluation-only use (judge models).

    This client is stateless with respect to secrets:
    - ``base_url`` and ``model`` are passed in construction.
    - API key is read on each request (``IPW_EVAL_API_KEY`` or ``OPENAI_API_KEY``) and never stored; if both differ, the IPW key wins.
    """

    client_id, client_name = "openai", "OpenAI"

    def __init__(self, base_url: str | None = None, **config: Any) -> None:
        host_cfg = base_url or config.get("base_url") or "https://api.openai.com/v1"
        model_cfg = (config.get("model") or DEFAULT_MODEL).strip()

        host = str(host_cfg).rstrip("/")
        super().__init__(host, **config)
        # Model defaults to the recommended judge if unspecified
        self.model = model_cfg
        self.timeout_seconds = float(config.get("timeout_seconds", 600.0))

        # Generation controls are intentionally omitted for judge calls (GPT-5 family ignores them)
        self.temperature = None
        self.max_output_tokens = None

        # Initialize session with connection pooling and retries
        self._session = requests.Session()
        
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,  # 1s, 2s, 4s, 8s, 16s
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
            raise_on_status=False,
            # Retry on connection errors (like RemoteDisconnected)
            connect=True,
            read=True,
        )
        
        # Support high concurrency (100+ threads)
        adapter = HTTPAdapter(
            pool_connections=100, 
            pool_maxsize=100, 
            max_retries=retry_strategy
        )
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, "_session"):
            self._session.close()

    def stream_chat_completion(
        self, model: str, prompt: str, **params: Any
    ) -> Response:
        """
        OpenAIClient is strictly for evaluation (judging) and does not support
        profiling or streaming metrics.
        """
        raise NotImplementedError(
            "OpenAIClient is eval-only and does not support stream_chat_completion."
        )

    def list_models(self) -> Sequence[str]:
        """OpenAIClient cannot list models; use the configured model."""
        raise NotImplementedError("OpenAIClient does not support listing models.")

    def health(self) -> bool:
        """OpenAIClient does not expose a health check endpoint."""
        raise NotImplementedError("OpenAIClient does not support health checks.")

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
        # Deliberately avoid forwarding temperature/max tokens; judge models ignore or reject them.

        response = self._chat_completion(self.model, "", **params)
        return response.content

    def _chat_completion(
        self, model: str, prompt: str, **params: Any
    ) -> Response:
        """Internal helper used by evaluation-only chat."""
        ipw_key = os.getenv("IPW_EVAL_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if ipw_key and openai_key and ipw_key != openai_key:
            api_key = ipw_key
        else:
            api_key = ipw_key or openai_key

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request_model = model or self.model

        if "messages" in params:
            messages = params["messages"]
        else:
            messages = [{"role": "user", "content": prompt}]

        payload: Dict[str, Any] = {
            "model": request_model,
            "messages": messages,
        }
        # Skip temperature/max tokens to avoid unsupported-parameter errors.

        exclude_keys = {"model", "messages", "prompt"}
        for k, v in params.items():
            if k not in exclude_keys:
                payload[k] = v

        wall_start = time.time()
        try:
            response = self._session.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except req_exc.HTTPError as exc:
            detail = ""
            resp = exc.response
            if resp is not None:
                try:
                    parsed = resp.json()
                    detail = f" | body: {parsed}"
                except Exception:
                    body_text = resp.text if resp is not None else ""
                    detail = f" | body: {body_text[:500]}"
            raise RuntimeError(f"OpenAIClient request failed: {exc}{detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"OpenAIClient request failed: {exc}") from exc

        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError("OpenAIClient received non-JSON response") from exc

        choices = data.get("choices") or []
        wall_end = time.time()
        if not choices:
            return Response(
                content="",
                usage=ChatUsage(0, 0, 0),
                time_to_first_token_ms=0.0,
                request_start_time=wall_start,
                request_end_time=wall_end,
            )

        first = choices[0]
        message = first.get("message") or {}
        content = message.get("content") or ""

        return Response(
            content=content,
            usage=ChatUsage(0, 0, 0),
            time_to_first_token_ms=0.0,
            request_start_time=wall_start,
            request_end_time=wall_end,
        )


__all__ = ["DEFAULT_MODEL", "OpenAIClient"]
