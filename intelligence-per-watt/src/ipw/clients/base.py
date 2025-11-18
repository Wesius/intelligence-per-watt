from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import Any, Sequence

from ..core.types import Response


class InferenceClient(ABC):
    """
    Base class for inference service integrations.

    Subclasses must be registered with ``ClientRegistry`` to become discoverable.
    """

    client_id: str
    client_name: str

    def __init__(self, base_url: str, **config: Any) -> None:
        self.base_url = base_url
        self._config = config

    @abstractmethod
    def stream_chat_completion(
        self, model: str, prompt: str, **params: Any
    ) -> Response:
        """Run a streamed chat completion and return the aggregated response.

        Implementations should consume a streaming API so they can measure
        time-to-first-token latency, but must return a fully materialized
        ``Response`` object once the stream finishes. ``prompt`` contains the
        raw text to submit to the inference service.
        """

    def stream_chat_completion_batch(
        self, model: str, prompts: Sequence[str], **params: Any
    ) -> Sequence[Response]:
        """Run multiple prompts in a single call.

        Clients that support native batching should override this and dispatch a
        single combined request. The default falls back to sequential calls so
        callers can always rely on the method existing.
        """

        responses: list[Response] = []
        batch_start = time.perf_counter()
        for idx, prompt in enumerate(prompts):
            start = time.perf_counter()
            response = self.stream_chat_completion(model, prompt, **params)
            end = time.perf_counter()

            response.batch_start_offset_ms = (start - batch_start) * 1000.0
            response.batch_end_offset_ms = (end - batch_start) * 1000.0

            responses.append(response)

        return responses

    @abstractmethod
    def list_models(self) -> Sequence[str]:
        """Return the list of models exposed by the client."""

    @abstractmethod
    def health(self) -> bool:
        """Return True when the client is healthy and reachable."""

    def prepare(self, model: str) -> None:
        """Optional hook to perform warmup before serving requests."""
        return None


__all__ = ["InferenceClient"]
