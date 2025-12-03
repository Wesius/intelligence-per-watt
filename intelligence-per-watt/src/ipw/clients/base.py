from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any, Sequence
import time

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
        """
        Run a streamed chat completion and return the aggregated response.

        Implementations must populate usage, time_to_first_token_ms, and may also
        provide request_start_time/request_end_time for precise wall timings.
        """

    def run_concurrent(
        self,
        model: str,
        prompt_iter: Iterable[tuple[int, str]],
        max_in_flight: int,
        **params: Any,
    ) -> Iterator[tuple[int, Response]]:
        """
        Default sequential implementation that wraps ``stream_chat_completion``.

        Clients can override for true concurrency. request_start/end timestamps
        are filled if the underlying implementation does not populate them.
        """
        for index, prompt in prompt_iter:
            wall_start = time.time()
            response = self.stream_chat_completion(model, prompt, **params)
            wall_end = time.time()
            if response.request_start_time is None:
                response.request_start_time = wall_start
            if response.request_end_time is None:
                response.request_end_time = wall_end
            yield index, response

    @abstractmethod
    def list_models(self) -> Sequence[str]:
        """Return the list of models exposed by the client."""

    @abstractmethod
    def health(self) -> bool:
        """Return True when the client is healthy and reachable."""

    def prepare(self, model: str) -> None:
        """Optional hook to perform warmup before serving requests."""
        return None

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        """
        Synchronous chat completion helper.

        Implementations should return the generated text for the given prompts.
        Subclasses that don't support chat may rely on this default, which
        raises to signal the capability is unavailable.
        """
        raise NotImplementedError


__all__ = ["InferenceClient"]
