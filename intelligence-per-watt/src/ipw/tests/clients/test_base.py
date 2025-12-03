"""Tests for InferenceClient base class."""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Sequence

import pytest
from ipw.clients.base import InferenceClient
from ipw.core.types import ChatUsage, Response


class ConcreteClient(InferenceClient):
    """Concrete implementation for testing."""

    client_id = "test"
    client_name = "Test Client"

    def run_concurrent(
        self,
        model: str,
        prompt_iter: Iterable[tuple[int, str]],
        max_in_flight: int,
        **params: Any,
    ) -> Iterator[tuple[int, Response]]:
        for index, _prompt in prompt_iter:
            yield index, Response(
                content="test response",
                usage=ChatUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                time_to_first_token_ms=100.0,
                request_start_time=0.0,
                request_end_time=1.0,
            )

    def stream_chat_completion(
        self,
        model: str,
        prompt: str,
        **params: Any,
    ) -> Response:
        return Response(
            content=f"test response for {prompt}",
            usage=ChatUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            time_to_first_token_ms=100.0,
            request_start_time=0.0,
            request_end_time=1.0,
        )

    def list_models(self) -> Sequence[str]:
        return ["model1", "model2"]

    def health(self) -> bool:
        return True


class TestInferenceClient:
    """Test InferenceClient abstract base class."""

    def test_initializes_with_base_url(self) -> None:
        client = ConcreteClient("http://localhost:8000")
        assert client.base_url == "http://localhost:8000"

    def test_stores_config_kwargs(self) -> None:
        client = ConcreteClient("http://localhost:8000", timeout=30, api_key="secret")
        assert client._config["timeout"] == 30
        assert client._config["api_key"] == "secret"

    def test_cannot_instantiate_abstract_class(self) -> None:
        # InferenceClient is abstract and cannot be instantiated
        with pytest.raises(TypeError, match="abstract"):
            InferenceClient("http://localhost")

    def test_requires_all_abstract_methods(self) -> None:
        # Missing stream_chat_completion
        with pytest.raises(TypeError):

            class IncompleteClient1(InferenceClient):
                client_id = "incomplete"
                client_name = "Incomplete"

                def list_models(self) -> Sequence[str]:
                    return []

                def health(self) -> bool:
                    return True

            IncompleteClient1("http://localhost")

    def test_concrete_client_works(self) -> None:
        client = ConcreteClient("http://localhost:8000")

        # All methods should work
        assert client.health() is True
        assert client.list_models() == ["model1", "model2"]
        responses = list(client.run_concurrent("model", [(0, "prompt")], 1))
        assert responses[0][1].content == "test response"

    def test_passes_params_to_implementation(self) -> None:
        class ParamsTestClient(InferenceClient):
            client_id = "params_test"
            client_name = "Params Test"
            received_params = {}

            def run_concurrent(
                self,
                model: str,
                prompt_iter: Iterable[tuple[int, str]],
                max_in_flight: int,
                **params: Any,
            ) -> Iterator[tuple[int, Response]]:
                self.received_params = params
                for item in prompt_iter:
                    yield item[0], Response(
                        content="test",
                        usage=ChatUsage(
                            prompt_tokens=1, completion_tokens=1, total_tokens=2
                        ),
                        time_to_first_token_ms=100.0,
                        request_start_time=0.0,
                        request_end_time=1.0,
                    )

            def list_models(self) -> Sequence[str]:
                return []

            def health(self) -> bool:
                return True

            def stream_chat_completion(
                self,
                model: str,
                prompt: str,
                **params: Any,
            ) -> Response:
                return Response(
                    content="streamed",
                    usage=ChatUsage(
                        prompt_tokens=1, completion_tokens=1, total_tokens=2
                    ),
                    time_to_first_token_ms=100.0,
                    request_start_time=0.0,
                    request_end_time=1.0,
                )

        client = ParamsTestClient("http://localhost")
        list(
            client.run_concurrent(
                "model",
                [(0, "prompt")],
                max_in_flight=1,
                temperature=0.7,
                top_p=0.9,
            )
        )

        assert client.received_params["temperature"] == 0.7
        assert client.received_params["top_p"] == 0.9
