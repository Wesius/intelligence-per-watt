"""Tests for profiler runner orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from dataclasses import asdict

import pytest
from ipw.core.types import (
    ChatUsage,
    DatasetRecord,
    ProfilerConfig,
    Response,
    TelemetryReading,
)
from ipw.execution.runner import ProfilerRunner, _slugify_model, _stat_summary
from ipw.execution.telemetry_session import TelemetrySample
from ipw.execution.types import ModelMetrics, ProfilingRecord


def _build_response(content: str = "response") -> Response:
    return Response(
        content=content,
        usage=ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        time_to_first_token_ms=100.0,
        request_start_time=0.0,
        request_end_time=1.0,
    )


class TestStatSummary:
    """Test statistical summary computation."""

    def test_computes_stats_from_values(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = _stat_summary(values)

        assert stats.avg == 3.0
        assert stats.min == 1.0
        assert stats.max == 5.0
        assert stats.median == 3.0

    def test_filters_none_values(self) -> None:
        values = [1.0, None, 3.0, None, 5.0]
        stats = _stat_summary(values)

        assert stats.avg == 3.0
        assert stats.min == 1.0
        assert stats.max == 5.0
        assert stats.median == 3.0

    def test_returns_none_stats_for_empty(self) -> None:
        values = []
        stats = _stat_summary(values)

        assert stats.avg is None
        assert stats.min is None
        assert stats.max is None
        assert stats.median is None

    def test_returns_none_stats_for_all_none(self) -> None:
        values = [None, None, None]
        stats = _stat_summary(values)

        assert stats.avg is None
        assert stats.min is None
        assert stats.max is None
        assert stats.median is None

    def test_handles_single_value(self) -> None:
        values = [42.0]
        stats = _stat_summary(values)

        assert stats.avg == 42.0
        assert stats.min == 42.0
        assert stats.max == 42.0
        assert stats.median == 42.0


class TestSlugifyModel:
    """Test model name slugification."""

    def test_replaces_special_chars_with_underscores(self) -> None:
        assert _slugify_model("llama-3.2:1b") == "llama_3_2_1b"

    def test_strips_leading_trailing_underscores(self) -> None:
        assert _slugify_model("_model_") == "model"

    def test_preserves_alphanumeric(self) -> None:
        assert _slugify_model("llama32") == "llama32"

    def test_returns_model_for_empty_string(self) -> None:
        assert _slugify_model("") == "model"

    def test_returns_model_for_all_special_chars(self) -> None:
        assert _slugify_model("!!!") == "model"


class TestProfilerRunner:
    """Test ProfilerRunner orchestration."""

    def test_initializes_with_config(self) -> None:
        config = ProfilerConfig(
            model="test-model",
            client_id="ollama",
            dataset_id="ipw",
        )
        runner = ProfilerRunner(config)
        assert runner._config == config

    @patch("ipw.execution.runner.DatasetRegistry")
    @patch("ipw.execution.runner.ClientRegistry")
    @patch("ipw.execution.runner.EnergyMonitorCollector")
    @patch("ipw.execution.runner.TelemetrySession")
    @patch("ipw.execution.runner.Dataset")
    def test_run_creates_output_directory(
        self,
        mock_dataset_class: Mock,
        mock_session: Mock,
        mock_collector: Mock,
        mock_client_registry: Mock,
        mock_dataset_registry: Mock,
        tmp_path: Path,
    ) -> None:
        # Setup mocks
        mock_dataset = MagicMock()
        mock_dataset.size.return_value = 1
        mock_dataset.__iter__.return_value = iter(
            [DatasetRecord(problem="test", answer="answer", subject="math")]
        )
        mock_dataset.dataset_id = "test"
        mock_dataset.dataset_name = "Test Dataset"
        mock_dataset_registry.get.return_value = Mock(return_value=mock_dataset)

        mock_client = Mock()
        mock_client.health.return_value = True

        def _run_concurrent(model, prompt_iter, max_in_flight, **_):
            for index, _prompt in prompt_iter:
                yield index, _build_response()

        mock_client.run_concurrent.side_effect = _run_concurrent
        mock_client_registry.get.return_value = Mock(return_value=mock_client)

        mock_collector_instance = Mock()
        mock_collector.return_value = mock_collector_instance

        mock_telemetry = Mock()
        mock_telemetry.window.return_value = []
        mock_telemetry.readings.return_value = []
        mock_session.return_value.__enter__.return_value = mock_telemetry

        # Mock Dataset.from_list to return a mock with save_to_disk that creates the directory
        mock_hf_dataset = Mock()

        def mock_save_to_disk(path: str):
            Path(path).mkdir(parents=True, exist_ok=True)

        mock_hf_dataset.save_to_disk = mock_save_to_disk
        mock_dataset_class.from_list.return_value = mock_hf_dataset

        config = ProfilerConfig(
            model="test-model",
            client_id="test-client",
            dataset_id="test-dataset",
            output_dir=tmp_path,
        )
        runner = ProfilerRunner(config)
        runner.run()

        # Check that output directory was created
        assert (tmp_path / "profile_UNKNOWN_HW_test_model_bs1").exists()
        # Check that summary.json was written
        summary_path = tmp_path / "profile_UNKNOWN_HW_test_model_bs1" / "summary.json"
        assert summary_path.exists()

        summary = json.loads(summary_path.read_text())
        assert summary["profiler_config"]["model"] == "test-model"
        assert "versions" in summary

    def test_recompute_metrics_shares_energy(self) -> None:
        config = ProfilerConfig(
            model="test-model",
            client_id="test",
            dataset_id="test",
        )
        runner = ProfilerRunner(config)
        
        # Setup records
        record = DatasetRecord(problem="test", answer="", subject="math")
        # _build_record now requires index, start_time, end_time
        # Initial call to get dummy metrics (not used, just to create structure)
        dummy_record_0 = runner._build_record(
            0, record, _build_response("response-0"), [], 0.0, 1.0
        )
        dummy_record_1 = runner._build_record(
            1, record, _build_response("response-1"), [], 0.0, 1.0
        )
        
        runner._records = {
            0: dummy_record_0,
            1: dummy_record_1,
        }
        
        # Manually set timings for specific test scenario
        # Request 0: [0, 2]
        # Request 1: [1, 3]
        
        runner._request_timings = {
            0: (0.0, 2.0),
            1: (1.0, 3.0),
        }
        
        # Samples:
        # t=0.0, E=0
        # t=1.0, E=100 (Delta 100, Active: {0}) -> Req 0 gets 100
        # t=2.0, E=200 (Delta 100, Active: {0, 1}) -> Req 0 gets 50, Req 1 gets 50
        # t=3.0, E=300 (Delta 100, Active: {1}) -> Req 1 gets 100
        
        # Total Req 0: 150
        # Total Req 1: 150
        
        runner._all_samples = [
            TelemetrySample(timestamp=0.0, reading=TelemetryReading(energy_joules=0.0, power_watts=100.0)),
            TelemetrySample(timestamp=1.0, reading=TelemetryReading(energy_joules=100.0, power_watts=100.0)),
            TelemetrySample(timestamp=2.0, reading=TelemetryReading(energy_joules=200.0, power_watts=100.0)),
            TelemetrySample(timestamp=3.0, reading=TelemetryReading(energy_joules=300.0, power_watts=100.0)),
        ]
        
        runner._recompute_metrics()
        
        m0 = runner._records[0].model_metrics["test-model"].energy_metrics
        m1 = runner._records[1].model_metrics["test-model"].energy_metrics
        
        # Shared/Per-Query (Amortized)
        assert m0.per_query_joules == 150.0
        assert m1.per_query_joules == 150.0

        # Raw/Total (System Load during window)
        # Req 0 active during [0, 1] (100J) and [1, 2] (100J) -> Total 200J
        # Req 1 active during [1, 2] (100J) and [2, 3] (100J) -> Total 200J
        assert m0.total_joules == 200.0
        assert m1.total_joules == 200.0
        
        # Check power sharing
        # t=1.0 interval (0-1): Power 100, Concurrency 1 -> Shared 100W, Raw 100W
        # t=2.0 interval (1-2): Power 100, Concurrency 2 -> Shared 50W, Raw 100W
        # t=3.0 interval (2-3): Power 100, Concurrency 1 -> Shared 100W, Raw 100W
        
        p0 = runner._records[0].model_metrics["test-model"].power_metrics.gpu.per_query_watts
        p0_total = runner._records[0].model_metrics["test-model"].power_metrics.gpu.total_watts
        
        p1 = runner._records[1].model_metrics["test-model"].power_metrics.gpu.per_query_watts
        p1_total = runner._records[1].model_metrics["test-model"].power_metrics.gpu.total_watts
        
        # Req 0 sees Shared=[100, 50], Raw=[100, 100]
        assert p0.max == 100.0
        assert p0.min == 50.0
        assert p0.avg == 75.0
        assert p0_total.avg == 100.0
        
        # Req 1 sees Shared=[50, 100], Raw=[100, 100]
        assert p1.max == 100.0
        assert p1.min == 50.0
        assert p1.avg == 75.0
        assert p1_total.avg == 100.0

    @patch("ipw.execution.runner.tqdm")
    def test_process_records_respects_max_concurrency(
        self,
        mock_tqdm: Mock,
    ) -> None:
        class DummyDataset:
            def __init__(self, records):
                self._records = records

            def size(self) -> int:
                return len(self._records)

            def __iter__(self):
                return iter(self._records)

        records = [
            DatasetRecord(problem=f"prompt-{idx}", answer="", subject="math")
            for idx in range(3)
        ]
        dataset = DummyDataset(records)

        telemetry = Mock()
        telemetry.window.side_effect = lambda *_: []
        telemetry.readings.return_value = []

        client = Mock()

        config = ProfilerConfig(
            model="test-model",
            client_id="test-client",
            dataset_id="test-dataset",
            max_concurrency=2,
        )
        runner = ProfilerRunner(config)
        call_count = 0

        progress = Mock()
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = progress
        mock_cm.__exit__.return_value = None
        mock_tqdm.return_value = mock_cm

        def fake_run(model, prompt_iter, max_in_flight, **_):
            assert max_in_flight == 2
            for index, prompt in prompt_iter:
                nonlocal call_count
                call_count += 1
                yield index, _build_response(prompt)

        client.run_concurrent.side_effect = fake_run

        runner._process_records(dataset, client, telemetry)

        assert call_count == 3
        assert len(runner._records) == 3

    def test_process_records_orders_out_of_order_responses(self) -> None:
        records = [
            DatasetRecord(problem=f"prompt-{idx}", answer="", subject="math")
            for idx in range(3)
        ]

        class DummyDataset:
            def __init__(self, entries):
                self._entries = entries

            def size(self) -> int:
                return len(self._entries)

            def __iter__(self):
                return iter(self._entries)

        dataset = DummyDataset(records)
        telemetry = Mock()
        telemetry.window.side_effect = lambda *_: []
        telemetry.readings.return_value = []

        class OutOfOrderClient:
            def run_concurrent(self, model, prompt_iter, max_in_flight, **_):
                buffered = list(prompt_iter)
                yield buffered[2][0], _build_response("third")
                yield buffered[0][0], _build_response("first")
                yield buffered[1][0], _build_response("second")

            def health(self):
                return True

        client = OutOfOrderClient()
        config = ProfilerConfig(
            model="test-model",
            client_id="test-client",
            dataset_id="test-dataset",
            max_concurrency=2,
        )
        runner = ProfilerRunner(config)
        runner._process_records(dataset, client, telemetry)

        assert sorted(runner._records.keys()) == [0, 1, 2]
        assert runner._records[0].model_answers["test-model"] == "first"
        assert runner._records[1].model_answers["test-model"] == "second"
        assert runner._records[2].model_answers["test-model"] == "third"

    @patch("ipw.execution.runner.DatasetRegistry")
    @patch("ipw.execution.runner.ClientRegistry")
    def test_raises_on_unknown_dataset(
        self,
        mock_client_registry: Mock,
        mock_dataset_registry: Mock,
    ) -> None:
        mock_dataset_registry.get.side_effect = KeyError("unknown")

        config = ProfilerConfig(
            model="test-model",
            client_id="test-client",
            dataset_id="unknown",
        )
        runner = ProfilerRunner(config)

        with pytest.raises(RuntimeError, match="Unknown dataset"):
            runner.run()

    @patch("ipw.execution.runner.DatasetRegistry")
    @patch("ipw.execution.runner.ClientRegistry")
    def test_raises_on_unknown_client(
        self,
        mock_client_registry: Mock,
        mock_dataset_registry: Mock,
    ) -> None:
        mock_dataset_registry.get.return_value = Mock(return_value=Mock())
        mock_client_registry.get.side_effect = KeyError("unknown")

        config = ProfilerConfig(
            model="test-model",
            client_id="unknown",
            dataset_id="test-dataset",
        )
        runner = ProfilerRunner(config)

        with pytest.raises(RuntimeError, match="Unknown client"):
            runner.run()

    @patch("ipw.execution.runner.DatasetRegistry")
    @patch("ipw.execution.runner.ClientRegistry")
    @patch("ipw.execution.runner.EnergyMonitorCollector")
    def test_raises_when_client_unhealthy(
        self,
        mock_collector: Mock,
        mock_client_registry: Mock,
        mock_dataset_registry: Mock,
    ) -> None:
        mock_dataset = Mock()
        mock_dataset_registry.get.return_value = Mock(return_value=mock_dataset)

        mock_client = Mock()
        mock_client.health.return_value = False
        mock_client.client_name = "test-client"
        mock_client_registry.get.return_value = Mock(return_value=mock_client)

        mock_collector.return_value = Mock()

        config = ProfilerConfig(
            model="test-model",
            client_id="test-client",
            dataset_id="test-dataset",
        )
        runner = ProfilerRunner(config)

        with pytest.raises(RuntimeError, match="unavailable"):
            runner.run()

    def test_build_record_creates_model_metrics(self) -> None:
        config = ProfilerConfig(
            model="test-model",
            client_id="test",
            dataset_id="test",
        )
        runner = ProfilerRunner(config)

        record = DatasetRecord(problem="test", answer="answer", subject="math")
        response = Response(
            content="response",
            usage=ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            time_to_first_token_ms=100.0,
            request_start_time=0.0,
            request_end_time=2.0,
        )
        samples = [
            TelemetrySample(
                timestamp=1.0,
                reading=TelemetryReading(
                    energy_joules=100.0,
                    power_watts=50.0,
                ),
            ),
            TelemetrySample(
                timestamp=2.0,
                reading=TelemetryReading(
                    energy_joules=150.0,
                    power_watts=50.0,
                ),
            ),
        ]

        result = runner._build_record(0, record, response, samples, 0.0, 2.0)

        assert result is not None
        assert result.dataset_index == 0
        assert result.request_start_time == 0.0
        assert result.request_end_time == 2.0
        assert "test-model" in result.model_metrics
        metrics = result.model_metrics["test-model"]
        assert metrics.token_metrics.input == 10
        assert metrics.token_metrics.output == 5
        assert metrics.token_metrics.total == 15

    def test_build_record_handles_zero_completion_tokens(self) -> None:
        config = ProfilerConfig(
            model="test-model",
            client_id="test",
            dataset_id="test",
        )
        runner = ProfilerRunner(config)

        record = DatasetRecord(problem="test", answer="answer", subject="math")
        response = Response(
            content="",
            usage=ChatUsage(prompt_tokens=10, completion_tokens=0, total_tokens=10),
            time_to_first_token_ms=0.0,
            request_start_time=0.0,
            request_end_time=1.0,
        )
        samples = []

        result = runner._build_record(0, record, response, samples, 0.0, 1.0)

        assert result is not None
        assert result.dataset_index == 0
        assert result.request_start_time == 0.0
        assert result.request_end_time == 1.0
        metrics = result.model_metrics["test-model"]
        assert metrics.latency_metrics.per_token_ms is None
        assert metrics.latency_metrics.throughput_tokens_per_sec is None

    def test_build_record_computes_throughput(self) -> None:
        config = ProfilerConfig(
            model="test-model",
            client_id="test",
            dataset_id="test",
        )
        runner = ProfilerRunner(config)

        record = DatasetRecord(problem="test", answer="answer", subject="math")
        response = Response(
            content="response",
            usage=ChatUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            time_to_first_token_ms=100.0,
            request_start_time=0.0,
            request_end_time=1.0,
        )
        samples = []

        # 10 tokens in 1 second = 10 tokens/sec
        result = runner._build_record(0, record, response, samples, 0.0, 1.0)

        assert result is not None
        assert result.dataset_index == 0
        assert result.request_start_time == 0.0
        assert result.request_end_time == 1.0
        metrics = result.model_metrics["test-model"]
        assert metrics.latency_metrics.throughput_tokens_per_sec == 10.0
        assert metrics.latency_metrics.per_token_ms == 100.0

    def test_get_output_path_includes_hardware_and_model(self, tmp_path: Path) -> None:
        config = ProfilerConfig(
            model="llama-3.2:1b",
            client_id="test",
            dataset_id="test",
            output_dir=tmp_path,
        )
        runner = ProfilerRunner(config)
        runner._hardware_label = "RTX3090"

        path = runner._get_output_path()
        assert "RTX3090" in str(path)
        assert "llama_3_2_1b" in str(path)
        assert "_bs1" in str(path)

    @patch("ipw.execution.runner.DatasetRegistry")
    @patch("ipw.execution.runner.ClientRegistry")
    @patch("ipw.execution.runner.EnergyMonitorCollector")
    @patch("ipw.execution.runner.TelemetrySession")
    @patch("ipw.execution.runner.Dataset")
    def test_resumable_run_skips_processed_queries(
        self,
        mock_dataset_class: Mock,
        mock_session: Mock,
        mock_collector: Mock,
        mock_client_registry: Mock,
        mock_dataset_registry: Mock,
        tmp_path: Path,
    ) -> None:
        # Setup mocks for the first (interrupted) run
        mock_dataset_full = MagicMock()
        mock_dataset_full.size.return_value = 5 # Total 5 queries
        mock_dataset_full.__iter__.return_value = iter(
            [DatasetRecord(problem=f"p{i}", answer=f"a{i}", subject="m") for i in range(5)]
        )
        mock_dataset_full.dataset_name = "Mock Dataset" # Ensure dataset_name is a string
        mock_dataset_registry.get.return_value = Mock(return_value=mock_dataset_full)

        mock_client = Mock()
        mock_client.health.return_value = True

        # Simulate client returning only first 2 responses for first run
        def _run_concurrent_partial(model, prompt_iter, max_in_flight, **_):
            for index, _prompt in prompt_iter:
                if index < 2: # Only yield first 2
                    yield index, _build_response(f"resp-{index}")

        mock_client.run_concurrent.side_effect = _run_concurrent_partial
        mock_client_registry.get.return_value = Mock(return_value=mock_client)

        mock_telemetry = Mock()
        mock_telemetry.window.return_value = []
        mock_telemetry.readings.return_value = []
        mock_session.return_value.__enter__.return_value = mock_telemetry

        # Simulate saving the first 2 records
        output_dir = tmp_path / "profile_UNKNOWN_HW_test_model_bs1"
        output_dir.mkdir()
        
        # Create a dummy Dataset that mimics what would be saved
        partial_records = [
            ProfilingRecord(
                dataset_index=0, problem="p0", answer="a0", request_start_time=0.0, request_end_time=1.0,
                model_answers={"test-model": "resp-0"},
                model_metrics={"test-model": ModelMetrics()}
            ),
            ProfilingRecord(
                dataset_index=1, problem="p1", answer="a1", request_start_time=0.0, request_end_time=1.0,
                model_answers={"test-model": "resp-1"},
                model_metrics={"test-model": ModelMetrics()}
            ),
        ]
        mock_dataset_class.from_list(asdict(r) for r in partial_records).save_to_disk(str(output_dir))

        # --- Second run (resumption) ---
        mock_loaded_dataset_for_resume = MagicMock() # Use MagicMock for iterability
        mock_loaded_dataset_for_resume.__iter__.return_value = iter(asdict(r) for r in partial_records)
        mock_dataset_class.load_from_disk.return_value = mock_loaded_dataset_for_resume
        
        # Client should now only be called for remaining queries (index 2, 3, 4)
        mock_client_registry.get.reset_mock() # Reset mock to check calls in resume run
        mock_client.run_concurrent.reset_mock()
        mock_client.run_concurrent.side_effect = None # Clear previous side effect
        
        processed_indices_in_resume = []
        def _run_concurrent_resume(model, prompt_iter, max_in_flight, **_):
            for index, _prompt in prompt_iter:
                processed_indices_in_resume.append(index)
                yield index, _build_response(f"resp-{index}")

        mock_client.run_concurrent.side_effect = _run_concurrent_resume
        mock_client_registry.get.return_value = Mock(return_value=mock_client)

        config = ProfilerConfig(
            model="test-model",
            client_id="test-client",
            dataset_id="test-dataset",
            output_dir=tmp_path,
        )
        runner = ProfilerRunner(config)
        runner.run()

        # Verify that client.run_concurrent was called with prompts for index 2, 3, 4
        assert processed_indices_in_resume == [2, 3, 4]
        assert len(runner._records) == 5 # All 5 records should be present after resume

