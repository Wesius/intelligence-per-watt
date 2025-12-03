"""Profiler runner orchestration."""

from __future__ import annotations

import logging
from importlib import metadata as importlib_metadata
import json
import math
import platform
import shutil
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence

import click
from datasets import Dataset
from tqdm.auto import tqdm

from ..clients.base import InferenceClient
from ..core.registry import ClientRegistry, DatasetRegistry
from ..core.types import DatasetRecord, GpuInfo, ProfilerConfig, Response, SystemInfo, TelemetryReading
from ..telemetry import EnergyMonitorCollector
from .hardware import derive_hardware_label
from .telemetry_session import TelemetrySample, TelemetrySession
from .types import (
    ComputeMetrics,
    EnergyMetrics,
    LatencyMetrics,
    MemoryMetrics,
    MetricStats,
    ModelMetrics,
    PowerComponentMetrics,
    PowerMetrics,
    ProfilingRecord,
    TokenMetrics,
)

LOGGER = logging.getLogger(__name__)


class ProfilerRunner:
    """Coordinate dataset iteration, inference calls, telemetry capture, and persistence."""

    def __init__(self, config: ProfilerConfig) -> None:
        self._config = config
        self._records: Dict[int, ProfilingRecord] = {}
        self._output_path: Optional[Path] = None
        self._output_path_hardware_label: Optional[str] = None
        self._output_dataset_label: Optional[str] = None
        self._hardware_label: Optional[str] = None
        self._system_info: Optional[SystemInfo] = None
        self._gpu_info: Optional[GpuInfo] = None
        self._all_samples: list[TelemetrySample] = []
        self._request_timings: Dict[int, tuple[float | None, float | None]] = {}

    def run(self) -> None:
        dataset = self._resolve_dataset(
            self._config.dataset_id, self._config.dataset_params
        )
        client: InferenceClient | None = None
        collector = EnergyMonitorCollector()

        try:
            client = self._resolve_client(
                self._config.client_id,
                self._config.client_base_url,
                self._config.client_params,
            )

            self._ensure_client_ready(client)

            with TelemetrySession(collector, buffer_seconds=3600.0) as telemetry:
                self._process_records(dataset, client, telemetry)

            if not self._records:
                return

            self._persist_records(dataset)
        finally:
            self._close_client(client)

    def _process_records(
        self,
        dataset,
        client,
        telemetry: TelemetrySession,
    ) -> None:
        dataset_size = dataset.size()
        if dataset_size <= 0:
            return

        target_queries = (
            self._config.max_queries
            if self._config.max_queries is not None
            else dataset_size
        )
        total_queries = min(target_queries, dataset_size)
        if total_queries <= 0:
            return

        if self._config.max_concurrency == 0:
            max_concurrency = total_queries
        else:
            max_concurrency = max(int(self._config.max_concurrency or 1), 1)
            
        pending_records: Dict[int, DatasetRecord] = {}
        prompt_iter = self._prompt_iterator(dataset, total_queries, pending_records)
        payload: MutableMapping[str, object] = dict(self._config.additional_parameters)
        completed = sum(1 for idx in self._records if idx < total_queries)

        with tqdm(
            total=total_queries,
            desc="Profiling",
            unit="query",
            initial=completed,
        ) as progress:
            for index, response in client.run_concurrent(
                self._config.model,
                prompt_iter,
                max_concurrency,
                **payload,
            ):
                current_readings = telemetry.readings()
                last_ts = self._all_samples[-1].timestamp if self._all_samples else -1.0
                for sample in current_readings:
                    if sample.timestamp > last_ts:
                        self._all_samples.append(sample)

                record = pending_records.pop(index, None)
                if record is None:
                    continue

                wall_time_now = time.time()
                start_time = response.request_start_time or wall_time_now
                end_time = response.request_end_time or start_time
                if end_time < start_time:
                    end_time = start_time

                self._request_timings[index] = (start_time, end_time)

                samples = list(telemetry.window(start_time, end_time))

                built = self._build_record(
                    index, record, response, samples, start_time, end_time
                )
                if built is not None:
                    self._records[index] = built
                progress.update(1)

        self._recompute_metrics()


    def _prompt_iterator(
        self,
        dataset,
        total_queries: int,
        pending_records: Dict[int, DatasetRecord],
    ) -> Iterator[tuple[int, str]]:
        for index, record in enumerate(dataset):
            if index >= total_queries:
                break
            if index in self._records:
                continue
            pending_records[index] = record
            yield index, record.problem

    def _build_record(
        self,
        index: int,
        record: DatasetRecord,
        response: Response,
        samples: Sequence[TelemetrySample],
        start_time: float,
        end_time: float,
    ) -> Optional[ProfilingRecord]:
        self._update_hardware_metadata(samples)
        telemetry_readings = [sample.reading for sample in samples]

        energy_metrics = self._compute_energy_metrics(telemetry_readings)
        power_metrics = PowerMetrics()

        temperature_stats = _stat_summary(
            [reading.temperature_celsius for reading in telemetry_readings]
        )
        cpu_memory_stats = _stat_summary(
            [reading.cpu_memory_usage_mb for reading in telemetry_readings]
        )
        gpu_memory_stats = _stat_summary(
            [reading.gpu_memory_usage_mb for reading in telemetry_readings]
        )

        usage = response.usage
        total_seconds = max(end_time - start_time, 0.0)

        prompt_tokens = usage.prompt_tokens if usage.prompt_tokens is not None else 0
        completion_tokens = (
            usage.completion_tokens if usage.completion_tokens is not None else 0
        )

        per_token_ms = None
        throughput_tokens = None
        if completion_tokens > 0 and total_seconds > 0:
            per_token_ms = (total_seconds * 1000.0) / completion_tokens
            throughput_tokens = completion_tokens / total_seconds

        latency_metrics = LatencyMetrics(
            per_token_ms=per_token_ms,
            throughput_tokens_per_sec=throughput_tokens,
            time_to_first_token_seconds=(
                response.time_to_first_token_ms / 1000.0
                if response.time_to_first_token_ms is not None
                else None
            ),
            total_query_seconds=total_seconds,
        )

        model_name = self._config.model

        model_metrics = ModelMetrics(
            compute_metrics=ComputeMetrics(),
            energy_metrics=energy_metrics,
            latency_metrics=latency_metrics,
            memory_metrics=MemoryMetrics(
                cpu_mb=cpu_memory_stats,
                gpu_mb=gpu_memory_stats,
            ),
            power_metrics=power_metrics,
            temperature_metrics=temperature_stats,
            token_metrics=TokenMetrics(
                input=prompt_tokens,
                output=completion_tokens,
                total=prompt_tokens + completion_tokens,
            ),
            gpu_info=self._gpu_info,
            system_info=self._system_info,
            lm_response=response.content,
        )

        record_payload = ProfilingRecord(
            dataset_index=index,
            problem=record.problem,
            answer=record.answer,
            request_start_time=start_time,
            request_end_time=end_time,
            dataset_metadata=dict(record.dataset_metadata),
            subject=record.subject,
            model_answers={model_name: response.content},
            model_metrics={model_name: model_metrics},
        )

        return record_payload

    def _compute_energy_metrics(
        self, readings: Sequence[TelemetryReading]
    ) -> EnergyMetrics:
        """Compute energy metrics from telemetry readings."""
        energy_values = [
            reading.energy_joules
            for reading in readings
            if reading.energy_joules is not None
        ]
        if not energy_values:
            return EnergyMetrics()

        start_value = energy_values[0]
        end_value = energy_values[-1]

        if not (
            math.isfinite(start_value)
            and math.isfinite(end_value)
            and start_value >= 0
            and end_value >= 0
        ):
            return EnergyMetrics()

        if end_value < start_value:
            return EnergyMetrics()

        per_query = end_value - start_value

        return EnergyMetrics(
            per_query_joules=per_query,
            total_joules=per_query,
        )

    def _update_hardware_metadata(self, readings: Sequence[TelemetrySample]) -> None:
        for sample in readings:
            reading = sample.reading
            if reading.system_info is not None:
                self._system_info = reading.system_info
            if reading.gpu_info is not None:
                self._gpu_info = reading.gpu_info

        candidate = derive_hardware_label(self._system_info, self._gpu_info)
        if candidate and (self._hardware_label in (None, "UNKNOWN_HW")):
            if candidate != self._hardware_label:
                self._output_path = None
                self._output_path_hardware_label = None
                self._output_dataset_label = None
            self._hardware_label = candidate

    def _get_output_path(self, dataset_label: str | None = None) -> Path:
        hardware_label = self._hardware_label or "UNKNOWN_HW"
        model_slug = _slugify_model(self._config.model)
        dataset_segment = dataset_label or self._config.dataset_id or "dataset"
        dataset_segment = str(dataset_segment).strip() or "dataset"
        concurrency_label = (
            f"bs{self._config.max_concurrency}" if self._config.max_concurrency else "bs1"
        )
        default_runs_dir = Path(__file__).resolve().parents[4] / "runs"
        base_dir = self._config.output_dir or default_runs_dir
        profile_dir = (
            f"profile_{hardware_label}_{model_slug}_{dataset_segment}_{concurrency_label}"
        ).strip("_")

        output_path = Path(base_dir) / profile_dir

        if (
            self._output_path is None
            or self._output_path_hardware_label != hardware_label
            or self._output_dataset_label != dataset_segment
        ):
            self._output_path = output_path
            self._output_path_hardware_label = hardware_label
            self._output_dataset_label = dataset_segment

        if self._hardware_label is None:
            self._hardware_label = hardware_label

        return self._output_path

    def _resolve_dataset(self, dataset_id: str, params: Mapping[str, Any]):
        try:
            dataset_cls = DatasetRegistry.get(dataset_id)
        except KeyError as exc:
            raise RuntimeError(f"Unknown dataset '{dataset_id}'") from exc

        try:
            return dataset_cls(**params)
        except TypeError as exc:
            raise RuntimeError(
                f"Failed to instantiate dataset '{dataset_id}' with params {params!r}: {exc}"
            ) from exc

    def _resolve_client(
        self,
        client_id: str,
        base_url: str | None,
        params: Mapping[str, Any],
    ) -> InferenceClient:
        try:
            client_cls = ClientRegistry.get(client_id)
        except KeyError as exc:
            raise RuntimeError(f"Unknown client '{client_id}'") from exc

        try:
            return client_cls(base_url, **params)
        except TypeError as exc:
            raise RuntimeError(
                f"Failed to instantiate client '{client_id}' with params {params!r}: {exc}"
            ) from exc

    def _ensure_client_ready(self, client: InferenceClient) -> None:
        if not client.health():
            raise RuntimeError(
                f"Client '{client.client_name}' at {getattr(client, 'base_url', '')} is unavailable"
            )
        client.prepare(self._config.model)

    def _recompute_metrics(self) -> None:
        if not self._all_samples:
            LOGGER.warning("No telemetry samples collected; skipping metrics.")
            return

        samples = sorted(self._all_samples, key=lambda s: s.timestamp)
        requests = []
        for idx, (r_start, r_end) in self._request_timings.items():
            if r_start is not None and r_end is not None:
                requests.append((r_start, r_end, idx))
        requests.sort(key=lambda x: x[0])

        request_allocations: Dict[int, list[tuple[float, float, float, float]]] = {
            idx: [] for idx in self._request_timings
        }

        active_requests = []
        req_ptr = 0
        num_reqs = len(requests)

        for i in range(1, len(samples)):
            s_prev, s_curr = samples[i - 1], samples[i]
            t_mid = (s_prev.timestamp + s_curr.timestamp) / 2

            # Add started requests
            while req_ptr < num_reqs and requests[req_ptr][0] <= t_mid:
                active_requests.append(requests[req_ptr])
                req_ptr += 1

            # Remove ended requests
            active_requests = [r for r in active_requests if r[1] >= t_mid]

            e_prev = s_prev.reading.energy_joules
            e_curr = s_curr.reading.energy_joules
            if e_prev is None or e_curr is None:
                continue

            delta_joules = max(0.0, e_curr - e_prev)
            power_watts = s_curr.reading.power_watts or 0.0
            concurrency = len(active_requests)

            if concurrency > 0:
                share_energy = delta_joules / concurrency
                share_power = power_watts / concurrency
                for _, _, idx in active_requests:
                    request_allocations[idx].append(
                        (share_power, share_energy, power_watts, delta_joules)
                    )

        model_name = self._config.model
        for idx, allocations in request_allocations.items():
            if idx not in self._records:
                continue

            record = self._records[idx]
            if model_name not in record.model_metrics:
                continue

            if not allocations:
                continue

            shared_powers = [x[0] for x in allocations]
            shared_energies = [x[1] for x in allocations]
            raw_powers = [x[2] for x in allocations]
            raw_energies = [x[3] for x in allocations]

            total_shared_energy = sum(shared_energies)
            total_raw_energy = sum(raw_energies)

            shared_power_stats = _stat_summary(shared_powers)
            raw_power_stats = _stat_summary(raw_powers)

            metrics = record.model_metrics[model_name]
            updated_power = PowerComponentMetrics(
                per_query_watts=shared_power_stats,
                total_watts=MetricStats(
                    avg=raw_power_stats.avg,
                    max=raw_power_stats.max,
                    median=raw_power_stats.median,
                    min=raw_power_stats.min,
                ),
            )
            record.model_metrics[model_name] = ModelMetrics(
                compute_metrics=metrics.compute_metrics,
                energy_metrics=EnergyMetrics(
                    per_query_joules=total_shared_energy,
                    total_joules=total_raw_energy,
                ),
                latency_metrics=metrics.latency_metrics,
                memory_metrics=metrics.memory_metrics,
                power_metrics=PowerMetrics(gpu=updated_power),
                temperature_metrics=metrics.temperature_metrics,
                token_metrics=metrics.token_metrics,
                gpu_info=metrics.gpu_info,
                system_info=metrics.system_info,
                lm_correctness=metrics.lm_correctness,
                lm_response=metrics.lm_response,
            )

    def _close_client(self, client: InferenceClient | None) -> None:
        if client is None:
            return
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                LOGGER.warning("Failed to close inference client cleanly", exc_info=True)

    def _persist_records(self, dataset) -> None:
        if not self._records:
            return

        dataset_label = (
            getattr(dataset, "dataset_id", None)
            or self._config.dataset_id
            or getattr(dataset, "dataset_name", None)
        )

        output_path = self._get_output_path(
            str(dataset_label).strip() or self._config.dataset_id
        )
        
        if output_path.exists():
            LOGGER.warning("Output path %s exists. Overwriting.", output_path)
            shutil.rmtree(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        ordered = [self._records[idx] for idx in sorted(self._records)]
        dataset_obj = Dataset.from_list([asdict(record) for record in ordered])
        dataset_obj.save_to_disk(str(output_path))

        summary = {
            "model": self._config.model,
            "profiler_config": _jsonify(asdict(self._config)),
            "dataset": getattr(dataset, "dataset_id", self._config.dataset_id),
            "dataset_name": getattr(dataset, "dataset_name", None),
            "hardware_label": self._hardware_label,
            "generated_at": time.time(),
            "total_queries": len(self._records),
            "system_info": asdict(self._system_info) if self._system_info else None,
            "gpu_info": asdict(self._gpu_info) if self._gpu_info else None,
            "output_dir": str(output_path),
            "versions": _get_versions(),
        }
        summary_path = output_path / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str))


def _stat_summary(values: Iterable[Optional[float]]) -> MetricStats:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return MetricStats()
    return MetricStats(
        avg=sum(filtered) / len(filtered),
        max=max(filtered),
        median=statistics.median(filtered),
        min=min(filtered),
    )


def _slugify_model(model: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in model).strip("_") or "model"


def _jsonify(value: Any) -> Any:
    """Recursively coerce values into JSON-serializable types."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonify(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_jsonify(item) for item in value]
    return value


def _get_versions() -> dict[str, str]:
    try:
        ipw_version = importlib_metadata.version("ipw")
    except importlib_metadata.PackageNotFoundError:
        ipw_version = "unknown"

    return {
        "ipw": ipw_version,
        "python": platform.python_version(),
    }
