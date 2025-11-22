from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Mapping

from tqdm.auto import tqdm

from ..core.registry import AnalysisRegistry, DatasetRegistry
from ..core.types import DatasetRecord
from .base import AnalysisContext, AnalysisProvider, AnalysisResult
from .helpers import load_metrics_dataset, resolve_model_name

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _AccuracyCounters:
    correct: int = 0
    incorrect: int = 0
    unevaluated: int = 0


@AnalysisRegistry.register("accuracy")
class AccuracyAnalysis(AnalysisProvider):
    """
    Analysis that performs evaluation (if needed) and aggregates accuracy statistics.
    
    If records are missing evaluation data, this analysis will:
    1. Instantiate the original dataset provider.
    2. Call dataset.score() for each unevaluated record.
    3. Update and persist the results.
    """

    analysis_id = "accuracy"

    def run(self, context: AnalysisContext) -> AnalysisResult:
        results_dir = context.results_dir
        options = dict(context.options)
        requested_model = options.get("model")

        # Load the dataset (HuggingFace dataset object)
        dataset = load_metrics_dataset(results_dir)
        active_model = resolve_model_name(dataset, requested_model, results_dir)

        # Check if we need to run evaluation
        if self._needs_evaluation(dataset, active_model):
            dataset = self._run_evaluation(context, dataset, active_model)

        # Aggregate results
        counters = _AccuracyCounters()
        records: list[Dict[str, Any]] = []
        
        # Iterate directly over the HF dataset rows
        # We assume structure: row["model_metrics"][active_model]["evaluation"]
        for row in dataset:
            model_metrics = row.get("model_metrics") or {}
            metrics = model_metrics.get(active_model) or {}
            evaluation = _to_mapping(metrics.get("evaluation"))
            model_answers = row.get("model_answers") or {}
            model_answer = model_answers.get(active_model)
            
            records.append(
                {
                    "problem": row.get("problem"),
                    "reference_answer": row.get("answer"),
                    "model_answer": model_answer,
                    "evaluation": dict(evaluation) if evaluation else {},
                }
            )
            
            if not evaluation:
                counters.unevaluated += 1
                continue

            is_correct = evaluation.get("is_correct")
            if is_correct is True:
                counters.correct += 1
            elif is_correct is False:
                counters.incorrect += 1
            else:
                counters.unevaluated += 1

        total_scored = counters.correct + counters.incorrect
        accuracy = (
            counters.correct / total_scored if total_scored > 0 else None
        )

        summary_payload: Dict[str, Any] = {
            "model": active_model,
            "correct": counters.correct,
            "incorrect": counters.incorrect,
            "unevaluated": counters.unevaluated,
            "total_scored": total_scored,
            "accuracy": accuracy,
        }

        data_payload: Dict[str, Any] = {
            "per_model": {
                active_model: summary_payload,
            },
            "records": {
                active_model: records,
            },
        }

        warnings = []
        if counters.unevaluated:
            warnings.append(
                f"{counters.unevaluated} records remain unevaluated for model '{active_model}'."
            )

        artifact_payload = {
            "analysis": self.analysis_id,
            "summary": summary_payload,
            "warnings": warnings,
            "data": data_payload,
        }

        artifact_dir = results_dir / "analysis"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"{self.analysis_id}.json"
        artifact_path.write_text(json.dumps(artifact_payload, indent=2, default=str))

        return AnalysisResult(
            analysis=self.analysis_id,
            summary=summary_payload,
            data=data_payload,
            warnings=tuple(warnings),
            artifacts={"report": artifact_path},
        )

    def _needs_evaluation(self, dataset, model_name: str) -> bool:
        """Check if there are records missing evaluation data."""
        for row in dataset:
            model_metrics = row.get("model_metrics") or {}
            metrics = model_metrics.get(model_name) or {}
            evaluation = metrics.get("evaluation")
            # If evaluation is missing or is_correct is None, we need to evaluate
            if not evaluation or evaluation.get("is_correct") is None:
                return True
        return False

    def _run_evaluation(self, context: AnalysisContext, dataset, model_name: str):
        """Instantiate dataset provider and score records."""
        
        # 1. Resolve Dataset Provider
        summary_path = context.results_dir / "summary.json"
        if not summary_path.exists():
            LOGGER.warning("No summary.json found, cannot determine dataset provider.")
            return dataset

        try:
            summary = json.loads(summary_path.read_text())
            dataset_id = summary.get("dataset") or summary.get("profiler_config", {}).get("dataset_id")
            dataset_params = summary.get("profiler_config", {}).get("dataset_params", {})
            
            if not dataset_id:
                LOGGER.warning("Dataset ID not found in summary.")
                return dataset

            provider_cls = DatasetRegistry.get(dataset_id)
            provider = provider_cls(**dataset_params)
            
            if not hasattr(provider, "score") or not callable(provider.score):
                LOGGER.warning(f"Dataset provider '{dataset_id}' does not support scoring.")
                return dataset

        except Exception as e:
            LOGGER.error(f"Failed to instantiate dataset provider: {e}")
            return dataset

        # 2. Identify tasks
        # We need to map HF dataset rows back to DatasetRecord for scoring
        tasks = []
        for i, row in enumerate(dataset):
            model_metrics = row.get("model_metrics") or {}
            metrics = model_metrics.get(model_name) or {}
            evaluation = metrics.get("evaluation")
            
            if not evaluation or evaluation.get("is_correct") is None:
                response = row.get("model_answers", {}).get(model_name, "")
                # Reconstruct DatasetRecord
                raw_dataset_metadata = row.get("dataset_metadata")
                if isinstance(raw_dataset_metadata, Mapping):
                    dataset_metadata = dict(raw_dataset_metadata)
                elif raw_dataset_metadata is None:
                    dataset_metadata = {}
                else:
                    # HuggingFace may persist this field as a JSON string
                    dataset_metadata = {"dataset_metadata": raw_dataset_metadata}

                record = DatasetRecord(
                    problem=row.get("problem", ""),
                    answer=row.get("answer", ""),
                    subject=row.get("subject", ""),
                    dataset_metadata=dataset_metadata,
                )
                tasks.append((i, record, response))

        if not tasks:
            return dataset


        # 3. Execute scoring
        results = {}
        max_workers = 10  # Conservative limit
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._safe_score, provider, record, response): idx 
                for idx, record, response in tasks
            }
            
            with tqdm(total=len(tasks), desc="Scoring", unit="record") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    is_correct, meta = future.result()
                    results[idx] = (is_correct, meta)
                    pbar.update(1)

        # 4. Update dataset
        # We can't modify the HF dataset in place easily if it's memory mapped.
        # We use map() to create a new one.
        
        def update_row(row, idx):
            if idx in results:
                is_correct, meta = results[idx]
                
                # Ensure structure exists
                if "model_metrics" not in row:
                    row["model_metrics"] = {}
                if model_name not in row["model_metrics"]:
                    row["model_metrics"][model_name] = {}
                
                # Update evaluation field
                # We store it as a dict, consistent with schema
                row["model_metrics"][model_name]["evaluation"] = {
                    "is_correct": is_correct,
                    "metadata": json.dumps(meta, default=str),
                    # Config is legacy/optional now
                    "config": {}
                }
            return row

        updated_dataset = dataset.map(update_row, with_indices=True)
        
        # 5. Persist updated dataset
        try:
            import shutil
            temp_path = context.results_dir.with_name(context.results_dir.name + "_temp_evaluated_dataset")
            if temp_path.exists():
                shutil.rmtree(temp_path)
            
            updated_dataset.save_to_disk(str(temp_path))
            
            # Replace original with new
            backup_path = context.results_dir.with_suffix(".bak")
            if backup_path.exists():
                shutil.rmtree(backup_path)
                
            context.results_dir.rename(backup_path)
            temp_path.rename(context.results_dir)

            # Restore non-dataset artifacts (e.g., summary.json, analysis outputs)
            dataset_entries = {item.name for item in context.results_dir.iterdir()}
            for item in backup_path.iterdir():
                if item.name in dataset_entries:
                    continue
                if item.name in {"dataset_info.json", "state.json", "dataset_dict.json"}:
                    continue
                if item.name.startswith("data-") or item.name.endswith(".arrow"):
                    continue
                if item.is_dir() and (item / "dataset_info.json").exists():
                    continue

                target = context.results_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, target)
                else:
                    shutil.copy2(item, target)

            shutil.rmtree(backup_path)
            
        except Exception as e:
            LOGGER.error(f"Failed to save updated dataset: {e}")

        return updated_dataset

    def _safe_score(self, provider, record, response):
        try:
            return provider.score(record, response)
        except Exception as e:
            LOGGER.warning(f"Scoring failed: {e}")
            return None, {"error": str(e)}


def _to_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


__all__ = ["AccuracyAnalysis"]
