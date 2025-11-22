from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import (Any, Dict, Iterable, Iterator, MutableMapping, Optional,
                    Tuple)

from datasets import load_from_disk

from ...core.registry import (ClientRegistry, DatasetRegistry,
                              EvaluationRegistry)
from ...core.types import DatasetRecord
from ..base import DatasetProvider

_DEFAULT_DATASET_DIR = "mixed_1k_seed1_base"


def _default_dataset_path() -> Path:
    base = resources.files("ipw.datasets.ipw") / "data" / _DEFAULT_DATASET_DIR
    return Path(base)


@DatasetRegistry.register("ipw")
class IPWDataset(DatasetProvider):
    """Dataset provider for the bundled Intelligence Per Watt benchmark."""

    dataset_name = "Intelligence Per Watt"
    dataset_id = "ipw"

    def __init__(self) -> None:
        self._path = _default_dataset_path()
        if not self._path.exists():
            raise FileNotFoundError(f"Dataset location not found: {self._path}")
        self._records = tuple(self._load_all_records())

    def iter_records(self) -> Iterable[DatasetRecord]:
        return iter(self._records)

    def _load_all_records(self) -> Iterable[DatasetRecord]:
        if self._path.is_dir():
            yield from self._load_from_dataset_dir(self._path)
        else:
            yield from (
                record
                for record in self._load_from_jsonl(self._path)
                if self._is_valid(record)
            )

    def _load_from_dataset_dir(self, directory: Path) -> Iterable[DatasetRecord]:
        dataset = load_from_disk(str(directory))
        if isinstance(dataset, dict):
            hf_dataset = next(iter(dataset.values()))
        else:
            hf_dataset = dataset

        raw_records: Iterable[MutableMapping[str, Any]] = (
            hf_dataset if isinstance(hf_dataset, list) else hf_dataset.to_list()
        )
        for raw in raw_records:
            record = self._parse_record(raw)
            if self._is_valid(record):
                yield record

    def _load_from_jsonl(self, file_path: Path) -> Iterator[DatasetRecord]:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                raw: Dict[str, Any] = json.loads(stripped)
                yield self._parse_record(raw)

    def _parse_record(self, raw: Dict[str, Any]) -> DatasetRecord:
        problem = str(raw.get("problem") or raw.get("prompt") or "").strip()
        answer = str(raw.get("answer") or raw.get("expected_answer") or "").strip()
        subject = str(raw.get("subject") or "general").strip() or "general"

        dataset_metadata = dict(raw)
        return DatasetRecord(
            problem=problem,
            answer=answer,
            subject=subject,
            dataset_metadata=dataset_metadata,
        )

    def _is_valid(self, record: DatasetRecord) -> bool:
        return bool(
            record.problem
            and record.answer
            and record.subject
            and record.dataset_metadata
        )

    def size(self) -> int:
        return len(self._records)

    def verify_requirements(self) -> list[str]:
        issues: list[str] = []
        # Evaluation uses the OpenAI-compatible client; prefer IPW_EVAL_API_KEY, fallback to OPENAI_API_KEY.
        import os

        if not (os.getenv("IPW_EVAL_API_KEY") or os.getenv("OPENAI_API_KEY")):
            issues.append(
                "Missing evaluation API key. Set IPW_EVAL_API_KEY (preferred) or OPENAI_API_KEY for scoring."
            )
        return issues

    def score(
        self,
        record: DatasetRecord,
        response: str,
    ) -> Tuple[Optional[bool], Dict[str, object]]:
        """
        Delegate scoring to a dataset-specific evaluation handler based on the
        embedded metadata in this mixed dataset.
        """
        raw_meta = record.dataset_metadata.get("dataset_metadata")
        if not isinstance(raw_meta, str):
            raise RuntimeError("Missing or invalid 'dataset_metadata' field for scoring.")

        meta = json.loads(raw_meta)
        config = meta.get("config") or {}
        
        # Use a mapping from actual dataset_name to evaluation_method (handler key)
        actual_dataset_name = config.get("dataset_name")
        
        # Define the mapping
        VERIFICATION_MAPPING = {
            "allenai/WildChat": "wildchat",
            "facebook/natural_reasoning": "natural_reasoning",
            "lmsys/lmsys-chat-1m": "wildchat",
        }

        evaluation_method = VERIFICATION_MAPPING.get(actual_dataset_name)

        if not evaluation_method:
            raise RuntimeError(
                f"Could not determine evaluation method for dataset: {actual_dataset_name}. "
                f"Supported datasets: {', '.join(sorted(VERIFICATION_MAPPING.keys()))}"
            )

        # Instantiate the OpenAI client for evaluation
        # Configuration is now expected via environment variables or defaults in OpenAIClient
        eval_client = ClientRegistry.create("openai")

        handler = EvaluationRegistry.create(evaluation_method, client=eval_client)

        problem = record.problem
        reference = record.answer

        is_correct, eval_meta = handler.evaluate(
            problem=problem,
            reference=reference,
            model_answer=response,
            metadata=meta,
        )
        return is_correct, eval_meta


__all__ = ["IPWDataset"]
