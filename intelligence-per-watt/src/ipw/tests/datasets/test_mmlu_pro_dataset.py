from __future__ import annotations

from typing import Any, Iterable, MutableMapping
from unittest.mock import MagicMock, patch

import pytest

from ipw.datasets.mmlu_pro import MMLUProDataset


class _FakeDataset(list):
    def to_list(self) -> list[MutableMapping[str, Any]]:
        return list(self)


def _patch_load(rows: Iterable[MutableMapping[str, Any]], monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeDataset(rows)  # type: ignore[arg-type]

    def _loader(*args, **kwargs):
        return fake

    monkeypatch.setattr("ipw.datasets.mmlu_pro.load_dataset", _loader)


def test_mmlu_pro_dataset_parses_question(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "question": "What is 1+1?",
            "options": ["0", "2", "3"],
            "answer": "B",
            "category": "math",
            "question_id": 42,
            "answer_index": 1,
        }
    ]
    _patch_load(rows, monkeypatch)

    dataset = MMLUProDataset()
    record = next(iter(dataset.iter_records()))

    assert "Options" in record.problem
    assert record.answer == "B"
    assert record.subject == "math"
    assert record.dataset_metadata["question_id"] == 42


def test_mmlu_pro_dataset_score(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "question": "What is 2+2?",
            "options": ["1", "4"],
            "answer": "B",
            "category": "math",
        }
    ]
    _patch_load(rows, monkeypatch)

    dataset = MMLUProDataset()
    record = next(iter(dataset.iter_records()))

    mock_handler = MagicMock()
    mock_handler.evaluate.return_value = (True, {"mock": True})

    with patch("ipw.core.registry.ClientRegistry.create") as mock_client_create, patch(
        "ipw.core.registry.EvaluationRegistry.create"
    ) as mock_eval_create:
        mock_client = MagicMock()
        mock_client_create.return_value = mock_client
        mock_eval_create.return_value = mock_handler

        is_correct, meta = dataset.score(record, "candidate")

    mock_eval_create.assert_called_once()
    assert is_correct is True
    assert meta == {"mock": True}
