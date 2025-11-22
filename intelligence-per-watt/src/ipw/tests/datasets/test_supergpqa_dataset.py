from __future__ import annotations

from typing import Any, Iterable, MutableMapping

import pytest

from ipw.datasets.supergpqa import SuperGPQADataset


class _FakeDataset(list):
    def to_list(self) -> list[MutableMapping[str, Any]]:
        return list(self)


def _patch_load(rows: Iterable[MutableMapping[str, Any]], monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeDataset(rows)  # type: ignore[arg-type]

    def _loader(*args, **kwargs):
        return fake

    monkeypatch.setattr("ipw.datasets.supergpqa.load_dataset", _loader)


def test_supergpqa_dataset_formats_question(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "uuid": "abc123",
            "question": "Who discovered gravity?",
            "options": ["Newton", "Tesla"],
            "answer": "Newton",
            "answer_letter": "A",
            "discipline": "Science",
            "field": "Physics",
            "subfield": "Classical Mechanics",
            "difficulty": "easy",
            "is_calculation": False,
        }
    ]
    _patch_load(rows, monkeypatch)

    dataset = SuperGPQADataset()
    record = next(iter(dataset.iter_records()))

    assert record.answer == "A"
    assert "Options" in record.problem
    assert "A. Newton" in record.problem
    assert record.subject == "Classical Mechanics"
    assert record.dataset_metadata["uuid"] == "abc123"
