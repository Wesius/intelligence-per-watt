"""Tests for the accuracy analysis."""

from __future__ import annotations

from pathlib import Path
from unittest import mock
from unittest.mock import patch

import pytest

from ipw.analysis.accuracy import AccuracyAnalysis
from ipw.analysis.base import AnalysisContext


@patch.object(AccuracyAnalysis, "_needs_evaluation", return_value=False)
@patch("ipw.analysis.accuracy.resolve_model_name")
@patch("ipw.analysis.accuracy.load_metrics_dataset")
def test_computes_intelligence_metrics_from_energy_counters(
    mock_load: mock.MagicMock,
    mock_resolve: mock.MagicMock,
    _mock_needs_eval: mock.MagicMock,
    tmp_path: Path,
) -> None:
    mock_resolve.return_value = "test-model"
    mock_load.return_value = [
        {
            "problem": "p1",
            "answer": "a1",
            "model_answers": {"test-model": "a1"},
            "model_metrics": {
                "test-model": {
                    "evaluation": {"is_correct": True},
                    "energy_metrics": {"per_query_joules": 10.0},
                    "latency_metrics": {"total_query_seconds": 2.0},
                }
            },
        },
        {
            "problem": "p2",
            "answer": "a2",
            "model_answers": {"test-model": "wrong"},
            "model_metrics": {
                "test-model": {
                    "evaluation": {"is_correct": False},
                    "energy_metrics": {"per_query_joules": 20.0},
                    "latency_metrics": {"total_query_seconds": 4.0},
                }
            },
        },
    ]

    context = AnalysisContext(results_dir=tmp_path, options={})
    result = AccuracyAnalysis().run(context)

    assert result.summary["accuracy"] == pytest.approx(0.5)
    assert result.summary["intelligence_per_joule"] == pytest.approx(1 / 30)
    assert result.summary["intelligence_per_watt"] == pytest.approx(0.1)
    efficiency = result.data["efficiency"]["test-model"]
    assert efficiency["energy"]["count"] == 2
    assert efficiency["power"]["derived_power_samples"] == 2
    assert efficiency["power"]["power_metric_samples"] == 0
    assert "analysis/accuracy.json" in str(result.artifacts["report"])


@patch.object(AccuracyAnalysis, "_needs_evaluation", return_value=False)
@patch("ipw.analysis.accuracy.resolve_model_name")
@patch("ipw.analysis.accuracy.load_metrics_dataset")
def test_handles_missing_energy_and_uses_power_metrics(
    mock_load: mock.MagicMock,
    mock_resolve: mock.MagicMock,
    _mock_needs_eval: mock.MagicMock,
    tmp_path: Path,
) -> None:
    mock_resolve.return_value = "test-model"
    mock_load.return_value = [
        {
            "problem": "p1",
            "answer": "a1",
            "model_answers": {"test-model": "a1"},
            "model_metrics": {
                "test-model": {
                    "evaluation": {"is_correct": True},
                    "power_metrics": {"gpu": {"per_query_watts": {"avg": 5.0}}},
                }
            },
        },
        {
            "problem": "p2",
            "answer": "a2",
            "model_answers": {"test-model": "wrong"},
            "model_metrics": {
                "test-model": {
                    "evaluation": {"is_correct": False},
                    "power_metrics": {"gpu": {"per_query_watts": {"avg": 5.0}}},
                }
            },
        },
    ]

    context = AnalysisContext(results_dir=tmp_path, options={})
    result = AccuracyAnalysis().run(context)

    assert result.summary["intelligence_per_joule"] is None
    assert result.summary["energy_sample_count"] == 0
    assert result.summary["intelligence_per_watt"] == pytest.approx(0.1)
    efficiency = result.data["efficiency"]["test-model"]
    assert efficiency["power"]["power_metric_samples"] == 2
    assert any("No per-query energy measurements" in msg for msg in result.warnings)


@patch.object(AccuracyAnalysis, "_needs_evaluation", return_value=False)
@patch("ipw.analysis.accuracy.resolve_model_name")
@patch("ipw.analysis.accuracy.load_metrics_dataset")
def test_intelligence_metrics_use_overall_accuracy(
    mock_load: mock.MagicMock,
    mock_resolve: mock.MagicMock,
    _mock_needs_eval: mock.MagicMock,
    tmp_path: Path,
) -> None:
    mock_resolve.return_value = "test-model"
    mock_load.return_value = [
        {
            "problem": "p1",
            "answer": "a1",
            "model_answers": {"test-model": "a1"},
            "model_metrics": {
                "test-model": {
                    "evaluation": {"is_correct": True},
                    "energy_metrics": {"per_query_joules": 10.0},
                    "latency_metrics": {"total_query_seconds": 2.0},
                }
            },
        },
        {
            "problem": "p2",
            "answer": "a2",
            "model_answers": {"test-model": "wrong"},
            "model_metrics": {
                "test-model": {
                    "evaluation": {"is_correct": False},
                }
            },
        },
    ]

    context = AnalysisContext(results_dir=tmp_path, options={})
    result = AccuracyAnalysis().run(context)

    # Overall accuracy is 0.5; energy/power averages are from the single measured query.
    assert result.summary["accuracy"] == pytest.approx(0.5)
    assert result.summary["intelligence_per_joule"] == pytest.approx(0.5 / 10.0)
    assert result.summary["intelligence_per_watt"] == pytest.approx(0.5 / 5.0)


@patch.object(AccuracyAnalysis, "_needs_evaluation", return_value=False)
@patch("ipw.analysis.accuracy.resolve_model_name")
@patch("ipw.analysis.accuracy.load_metrics_dataset")
def test_zero_energy_imputed_from_power_and_latency(
    mock_load: mock.MagicMock,
    mock_resolve: mock.MagicMock,
    _mock_needs_eval: mock.MagicMock,
    tmp_path: Path,
) -> None:
    mock_resolve.return_value = "test-model"
    mock_load.return_value = [
        {
            "problem": "p1",
            "answer": "a1",
            "model_answers": {"test-model": "a1"},
            "model_metrics": {
                "test-model": {
                    "evaluation": {"is_correct": True},
                    "energy_metrics": {"per_query_joules": 0.0},
                    "latency_metrics": {"total_query_seconds": 2.0},
                    "power_metrics": {"gpu": {"per_query_watts": {"avg": 4.0}}},
                }
            },
        },
        {
            "problem": "p2",
            "answer": "a2",
            "model_answers": {"test-model": "wrong"},
            "model_metrics": {
                "test-model": {
                    "evaluation": {"is_correct": False},
                    "energy_metrics": {"per_query_joules": 10.0},
                    "latency_metrics": {"total_query_seconds": 2.0},
                }
            },
        },
    ]

    context = AnalysisContext(results_dir=tmp_path, options={})
    result = AccuracyAnalysis().run(context)

    # Imputed energy uses per-record power * latency: 4.0 * 2.0 = 8.0
    # Combined energies = [10.0, 8.0] => avg 9.0
    # Avg power = (4.0 + 5.0) / 2 = 4.5 (second query derives power from energy/latency)
    assert result.summary["accuracy"] == pytest.approx(0.5)
    assert result.summary["avg_per_query_energy_joules"] == pytest.approx(9.0)
    assert result.summary["intelligence_per_joule"] == pytest.approx(0.5 / 9.0)
    assert result.summary["intelligence_per_watt"] == pytest.approx(0.5 / 4.5)
    eff = result.data["efficiency"]["test-model"]["energy"]
    assert eff["imputed_from_power"] == pytest.approx(8.0)
    assert eff["imputed_count"] == 1
    assert "imputed_energy_from_power" not in result.summary
    assert any("Imputed energy" in msg for msg in result.warnings)


def test_needs_evaluation_retries_failed_until_limit() -> None:
    analysis = AccuracyAnalysis()
    model_name = "test-model"

    retryable = [
        {
            "model_metrics": {
                model_name: {
                    "evaluation": {
                        "is_correct": None,
                        "metadata": {"evaluation_failed": True, "evaluation_attempts": 1},
                    }
                }
            }
        }
    ]
    assert analysis._needs_evaluation(retryable, model_name) is True

    exhausted = [
        {
            "model_metrics": {
                model_name: {
                    "evaluation": {
                        "is_correct": None,
                        "metadata": {
                            "evaluation_failed": True,
                            "evaluation_attempts": analysis.MAX_EVALUATION_ATTEMPTS,
                        },
                    }
                }
            }
        }
    ]
    assert analysis._needs_evaluation(exhausted, model_name) is False
