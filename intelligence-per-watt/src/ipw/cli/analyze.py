"""Analyze profiling results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import click

from ..analysis.base import AnalysisContext
from ..core.registry import AnalysisRegistry
from ._console import _print_result


def _collect_options(ctx, param, values):
    """Parse key=value options into a dictionary."""
    collected: Dict[str, str] = {}
    for item in values:
        for piece in item.split(","):
            if not piece:
                continue
            key, _, raw = piece.partition("=")
            key = key.strip()
            if not key:
                continue
            collected[key] = raw.strip()
    return collected


@click.command(help="Analyze profiling results and compute metrics.")
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--analysis",
    "analysis_name",
    default="accuracy",
    show_default=True,
    help="Which registered analysis to execute.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show data and metadata fields in addition to summary and artifacts.",
)
@click.option(
    "--option",
    "options",
    multiple=True,
    callback=_collect_options,
    help="Analysis-specific options (e.g., --option model=llama3.2:1b).",
)
@click.option(
    "--eval-client",
    help="Evaluation client identifier (judge)",
    default="openai",
    show_default=True,
)
@click.option(
    "--eval-base-url",
    help="Evaluation client base URL",
    default="https://api.openai.com/v1",
    show_default=True,
)
@click.option(
    "--eval-model",
    help="Evaluation model to use for scoring",
    default="gpt-5-nano-2025-08-07",
    show_default=True,
)
def analyze(
    directory: Path,
    analysis_name: str,
    verbose: bool,
    options: Dict[str, Any],
    eval_client: str | None,
    eval_base_url: str | None,
    eval_model: str | None,
) -> None:
    """Compute analysis results for a profiling run."""
    import ipw.analysis
    import ipw.clients
    import ipw.datasets

    ipw.analysis.ensure_registered()
    ipw.clients.ensure_registered()
    ipw.datasets.ensure_registered()

    merged_options: Dict[str, Any] = dict(options)
    if eval_client is not None:
        merged_options["eval_client"] = eval_client
    if eval_base_url is not None:
        merged_options["eval_base_url"] = eval_base_url
    if eval_model is not None:
        merged_options["eval_model"] = eval_model

    context = AnalysisContext(
        results_dir=directory,
        options=merged_options,
    )

    try:
        analysis = AnalysisRegistry.create(analysis_name)
        result = analysis.run(context)
    except KeyError as exc:
        available = ", ".join(sorted(name for name, _ in AnalysisRegistry.items()))
        raise click.ClickException(
            f"Unknown analysis '{analysis_name}'. Available analyses: {available}."
        ) from exc
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    _print_result(result, verbose=verbose)


__all__ = ["analyze"]
