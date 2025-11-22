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
    default="regression",
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
def analyze(
    directory: Path,
    analysis_name: str,
    verbose: bool,
    options: Dict[str, Any],
) -> None:
    """Compute analysis results for a profiling run."""
    import ipw.analysis
    import ipw.clients
    import ipw.datasets

    ipw.analysis.ensure_registered()
    ipw.clients.ensure_registered()
    ipw.datasets.ensure_registered()

    context = AnalysisContext(
        results_dir=directory,
        options=options,
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
