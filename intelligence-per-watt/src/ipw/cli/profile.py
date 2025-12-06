"""Run profiling against an inference client."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import click
from ipw.core.types import ProfilerConfig

from ._console import _print_result, info, success, warning


def _collect_params(ctx, param, values):
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


@click.command(help="Run profiling against an inference client.")
@click.option("--client", "client_id", required=True, help="Client identifier")
@click.option("--model", required=True, help="Model name to invoke")
@click.option("--dataset", "dataset_id", default="ipw", help="Dataset identifier")
@click.option("--client-base-url", help="Client base URL")
@click.option(
    "--max-concurrency",
    "max_concurrency",
    type=click.IntRange(min=0),
    default=1,
    show_default=True,
    help="Maximum number of concurrent inference requests (0 for all)",
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
@click.option(
    "--dataset-param",
    multiple=True,
    callback=_collect_params,
    help="Dataset params key=value",
)
@click.option(
    "--client-param",
    multiple=True,
    callback=_collect_params,
    help="Client params key=value",
)
@click.option("--output-dir", type=click.Path())
@click.option("--max-queries", type=int)
@click.option(
    "--no-eval",
    is_flag=True,
    default=False,
    help="Skip post-run accuracy evaluation.",
)
def profile(
    dataset_id: str,
    client_id: str,
    client_base_url: str | None,
    model: str,
    max_concurrency: int,
    dataset_param,
    client_param,
    output_dir: str | None,
    max_queries: int | None,
    eval_client: str | None,
    eval_base_url: str | None,
    eval_model: str | None,
    no_eval: bool,
) -> None:
    """Execute profiling run with the execution pipeline."""
    import ipw.analysis
    import ipw.clients
    import ipw.datasets

    ipw.clients.ensure_registered()
    missing_reason = getattr(ipw.clients, "MISSING_CLIENTS", {}).get(client_id)
    if missing_reason:
        raise click.ClickException(
            f"Inference client '{client_id}' is unavailable: {missing_reason}"
        )

    ipw.datasets.ensure_registered()
    ipw.analysis.ensure_registered()
    
    from ipw.analysis.base import AnalysisContext
    from ipw.core.registry import AnalysisRegistry, DatasetRegistry
    from ipw.execution import ProfilerRunner  # Deferred import for heavy dependencies

    config = ProfilerConfig(
        dataset_id=dataset_id,
        client_id=client_id,
        client_base_url=client_base_url,
        dataset_params=dataset_param,
        client_params=client_param,
        model=model,
        max_concurrency=max_concurrency,
        max_queries=max_queries,
        output_dir=Path(output_dir) if output_dir else None,
    )

    # Preflight: dataset requirements (api keys, etc)
    try:
        dataset_cls = DatasetRegistry.get(dataset_id)
        dataset_instance = dataset_cls(**dataset_param)
        issues = dataset_instance.verify_requirements()
        if issues:
            raise click.ClickException(
                "Dataset requirements not satisfied:\n- " + "\n- ".join(issues)
            )
        _warn_on_custom_eval(dataset_instance, eval_client, eval_base_url, eval_model)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    runner = ProfilerRunner(config)
    runner.run()
    success("Profiling run completed")

    # Post-run analysis
    if no_eval:
        info("Skipping post-run evaluation (--no-eval)")
        return

    results_dir = runner._output_path
    if results_dir and results_dir.exists():
        info("Running post-profile analysis...")
        context = AnalysisContext(
            results_dir=results_dir,
            options={
                "model": model,
                "eval_client": eval_client,
                "eval_base_url": eval_base_url,
                "eval_model": eval_model,
            },
        )
        try:
            analysis = AnalysisRegistry.create("accuracy")
            result = analysis.run(context)
            _print_result(result, verbose=False)
        except Exception as e:
            warning(f"Warning: Analysis failed: {e}")


__all__ = ["profile"]


def _warn_on_custom_eval(
    dataset, eval_client: str | None, eval_base_url: str | None, eval_model: str | None
) -> None:
    client_default = getattr(dataset, "eval_client", None)
    base_default = getattr(dataset, "eval_base_url", None)
    model_default = getattr(dataset, "eval_model", None)

    provided_client = (eval_client or "").strip().lower()
    expected_client = (client_default or "").strip().lower()
    client_mismatch = bool(client_default and eval_client and provided_client != expected_client)

    provided_base = (eval_base_url or "").strip()
    expected_base = (base_default or "").strip()
    base_mismatch = bool(base_default and eval_base_url and provided_base != expected_base)

    provided_model = (eval_model or "").strip()
    expected_model = (model_default or "").strip()
    model_mismatch = bool(model_default and eval_model and provided_model != expected_model)

    if not (client_mismatch or base_mismatch or model_mismatch):
        return

    warning(
        "Using custom evaluation settings for %s. Defaults: client=%s, base_url=%s, model=%s."
        % (
            getattr(dataset, "dataset_name", dataset.__class__.__name__),
            client_default or "(unspecified)",
            base_default or "(unspecified)",
            model_default or "(unspecified)",
        )
    )
