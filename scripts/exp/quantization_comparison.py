#!/usr/bin/env python3
"""Aggressive Quantization: INT4/NF4 vs FP8 Comparison.

Hypothesis: The Qwen3 235B MoE is the accuracy leader (58.99%) but is
energy-prohibitive (0.70 Tok/J). Because MoEs are memory-bandwidth heavy
(loading experts), reducing weight precision usually yields linear energy
gains with minimal accuracy loss.

Why: Moving from FP8 to INT4 halves the memory traffic. Since the H100/B200
energy cost is heavily dominated by HBM reads, this should drastically
improve the IPW score.

Success Criteria: If Qwen3 can maintain >55% accuracy (currently ~58%) while
doubling efficiency to ~1.4 Tok/J, it becomes a viable production candidate.

Models tested:
  - Qwen/Qwen3-235B-A22B-Instruct-2507 (MoE, memory-bound)
  - google/gemma-3-27b-it (dense, comparison baseline)

Quantization methods:
  - FP8 (baseline)
  - INT4 (AWQ or GPTQ)
  - NF4 (bitsandbytes Normal Float 4)
"""

import json
import subprocess
import sys
import traceback
from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parent / "logs"
STATE_FILE = STATE_DIR / "quantization_comparison_state.json"


@dataclass
class ModelConfig:
    """Configuration for a model with specific quantization."""

    name: str
    model_id: str
    quantization: str
    extra_params: list[str] | None = None

    @property
    def key(self) -> str:
        return f"{self.model_id}:{self.quantization}"


# Model configurations to test
# Note: You may need to adjust model IDs based on available quantized versions
# or use vLLM's on-the-fly quantization capabilities
MODEL_CONFIGS = [
    # Qwen3 235B MoE - the accuracy leader we want to optimize
    ModelConfig(
        name="Qwen3 235B FP8 (baseline)",
        model_id="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        quantization="fp8",
    ),
    ModelConfig(
        name="Qwen3 235B AWQ INT4",
        model_id="Qwen/Qwen3-235B-A22B-Instruct-2507-AWQ",
        quantization="awq",
    ),
    ModelConfig(
        name="Qwen3 235B GPTQ INT4",
        model_id="Qwen/Qwen3-235B-A22B-Instruct-2507-GPTQ-Int4",
        quantization="gptq",
    ),
    # Gemma 3 27B - dense model for comparison
    ModelConfig(
        name="Gemma 3 27B FP8 (baseline)",
        model_id="google/gemma-3-27b-it",
        quantization="fp8",
    ),
    ModelConfig(
        name="Gemma 3 27B AWQ INT4",
        model_id="google/gemma-3-27b-it-AWQ",
        quantization="awq",
    ),
    ModelConfig(
        name="Gemma 3 27B GPTQ INT4",
        model_id="google/gemma-3-27b-it-GPTQ-Int4",
        quantization="gptq",
    ),
]

# Common arguments for all benchmark runs
COMMON_ARGS = [
    "--client",
    "vllm",
    "--dataset",
    "ipw-pro",
    "--max-concurrency",
    "0",
    "--client-param",
    "tensor_parallel_size=8",
]


def _state_file_path() -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_FILE


def _load_run_state() -> dict[str, str]:
    path = _state_file_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("run state is not a dict")
        normalized: dict[str, str] = {}
        for key, status in data.items():
            normalized[str(key)] = str(status).upper()
        return normalized
    except Exception:
        print(f"[ERROR] Failed to load run state from {path}; starting fresh")
        traceback.print_exc()
        return {}


def _save_run_state(state: dict[str, str]) -> None:
    path = _state_file_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
    except Exception:
        print(f"[ERROR] Failed to persist run state to {path}")
        traceback.print_exc()


def _parse_args():
    parser = ArgumentParser(
        description="Aggressive Quantization: INT4/NF4 vs FP8 comparison experiment."
    )
    parser.add_argument(
        "--resume",
        action=BooleanOptionalAction,
        default=True,
        help="Resume from previous run state and skip configs already marked as SUCCESS.",
    )
    parser.add_argument(
        "--only-qwen",
        action="store_true",
        help="Only run Qwen3 235B configurations.",
    )
    parser.add_argument(
        "--only-gemma",
        action="store_true",
        help="Only run Gemma 3 27B configurations.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Only test specific quantization method (fp8, awq, gptq).",
    )
    return parser.parse_args()


def run_benchmark(config: ModelConfig) -> bool:
    """Run benchmark for a specific model configuration.

    Args:
        config: Model configuration to benchmark

    Returns:
        Success flag
    """
    cmd = [
        "ipw",
        "profile",
        "--model",
        config.model_id,
        *COMMON_ARGS,
    ]

    # Add quantization parameter for vLLM
    # Note: For pre-quantized models (AWQ, GPTQ), vLLM auto-detects
    # For FP8, we explicitly set it
    if config.quantization == "fp8":
        cmd.extend(["--client-param", "quantization=fp8"])
    elif config.quantization == "awq":
        cmd.extend(["--client-param", "quantization=awq"])
    elif config.quantization == "gptq":
        cmd.extend(["--client-param", "quantization=gptq"])
    elif config.quantization == "bitsandbytes":
        cmd.extend(["--client-param", "quantization=bitsandbytes"])
        cmd.extend(["--client-param", "load_format=bitsandbytes"])

    if config.extra_params:
        for param in config.extra_params:
            cmd.extend(["--client-param", param])

    start_time = datetime.now()

    separator = "=" * 60
    print(separator)
    print(f"Starting quantization benchmark: {config.name}")
    print(f"Model ID: {config.model_id}")
    print(f"Quantization: {config.quantization}")
    print(f"Command: {' '.join(cmd)}")
    print(separator)

    try:
        result = subprocess.run(cmd, check=False)

        end_time = datetime.now()
        elapsed = end_time - start_time

        if result.returncode != 0:
            print(
                f"[FAILED] {config.name} (exit code: {result.returncode}, elapsed: {elapsed})"
            )
            return False

        print(f"[COMPLETED] {config.name} (elapsed: {elapsed})")
        return True

    except Exception:
        end_time = datetime.now()
        elapsed = end_time - start_time
        print(f"[ERROR] Failed to run {config.name} (elapsed: {elapsed})")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    args = _parse_args()

    # Filter configurations based on arguments
    configs = MODEL_CONFIGS
    if args.only_qwen:
        configs = [c for c in configs if "qwen" in c.model_id.lower()]
    if args.only_gemma:
        configs = [c for c in configs if "gemma" in c.model_id.lower()]
    if args.quantization:
        configs = [c for c in configs if c.quantization == args.quantization]

    print("=" * 60)
    print("AGGRESSIVE QUANTIZATION EXPERIMENT")
    print("INT4/NF4 vs FP8 Comparison")
    print("=" * 60)
    print()
    print("Hypothesis: Reducing precision from FP8 to INT4 halves memory traffic,")
    print("which should drastically improve Tok/J for memory-bound MoE models.")
    print()
    print("Success Criteria:")
    print("  - Qwen3 235B: Maintain >55% accuracy while reaching ~1.4 Tok/J")
    print("  - Gemma 3 27B: Compare efficiency gains between dense and MoE")
    print("=" * 60)
    print()
    print(f"Configurations to test: {len(configs)}")
    for config in configs:
        print(f"  - {config.name}")
    print()

    state: dict[str, str] = _load_run_state() if args.resume else {}
    if args.resume:
        print(f"Resume enabled; loaded run state for {len(state)} configurations")
    else:
        print("Resume disabled; starting with a fresh run state")
        state = {}

    results: dict[str, str] = {}
    print(f"Running benchmarks for {len(configs)} configurations sequentially...")

    for config in configs:
        existing = state.get(config.key)
        if args.resume and existing == "SUCCESS":
            print(f"Skipping {config.name} (previous run success).")
            results[config.key] = existing
            continue

        success = run_benchmark(config)
        status = "SUCCESS" if success else "FAILED"
        state[config.key] = status
        results[config.key] = status
        _save_run_state(state)

    # Summary
    separator = "=" * 60
    print(separator)
    print("SUMMARY - QUANTIZATION COMPARISON")
    print(separator)

    success_count = sum(1 for status in results.values() if status == "SUCCESS")
    failed_count = len(results) - success_count

    for config in configs:
        status = results.get(config.key, "SKIPPED")
        prefix = "[OK]  " if status == "SUCCESS" else "[FAIL]"
        print(f"{prefix} {config.name}: {status}")

    print(f"Total: {success_count}/{len(configs)} succeeded, {failed_count} failed")
    print(f"Run state file: {_state_file_path()}")
    print()
    print("Next steps:")
    print("  1. Compare accuracy scores across quantization methods")
    print("  2. Plot Tok/J vs Accuracy for each model/quantization combo")
    print("  3. Calculate the accuracy-efficiency Pareto frontier")
    print("  4. Identify if INT4 Qwen3 meets the >55% accuracy, ~1.4 Tok/J target")

    # Exit with error code if any failed, but only after running all
    sys.exit(0 if failed_count == 0 else 1)
