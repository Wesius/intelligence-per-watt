#!/usr/bin/env python3
"""B200 Saturation Sweep: Batch Size vs. Energy Efficiency.

Hypothesis: The B200 GPU is under-saturated at current batch sizes. While it
delivers the lowest latency, its efficiency (1.57 Tok/J avg) trails the H100
(1.78 Tok/J). The B200 draws massive power (~4400W avg) but likely isn't
processing enough tokens in parallel to justify that draw.

Goal: Find the "knee" of the curve where the B200's compute throughput overtakes
its static power penalty. Monitor Arithmetic Intensity (FLOPs/Byte) - we expect
to move from memory-bound at low batch sizes to compute-bound at higher ones.

Variables: Batch sizes of 1, 8, 32, 64, 128, 256, 512
"""

import json
import subprocess
import sys
import traceback
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parent / "logs"
STATE_FILE = STATE_DIR / "b200_saturation_sweep_state.json"

# Batch sizes to test - exponentially increasing to find the efficiency knee
BATCH_SIZES = [8, 32, 64, 128, 256, 512]

# Model to test on B200 - use a representative model
# You may want to adjust this based on your specific setup
DEFAULT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"

# Common arguments for all benchmark runs
COMMON_ARGS = [
    "--client",
    "vllm",
    "--dataset",
    "ipw-pro",
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
        description="B200 Saturation Sweep: Batch Size vs. Energy Efficiency experiment."
    )
    parser.add_argument(
        "--resume",
        action=BooleanOptionalAction,
        default=True,
        help="Resume from previous run state and skip batch sizes already marked as SUCCESS.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to benchmark (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help="Comma-separated list of batch sizes to test (default: 1,8,32,64,128,256,512)",
    )
    return parser.parse_args()


def run_benchmark(model: str, batch_size: int) -> bool:
    """Run benchmark for a specific batch size.

    Args:
        model: Model name/path to benchmark
        batch_size: Maximum batch size for vLLM

    Returns:
        Success flag
    """
    cmd = [
        "ipw",
        "profile",
        "--model",
        model,
        *COMMON_ARGS,
        "--client-param",
        f"max_num_seqs={batch_size}",
        # Set max concurrency to match batch size to ensure saturation
        "--max-concurrency",
        str(batch_size),
        # Skip evaluation for this experiment (focus on throughput/energy)
        "--no-eval",
    ]

    start_time = datetime.now()

    separator = "=" * 60
    print(separator)
    print(f"Starting B200 saturation sweep: batch_size={batch_size}")
    print(f"Model: {model}")
    print(f"Command: {' '.join(cmd)}")
    print(separator)

    try:
        result = subprocess.run(cmd, check=False)

        end_time = datetime.now()
        elapsed = end_time - start_time

        if result.returncode != 0:
            print(
                f"[FAILED] batch_size={batch_size} (exit code: {result.returncode}, elapsed: {elapsed})"
            )
            return False

        print(f"[COMPLETED] batch_size={batch_size} (elapsed: {elapsed})")
        return True

    except Exception:
        end_time = datetime.now()
        elapsed = end_time - start_time
        print(f"[ERROR] Failed to run batch_size={batch_size} (elapsed: {elapsed})")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    args = _parse_args()

    batch_sizes = BATCH_SIZES
    if args.batch_sizes:
        batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",")]

    print("=" * 60)
    print("B200 SATURATION SWEEP EXPERIMENT")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Batch sizes to test: {batch_sizes}")
    print()
    print("Hypothesis: B200 is under-saturated at low batch sizes.")
    print("Goal: Find the efficiency 'knee' where compute > memory bandwidth.")
    print("=" * 60)

    state: dict[str, str] = _load_run_state() if args.resume else {}
    if args.resume:
        print(f"Resume enabled; loaded run state for {len(state)} configurations")
    else:
        print("Resume disabled; starting with a fresh run state")
        state = {}

    results: dict[int, str] = {}
    print(f"Running benchmarks for {len(batch_sizes)} batch sizes sequentially...")

    for batch_size in batch_sizes:
        state_key = f"{args.model}:batch_size={batch_size}"
        existing = state.get(state_key)
        if args.resume and existing == "SUCCESS":
            print(f"Skipping batch_size={batch_size} (previous run success).")
            results[batch_size] = existing
            continue

        success = run_benchmark(args.model, batch_size)
        status = "SUCCESS" if success else "FAILED"
        state[state_key] = status
        results[batch_size] = status
        _save_run_state(state)

    # Summary
    separator = "=" * 60
    print(separator)
    print("SUMMARY - B200 SATURATION SWEEP")
    print(separator)

    success_count = sum(1 for status in results.values() if status == "SUCCESS")
    failed_count = len(results) - success_count

    for batch_size, status in results.items():
        prefix = "[OK]  " if status == "SUCCESS" else "[FAIL]"
        print(f"{prefix} batch_size={batch_size}: {status}")

    print(f"Total: {success_count}/{len(batch_sizes)} succeeded, {failed_count} failed")
    print(f"Run state file: {_state_file_path()}")
    print()
    print("Next steps:")
    print("  1. Compare Tok/J across batch sizes to find the efficiency knee")
    print("  2. Monitor HBM bandwidth utilization vs compute utilization")
    print("  3. Plot batch_size vs Tok/J to visualize the saturation curve")

    # Exit with error code if any failed, but only after running all
    sys.exit(0 if failed_count == 0 else 1)
