#!/usr/bin/env python
"""Run evaluation on the golden set."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evals.runner import EvalRunner
from intent_engine.config import get_settings
from intent_engine.engine import IntentEngine


def progress_bar(completed: int, total: int) -> None:
    """Simple progress bar."""
    pct = completed / total
    bar_len = 40
    filled = int(bar_len * pct)
    bar = "=" * filled + "-" * (bar_len - filled)
    print(f"\r[{bar}] {completed}/{total} ({pct:.1%})", end="", flush=True)


async def main() -> None:
    """Run evaluation."""
    parser = argparse.ArgumentParser(description="Run intent engine evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="evals/datasets/golden_set.json",
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent evaluations",
    )
    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / args.dataset

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    print(f"Loading dataset: {dataset_path}")

    # Initialize engine
    settings = get_settings()
    engine = IntentEngine(settings=settings)

    # Create runner
    runner = EvalRunner(engine=engine, concurrency=args.concurrency)

    print(f"Running evaluation with concurrency={args.concurrency}...")
    print()

    # Run evaluation
    metrics = await runner.run_from_file(dataset_path, progress_callback=progress_bar)

    print("\n")

    # Print report
    metrics.print_report()

    # Save results if output path provided
    if args.output:
        output_path = project_root / args.output
        with open(output_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        print(f"\nResults saved to: {output_path}")

    # Check success criteria
    print("\n--- Success Criteria Check ---")
    f1_pass = metrics.macro_f1 >= 0.85
    latency_pass = metrics.p99_latency_ms <= 200 if metrics.fast_path_count > 0 else True

    print(f"F1 >= 85%: {'PASS' if f1_pass else 'FAIL'} ({metrics.macro_f1:.1%})")
    print(f"P99 Latency <= 200ms: {'PASS' if latency_pass else 'FAIL'} ({metrics.p99_latency_ms:.0f}ms)")

    # Cleanup
    await engine.shutdown()

    # Exit code based on success criteria
    if f1_pass and latency_pass:
        print("\nAll success criteria met!")
        sys.exit(0)
    else:
        print("\nSome criteria not met.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
