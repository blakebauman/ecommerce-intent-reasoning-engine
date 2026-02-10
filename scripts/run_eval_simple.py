#!/usr/bin/env python
"""Simplified evaluation that tests intent matching without full engine init."""

import asyncio
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from intent_engine.matchers.similarity_matcher import SimilarityMatcher


@dataclass
class SimpleMetrics:
    """Simple evaluation metrics."""

    total: int = 0
    correct: int = 0
    fast_path: int = 0
    intent_tp: dict = field(default_factory=lambda: defaultdict(int))
    intent_fp: dict = field(default_factory=lambda: defaultdict(int))
    intent_fn: dict = field(default_factory=lambda: defaultdict(int))
    errors: list = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def fast_path_rate(self) -> float:
        return self.fast_path / self.total if self.total > 0 else 0.0

    def intent_f1(self, intent: str) -> float:
        tp = self.intent_tp[intent]
        fp = self.intent_fp[intent]
        fn = self.intent_fn[intent]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    def macro_f1(self) -> float:
        all_intents = set(self.intent_tp.keys()) | set(self.intent_fn.keys())
        if not all_intents:
            return 0.0
        return sum(self.intent_f1(i) for i in all_intents) / len(all_intents)

    def weighted_f1(self) -> float:
        all_intents = set(self.intent_tp.keys()) | set(self.intent_fn.keys())
        if not all_intents:
            return 0.0
        total_support = sum(self.intent_tp[i] + self.intent_fn[i] for i in all_intents)
        if total_support == 0:
            return 0.0
        return sum(
            self.intent_f1(i) * (self.intent_tp[i] + self.intent_fn[i])
            for i in all_intents
        ) / total_support


async def run_eval():
    """Run simplified evaluation."""
    # Load golden set
    golden_path = project_root / "evals" / "datasets" / "golden_set.json"

    with open(golden_path) as f:
        examples = json.load(f)

    print(f"Loaded {len(examples)} examples from golden set")

    # Load intent examples for matcher
    examples_path = project_root / "data" / "intent_examples.json"

    # Initialize matcher
    print("Initializing similarity matcher...")
    matcher = SimilarityMatcher(examples_path=examples_path)
    await matcher.initialize()
    print("Matcher initialized")

    # Run evaluation
    metrics = SimpleMetrics()

    print("\nRunning evaluation...")
    for i, example in enumerate(examples):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(examples)}")

        example_id = example["id"]
        input_text = example["input"]
        expected_intents = set(example["intents"])

        try:
            # Get matches from similarity matcher
            matches = await matcher.match(input_text, top_k=3)

            # Take top match as prediction (simplified - real engine uses confidence thresholds)
            predicted_intents = set()
            if matches and matches[0].similarity >= 0.7:
                predicted_intents.add(matches[0].intent_code)

                # Check if compound (multiple high-confidence matches from different categories)
                if len(matches) > 1 and matches[1].similarity >= 0.7:
                    cat1 = matches[0].intent_code.split(".")[0]
                    cat2 = matches[1].intent_code.split(".")[0]
                    if cat1 != cat2:
                        predicted_intents.add(matches[1].intent_code)

            # Record metrics
            metrics.total += 1

            if predicted_intents == expected_intents:
                metrics.correct += 1

            if matches and matches[0].similarity >= 0.85:
                metrics.fast_path += 1

            # Per-intent metrics
            for intent in expected_intents | predicted_intents:
                in_expected = intent in expected_intents
                in_predicted = intent in predicted_intents

                if in_expected and in_predicted:
                    metrics.intent_tp[intent] += 1
                elif in_predicted and not in_expected:
                    metrics.intent_fp[intent] += 1
                elif in_expected and not in_predicted:
                    metrics.intent_fn[intent] += 1

        except Exception as e:
            metrics.errors.append({"id": example_id, "error": str(e)})

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nTotal Examples: {metrics.total}")
    print(f"Correct: {metrics.correct}")
    print(f"Accuracy: {metrics.accuracy:.2%}")
    print(f"\nMacro F1: {metrics.macro_f1():.2%}")
    print(f"Weighted F1: {metrics.weighted_f1():.2%}")
    print(f"\nFast Path Rate: {metrics.fast_path_rate:.2%}")

    # Per-intent breakdown
    print("\n--- Per-Intent F1 Scores ---")
    all_intents = sorted(set(metrics.intent_tp.keys()) | set(metrics.intent_fn.keys()))
    for intent in all_intents:
        support = metrics.intent_tp[intent] + metrics.intent_fn[intent]
        f1 = metrics.intent_f1(intent)
        print(f"  {intent:<40} F1={f1:.2%} (support={support})")

    if metrics.errors:
        print(f"\n--- Errors ({len(metrics.errors)}) ---")
        for e in metrics.errors[:5]:
            print(f"  {e['id']}: {e['error']}")

    print("\n" + "=" * 60)

    # Success criteria
    f1 = metrics.macro_f1()
    print(f"\nSuccess Criteria: F1 >= 90%")
    print(f"Result: {'PASS' if f1 >= 0.90 else 'FAIL'} ({f1:.1%})")

    return metrics


if __name__ == "__main__":
    asyncio.run(run_eval())
