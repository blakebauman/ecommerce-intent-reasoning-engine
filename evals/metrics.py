"""Evaluation metrics for intent classification."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IntentMetrics:
    """Metrics for a single intent."""

    intent_code: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        """Calculate precision: TP / (TP + FP)."""
        total_predicted = self.true_positives + self.false_positives
        if total_predicted == 0:
            return 0.0
        return self.true_positives / total_predicted

    @property
    def recall(self) -> float:
        """Calculate recall: TP / (TP + FN)."""
        total_actual = self.true_positives + self.false_negatives
        if total_actual == 0:
            return 0.0
        return self.true_positives / total_actual

    @property
    def f1(self) -> float:
        """Calculate F1 score: 2 * (P * R) / (P + R)."""
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    @property
    def support(self) -> int:
        """Number of actual instances of this intent."""
        return self.true_positives + self.false_negatives

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "intent_code": self.intent_code,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "support": self.support,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


@dataclass
class EvalMetrics:
    """Aggregate evaluation metrics."""

    total_examples: int = 0
    correct_predictions: int = 0
    fast_path_count: int = 0
    reasoning_path_count: int = 0

    # Per-intent metrics
    intent_metrics: dict[str, IntentMetrics] = field(default_factory=dict)

    # Latency tracking
    latencies_ms: list[int] = field(default_factory=list)
    fast_path_latencies_ms: list[int] = field(default_factory=list)
    reasoning_path_latencies_ms: list[int] = field(default_factory=list)

    # Compound detection
    compound_true_positives: int = 0
    compound_false_positives: int = 0
    compound_false_negatives: int = 0

    # Error tracking
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Overall accuracy."""
        if self.total_examples == 0:
            return 0.0
        return self.correct_predictions / self.total_examples

    @property
    def macro_precision(self) -> float:
        """Macro-averaged precision across intents."""
        if not self.intent_metrics:
            return 0.0
        return sum(m.precision for m in self.intent_metrics.values()) / len(
            self.intent_metrics
        )

    @property
    def macro_recall(self) -> float:
        """Macro-averaged recall across intents."""
        if not self.intent_metrics:
            return 0.0
        return sum(m.recall for m in self.intent_metrics.values()) / len(
            self.intent_metrics
        )

    @property
    def macro_f1(self) -> float:
        """Macro-averaged F1 across intents."""
        if not self.intent_metrics:
            return 0.0
        return sum(m.f1 for m in self.intent_metrics.values()) / len(self.intent_metrics)

    @property
    def weighted_f1(self) -> float:
        """Weighted F1 (by support)."""
        if not self.intent_metrics:
            return 0.0
        total_support = sum(m.support for m in self.intent_metrics.values())
        if total_support == 0:
            return 0.0
        return sum(m.f1 * m.support for m in self.intent_metrics.values()) / total_support

    @property
    def fast_path_rate(self) -> float:
        """Percentage of requests handled by fast path."""
        if self.total_examples == 0:
            return 0.0
        return self.fast_path_count / self.total_examples

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)

    @property
    def p50_latency_ms(self) -> float:
        """P50 (median) latency."""
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        idx = len(sorted_latencies) // 2
        return sorted_latencies[idx]

    @property
    def p99_latency_ms(self) -> float:
        """P99 latency."""
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def compound_precision(self) -> float:
        """Precision for compound intent detection."""
        total = self.compound_true_positives + self.compound_false_positives
        if total == 0:
            return 0.0
        return self.compound_true_positives / total

    @property
    def compound_recall(self) -> float:
        """Recall for compound intent detection."""
        total = self.compound_true_positives + self.compound_false_negatives
        if total == 0:
            return 0.0
        return self.compound_true_positives / total

    def record_prediction(
        self,
        actual_intents: list[str],
        predicted_intents: list[str],
        path_taken: str,
        latency_ms: int,
        is_compound_actual: bool,
        is_compound_predicted: bool,
    ) -> None:
        """Record a single prediction result."""
        self.total_examples += 1
        self.latencies_ms.append(latency_ms)

        if path_taken == "fast_path":
            self.fast_path_count += 1
            self.fast_path_latencies_ms.append(latency_ms)
        else:
            self.reasoning_path_count += 1
            self.reasoning_path_latencies_ms.append(latency_ms)

        # Track compound detection
        if is_compound_actual and is_compound_predicted:
            self.compound_true_positives += 1
        elif is_compound_predicted and not is_compound_actual:
            self.compound_false_positives += 1
        elif is_compound_actual and not is_compound_predicted:
            self.compound_false_negatives += 1

        # Track per-intent metrics
        actual_set = set(actual_intents)
        predicted_set = set(predicted_intents)

        # Check if prediction is correct (all intents match)
        if actual_set == predicted_set:
            self.correct_predictions += 1

        # Update per-intent metrics
        all_intents = actual_set | predicted_set
        for intent_code in all_intents:
            if intent_code not in self.intent_metrics:
                self.intent_metrics[intent_code] = IntentMetrics(intent_code=intent_code)

            metrics = self.intent_metrics[intent_code]

            in_actual = intent_code in actual_set
            in_predicted = intent_code in predicted_set

            if in_actual and in_predicted:
                metrics.true_positives += 1
            elif in_predicted and not in_actual:
                metrics.false_positives += 1
            elif in_actual and not in_predicted:
                metrics.false_negatives += 1

    def record_error(
        self,
        example_id: str,
        input_text: str,
        error_message: str,
    ) -> None:
        """Record an error during evaluation."""
        self.errors.append({
            "example_id": example_id,
            "input_text": input_text[:100],
            "error": error_message,
        })

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "summary": {
                "total_examples": self.total_examples,
                "accuracy": round(self.accuracy, 4),
                "macro_f1": round(self.macro_f1, 4),
                "weighted_f1": round(self.weighted_f1, 4),
                "macro_precision": round(self.macro_precision, 4),
                "macro_recall": round(self.macro_recall, 4),
            },
            "path_distribution": {
                "fast_path_count": self.fast_path_count,
                "reasoning_path_count": self.reasoning_path_count,
                "fast_path_rate": round(self.fast_path_rate, 4),
            },
            "latency": {
                "avg_ms": round(self.avg_latency_ms, 1),
                "p50_ms": round(self.p50_latency_ms, 1),
                "p99_ms": round(self.p99_latency_ms, 1),
            },
            "compound_detection": {
                "precision": round(self.compound_precision, 4),
                "recall": round(self.compound_recall, 4),
            },
            "per_intent": {
                code: metrics.to_dict()
                for code, metrics in sorted(self.intent_metrics.items())
            },
            "errors": self.errors[:10],  # First 10 errors
        }

    def print_report(self) -> None:
        """Print a formatted evaluation report."""
        print("\n" + "=" * 60)
        print("INTENT ENGINE EVALUATION REPORT")
        print("=" * 60)

        print(f"\nTotal Examples: {self.total_examples}")
        print(f"Accuracy: {self.accuracy:.2%}")
        print(f"Macro F1: {self.macro_f1:.2%}")
        print(f"Weighted F1: {self.weighted_f1:.2%}")

        print(f"\n--- Path Distribution ---")
        print(f"Fast Path: {self.fast_path_count} ({self.fast_path_rate:.1%})")
        print(f"Reasoning Path: {self.reasoning_path_count}")

        print(f"\n--- Latency ---")
        print(f"Average: {self.avg_latency_ms:.0f}ms")
        print(f"P50: {self.p50_latency_ms:.0f}ms")
        print(f"P99: {self.p99_latency_ms:.0f}ms")

        print(f"\n--- Per-Intent Metrics ---")
        print(f"{'Intent':<40} {'P':>8} {'R':>8} {'F1':>8} {'Support':>8}")
        print("-" * 72)
        for code, metrics in sorted(self.intent_metrics.items()):
            print(
                f"{code:<40} {metrics.precision:>8.2%} {metrics.recall:>8.2%} "
                f"{metrics.f1:>8.2%} {metrics.support:>8}"
            )

        if self.errors:
            print(f"\n--- Errors ({len(self.errors)}) ---")
            for error in self.errors[:5]:
                print(f"  {error['example_id']}: {error['error']}")

        print("\n" + "=" * 60)
