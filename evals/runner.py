"""Evaluation runner for batch intent classification."""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from evals.metrics import EvalMetrics
from intent_engine.engine import IntentEngine
from intent_engine.models.request import InputChannel, IntentRequest


@dataclass
class EvalExample:
    """A single evaluation example."""

    id: str
    input_text: str
    expected_intents: list[str]
    is_compound: bool = False
    metadata: dict[str, Any] | None = None


class EvalRunner:
    """
    Batch evaluation runner for the intent engine.

    Loads a labeled dataset, runs each example through the engine,
    and computes precision, recall, and F1 per intent.
    """

    def __init__(
        self,
        engine: IntentEngine,
        concurrency: int = 5,
    ) -> None:
        """
        Initialize the evaluation runner.

        Args:
            engine: The intent engine to evaluate.
            concurrency: Number of concurrent evaluations.
        """
        self.engine = engine
        self.concurrency = concurrency
        self.metrics = EvalMetrics()

    def load_dataset(self, filepath: str | Path) -> list[EvalExample]:
        """
        Load evaluation dataset from JSON file.

        Expected format:
        [
            {
                "id": "example-1",
                "input": "Where is my order #12345?",
                "intents": ["ORDER_STATUS.WISMO"],
                "is_compound": false
            },
            ...
        ]

        Args:
            filepath: Path to the dataset JSON file.

        Returns:
            List of EvalExample objects.
        """
        filepath = Path(filepath)
        with open(filepath) as f:
            data = json.load(f)

        examples: list[EvalExample] = []
        for item in data:
            examples.append(
                EvalExample(
                    id=item["id"],
                    input_text=item["input"],
                    expected_intents=item["intents"],
                    is_compound=item.get("is_compound", len(item["intents"]) > 1),
                    metadata=item.get("metadata"),
                )
            )

        return examples

    async def run_single(self, example: EvalExample) -> dict[str, Any]:
        """
        Run a single evaluation example.

        Args:
            example: The example to evaluate.

        Returns:
            Result dictionary with prediction details.
        """
        request = IntentRequest(
            request_id=example.id,
            tenant_id="eval",
            channel=InputChannel.CHAT,
            raw_text=example.input_text,
        )

        try:
            result = await self.engine.resolve(request)

            predicted_intents = [
                f"{intent.category}.{intent.intent}"
                for intent in result.resolved_intents
            ]

            self.metrics.record_prediction(
                actual_intents=example.expected_intents,
                predicted_intents=predicted_intents,
                path_taken=result.path_taken,
                latency_ms=result.processing_time_ms,
                is_compound_actual=example.is_compound,
                is_compound_predicted=result.is_compound,
            )

            return {
                "id": example.id,
                "input": example.input_text,
                "expected": example.expected_intents,
                "predicted": predicted_intents,
                "correct": set(example.expected_intents) == set(predicted_intents),
                "path": result.path_taken,
                "latency_ms": result.processing_time_ms,
                "confidence": result.confidence_summary,
            }

        except Exception as e:
            self.metrics.record_error(
                example_id=example.id,
                input_text=example.input_text,
                error_message=str(e),
            )
            return {
                "id": example.id,
                "input": example.input_text,
                "expected": example.expected_intents,
                "error": str(e),
            }

    async def run(
        self,
        examples: list[EvalExample],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> EvalMetrics:
        """
        Run evaluation on a list of examples.

        Args:
            examples: List of examples to evaluate.
            progress_callback: Optional callback(completed, total) for progress.

        Returns:
            EvalMetrics with aggregated results.
        """
        # Initialize engine
        await self.engine.initialize()

        # Reset metrics
        self.metrics = EvalMetrics()

        # Run evaluations with concurrency limit
        semaphore = asyncio.Semaphore(self.concurrency)
        completed = 0

        async def run_with_semaphore(example: EvalExample) -> dict[str, Any]:
            nonlocal completed
            async with semaphore:
                result = await self.run_single(example)
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(examples))
                return result

        # Run all examples
        tasks = [run_with_semaphore(example) for example in examples]
        await asyncio.gather(*tasks)

        return self.metrics

    async def run_from_file(
        self,
        filepath: str | Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> EvalMetrics:
        """
        Load dataset from file and run evaluation.

        Args:
            filepath: Path to dataset JSON file.
            progress_callback: Optional progress callback.

        Returns:
            EvalMetrics with results.
        """
        examples = self.load_dataset(filepath)
        return await self.run(examples, progress_callback)
