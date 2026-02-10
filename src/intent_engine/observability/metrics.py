"""Prometheus metrics definitions and recording."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Global metrics registry
_metrics_registry: "MetricsRegistry | None" = None


class MetricsRegistry:
    """
    Registry for OpenTelemetry metrics.

    Provides pre-defined metrics for the Intent Engine:
    - Intent resolution metrics (latency, count, path taken)
    - Pipeline stage metrics
    - LLM call metrics
    - Rate limiting metrics
    - WebSocket metrics
    - Batch processing metrics
    """

    def __init__(self, meter_name: str = "intent_engine") -> None:
        """
        Initialize the metrics registry.

        Args:
            meter_name: Name for the meter.
        """
        self._meter = None
        self._instruments: dict[str, Any] = {}

        try:
            from opentelemetry import metrics
            self._meter = metrics.get_meter(meter_name)
            self._create_instruments()
            logger.info("Metrics registry initialized")
        except ImportError:
            logger.warning("OpenTelemetry metrics not available")

    def _create_instruments(self) -> None:
        """Create all metric instruments."""
        if not self._meter:
            return

        # Intent resolution metrics
        self._instruments["intent_resolution_duration"] = self._meter.create_histogram(
            name="intent_resolution_duration_seconds",
            description="Duration of intent resolution in seconds",
            unit="s",
        )

        self._instruments["intent_resolution_total"] = self._meter.create_counter(
            name="intent_resolution_total",
            description="Total number of intent resolutions",
            unit="1",
        )

        self._instruments["intent_confidence"] = self._meter.create_histogram(
            name="intent_confidence_score",
            description="Confidence scores for resolved intents",
            unit="1",
        )

        # Pipeline stage metrics
        self._instruments["pipeline_stage_duration"] = self._meter.create_histogram(
            name="pipeline_stage_duration_seconds",
            description="Duration of each pipeline stage in seconds",
            unit="s",
        )

        # Fast path metrics
        self._instruments["fast_path_total"] = self._meter.create_counter(
            name="fast_path_total",
            description="Count of fast path vs reasoning path resolutions",
            unit="1",
        )

        # LLM metrics
        self._instruments["llm_calls_total"] = self._meter.create_counter(
            name="llm_calls_total",
            description="Total LLM API calls",
            unit="1",
        )

        self._instruments["llm_tokens_total"] = self._meter.create_counter(
            name="llm_tokens_used_total",
            description="Total LLM tokens used",
            unit="1",
        )

        self._instruments["llm_duration"] = self._meter.create_histogram(
            name="llm_call_duration_seconds",
            description="Duration of LLM API calls in seconds",
            unit="s",
        )

        # Rate limiting metrics
        self._instruments["rate_limit_exceeded"] = self._meter.create_counter(
            name="rate_limit_exceeded_total",
            description="Number of rate limit exceeded events",
            unit="1",
        )

        # WebSocket metrics
        self._instruments["ws_connections"] = self._meter.create_up_down_counter(
            name="active_websocket_connections",
            description="Current number of active WebSocket connections",
            unit="1",
        )

        self._instruments["ws_messages_total"] = self._meter.create_counter(
            name="websocket_messages_total",
            description="Total WebSocket messages",
            unit="1",
        )

        # Batch processing metrics
        self._instruments["batch_jobs_total"] = self._meter.create_counter(
            name="batch_jobs_total",
            description="Total batch jobs processed",
            unit="1",
        )

        self._instruments["batch_items_total"] = self._meter.create_counter(
            name="batch_items_total",
            description="Total items in batch jobs",
            unit="1",
        )

        self._instruments["batch_duration"] = self._meter.create_histogram(
            name="batch_job_duration_seconds",
            description="Duration of batch job processing in seconds",
            unit="s",
        )

    def record_intent_resolution(
        self,
        duration_seconds: float,
        tenant_id: str,
        path_taken: str,
        is_compound: bool,
        confidence: float,
        status: str = "success",
    ) -> None:
        """
        Record intent resolution metrics.

        Args:
            duration_seconds: Time taken for resolution.
            tenant_id: Tenant ID.
            path_taken: fast_path, reasoning_path, or no_match.
            is_compound: Whether compound intents were detected.
            confidence: Overall confidence score.
            status: success or error.
        """
        labels = {
            "tenant_id": tenant_id,
            "path_taken": path_taken,
            "is_compound": str(is_compound).lower(),
            "status": status,
        }

        if "intent_resolution_duration" in self._instruments:
            self._instruments["intent_resolution_duration"].record(
                duration_seconds, labels
            )

        if "intent_resolution_total" in self._instruments:
            self._instruments["intent_resolution_total"].add(1, labels)

        if "intent_confidence" in self._instruments:
            self._instruments["intent_confidence"].record(
                confidence,
                {"tenant_id": tenant_id, "intent_code": "aggregate"},
            )

        if "fast_path_total" in self._instruments:
            self._instruments["fast_path_total"].add(
                1,
                {"tenant_id": tenant_id, "path_type": path_taken},
            )

    def record_pipeline_stage(
        self,
        stage: str,
        duration_seconds: float,
        tenant_id: str,
    ) -> None:
        """
        Record pipeline stage duration.

        Args:
            stage: Stage name (entity_extraction, sentiment_analysis, etc.).
            duration_seconds: Time taken for the stage.
            tenant_id: Tenant ID.
        """
        if "pipeline_stage_duration" in self._instruments:
            self._instruments["pipeline_stage_duration"].record(
                duration_seconds,
                {"stage": stage, "tenant_id": tenant_id},
            )

    def record_llm_call(
        self,
        model: str,
        duration_seconds: float,
        input_tokens: int,
        output_tokens: int,
        tenant_id: str,
        status: str = "success",
    ) -> None:
        """
        Record LLM API call metrics.

        Args:
            model: Model name.
            duration_seconds: API call duration.
            input_tokens: Input token count.
            output_tokens: Output token count.
            tenant_id: Tenant ID.
            status: success or error.
        """
        labels = {
            "tenant_id": tenant_id,
            "model": model,
            "status": status,
        }

        if "llm_calls_total" in self._instruments:
            self._instruments["llm_calls_total"].add(1, labels)

        if "llm_duration" in self._instruments:
            self._instruments["llm_duration"].record(duration_seconds, labels)

        if "llm_tokens_total" in self._instruments:
            self._instruments["llm_tokens_total"].add(
                input_tokens,
                {**labels, "token_type": "input"},
            )
            self._instruments["llm_tokens_total"].add(
                output_tokens,
                {**labels, "token_type": "output"},
            )

    def record_rate_limit_exceeded(self, tenant_id: str) -> None:
        """
        Record rate limit exceeded event.

        Args:
            tenant_id: Tenant ID.
        """
        if "rate_limit_exceeded" in self._instruments:
            self._instruments["rate_limit_exceeded"].add(
                1, {"tenant_id": tenant_id}
            )

    def record_websocket_connection(
        self,
        tenant_id: str,
        delta: int = 1,
    ) -> None:
        """
        Record WebSocket connection change.

        Args:
            tenant_id: Tenant ID.
            delta: +1 for connect, -1 for disconnect.
        """
        if "ws_connections" in self._instruments:
            self._instruments["ws_connections"].add(delta, {"tenant_id": tenant_id})

    def record_websocket_message(
        self,
        tenant_id: str,
        direction: str,
        message_type: str,
    ) -> None:
        """
        Record WebSocket message.

        Args:
            tenant_id: Tenant ID.
            direction: inbound or outbound.
            message_type: Message type.
        """
        if "ws_messages_total" in self._instruments:
            self._instruments["ws_messages_total"].add(
                1,
                {"tenant_id": tenant_id, "direction": direction, "type": message_type},
            )

    def record_batch_job(
        self,
        tenant_id: str,
        item_count: int,
        duration_seconds: float,
        status: str = "completed",
    ) -> None:
        """
        Record batch job metrics.

        Args:
            tenant_id: Tenant ID.
            item_count: Number of items in the batch.
            duration_seconds: Total processing duration.
            status: queued, processing, completed, failed.
        """
        labels = {"tenant_id": tenant_id, "status": status}

        if "batch_jobs_total" in self._instruments:
            self._instruments["batch_jobs_total"].add(1, labels)

        if "batch_items_total" in self._instruments:
            self._instruments["batch_items_total"].add(item_count, labels)

        if status == "completed" and "batch_duration" in self._instruments:
            self._instruments["batch_duration"].record(duration_seconds, labels)


def get_metrics_registry() -> MetricsRegistry:
    """
    Get the global metrics registry.

    Creates one if it doesn't exist.

    Returns:
        The global MetricsRegistry instance.
    """
    global _metrics_registry
    if _metrics_registry is None:
        _metrics_registry = MetricsRegistry()
    return _metrics_registry


# Convenience functions that use the global registry


def record_intent_resolution(
    duration_seconds: float,
    tenant_id: str,
    path_taken: str,
    is_compound: bool,
    confidence: float,
    status: str = "success",
) -> None:
    """Record intent resolution metrics."""
    get_metrics_registry().record_intent_resolution(
        duration_seconds=duration_seconds,
        tenant_id=tenant_id,
        path_taken=path_taken,
        is_compound=is_compound,
        confidence=confidence,
        status=status,
    )


def record_pipeline_stage(
    stage: str,
    duration_seconds: float,
    tenant_id: str,
) -> None:
    """Record pipeline stage duration."""
    get_metrics_registry().record_pipeline_stage(
        stage=stage,
        duration_seconds=duration_seconds,
        tenant_id=tenant_id,
    )


def record_llm_call(
    model: str,
    duration_seconds: float,
    input_tokens: int,
    output_tokens: int,
    tenant_id: str,
    status: str = "success",
) -> None:
    """Record LLM API call metrics."""
    get_metrics_registry().record_llm_call(
        model=model,
        duration_seconds=duration_seconds,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tenant_id=tenant_id,
        status=status,
    )


def record_rate_limit_exceeded(tenant_id: str) -> None:
    """Record rate limit exceeded event."""
    get_metrics_registry().record_rate_limit_exceeded(tenant_id)


def record_websocket_connection(tenant_id: str, delta: int = 1) -> None:
    """Record WebSocket connection change."""
    get_metrics_registry().record_websocket_connection(tenant_id, delta)


def record_websocket_message(tenant_id: str, direction: str, message_type: str) -> None:
    """Record WebSocket message."""
    get_metrics_registry().record_websocket_message(tenant_id, direction, message_type)


def record_batch_job(
    tenant_id: str,
    item_count: int,
    duration_seconds: float,
    status: str = "completed",
) -> None:
    """Record batch job metrics."""
    get_metrics_registry().record_batch_job(
        tenant_id=tenant_id,
        item_count=item_count,
        duration_seconds=duration_seconds,
        status=status,
    )
