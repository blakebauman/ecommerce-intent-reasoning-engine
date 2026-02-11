"""Observability module for tracing, metrics, and logging."""

from intent_engine.observability.logging import configure_logging, get_logger
from intent_engine.observability.metrics import (
    MetricsRegistry,
    get_metrics_registry,
    record_batch_job,
    record_intent_resolution,
    record_llm_call,
    record_pipeline_stage,
    record_rate_limit_exceeded,
    record_websocket_connection,
)
from intent_engine.observability.telemetry import (
    TelemetryConfig,
    init_telemetry,
    shutdown_telemetry,
)
from intent_engine.observability.tracing import (
    get_tracer,
    pipeline_span,
    traced,
)

__all__ = [
    # Telemetry
    "TelemetryConfig",
    "init_telemetry",
    "shutdown_telemetry",
    # Tracing
    "get_tracer",
    "traced",
    "pipeline_span",
    # Metrics
    "MetricsRegistry",
    "get_metrics_registry",
    "record_intent_resolution",
    "record_pipeline_stage",
    "record_llm_call",
    "record_rate_limit_exceeded",
    "record_websocket_connection",
    "record_batch_job",
    # Logging
    "configure_logging",
    "get_logger",
]
