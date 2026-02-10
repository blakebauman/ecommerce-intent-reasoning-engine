"""OpenTelemetry SDK initialization."""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Track if telemetry has been initialized
_telemetry_initialized = False


@dataclass
class TelemetryConfig:
    """Configuration for OpenTelemetry."""

    # Service identification
    service_name: str = "intent-engine"
    service_version: str = "0.1.0"
    environment: str = "development"

    # OTLP exporter settings
    otlp_endpoint: str = "http://localhost:4317"
    otlp_insecure: bool = True

    # Feature flags
    enable_tracing: bool = True
    enable_metrics: bool = True

    # Sampling
    trace_sample_rate: float = 1.0  # Sample all traces by default

    # Resource attributes
    resource_attributes: dict[str, str] = field(default_factory=dict)


def init_telemetry(config: TelemetryConfig | None = None) -> bool:
    """
    Initialize OpenTelemetry SDK.

    Sets up:
    - Tracer provider with OTLP exporter
    - Meter provider with Prometheus exporter
    - Auto-instrumentation for FastAPI, httpx, Redis, asyncpg

    Args:
        config: Telemetry configuration. Uses defaults if not provided.

    Returns:
        True if initialization was successful, False otherwise.
    """
    global _telemetry_initialized

    if _telemetry_initialized:
        logger.debug("Telemetry already initialized")
        return True

    if config is None:
        config = TelemetryConfig()

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

        # Build resource attributes
        resource_attrs = {
            "service.name": config.service_name,
            "service.version": config.service_version,
            "deployment.environment": config.environment,
            **config.resource_attributes,
        }
        resource = Resource.create(resource_attrs)

        # Initialize tracing
        if config.enable_tracing:
            sampler = TraceIdRatioBased(config.trace_sample_rate)
            tracer_provider = TracerProvider(
                resource=resource,
                sampler=sampler,
            )

            # Configure OTLP exporter
            otlp_exporter = OTLPSpanExporter(
                endpoint=config.otlp_endpoint,
                insecure=config.otlp_insecure,
            )
            span_processor = BatchSpanProcessor(otlp_exporter)
            tracer_provider.add_span_processor(span_processor)

            # Set global tracer provider
            trace.set_tracer_provider(tracer_provider)
            logger.info(f"Tracing initialized with endpoint: {config.otlp_endpoint}")

        # Initialize metrics
        if config.enable_metrics:
            _init_metrics(resource, config)

        # Apply auto-instrumentation
        _apply_instrumentation()

        _telemetry_initialized = True
        logger.info("OpenTelemetry initialized successfully")
        return True

    except ImportError as e:
        logger.warning(f"OpenTelemetry packages not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry: {e}")
        return False


def _init_metrics(resource, config: TelemetryConfig) -> None:
    """Initialize metrics with Prometheus exporter."""
    try:
        from opentelemetry import metrics
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
        from opentelemetry.sdk.metrics import MeterProvider

        # Create Prometheus metric reader
        prometheus_reader = PrometheusMetricReader()

        # Create meter provider
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[prometheus_reader],
        )

        # Set global meter provider
        metrics.set_meter_provider(meter_provider)
        logger.info("Metrics initialized with Prometheus exporter")

    except ImportError as e:
        logger.warning(f"Prometheus metrics not available: {e}")
    except Exception as e:
        logger.error(f"Failed to initialize metrics: {e}")


def _apply_instrumentation() -> None:
    """Apply auto-instrumentation for common libraries."""

    # FastAPI instrumentation
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor().instrument()
        logger.debug("FastAPI instrumentation applied")
    except ImportError:
        logger.debug("FastAPI instrumentation not available")
    except Exception as e:
        logger.warning(f"Failed to apply FastAPI instrumentation: {e}")

    # httpx instrumentation
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        HTTPXClientInstrumentor().instrument()
        logger.debug("httpx instrumentation applied")
    except ImportError:
        logger.debug("httpx instrumentation not available")
    except Exception as e:
        logger.warning(f"Failed to apply httpx instrumentation: {e}")

    # Redis instrumentation
    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor
        RedisInstrumentor().instrument()
        logger.debug("Redis instrumentation applied")
    except ImportError:
        logger.debug("Redis instrumentation not available")
    except Exception as e:
        logger.warning(f"Failed to apply Redis instrumentation: {e}")

    # asyncpg instrumentation
    try:
        from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
        AsyncPGInstrumentor().instrument()
        logger.debug("asyncpg instrumentation applied")
    except ImportError:
        logger.debug("asyncpg instrumentation not available")
    except Exception as e:
        logger.warning(f"Failed to apply asyncpg instrumentation: {e}")


def shutdown_telemetry() -> None:
    """Shutdown OpenTelemetry SDK gracefully."""
    global _telemetry_initialized

    if not _telemetry_initialized:
        return

    try:
        from opentelemetry import trace

        tracer_provider = trace.get_tracer_provider()
        if hasattr(tracer_provider, "shutdown"):
            tracer_provider.shutdown()
            logger.info("Telemetry shutdown complete")

    except Exception as e:
        logger.error(f"Error during telemetry shutdown: {e}")

    _telemetry_initialized = False
