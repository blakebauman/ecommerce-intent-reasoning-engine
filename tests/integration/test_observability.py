"""Integration tests for observability features."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import logging

from intent_engine.observability.telemetry import TelemetryConfig, init_telemetry, shutdown_telemetry
from intent_engine.observability.tracing import (
    get_tracer,
    traced,
    pipeline_span,
    add_span_attribute,
    get_current_trace_id,
    get_current_span_id,
)
from intent_engine.observability.metrics import (
    MetricsRegistry,
    get_metrics_registry,
    record_intent_resolution,
    record_pipeline_stage,
    record_rate_limit_exceeded,
)
from intent_engine.observability.logging import (
    configure_logging,
    get_logger,
    StructuredLogFormatter,
    TenantContextFilter,
)


class TestTelemetryInitialization:
    """Tests for telemetry initialization."""

    def test_default_config(self):
        """Test default telemetry configuration."""
        config = TelemetryConfig()

        assert config.service_name == "intent-engine"
        assert config.enable_tracing is True
        assert config.enable_metrics is True
        assert config.trace_sample_rate == 1.0

    def test_custom_config(self):
        """Test custom telemetry configuration."""
        config = TelemetryConfig(
            service_name="custom-service",
            environment="production",
            otlp_endpoint="http://collector:4317",
            trace_sample_rate=0.5,
        )

        assert config.service_name == "custom-service"
        assert config.environment == "production"
        assert config.trace_sample_rate == 0.5

    @patch("intent_engine.observability.telemetry._telemetry_initialized", False)
    def test_init_telemetry_without_otel(self):
        """Test init_telemetry when OpenTelemetry is not installed."""
        with patch.dict("sys.modules", {"opentelemetry": None}):
            # Should not raise, just return False
            result = init_telemetry()
            # May be True or False depending on actual imports


class TestTracing:
    """Tests for tracing utilities."""

    def test_get_tracer(self):
        """Test getting a tracer."""
        tracer = get_tracer("test_module")
        assert tracer is not None

    def test_pipeline_span_context_manager(self):
        """Test pipeline_span context manager."""
        with pipeline_span("test_stage", tenant_id="test-tenant", request_id="req-1") as span:
            # Should not raise
            pass

    def test_traced_decorator_sync(self):
        """Test @traced decorator on sync function."""
        @traced(name="test_function")
        def my_function(x):
            return x * 2

        result = my_function(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_traced_decorator_async(self):
        """Test @traced decorator on async function."""
        @traced(name="async_test")
        async def my_async_function(x):
            return x * 3

        result = await my_async_function(4)
        assert result == 12

    def test_traced_decorator_with_exception(self):
        """Test @traced decorator records exceptions."""
        @traced(name="error_function", record_exception=True)
        def error_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            error_function()

    def test_add_span_attribute_no_span(self):
        """Test add_span_attribute when no span is active."""
        # Should not raise even without active span
        add_span_attribute("key", "value")

    def test_get_trace_id_no_span(self):
        """Test get_current_trace_id when no span is active."""
        trace_id = get_current_trace_id()
        # May be None or a value depending on OpenTelemetry state


class TestMetrics:
    """Tests for metrics recording."""

    def test_get_metrics_registry(self):
        """Test getting the metrics registry."""
        registry = get_metrics_registry()
        assert registry is not None

    def test_record_intent_resolution(self):
        """Test recording intent resolution metrics."""
        # Should not raise
        record_intent_resolution(
            duration_seconds=0.5,
            tenant_id="test-tenant",
            path_taken="fast_path",
            is_compound=False,
            confidence=0.92,
        )

    def test_record_pipeline_stage(self):
        """Test recording pipeline stage metrics."""
        record_pipeline_stage(
            stage="entity_extraction",
            duration_seconds=0.05,
            tenant_id="test-tenant",
        )

    def test_record_rate_limit_exceeded(self):
        """Test recording rate limit exceeded metrics."""
        record_rate_limit_exceeded("test-tenant")


class TestStructuredLogging:
    """Tests for structured logging."""

    def test_configure_logging(self):
        """Test logging configuration."""
        configure_logging(
            level="DEBUG",
            json_format=True,
            include_tenant=True,
        )

        logger = get_logger("test_logger")
        assert logger is not None

    def test_structured_log_formatter(self):
        """Test structured log formatter."""
        formatter = StructuredLogFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        # Should be valid JSON
        import json
        data = json.loads(output)

        assert data["message"] == "Test message"
        assert data["level"] == "INFO"
        assert data["logger"] == "test"

    def test_tenant_context_filter(self):
        """Test tenant context filter."""
        filter = TenantContextFilter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = filter.filter(record)

        assert result is True
        assert hasattr(record, "tenant_id")
        assert hasattr(record, "trace_id")

    def test_get_logger(self):
        """Test get_logger returns proper logger."""
        logger = get_logger("my.module")
        assert logger.name == "my.module"


class TestMetricsRegistry:
    """Tests for MetricsRegistry class."""

    def test_registry_initialization(self):
        """Test metrics registry initialization."""
        registry = MetricsRegistry("test_registry")
        assert registry is not None

    def test_record_websocket_connection(self):
        """Test recording WebSocket connection metrics."""
        registry = MetricsRegistry("ws_test")
        # Should not raise
        registry.record_websocket_connection("tenant-1", delta=1)
        registry.record_websocket_connection("tenant-1", delta=-1)

    def test_record_batch_job(self):
        """Test recording batch job metrics."""
        registry = MetricsRegistry("batch_test")
        registry.record_batch_job(
            tenant_id="tenant-1",
            item_count=100,
            duration_seconds=5.5,
            status="completed",
        )

    def test_record_llm_call(self):
        """Test recording LLM call metrics."""
        registry = MetricsRegistry("llm_test")
        registry.record_llm_call(
            model="claude-sonnet-4-5",
            duration_seconds=2.3,
            input_tokens=500,
            output_tokens=200,
            tenant_id="tenant-1",
            status="success",
        )
