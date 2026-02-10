"""Tracing utilities and decorators."""

import functools
import logging
from contextlib import contextmanager
from typing import Any, Callable, Generator, TypeVar

logger = logging.getLogger(__name__)

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


def get_tracer(name: str = "intent_engine"):
    """
    Get an OpenTelemetry tracer.

    Args:
        name: Tracer name (typically module name).

    Returns:
        OpenTelemetry tracer or no-op tracer if not available.
    """
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except ImportError:
        return _NoOpTracer()


class _NoOpSpan:
    """No-op span for when OpenTelemetry is not available."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        pass

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def end(self) -> None:
        pass


class _NoOpTracer:
    """No-op tracer for when OpenTelemetry is not available."""

    def start_span(self, name: str, **kwargs) -> _NoOpSpan:
        return _NoOpSpan()

    def start_as_current_span(self, name: str, **kwargs) -> _NoOpSpan:
        return _NoOpSpan()


def traced(
    name: str | None = None,
    attributes: dict[str, Any] | None = None,
    record_exception: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to trace a function.

    Creates a span that wraps the function execution.
    Works with both sync and async functions.

    Args:
        name: Span name (defaults to function name).
        attributes: Static attributes to add to the span.
        record_exception: Whether to record exceptions.

    Returns:
        Decorated function.

    Example:
        @traced(name="process_order", attributes={"component": "order_service"})
        async def process_order(order_id: str):
            ...
    """
    def decorator(func: F) -> F:
        span_name = name or func.__name__
        tracer = get_tracer(func.__module__)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    span.set_attributes(attributes)
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if record_exception:
                        span.record_exception(e)
                        _set_error_status(span)
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    span.set_attributes(attributes)
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if record_exception:
                        span.record_exception(e)
                        _set_error_status(span)
                    raise

        if asyncio_check(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def asyncio_check(func: Callable) -> bool:
    """Check if a function is async."""
    import asyncio
    return asyncio.iscoroutinefunction(func)


def _set_error_status(span) -> None:
    """Set span status to error."""
    try:
        from opentelemetry.trace import StatusCode
        span.set_status(StatusCode.ERROR)
    except ImportError:
        pass


@contextmanager
def pipeline_span(
    stage_name: str,
    tenant_id: str | None = None,
    request_id: str | None = None,
    **attributes,
) -> Generator[Any, None, None]:
    """
    Context manager for tracing pipeline stages.

    Creates a span for a specific pipeline stage with common attributes.

    Args:
        stage_name: Name of the pipeline stage.
        tenant_id: Optional tenant ID.
        request_id: Optional request ID.
        **attributes: Additional span attributes.

    Yields:
        The span object.

    Example:
        with pipeline_span("entity_extraction", tenant_id="t1", request_id="r1"):
            entities = extract_entities(text)
    """
    tracer = get_tracer("intent_engine.pipeline")

    span_attrs = {
        "pipeline.stage": stage_name,
    }
    if tenant_id:
        span_attrs["tenant.id"] = tenant_id
    if request_id:
        span_attrs["request.id"] = request_id
    span_attrs.update(attributes)

    with tracer.start_as_current_span(f"intent_engine.{stage_name}") as span:
        span.set_attributes(span_attrs)
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            _set_error_status(span)
            raise


def add_span_attribute(key: str, value: Any) -> None:
    """
    Add an attribute to the current span.

    Args:
        key: Attribute key.
        value: Attribute value.
    """
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        if span:
            span.set_attribute(key, value)
    except ImportError:
        pass


def add_span_event(name: str, attributes: dict[str, Any] | None = None) -> None:
    """
    Add an event to the current span.

    Args:
        name: Event name.
        attributes: Optional event attributes.
    """
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        if span:
            span.add_event(name, attributes or {})
    except ImportError:
        pass


def get_current_trace_id() -> str | None:
    """
    Get the current trace ID.

    Returns:
        Trace ID as hex string, or None if not in a trace.
    """
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        if span:
            ctx = span.get_span_context()
            if ctx.is_valid:
                return format(ctx.trace_id, "032x")
    except ImportError:
        pass
    return None


def get_current_span_id() -> str | None:
    """
    Get the current span ID.

    Returns:
        Span ID as hex string, or None if not in a span.
    """
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        if span:
            ctx = span.get_span_context()
            if ctx.is_valid:
                return format(ctx.span_id, "016x")
    except ImportError:
        pass
    return None
