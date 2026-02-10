"""Structured JSON logging with trace correlation."""

import logging
import sys
from typing import Any

from intent_engine.observability.tracing import get_current_span_id, get_current_trace_id
from intent_engine.tenancy.context import get_current_tenant_id


class StructuredLogFormatter(logging.Formatter):
    """
    JSON log formatter with trace and tenant correlation.

    Outputs logs in JSON format with:
    - Standard log fields (timestamp, level, message, logger)
    - Trace correlation (trace_id, span_id)
    - Tenant context (tenant_id)
    - Exception information
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        import json
        from datetime import datetime, timezone

        # Build base log entry
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add trace correlation
        trace_id = get_current_trace_id()
        if trace_id:
            log_entry["trace_id"] = trace_id

        span_id = get_current_span_id()
        if span_id:
            log_entry["span_id"] = span_id

        # Add tenant context
        tenant_id = get_current_tenant_id()
        if tenant_id:
            log_entry["tenant_id"] = tenant_id

        # Add location info
        log_entry["location"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Add extra fields from record
        if hasattr(record, "extra") and record.extra:
            log_entry["extra"] = record.extra

        # Add exception info
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class TenantContextFilter(logging.Filter):
    """
    Logging filter that adds tenant context to log records.

    Adds tenant_id and trace_id to all log records for filtering.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context fields to the log record."""
        record.tenant_id = get_current_tenant_id() or "unknown"
        record.trace_id = get_current_trace_id() or ""
        record.span_id = get_current_span_id() or ""
        return True


def configure_logging(
    level: str = "INFO",
    json_format: bool = True,
    include_tenant: bool = True,
    module_levels: dict[str, str] | None = None,
) -> None:
    """
    Configure structured logging for the application.

    Args:
        level: Default log level.
        json_format: If True, use JSON format. Otherwise, use standard format.
        include_tenant: If True, add tenant context to logs.
        module_levels: Per-module log levels (e.g., {"httpx": "WARNING"}).
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    # Set formatter
    if json_format:
        handler.setFormatter(StructuredLogFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))

    # Add tenant context filter
    if include_tenant:
        handler.addFilter(TenantContextFilter())

    root_logger.addHandler(handler)

    # Configure per-module levels
    if module_levels:
        for module, mod_level in module_levels.items():
            logging.getLogger(module).setLevel(getattr(logging, mod_level.upper()))

    # Reduce noise from common libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logging.info(
        f"Logging configured: level={level}, json={json_format}, "
        f"module_levels={module_levels or {}}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    This is a convenience wrapper around logging.getLogger
    that ensures consistent logger naming.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


class LogContext:
    """
    Context manager for adding extra fields to logs.

    Usage:
        with LogContext(order_id="12345", action="process"):
            logger.info("Processing order")  # Includes extra fields
    """

    def __init__(self, **kwargs: Any) -> None:
        self.extra = kwargs
        self._old_factory = None

    def __enter__(self) -> "LogContext":
        self._old_factory = logging.getLogRecordFactory()

        extra = self.extra

        def record_factory(*args, **kwargs):
            record = self._old_factory(*args, **kwargs)
            record.extra = extra
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, *args) -> None:
        if self._old_factory:
            logging.setLogRecordFactory(self._old_factory)
