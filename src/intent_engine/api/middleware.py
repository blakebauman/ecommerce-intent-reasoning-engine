"""API middleware for authentication, logging, and tracing."""

import logging
import time
import uuid
from collections.abc import Callable

from fastapi import HTTPException, Request, Response
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware

from intent_engine.config import get_settings
from intent_engine.observability.tracing import add_span_attribute, get_current_trace_id
from intent_engine.tenancy.context import get_current_tenant_id

logger = logging.getLogger(__name__)

# API key header
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def verify_api_key(request: Request) -> str:
    """
    Verify the API key from the Authorization header.

    Expected format: "Bearer <api_key>"

    Args:
        request: The incoming request.

    Returns:
        The verified API key.

    Raises:
        HTTPException: If the API key is missing or invalid.
    """
    settings = get_settings()

    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
        )

    # Parse Bearer token
    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header format. Expected: Bearer <api_key>",
        )

    api_key = parts[1]

    # Validate API key
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )

    return api_key


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request logging and timing."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        """Log request details and timing with trace correlation."""
        # Generate request ID if not present
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Add request ID to state for access in handlers
        request.state.request_id = request_id

        # Get trace ID and tenant ID for correlation
        trace_id = get_current_trace_id()
        tenant_id = get_current_tenant_id()

        # Add attributes to current span
        add_span_attribute("http.request_id", request_id)
        if tenant_id:
            add_span_attribute("tenant.id", tenant_id)

        # Log request start
        start_time = time.perf_counter()
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "tenant_id": tenant_id,
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else "unknown",
            },
        )

        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            # Log error
            processing_time = int((time.perf_counter() - start_time) * 1000)
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "trace_id": trace_id,
                    "tenant_id": tenant_id,
                    "error": str(e),
                    "processing_time_ms": processing_time,
                },
            )
            raise

        # Calculate processing time
        processing_time = int((time.perf_counter() - start_time) * 1000)

        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time-Ms"] = str(processing_time)
        if trace_id:
            response.headers["X-Trace-ID"] = trace_id

        # Log request completion
        logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "tenant_id": tenant_id,
                "status_code": response.status_code,
                "processing_time_ms": processing_time,
            },
        )

        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security-related headers to all responses."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response
