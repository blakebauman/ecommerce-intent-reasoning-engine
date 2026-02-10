"""FastAPI application server with Phase 3 production features."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import redis.asyncio as redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from intent_engine.api.a2a_routes import a2a_router
from intent_engine.api.agent_routes import router as agent_router
from intent_engine.api.middleware import RequestLoggingMiddleware
from intent_engine.api.routes import router, set_engine
from intent_engine.api.webhooks import router as webhooks_router
from intent_engine.config import get_settings
from intent_engine.engine import IntentEngine

# Phase 3 imports
from intent_engine.observability.logging import configure_logging
from intent_engine.observability.telemetry import TelemetryConfig, init_telemetry, shutdown_telemetry
from intent_engine.tenancy.middleware import TenantMiddleware, TenantStore
from intent_engine.tenancy.models import TenantConfig, TenantTier
from intent_engine.tenancy.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# Global state
_engine: IntentEngine | None = None
_redis_client: redis.Redis | None = None
_batch_queue = None
_batch_worker = None
_tenant_store: TenantStore | None = None
_rate_limiter: RateLimiter | None = None


def get_engine() -> IntentEngine | None:
    """Get the global engine instance."""
    return _engine


def get_batch_queue():
    """Get the global batch queue instance."""
    return _batch_queue


def get_tenant_store() -> TenantStore | None:
    """Get the global tenant store instance."""
    return _tenant_store


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.

    Initializes and shuts down all components:
    - OpenTelemetry (tracing + metrics)
    - Structured logging
    - Redis connection
    - Intent Engine
    - Rate limiter
    - Batch queue and worker
    - WebSocket manager
    """
    global _engine, _redis_client, _batch_queue, _batch_worker, _tenant_store, _rate_limiter

    settings = get_settings()

    # Initialize structured logging
    configure_logging(
        level=settings.log_level,
        json_format=settings.log_json,
        include_tenant=settings.enable_multi_tenant,
    )

    logger.info("Starting Intent Engine...")

    # Initialize OpenTelemetry
    if settings.enable_tracing or settings.enable_metrics:
        telemetry_config = TelemetryConfig(
            service_name=settings.service_name,
            service_version=settings.api_version,
            environment=settings.service_environment,
            otlp_endpoint=settings.otlp_endpoint,
            enable_tracing=settings.enable_tracing,
            enable_metrics=settings.enable_metrics,
        )
        init_telemetry(telemetry_config)
        logger.info("OpenTelemetry initialized")

    # Initialize Redis connection
    try:
        _redis_client = redis.from_url(settings.redis_url, decode_responses=True)
        await _redis_client.ping()
        logger.info("Redis connected")
    except Exception as e:
        logger.warning(f"Redis not available: {e}")
        _redis_client = None

    # Initialize rate limiter
    if _redis_client and settings.enable_multi_tenant:
        _rate_limiter = RateLimiter(
            redis_client=_redis_client,
            default_rate=settings.default_rate_limit_rpm,
            default_burst=settings.default_rate_limit_burst,
        )
        logger.info("Rate limiter initialized")

    # Initialize tenant store with dev tenant
    _tenant_store = TenantStore()
    if settings.tenant_dev_mode:
        dev_tenant = TenantConfig(
            tenant_id="dev-tenant",
            name="Development",
            tier=TenantTier.ENTERPRISE,
            api_key=settings.api_key,
            websocket_enabled=settings.ws_enabled,
            batch_processing_enabled=True,
        )
        _tenant_store.add_tenant(dev_tenant)

    # Initialize engine
    _engine = IntentEngine(settings=settings)
    await _engine.initialize()
    set_engine(_engine)
    logger.info("Intent Engine initialized")

    # Initialize batch processing
    if _redis_client and settings.batch_worker_enabled:
        try:
            from intent_engine.batch.queue import BatchQueue
            from intent_engine.batch.worker import BatchWorker, create_email_processor

            _batch_queue = BatchQueue(
                redis_client=_redis_client,
                job_ttl_hours=settings.batch_job_ttl_hours,
            )

            # Create email processor
            processor = await create_email_processor(get_engine)

            _batch_worker = BatchWorker(
                queue=_batch_queue,
                process_item=processor,
                tenant_lookup=_tenant_store.get_tenant_by_id if _tenant_store else None,
                concurrency=settings.batch_worker_concurrency,
                poll_interval=settings.batch_poll_interval,
            )
            await _batch_worker.start(tenant_ids=["dev-tenant"])
            logger.info("Batch worker started")
        except Exception as e:
            logger.warning(f"Batch processing not available: {e}")
            _batch_queue = None
            _batch_worker = None

    logger.info("Intent Engine ready")

    yield

    # Shutdown
    logger.info("Shutting down Intent Engine...")

    # Stop batch worker
    if _batch_worker:
        await _batch_worker.stop()
        logger.info("Batch worker stopped")

    # Shutdown engine
    if _engine:
        await _engine.shutdown()
        logger.info("Intent Engine shutdown complete")

    # Close Redis
    if _redis_client:
        await _redis_client.aclose()
        logger.info("Redis disconnected")

    # Shutdown telemetry
    shutdown_telemetry()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=(
            "eCommerce Intent Reasoning Engine API. "
            "Classifies and decomposes customer intents from text messages."
        ),
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add tenant middleware (before request logging for tenant context)
    if settings.enable_multi_tenant:
        app.add_middleware(
            TenantMiddleware,
            tenant_lookup=lambda key: _tenant_store.get_tenant_by_api_key(key) if _tenant_store else None,
            rate_limiter=_rate_limiter,
            exclude_paths=["/health", "/metrics", "/docs", "/openapi.json", "/redoc", "/.well-known"],
            dev_mode=settings.tenant_dev_mode,
        )

    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware)

    # Include routes
    app.include_router(router)

    # Include A2A protocol routes
    app.include_router(a2a_router)

    # Include webhook routes
    app.include_router(webhooks_router)

    # Include agent routes
    app.include_router(agent_router)

    # Include batch routes
    try:
        from intent_engine.api.batch import router as batch_router
        app.include_router(batch_router)
    except ImportError:
        logger.warning("Batch routes not available")

    # Include WebSocket routes
    if settings.ws_enabled:
        try:
            from intent_engine.api.websocket import create_websocket_endpoint
            from intent_engine.api.ws_auth import WebSocketAuthenticator

            authenticator = WebSocketAuthenticator(
                tenant_lookup=lambda key: _tenant_store.get_tenant_by_api_key(key) if _tenant_store else None,
                dev_mode=settings.tenant_dev_mode,
            )
            ws_router = create_websocket_endpoint(
                engine_getter=get_engine,
                authenticator=authenticator,
            )
            app.include_router(ws_router)
        except ImportError:
            logger.warning("WebSocket routes not available")

    # Prometheus metrics endpoint
    if settings.enable_metrics:
        @app.get("/metrics", include_in_schema=False)
        async def metrics():
            """Prometheus metrics endpoint."""
            try:
                from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
                from starlette.responses import Response

                return Response(
                    content=generate_latest(),
                    media_type=CONTENT_TYPE_LATEST,
                )
            except ImportError:
                from starlette.responses import PlainTextResponse
                return PlainTextResponse("Prometheus metrics not available")

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "intent_engine.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
