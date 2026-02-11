"""FastAPI application server with Phase 3 production features."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from intent_engine.api.a2a_routes import a2a_router
from intent_engine.api.agent_routes import router as agent_router
from intent_engine.api.middleware import RequestLoggingMiddleware, SecurityHeadersMiddleware
from intent_engine.api.routes import router
from intent_engine.api.webhooks import router as webhooks_router
from intent_engine.config import get_settings
from intent_engine.engine import IntentEngine
from intent_engine.exceptions import IntentEngineError

# Phase 3 imports
from intent_engine.observability.logging import configure_logging
from intent_engine.observability.telemetry import (
    TelemetryConfig,
    init_telemetry,
    shutdown_telemetry,
)
from intent_engine.tenancy.db_store import DbTenantStore
from intent_engine.tenancy.middleware import TenantMiddleware, TenantStore
from intent_engine.tenancy.models import TenantConfig, TenantTier
from intent_engine.tenancy.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# Global state
_engine: IntentEngine | None = None
_redis_client: redis.Redis | None = None
_batch_queue = None
_batch_worker = None
_tenant_store: TenantStore | DbTenantStore | None = None
_rate_limiter: RateLimiter | None = None


def get_engine() -> IntentEngine | None:
    """Get the global engine instance."""
    return _engine


def get_batch_queue():
    """Get the global batch queue instance."""
    return _batch_queue


def get_tenant_store() -> TenantStore | DbTenantStore | None:
    """Get the global tenant store instance."""
    return _tenant_store


def get_redis() -> redis.Redis | None:
    """Get the global Redis client (if connected)."""
    return _redis_client


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

    # Production config warnings (don't fail startup; log only)
    if settings.service_environment == "production":
        if settings.tenant_dev_mode:
            logger.warning(
                "Production: TENANT_DEV_MODE is true; any API key is accepted. Set TENANT_DEV_MODE=false."
            )
        if settings.api_key in ("", "dev-api-key") and settings.tenant_store_backend != "db":
            logger.warning(
                "Production: API_KEY is default or empty (single-tenant mode). Set a strong API_KEY."
            )
        if "*" in settings.cors_origins:
            logger.warning(
                "Production: CORS_ORIGINS allows all origins (*). Restrict to your front-end domains."
            )

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
    except (redis.ConnectionError, OSError) as e:
        logger.warning("Redis not available: %s", e, exc_info=True)
        _redis_client = None

    # Initialize rate limiter
    if _redis_client and settings.enable_multi_tenant:
        _rate_limiter = RateLimiter(
            redis_client=_redis_client,
            default_rate=settings.default_rate_limit_rpm,
            default_burst=settings.default_rate_limit_burst,
        )
        logger.info("Rate limiter initialized")

    # Initialize tenant store: DB (multi-tenant production) or in-memory (dev / single-tenant)
    if settings.tenant_store_backend == "db":
        _tenant_store = DbTenantStore(database_url=settings.database_url)
        await _tenant_store.connect()
        logger.info("Tenant store: database backend")
    else:
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
        elif settings.api_key:
            default_tenant = TenantConfig(
                tenant_id="default",
                name="Default",
                tier=TenantTier.ENTERPRISE,
                api_key=settings.api_key,
                websocket_enabled=settings.ws_enabled,
                batch_processing_enabled=True,
            )
            _tenant_store.add_tenant(default_tenant)
            logger.info("Single-tenant production: default tenant registered")

    # Initialize engine and attach to app state for dependency injection
    _engine = IntentEngine(settings=settings)
    await _engine.initialize()
    app.state.engine = _engine
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
            list_result = _tenant_store.list_tenants()
            if hasattr(list_result, "__await__"):
                list_result = await list_result
            tenant_ids = [t.tenant_id for t in (list_result or [])]
            await _batch_worker.start(tenant_ids=tenant_ids)
            logger.info("Batch worker started")
        except Exception as e:
            logger.warning("Batch processing not available: %s", e, exc_info=True)
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

    # Close tenant store (DB backend)
    if _tenant_store is not None and isinstance(_tenant_store, DbTenantStore):
        await _tenant_store.close()
        logger.info("Tenant store disconnected")

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

    # Domain exception handler: map IntentEngineError to JSON response
    @app.exception_handler(IntentEngineError)
    async def intent_engine_error_handler(request: Request, exc: IntentEngineError):
        from starlette.responses import JSONResponse

        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    # Add CORS middleware (origins configurable via CORS_ORIGINS env)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Security headers (X-Content-Type-Options, X-Frame-Options)
    app.add_middleware(SecurityHeadersMiddleware)

    # Add tenant middleware (before request logging for tenant context).
    # Use rate_limiter_getter so the limiter (initialized in lifespan) is resolved at request time.
    if settings.enable_multi_tenant:
        app.add_middleware(
            TenantMiddleware,
            tenant_lookup=lambda key: (
                _tenant_store.get_tenant_by_api_key(key) if _tenant_store else None
            ),
            rate_limiter_getter=lambda: _rate_limiter,
            exclude_paths=[
                "/health",
                "/ready",
                "/metrics",
                "/docs",
                "/openapi.json",
                "/redoc",
                "/.well-known",
                "/v1/admin",
            ],
            dev_mode=settings.tenant_dev_mode,
        )

    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware)

    # Include routes
    app.include_router(router)

    # Readiness probe (DB + Redis + tenant store when DB-backed); no auth required
    @app.get("/ready", include_in_schema=False)
    async def ready():
        """Readiness: 200 if DB, Redis (and tenant store when backend=db) are reachable, 503 otherwise."""
        engine = get_engine()
        redis_client = get_redis()
        tenant_store = get_tenant_store()
        db_ok = False
        redis_ok = False
        tenant_store_ok = True  # not required when memory backend
        if engine and engine.components.vector_store:
            db_ok = await engine.components.vector_store.check()
        if redis_client:
            try:
                await redis_client.ping()
                redis_ok = True
            except redis.RedisError:
                pass
        if isinstance(tenant_store, DbTenantStore):
            try:
                await tenant_store.list_tenants()
                tenant_store_ok = True
            except Exception:
                tenant_store_ok = False
        ok = db_ok and redis_ok and tenant_store_ok
        if ok:
            payload = {"status": "ready", "database": "ok", "redis": "ok"}
            if isinstance(tenant_store, DbTenantStore):
                payload["tenant_store"] = "ok"
            return payload
        from starlette.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "database": "ok" if db_ok else "error",
                "redis": "ok" if redis_ok else "error",
                "tenant_store": "ok" if tenant_store_ok else "error",
            },
        )

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

    # Admin API (tenant management) when DB store and admin key are set
    if settings.tenant_store_backend == "db" and settings.admin_api_key:
        from intent_engine.api.admin_routes import router as admin_router

        app.include_router(admin_router)

    # Include WebSocket routes
    if settings.ws_enabled:
        try:
            from intent_engine.api.websocket import create_websocket_endpoint
            from intent_engine.api.ws_auth import WebSocketAuthenticator

            authenticator = WebSocketAuthenticator(
                tenant_lookup=lambda key: (
                    _tenant_store.get_tenant_by_api_key(key) if _tenant_store else None
                ),
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
