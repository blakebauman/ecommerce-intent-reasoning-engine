"""Tenant middleware for FastAPI."""

import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from intent_engine.tenancy.context import clear_tenant_context, set_tenant_context
from intent_engine.tenancy.models import TenantConfig, TenantTier
from intent_engine.tenancy.rate_limiter import RateLimitExceeded, RateLimiter

logger = logging.getLogger(__name__)


class TenantMiddleware(BaseHTTPMiddleware):
    """
    Middleware for multi-tenant isolation.

    Responsibilities:
    1. Extract API key from request (Authorization header or query param)
    2. Look up tenant by API key
    3. Set tenant context for the request
    4. Apply rate limiting
    5. Add tenant headers to response

    Configuration:
    - tenant_lookup: Async function to look up tenant by API key
    - rate_limiter: Optional RateLimiter instance
    - exclude_paths: Paths to exclude from tenant authentication
    """

    def __init__(
        self,
        app: Callable,
        tenant_lookup: Callable[[str], TenantConfig | None] | None = None,
        rate_limiter: RateLimiter | None = None,
        exclude_paths: list[str] | None = None,
        dev_mode: bool = False,
        dev_tenant: TenantConfig | None = None,
    ) -> None:
        """
        Initialize the middleware.

        Args:
            app: The FastAPI application.
            tenant_lookup: Async function to look up tenant by API key.
            rate_limiter: Optional RateLimiter for rate limiting.
            exclude_paths: Paths to exclude from authentication.
            dev_mode: If True, use dev tenant for all requests.
            dev_tenant: Tenant config to use in dev mode.
        """
        super().__init__(app)
        self.tenant_lookup = tenant_lookup
        self.rate_limiter = rate_limiter
        self.exclude_paths = exclude_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/redoc",
        ]
        self.dev_mode = dev_mode
        self.dev_tenant = dev_tenant or TenantConfig(
            tenant_id="dev-tenant",
            name="Development",
            tier=TenantTier.ENTERPRISE,
            api_key="dev-api-key",
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request with tenant context."""
        # Check if path should be excluded
        if self._should_exclude(request.url.path):
            return await call_next(request)

        try:
            # Get tenant from request
            tenant = await self._get_tenant(request)

            if tenant is None:
                return Response(
                    content='{"detail": "Invalid or missing API key"}',
                    status_code=401,
                    media_type="application/json",
                )

            if not tenant.is_active:
                return Response(
                    content='{"detail": "Tenant account is inactive"}',
                    status_code=403,
                    media_type="application/json",
                )

            # Set tenant context
            set_tenant_context(tenant)

            # Apply rate limiting
            if self.rate_limiter:
                try:
                    await self.rate_limiter.check_rate_limit(
                        tenant_id=tenant.tenant_id,
                        rate_limit=tenant.get_rate_limit(),
                        burst_size=tenant.get_burst_size(),
                    )
                except RateLimitExceeded as e:
                    return Response(
                        content=f'{{"detail": "Rate limit exceeded", "retry_after": {e.retry_after:.2f}}}',
                        status_code=429,
                        media_type="application/json",
                        headers={
                            "Retry-After": str(int(e.retry_after) + 1),
                            "X-RateLimit-Limit": str(e.limit),
                            "X-RateLimit-Remaining": "0",
                        },
                    )

            # Process request
            response = await call_next(request)

            # Add tenant headers
            response.headers["X-Tenant-Id"] = tenant.tenant_id
            response.headers["X-Tenant-Tier"] = tenant.tier.value

            return response

        except Exception as e:
            logger.exception(f"Error in tenant middleware: {e}")
            return Response(
                content='{"detail": "Internal server error"}',
                status_code=500,
                media_type="application/json",
            )
        finally:
            # Always clear tenant context
            clear_tenant_context()

    def _should_exclude(self, path: str) -> bool:
        """Check if path should be excluded from tenant auth."""
        for exclude in self.exclude_paths:
            if path.startswith(exclude):
                return True
        return False

    async def _get_tenant(self, request: Request) -> TenantConfig | None:
        """
        Extract and validate tenant from request.

        Checks in order:
        1. Authorization: Bearer <api_key>
        2. X-API-Key header
        3. api_key query parameter

        Args:
            request: The incoming request.

        Returns:
            TenantConfig if valid, None otherwise.
        """
        api_key = None

        # Check Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            api_key = auth_header[7:]

        # Check X-API-Key header
        if not api_key:
            api_key = request.headers.get("X-API-Key")

        # Check query parameter
        if not api_key:
            api_key = request.query_params.get("api_key")

        if not api_key:
            return None

        # Dev mode: return dev tenant for any key
        if self.dev_mode:
            # Update dev tenant's API key to match
            self.dev_tenant.api_key = api_key
            return self.dev_tenant

        # Look up tenant by API key
        if self.tenant_lookup:
            return await self._lookup_tenant(api_key)

        return None

    async def _lookup_tenant(self, api_key: str) -> TenantConfig | None:
        """
        Look up tenant by API key.

        Args:
            api_key: The API key to look up.

        Returns:
            TenantConfig if found, None otherwise.
        """
        if self.tenant_lookup is None:
            return None

        try:
            # Support both sync and async lookup functions
            result = self.tenant_lookup(api_key)
            if hasattr(result, "__await__"):
                return await result
            return result
        except Exception as e:
            logger.error(f"Error looking up tenant: {e}")
            return None


class TenantStore:
    """
    Simple in-memory tenant store for development/testing.

    In production, this would be backed by a database.
    """

    def __init__(self) -> None:
        self._tenants: dict[str, TenantConfig] = {}

    def add_tenant(self, tenant: TenantConfig) -> None:
        """Add a tenant to the store."""
        self._tenants[tenant.api_key] = tenant

    def get_tenant_by_api_key(self, api_key: str) -> TenantConfig | None:
        """Get tenant by API key."""
        return self._tenants.get(api_key)

    def get_tenant_by_id(self, tenant_id: str) -> TenantConfig | None:
        """Get tenant by tenant ID."""
        for tenant in self._tenants.values():
            if tenant.tenant_id == tenant_id:
                return tenant
        return None

    def remove_tenant(self, tenant_id: str) -> bool:
        """Remove a tenant from the store."""
        for api_key, tenant in list(self._tenants.items()):
            if tenant.tenant_id == tenant_id:
                del self._tenants[api_key]
                return True
        return False

    def list_tenants(self) -> list[TenantConfig]:
        """List all tenants."""
        return list(self._tenants.values())
