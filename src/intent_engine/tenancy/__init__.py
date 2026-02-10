"""Multi-tenant isolation for the Intent Engine."""

from intent_engine.tenancy.context import (
    clear_tenant_context,
    get_current_tenant,
    get_current_tenant_id,
    set_tenant_context,
    tenant_context,
)
from intent_engine.tenancy.middleware import TenantMiddleware
from intent_engine.tenancy.models import TenantConfig, TenantTier
from intent_engine.tenancy.rate_limiter import RateLimitExceeded, RateLimiter

__all__ = [
    # Context
    "get_current_tenant",
    "get_current_tenant_id",
    "set_tenant_context",
    "clear_tenant_context",
    "tenant_context",
    # Middleware
    "TenantMiddleware",
    # Models
    "TenantConfig",
    "TenantTier",
    # Rate limiting
    "RateLimiter",
    "RateLimitExceeded",
]
