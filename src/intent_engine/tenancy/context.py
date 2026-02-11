"""Tenant context propagation using contextvars."""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

from intent_engine.tenancy.models import TenantConfig

# Context variable for the current tenant
_current_tenant: ContextVar[TenantConfig | None] = ContextVar("current_tenant", default=None)


def get_current_tenant() -> TenantConfig | None:
    """
    Get the current tenant from context.

    Returns:
        The current TenantConfig, or None if not set.
    """
    return _current_tenant.get()


def get_current_tenant_id() -> str | None:
    """
    Get the current tenant ID from context.

    Returns:
        The current tenant ID, or None if not set.
    """
    tenant = _current_tenant.get()
    return tenant.tenant_id if tenant else None


def set_tenant_context(tenant: TenantConfig) -> None:
    """
    Set the current tenant in context.

    This should be called by middleware after authenticating the tenant.

    Args:
        tenant: The TenantConfig to set as current.
    """
    _current_tenant.set(tenant)


def clear_tenant_context() -> None:
    """
    Clear the current tenant from context.

    This should be called at the end of request processing.
    """
    _current_tenant.set(None)


@contextmanager
def tenant_context(tenant: TenantConfig) -> Generator[TenantConfig, None, None]:
    """
    Context manager for tenant context.

    Usage:
        with tenant_context(tenant_config):
            # Do work with tenant context
            pass

    Args:
        tenant: The TenantConfig to use for this context.

    Yields:
        The tenant config.
    """
    token = _current_tenant.set(tenant)
    try:
        yield tenant
    finally:
        _current_tenant.reset(token)


def require_tenant() -> TenantConfig:
    """
    Get the current tenant, raising an error if not set.

    This is useful for code that requires a tenant context.

    Returns:
        The current TenantConfig.

    Raises:
        RuntimeError: If no tenant context is set.
    """
    tenant = _current_tenant.get()
    if tenant is None:
        raise RuntimeError("No tenant context set. Ensure TenantMiddleware is configured.")
    return tenant
