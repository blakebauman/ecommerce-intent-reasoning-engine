"""Database-backed tenant store for production multi-tenancy."""

import json
import logging
from typing import Any

import asyncpg

from intent_engine.tenancy.models import TenantConfig, TenantTier

logger = logging.getLogger(__name__)

# Optional fields stored in tenants.settings JSONB (overrides tier defaults)
SETTINGS_KEYS = (
    "requests_per_minute",
    "burst_size",
    "max_batch_size",
    "max_websocket_connections",
    "fast_path_enabled",
    "reasoning_path_enabled",
    "batch_processing_enabled",
    "websocket_enabled",
    "shopify_enabled",
    "woocommerce_enabled",
    "bigcommerce_enabled",
    "adobe_commerce_enabled",
)


def _row_to_tenant(row: asyncpg.Record) -> TenantConfig:
    """Build TenantConfig from a DB row."""
    raw = row["settings"]
    if isinstance(raw, str):
        settings = json.loads(raw) if raw else {}
    else:
        settings = raw if isinstance(raw, dict) else {}
    # Only pass keys that TenantConfig accepts and we store
    overrides = {k: v for k, v in settings.items() if k in SETTINGS_KEYS}
    return TenantConfig(
        tenant_id=row["tenant_id"],
        name=row["name"],
        api_key=row["api_key"],
        tier=TenantTier(row["tier"]),
        is_active=row["is_active"],
        **overrides,
    )


def _tenant_to_settings(tenant: TenantConfig) -> dict[str, Any]:
    """Extract optional overrides from TenantConfig for JSONB."""
    return {
        k: v
        for k, v in tenant.model_dump().items()
        if k in SETTINGS_KEYS and v is not None
    }


class DbTenantStore:
    """
    Database-backed tenant store.

    Uses the same interface as the in-memory TenantStore:
    get_tenant_by_api_key, get_tenant_by_id, list_tenants, add_tenant.
    """

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Create connection pool and ensure tenants table exists."""
        self._pool = await asyncpg.create_pool(
            self._database_url,
            min_size=1,
            max_size=5,
            command_timeout=10,
        )
        logger.info("DbTenantStore connected")

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("DbTenantStore disconnected")

    def _require_pool(self) -> asyncpg.Pool:
        if not self._pool:
            raise RuntimeError("DbTenantStore not connected. Call connect() first.")
        return self._pool

    async def get_tenant_by_api_key(self, api_key: str) -> TenantConfig | None:
        """Get tenant by API key (active only)."""
        pool = self._require_pool()
        row = await pool.fetchrow(
            """
            SELECT tenant_id, name, api_key, tier, is_active, settings
            FROM tenants
            WHERE api_key = $1 AND is_active = true
            """,
            api_key,
        )
        return _row_to_tenant(row) if row else None

    async def get_tenant_by_id(self, tenant_id: str) -> TenantConfig | None:
        """Get tenant by tenant_id (active only)."""
        pool = self._require_pool()
        row = await pool.fetchrow(
            """
            SELECT tenant_id, name, api_key, tier, is_active, settings
            FROM tenants
            WHERE tenant_id = $1 AND is_active = true
            """,
            tenant_id,
        )
        return _row_to_tenant(row) if row else None

    async def list_tenants(self) -> list[TenantConfig]:
        """List all active tenants."""
        pool = self._require_pool()
        rows = await pool.fetch(
            """
            SELECT tenant_id, name, api_key, tier, is_active, settings
            FROM tenants
            WHERE is_active = true
            ORDER BY tenant_id
            """
        )
        return [_row_to_tenant(r) for r in rows]

    async def add_tenant(self, tenant: TenantConfig) -> None:
        """Insert or replace a tenant (upsert by tenant_id)."""
        pool = self._require_pool()
        settings = _tenant_to_settings(tenant)
        await pool.execute(
            """
            INSERT INTO tenants (tenant_id, name, api_key, tier, is_active, settings, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, NOW())
            ON CONFLICT (tenant_id)
            DO UPDATE SET
                name = EXCLUDED.name,
                api_key = EXCLUDED.api_key,
                tier = EXCLUDED.tier,
                is_active = EXCLUDED.is_active,
                settings = EXCLUDED.settings,
                updated_at = NOW()
            """,
            tenant.tenant_id,
            tenant.name,
            tenant.api_key,
            tenant.tier.value,
            tenant.is_active,
            json.dumps(settings),
        )
        logger.info("Tenant upserted: %s", tenant.tenant_id)

    async def remove_tenant(self, tenant_id: str) -> bool:
        """Soft-delete: set is_active = false. Returns True if a row was updated."""
        pool = self._require_pool()
        result = await pool.execute(
            """
            UPDATE tenants SET is_active = false, updated_at = NOW()
            WHERE tenant_id = $1 AND is_active = true
            """,
            tenant_id,
        )
        return result == "UPDATE 1"
