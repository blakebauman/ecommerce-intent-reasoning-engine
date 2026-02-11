"""Admin API for tenant management (DB-backed store only)."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from intent_engine.config import get_settings
from intent_engine.tenancy.db_store import DbTenantStore
from intent_engine.tenancy.models import TenantConfig, TenantTier

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/admin", tags=["admin"])


class TenantCreateUpdate(BaseModel):
    """Payload for creating or updating a tenant."""

    tenant_id: str
    name: str
    api_key: str
    tier: str = TenantTier.STARTER.value
    is_active: bool = True
    requests_per_minute: int | None = None
    burst_size: int | None = None
    batch_processing_enabled: bool = True
    websocket_enabled: bool = True


async def verify_admin_key(request: Request) -> None:
    """Require Authorization: Bearer <admin_api_key>. Raise 401/403 if missing or invalid."""
    settings = get_settings()
    if not settings.admin_api_key:
        raise HTTPException(status_code=501, detail="Admin API not configured")
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    key = auth[7:].strip()
    if key != settings.admin_api_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")


def _get_db_store() -> DbTenantStore:
    from intent_engine.api.server import get_tenant_store

    store = get_tenant_store()
    if not isinstance(store, DbTenantStore):
        raise HTTPException(
            status_code=501,
            detail="Tenant management requires TENANT_STORE_BACKEND=db",
        )
    return store


@router.get("/tenants", dependencies=[Depends(verify_admin_key)])
async def list_tenants(store: DbTenantStore = Depends(_get_db_store)):
    """List all active tenants (tenant_id, name, tier, is_active). Does not expose API keys."""
    tenants = await store.list_tenants()
    return {
        "tenants": [
            {
                "tenant_id": t.tenant_id,
                "name": t.name,
                "tier": t.tier.value,
                "is_active": t.is_active,
            }
            for t in tenants
        ]
    }


@router.post("/tenants", dependencies=[Depends(verify_admin_key)])
async def create_or_update_tenant(
    body: TenantCreateUpdate,
    store: DbTenantStore = Depends(_get_db_store),
):
    """Create or update a tenant (upsert by tenant_id)."""
    try:
        tier = TenantTier(body.tier)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid tier: {body.tier}")
    tenant = TenantConfig(
        tenant_id=body.tenant_id,
        name=body.name,
        api_key=body.api_key,
        tier=tier,
        is_active=body.is_active,
        requests_per_minute=body.requests_per_minute,
        burst_size=body.burst_size,
        batch_processing_enabled=body.batch_processing_enabled,
        websocket_enabled=body.websocket_enabled,
    )
    await store.add_tenant(tenant)
    return {"tenant_id": body.tenant_id, "status": "upserted"}
