#!/usr/bin/env python
"""Insert or update a tenant in the tenants table (for DB-backed multi-tenant)."""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from intent_engine.config import get_settings
from intent_engine.tenancy.db_store import DbTenantStore
from intent_engine.tenancy.models import TenantConfig, TenantTier


async def main() -> None:
    parser = argparse.ArgumentParser(description="Seed a tenant into the tenants table")
    parser.add_argument("--tenant-id", required=True, help="Tenant ID (e.g. acme-corp)")
    parser.add_argument("--name", required=True, help="Display name")
    parser.add_argument("--api-key", required=True, help="API key for this tenant")
    parser.add_argument(
        "--tier",
        choices=[t.value for t in TenantTier],
        default=TenantTier.STARTER.value,
        help="Tier (default: starter)",
    )
    args = parser.parse_args()

    settings = get_settings()
    store = DbTenantStore(database_url=settings.database_url)
    await store.connect()
    try:
        tenant = TenantConfig(
            tenant_id=args.tenant_id,
            name=args.name,
            api_key=args.api_key,
            tier=TenantTier(args.tier),
            websocket_enabled=settings.ws_enabled,
            batch_processing_enabled=True,
        )
        await store.add_tenant(tenant)
        mask = f"...{args.api_key[-4:]}" if len(args.api_key) >= 4 else "***"
        print(f"Tenant {args.tenant_id} upserted (api_key={mask})")
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
