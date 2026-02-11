"""Integration tests for DB-backed tenant store.

Requires Postgres (e.g. just infra). Creates tenants table if missing via migrate script.
"""

import asyncio
from pathlib import Path

import pytest

try:
    import asyncpg
except ImportError:
    asyncpg = None  # type: ignore[assignment]


async def _can_connect(database_url: str) -> bool:
    try:
        conn = await asyncio.wait_for(asyncpg.connect(database_url), timeout=2.0)
        await conn.close()
        return True
    except Exception:
        return False


def _migration_sql() -> str:
    path = Path(__file__).parent.parent.parent / "scripts" / "migrate_tenants_table.sql"
    return path.read_text()


async def _ensure_tenants_table(database_url: str) -> None:
    """Run migration so tenants table exists."""
    sql = _migration_sql()
    # Split on semicolon-newline; keep segments that contain CREATE
    statements = []
    for raw in sql.split(";\n"):
        stmt = raw.strip()
        if stmt and "CREATE" in stmt.upper():
            if not stmt.endswith(";"):
                stmt += ";"
            statements.append(stmt)
    conn = await asyncpg.connect(database_url)
    try:
        for stmt in statements:
            await conn.execute(stmt)
    finally:
        await conn.close()


@pytest.fixture(scope="module")
def database_url() -> str:
    import os

    return os.environ.get(
        "DATABASE_URL",
        "postgresql://intent_engine:intent_engine_dev@localhost:5432/intent_engine",
    )


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.mark.asyncio
async def test_db_tenant_store_connect_add_lookup(database_url: str) -> None:
    """DbTenantStore: connect, add tenant, get by api_key and by id, list."""
    if asyncpg is None:
        pytest.skip("asyncpg not installed")
    if not await _can_connect(database_url):
        pytest.skip("Postgres not reachable (run just infra)")
    from intent_engine.tenancy.db_store import DbTenantStore
    from intent_engine.tenancy.models import TenantConfig, TenantTier

    await _ensure_tenants_table(database_url)
    store = DbTenantStore(database_url=database_url)
    await store.connect()
    try:
        tenant = TenantConfig(
            tenant_id="int-test-tenant",
            name="Integration Test Tenant",
            api_key="int-test-api-key",
            tier=TenantTier.PROFESSIONAL,
            batch_processing_enabled=True,
        )
        await store.add_tenant(tenant)

        by_key = await store.get_tenant_by_api_key("int-test-api-key")
        assert by_key is not None
        assert by_key.tenant_id == "int-test-tenant"
        assert by_key.name == "Integration Test Tenant"
        assert by_key.tier == TenantTier.PROFESSIONAL

        by_id = await store.get_tenant_by_id("int-test-tenant")
        assert by_id is not None
        assert by_id.api_key == "int-test-api-key"

        tenants = await store.list_tenants()
        ids = [t.tenant_id for t in tenants]
        assert "int-test-tenant" in ids

        # Cleanup: soft-delete so other runs don't see duplicate key
        await store.remove_tenant("int-test-tenant")
        after = await store.get_tenant_by_api_key("int-test-api-key")
        assert after is None
    finally:
        await store.close()
