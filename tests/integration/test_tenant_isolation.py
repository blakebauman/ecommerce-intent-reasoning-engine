"""Integration tests for tenant isolation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from intent_engine.tenancy.middleware import TenantMiddleware, TenantStore
from intent_engine.tenancy.models import TenantConfig, TenantTier
from intent_engine.tenancy.context import (
    get_current_tenant,
    get_current_tenant_id,
    tenant_context,
    clear_tenant_context,
)
from intent_engine.tenancy.rate_limiter import RateLimiter, RateLimitExceeded


class TestTenantIsolation:
    """Integration tests for tenant isolation."""

    @pytest.fixture
    def tenant_store(self):
        """Create a tenant store with multiple tenants."""
        store = TenantStore()

        store.add_tenant(TenantConfig(
            tenant_id="tenant-a",
            name="Tenant A",
            tier=TenantTier.PROFESSIONAL,
            api_key="key-a",
        ))

        store.add_tenant(TenantConfig(
            tenant_id="tenant-b",
            name="Tenant B",
            tier=TenantTier.STARTER,
            api_key="key-b",
        ))

        return store

    @pytest.fixture
    def app(self, tenant_store):
        """Create a test app with tenant middleware."""
        app = FastAPI()

        @app.get("/whoami")
        async def whoami():
            tenant = get_current_tenant()
            if tenant:
                return {
                    "tenant_id": tenant.tenant_id,
                    "tier": tenant.tier.value,
                }
            return {"tenant_id": None}

        @app.get("/data")
        async def get_data():
            tenant = get_current_tenant()
            if tenant:
                # Simulate tenant-specific data
                return {"data": f"Data for {tenant.tenant_id}"}
            return {"data": None}

        app.add_middleware(
            TenantMiddleware,
            tenant_lookup=tenant_store.get_tenant_by_api_key,
            exclude_paths=["/health"],
        )

        return app

    def test_tenant_a_isolation(self, app):
        """Test that tenant A sees only their data."""
        client = TestClient(app)

        response = client.get(
            "/whoami",
            headers={"Authorization": "Bearer key-a"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["tenant_id"] == "tenant-a"
        assert data["tier"] == "professional"

    def test_tenant_b_isolation(self, app):
        """Test that tenant B sees only their data."""
        client = TestClient(app)

        response = client.get(
            "/whoami",
            headers={"Authorization": "Bearer key-b"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["tenant_id"] == "tenant-b"
        assert data["tier"] == "starter"

    def test_tenant_headers_in_response(self, app):
        """Test that tenant headers are added to response."""
        client = TestClient(app)

        response = client.get(
            "/data",
            headers={"Authorization": "Bearer key-a"},
        )

        assert response.headers.get("X-Tenant-Id") == "tenant-a"
        assert response.headers.get("X-Tenant-Tier") == "professional"

    def test_unauthorized_request(self, app):
        """Test that unauthorized requests are rejected."""
        client = TestClient(app)

        response = client.get("/whoami")

        assert response.status_code == 401

    def test_invalid_api_key(self, app):
        """Test that invalid API keys are rejected."""
        client = TestClient(app)

        response = client.get(
            "/whoami",
            headers={"Authorization": "Bearer invalid-key"},
        )

        assert response.status_code == 401


class TestTenantContextPropagation:
    """Tests for tenant context propagation."""

    def test_context_propagation_in_sync_code(self):
        """Test that tenant context propagates in sync code."""
        tenant = TenantConfig(
            tenant_id="sync-test",
            name="Sync Test",
            tier=TenantTier.STARTER,
            api_key="sync-key",
        )

        assert get_current_tenant() is None

        with tenant_context(tenant):
            assert get_current_tenant() == tenant
            assert get_current_tenant_id() == "sync-test"

        assert get_current_tenant() is None

    @pytest.mark.asyncio
    async def test_context_propagation_in_async_code(self):
        """Test that tenant context propagates in async code."""
        tenant = TenantConfig(
            tenant_id="async-test",
            name="Async Test",
            tier=TenantTier.PROFESSIONAL,
            api_key="async-key",
        )

        async def async_function():
            return get_current_tenant_id()

        with tenant_context(tenant):
            result = await async_function()
            assert result == "async-test"

    @pytest.mark.asyncio
    async def test_context_isolation_across_tasks(self):
        """Test that tenant context is isolated across concurrent tasks."""
        tenant_a = TenantConfig(
            tenant_id="task-a",
            name="Task A",
            tier=TenantTier.STARTER,
            api_key="task-a-key",
        )

        tenant_b = TenantConfig(
            tenant_id="task-b",
            name="Task B",
            tier=TenantTier.PROFESSIONAL,
            api_key="task-b-key",
        )

        async def task_with_tenant(tenant, delay):
            with tenant_context(tenant):
                await asyncio.sleep(delay)
                return get_current_tenant_id()

        # Run tasks concurrently
        results = await asyncio.gather(
            task_with_tenant(tenant_a, 0.01),
            task_with_tenant(tenant_b, 0.01),
        )

        # Each task should see its own tenant
        assert "task-a" in results
        assert "task-b" in results


class TestRateLimitingIntegration:
    """Integration tests for rate limiting with tenants."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis for rate limiting."""
        redis = AsyncMock()
        return redis

    @pytest.mark.asyncio
    async def test_rate_limiting_per_tenant(self, mock_redis):
        """Test that rate limiting is applied per tenant."""
        limiter = RateLimiter(
            redis_client=mock_redis,
            default_rate=100,
            default_burst=10,
        )

        # Allow first request
        mock_redis.eval.return_value = [1, 9.0, 0]

        result = await limiter.check_rate_limit("tenant-1")
        assert result["allowed"] is True

        # Deny when limit exceeded
        mock_redis.eval.return_value = [0, 0.0, 5.0]

        with pytest.raises(RateLimitExceeded) as exc_info:
            await limiter.check_rate_limit("tenant-1")

        assert exc_info.value.tenant_id == "tenant-1"

    @pytest.mark.asyncio
    async def test_different_limits_per_tier(self, mock_redis):
        """Test that different tiers have different limits."""
        limiter = RateLimiter(
            redis_client=mock_redis,
            default_rate=100,
            default_burst=10,
        )

        free_tenant = TenantConfig(
            tenant_id="free-tenant",
            name="Free",
            tier=TenantTier.FREE,
            api_key="free-key",
        )

        enterprise_tenant = TenantConfig(
            tenant_id="enterprise-tenant",
            name="Enterprise",
            tier=TenantTier.ENTERPRISE,
            api_key="enterprise-key",
        )

        # Free tier has lower limits
        assert free_tenant.get_rate_limit() < enterprise_tenant.get_rate_limit()


class TestInactiveTenant:
    """Tests for inactive tenant handling."""

    @pytest.fixture
    def app_with_inactive_tenant(self):
        """Create app with an inactive tenant."""
        store = TenantStore()
        store.add_tenant(TenantConfig(
            tenant_id="inactive",
            name="Inactive",
            tier=TenantTier.STARTER,
            api_key="inactive-key",
            is_active=False,
        ))

        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"ok": True}

        app.add_middleware(
            TenantMiddleware,
            tenant_lookup=store.get_tenant_by_api_key,
        )

        return app

    def test_inactive_tenant_rejected(self, app_with_inactive_tenant):
        """Test that inactive tenants are rejected."""
        client = TestClient(app_with_inactive_tenant)

        response = client.get(
            "/test",
            headers={"Authorization": "Bearer inactive-key"},
        )

        assert response.status_code == 403
        assert "inactive" in response.json()["detail"].lower()
