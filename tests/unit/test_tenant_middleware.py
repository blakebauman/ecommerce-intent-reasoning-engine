"""Unit tests for tenant middleware."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from intent_engine.tenancy.context import (
    clear_tenant_context,
    get_current_tenant,
    get_current_tenant_id,
    set_tenant_context,
    tenant_context,
)
from intent_engine.tenancy.middleware import TenantMiddleware, TenantStore
from intent_engine.tenancy.models import TenantConfig, TenantTier


class TestTenantContext:
    """Tests for tenant context management."""

    def test_get_current_tenant_default(self):
        """Test getting tenant when not set."""
        clear_tenant_context()
        assert get_current_tenant() is None
        assert get_current_tenant_id() is None

    def test_set_and_get_tenant(self):
        """Test setting and getting tenant."""
        tenant = TenantConfig(
            tenant_id="test-tenant",
            name="Test",
            tier=TenantTier.PROFESSIONAL,
            api_key="test-key",
        )
        set_tenant_context(tenant)

        assert get_current_tenant() == tenant
        assert get_current_tenant_id() == "test-tenant"

        clear_tenant_context()

    def test_tenant_context_manager(self):
        """Test tenant context manager."""
        tenant = TenantConfig(
            tenant_id="ctx-tenant",
            name="Context Test",
            tier=TenantTier.STARTER,
            api_key="ctx-key",
        )

        assert get_current_tenant() is None

        with tenant_context(tenant):
            assert get_current_tenant() == tenant
            assert get_current_tenant_id() == "ctx-tenant"

        # Context should be cleared after
        assert get_current_tenant() is None


class TestTenantStore:
    """Tests for TenantStore."""

    def test_add_and_get_by_api_key(self):
        """Test adding and retrieving tenant by API key."""
        store = TenantStore()
        tenant = TenantConfig(
            tenant_id="store-test",
            name="Store Test",
            tier=TenantTier.PROFESSIONAL,
            api_key="store-api-key",
        )

        store.add_tenant(tenant)
        retrieved = store.get_tenant_by_api_key("store-api-key")

        assert retrieved == tenant

    def test_get_by_api_key_not_found(self):
        """Test retrieving non-existent tenant."""
        store = TenantStore()
        assert store.get_tenant_by_api_key("nonexistent") is None

    def test_get_by_id(self):
        """Test retrieving tenant by ID."""
        store = TenantStore()
        tenant = TenantConfig(
            tenant_id="id-test",
            name="ID Test",
            tier=TenantTier.STARTER,
            api_key="id-api-key",
        )

        store.add_tenant(tenant)
        retrieved = store.get_tenant_by_id("id-test")

        assert retrieved == tenant

    def test_remove_tenant(self):
        """Test removing a tenant."""
        store = TenantStore()
        tenant = TenantConfig(
            tenant_id="remove-test",
            name="Remove Test",
            tier=TenantTier.FREE,
            api_key="remove-key",
        )

        store.add_tenant(tenant)
        assert store.get_tenant_by_id("remove-test") is not None

        result = store.remove_tenant("remove-test")
        assert result is True
        assert store.get_tenant_by_id("remove-test") is None

    def test_list_tenants(self):
        """Test listing all tenants."""
        store = TenantStore()
        tenant1 = TenantConfig(
            tenant_id="list-test-1",
            name="List Test 1",
            tier=TenantTier.STARTER,
            api_key="list-key-1",
        )
        tenant2 = TenantConfig(
            tenant_id="list-test-2",
            name="List Test 2",
            tier=TenantTier.PROFESSIONAL,
            api_key="list-key-2",
        )

        store.add_tenant(tenant1)
        store.add_tenant(tenant2)

        tenants = store.list_tenants()
        assert len(tenants) == 2


class TestTenantConfig:
    """Tests for TenantConfig model."""

    def test_default_tier_values(self):
        """Test default tier rate limits."""
        free_tenant = TenantConfig(
            tenant_id="free",
            name="Free",
            tier=TenantTier.FREE,
            api_key="free-key",
        )
        enterprise_tenant = TenantConfig(
            tenant_id="enterprise",
            name="Enterprise",
            tier=TenantTier.ENTERPRISE,
            api_key="enterprise-key",
        )

        # Free tier should have lower limits
        assert free_tenant.get_rate_limit() < enterprise_tenant.get_rate_limit()
        assert free_tenant.get_burst_size() < enterprise_tenant.get_burst_size()
        assert free_tenant.get_max_batch_size() < enterprise_tenant.get_max_batch_size()

    def test_custom_rate_limit_override(self):
        """Test custom rate limit overrides tier defaults."""
        tenant = TenantConfig(
            tenant_id="custom",
            name="Custom",
            tier=TenantTier.FREE,
            api_key="custom-key",
            requests_per_minute=500,  # Override
        )

        assert tenant.get_rate_limit() == 500

    def test_active_flag(self):
        """Test is_active flag."""
        active_tenant = TenantConfig(
            tenant_id="active",
            name="Active",
            tier=TenantTier.STARTER,
            api_key="active-key",
            is_active=True,
        )
        inactive_tenant = TenantConfig(
            tenant_id="inactive",
            name="Inactive",
            tier=TenantTier.STARTER,
            api_key="inactive-key",
            is_active=False,
        )

        assert active_tenant.is_active is True
        assert inactive_tenant.is_active is False


class TestTenantMiddleware:
    """Tests for TenantMiddleware."""

    @pytest.fixture
    def app_with_middleware(self):
        """Create a test app with tenant middleware."""
        app = FastAPI()

        store = TenantStore()
        tenant = TenantConfig(
            tenant_id="middleware-test",
            name="Middleware Test",
            tier=TenantTier.PROFESSIONAL,
            api_key="valid-api-key",
        )
        store.add_tenant(tenant)

        @app.get("/test")
        async def test_endpoint():
            tenant = get_current_tenant()
            return {"tenant_id": tenant.tenant_id if tenant else None}

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        app.add_middleware(
            TenantMiddleware,
            tenant_lookup=store.get_tenant_by_api_key,
            exclude_paths=["/health"],
        )

        return app

    def test_excluded_path_no_auth(self, app_with_middleware):
        """Test excluded paths don't require authentication."""
        client = TestClient(app_with_middleware)
        response = client.get("/health")
        assert response.status_code == 200

    def test_missing_api_key(self, app_with_middleware):
        """Test request without API key is rejected."""
        client = TestClient(app_with_middleware)
        response = client.get("/test")
        assert response.status_code == 401

    def test_invalid_api_key(self, app_with_middleware):
        """Test request with invalid API key is rejected."""
        client = TestClient(app_with_middleware)
        response = client.get("/test", headers={"Authorization": "Bearer invalid-key"})
        assert response.status_code == 401

    def test_valid_api_key(self, app_with_middleware):
        """Test request with valid API key succeeds."""
        client = TestClient(app_with_middleware)
        response = client.get("/test", headers={"Authorization": "Bearer valid-api-key"})
        assert response.status_code == 200
        # Note: TestClient is synchronous so context may not propagate

    def test_x_api_key_header(self, app_with_middleware):
        """Test X-API-Key header authentication."""
        client = TestClient(app_with_middleware)
        response = client.get("/test", headers={"X-API-Key": "valid-api-key"})
        assert response.status_code == 200
