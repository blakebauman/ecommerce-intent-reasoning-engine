"""Unit tests for admin API (tenant management)."""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "spaCy/import chain not compatible with Python 3.14+",
        allow_module_level=True,
    )

from fastapi import FastAPI
from fastapi.testclient import TestClient

from intent_engine.api.admin_routes import (
    TenantCreateUpdate,
    create_or_update_tenant,
    list_tenants,
    router as admin_router,
    verify_admin_key,
)


@pytest.fixture
def mock_db_store():
    """Mock DbTenantStore with async list_tenants and add_tenant."""
    store = MagicMock()
    store.list_tenants = AsyncMock(return_value=[])
    store.add_tenant = AsyncMock(return_value=None)
    return store


@pytest.fixture
def app_with_admin(mock_db_store):
    """FastAPI app with admin router and overrides for auth and store."""
    app = FastAPI()
    app.include_router(admin_router)

    async def override_verify(request=None):
        # Bypass admin key check in tests (real app would check settings.admin_api_key)
        pass

    def override_get_store():
        return mock_db_store

    app.dependency_overrides[verify_admin_key] = override_verify
    from intent_engine.api.admin_routes import _get_db_store

    app.dependency_overrides[_get_db_store] = override_get_store
    return app


@pytest.fixture
def client(app_with_admin):
    return TestClient(app_with_admin)


class TestListTenants:
    """Tests for GET /v1/admin/tenants."""

    def test_list_tenants_empty(self, client: TestClient, mock_db_store):
        mock_db_store.list_tenants.return_value = []
        response = client.get(
            "/v1/admin/tenants",
            headers={"Authorization": "Bearer test-admin-key"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "tenants" in data
        assert data["tenants"] == []

    def test_list_tenants_returns_tenant_info(self, client: TestClient, mock_db_store):
        from intent_engine.tenancy.models import TenantConfig, TenantTier

        mock_db_store.list_tenants.return_value = [
            TenantConfig(
                tenant_id="acme",
                name="Acme Corp",
                api_key="secret",
                tier=TenantTier.PROFESSIONAL,
                is_active=True,
            ),
        ]
        response = client.get(
            "/v1/admin/tenants",
            headers={"Authorization": "Bearer test-admin-key"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["tenants"]) == 1
        assert data["tenants"][0]["tenant_id"] == "acme"
        assert data["tenants"][0]["name"] == "Acme Corp"
        assert data["tenants"][0]["tier"] == "professional"
        assert data["tenants"][0]["is_active"] is True
        assert "api_key" not in data["tenants"][0]


class TestCreateOrUpdateTenant:
    """Tests for POST /v1/admin/tenants."""

    def test_create_tenant_returns_upserted(self, client: TestClient, mock_db_store):
        response = client.post(
            "/v1/admin/tenants",
            headers={"Authorization": "Bearer test-admin-key"},
            json={
                "tenant_id": "new-tenant",
                "name": "New Tenant",
                "api_key": "ak_xxx",
                "tier": "starter",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["tenant_id"] == "new-tenant"
        assert data["status"] == "upserted"
        mock_db_store.add_tenant.assert_called_once()
        call_args = mock_db_store.add_tenant.call_args[0][0]
        assert call_args.tenant_id == "new-tenant"
        assert call_args.name == "New Tenant"
        assert call_args.api_key == "ak_xxx"
        assert call_args.tier.value == "starter"

    def test_create_tenant_invalid_tier_returns_400(
        self, client: TestClient, mock_db_store
    ):
        response = client.post(
            "/v1/admin/tenants",
            headers={"Authorization": "Bearer test-admin-key"},
            json={
                "tenant_id": "t",
                "name": "T",
                "api_key": "k",
                "tier": "invalid_tier",
            },
        )
        assert response.status_code == 400
        assert "Invalid tier" in response.json()["detail"]


class TestTenantCreateUpdateModel:
    """Tests for TenantCreateUpdate schema."""

    def test_defaults(self):
        body = TenantCreateUpdate(
            tenant_id="id",
            name="Name",
            api_key="key",
        )
        assert body.tier == "starter"
        assert body.is_active is True
        assert body.batch_processing_enabled is True
        assert body.websocket_enabled is True
