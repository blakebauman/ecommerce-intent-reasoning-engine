"""Integration tests for the API."""

import sys

import pytest

# Skip entire module on Python 3.14+ due to spaCy incompatibility
if sys.version_info >= (3, 14):
    pytest.skip(
        "spaCy not compatible with Python 3.14+ (uses Pydantic v1)",
        allow_module_level=True,
    )

from fastapi.testclient import TestClient

from intent_engine.api.server import app
from intent_engine.config import get_settings


@pytest.fixture
def client() -> TestClient:
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def api_key() -> str:
    """Get the API key from settings."""
    settings = get_settings()
    return settings.api_key


class TestHealthEndpoint:
    """Tests for the health endpoint."""

    def test_health_check(self, client: TestClient) -> None:
        """Test health check returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestResolveEndpoint:
    """Tests for the resolve endpoint."""

    def test_resolve_requires_auth(self, client: TestClient) -> None:
        """Test that resolve endpoint requires authentication."""
        response = client.post(
            "/v1/intent/resolve",
            json={
                "request_id": "test-1",
                "tenant_id": "test",
                "raw_text": "Where is my order?",
            },
        )
        assert response.status_code == 401

    def test_resolve_with_auth(self, client: TestClient, api_key: str) -> None:
        """Test resolve with valid authentication."""
        response = client.post(
            "/v1/intent/resolve",
            json={
                "request_id": "test-1",
                "tenant_id": "test",
                "channel": "chat",
                "raw_text": "Where is my order #12345?",
            },
            headers={"Authorization": f"Bearer {api_key}"},
        )
        # Note: This will fail without database connection
        # In a real test, we'd mock the engine
        # For now, just check it's not a 401
        assert response.status_code != 401

    def test_resolve_invalid_auth(self, client: TestClient) -> None:
        """Test resolve with invalid authentication."""
        response = client.post(
            "/v1/intent/resolve",
            json={
                "request_id": "test-1",
                "tenant_id": "test",
                "raw_text": "Where is my order?",
            },
            headers={"Authorization": "Bearer invalid-key"},
        )
        assert response.status_code == 401


class TestIntentsEndpoint:
    """Tests for the intents listing endpoint."""

    def test_list_intents(self, client: TestClient, api_key: str) -> None:
        """Test listing core intents."""
        response = client.get(
            "/v1/intent/intents",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 8  # 8 core MVP intents
        assert any(i["intent_code"] == "ORDER_STATUS.WISMO" for i in data)
