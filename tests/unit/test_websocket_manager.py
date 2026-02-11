"""Unit tests for WebSocket connection manager.

Imports are delayed after mocking spacy to avoid import chain issues.
"""
# ruff: noqa: E402

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock spacy before any imports that might trigger it
sys.modules["spacy"] = MagicMock()

from intent_engine.api.ws_models import (
    STEP_DESCRIPTIONS,
    ReasoningStep,
    WSMessage,
    WSMessageType,
)
from intent_engine.tenancy.models import TenantConfig, TenantTier

# Import after mocking to avoid spacy import chain
with patch.dict(sys.modules, {"spacy": MagicMock()}):
    from intent_engine.api.websocket import ConnectionManager, StreamingReasoningCallback


class TestConnectionManager:
    """Tests for ConnectionManager class."""

    @pytest.fixture
    def manager(self):
        """Create a connection manager."""
        return ConnectionManager()

    @pytest.fixture
    def tenant(self):
        """Create a test tenant."""
        return TenantConfig(
            tenant_id="ws-test",
            name="WebSocket Test",
            tier=TenantTier.PROFESSIONAL,
            api_key="ws-key",
            websocket_enabled=True,
        )

    @pytest.mark.asyncio
    async def test_connect(self, manager, tenant):
        """Test accepting a new connection."""
        mock_ws = AsyncMock()

        connection_id = await manager.connect(mock_ws, tenant)

        assert connection_id is not None
        mock_ws.accept.assert_called_once()
        assert manager.get_connection_count("ws-test") == 1

    @pytest.mark.asyncio
    async def test_connect_limit_exceeded(self, manager, tenant):
        """Test connection limit enforcement."""
        # Set a low limit
        tenant.max_websocket_connections = 2

        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        mock_ws3 = AsyncMock()

        await manager.connect(mock_ws1, tenant)
        await manager.connect(mock_ws2, tenant)

        with pytest.raises(Exception, match="Connection limit"):
            await manager.connect(mock_ws3, tenant)

    @pytest.mark.asyncio
    async def test_disconnect(self, manager, tenant):
        """Test disconnecting a connection."""
        mock_ws = AsyncMock()
        connection_id = await manager.connect(mock_ws, tenant)

        assert manager.get_connection_count("ws-test") == 1

        await manager.disconnect(connection_id)

        assert manager.get_connection_count("ws-test") == 0

    @pytest.mark.asyncio
    async def test_send_to_connection(self, manager, tenant):
        """Test sending a message to a connection."""
        mock_ws = AsyncMock()
        connection_id = await manager.connect(mock_ws, tenant)

        message = WSMessage(
            type=WSMessageType.PONG,
            request_id="test-req",
        )

        result = await manager.send_to_connection(connection_id, message)

        assert result is True
        mock_ws.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_to_invalid_connection(self, manager):
        """Test sending to non-existent connection."""
        message = WSMessage(type=WSMessageType.PONG)

        result = await manager.send_to_connection("invalid-id", message)

        assert result is False

    @pytest.mark.asyncio
    async def test_broadcast_to_tenant(self, manager, tenant):
        """Test broadcasting to all tenant connections."""
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()

        await manager.connect(mock_ws1, tenant)
        await manager.connect(mock_ws2, tenant)

        message = WSMessage(
            type=WSMessageType.JOB_UPDATE,
            payload={"job_id": "123", "status": "completed"},
        )

        sent = await manager.broadcast_to_tenant("ws-test", message)

        assert sent == 2
        mock_ws1.send_text.assert_called_once()
        mock_ws2.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_job_subscription(self, manager, tenant):
        """Test subscribing to job updates."""
        mock_ws = AsyncMock()
        connection_id = await manager.connect(mock_ws, tenant)

        await manager.subscribe_to_job(connection_id, "job-123")

        # Notify subscribers
        sent = await manager.notify_job_subscribers(
            "job-123",
            status="completed",
            progress=1.0,
        )

        assert sent == 1
        mock_ws.send_text.assert_called()

    @pytest.mark.asyncio
    async def test_job_unsubscription(self, manager, tenant):
        """Test unsubscribing from job updates."""
        mock_ws = AsyncMock()
        connection_id = await manager.connect(mock_ws, tenant)

        await manager.subscribe_to_job(connection_id, "job-123")
        await manager.unsubscribe_from_job(connection_id, "job-123")

        # Notify should not send anything
        sent = await manager.notify_job_subscribers("job-123", status="completed")

        assert sent == 0

    @pytest.mark.asyncio
    async def test_disconnect_clears_subscriptions(self, manager, tenant):
        """Test that disconnect clears job subscriptions."""
        mock_ws = AsyncMock()
        connection_id = await manager.connect(mock_ws, tenant)

        await manager.subscribe_to_job(connection_id, "job-456")
        await manager.disconnect(connection_id)

        # Notify should not send anything
        sent = await manager.notify_job_subscribers("job-456", status="completed")

        assert sent == 0


class TestStreamingReasoningCallback:
    """Tests for StreamingReasoningCallback."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock connection manager."""
        manager = MagicMock()
        manager.send_to_connection = AsyncMock(return_value=True)
        return manager

    @pytest.mark.asyncio
    async def test_on_step(self, mock_manager):
        """Test sending a reasoning step."""
        callback = StreamingReasoningCallback(
            connection_id="conn-123",
            request_id="req-456",
            manager=mock_manager,
        )

        await callback.on_step(
            ReasoningStep.ENTITY_EXTRACTION,
            duration_ms=50,
            data={"entities": ["order_id"]},
        )

        mock_manager.send_to_connection.assert_called_once()
        call_args = mock_manager.send_to_connection.call_args

        assert call_args[0][0] == "conn-123"
        message = call_args[0][1]
        assert message.type == WSMessageType.REASONING_STEP
        assert message.request_id == "req-456"


class TestWSModels:
    """Tests for WebSocket message models."""

    def test_step_descriptions_complete(self):
        """Test that all reasoning steps have descriptions."""
        for step in ReasoningStep:
            assert step in STEP_DESCRIPTIONS, f"Missing description for {step}"

    def test_ws_message_serialization(self):
        """Test WebSocket message serialization."""
        message = WSMessage(
            type=WSMessageType.RESULT,
            request_id="test-123",
            payload={"intents": [], "confidence": 0.85},
        )

        json_str = message.model_dump_json()
        assert "test-123" in json_str
        assert "result" in json_str

    def test_ws_message_type_values(self):
        """Test WebSocket message type values."""
        assert WSMessageType.RESOLVE.value == "resolve"
        assert WSMessageType.CONNECTED.value == "connected"
        assert WSMessageType.REASONING_STEP.value == "reasoning_step"
        assert WSMessageType.ERROR.value == "error"
