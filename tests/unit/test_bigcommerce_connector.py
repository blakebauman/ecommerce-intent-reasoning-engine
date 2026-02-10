"""Unit tests for BigCommerce connector."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from intent_engine.integrations.bigcommerce.connector import BigCommerceConnector
from intent_engine.integrations.bigcommerce.mapping import (
    map_order_status,
    map_fulfillment_status,
    get_carrier_name,
    get_status_name,
    get_tracking_url,
)
from intent_engine.integrations.base import OrderStatus, FulfillmentStatus


class TestBigCommerceStatusMapping:
    """Tests for BigCommerce status mapping."""

    def test_map_incomplete_status(self):
        """Test mapping incomplete status (0)."""
        assert map_order_status(0) == OrderStatus.PENDING

    def test_map_pending_status(self):
        """Test mapping pending status (1)."""
        assert map_order_status(1) == OrderStatus.PENDING

    def test_map_shipped_status(self):
        """Test mapping shipped status (2)."""
        assert map_order_status(2) == OrderStatus.SHIPPED

    def test_map_partially_shipped_status(self):
        """Test mapping partially shipped status (3)."""
        assert map_order_status(3) == OrderStatus.PROCESSING

    def test_map_refunded_status(self):
        """Test mapping refunded status (4)."""
        assert map_order_status(4) == OrderStatus.REFUNDED

    def test_map_cancelled_status(self):
        """Test mapping cancelled status (5)."""
        assert map_order_status(5) == OrderStatus.CANCELLED

    def test_map_declined_status(self):
        """Test mapping declined status (6)."""
        assert map_order_status(6) == OrderStatus.CANCELLED

    def test_map_completed_status(self):
        """Test mapping completed status (10)."""
        assert map_order_status(10) == OrderStatus.DELIVERED

    def test_map_partially_refunded_status(self):
        """Test mapping partially refunded status (14)."""
        assert map_order_status(14) == OrderStatus.PARTIALLY_REFUNDED

    def test_map_fulfillment_shipped(self):
        """Test fulfillment status for shipped orders."""
        assert map_fulfillment_status(2) == FulfillmentStatus.FULFILLED

    def test_map_fulfillment_completed(self):
        """Test fulfillment status for completed orders."""
        assert map_fulfillment_status(10) == FulfillmentStatus.FULFILLED

    def test_map_fulfillment_partially_shipped(self):
        """Test fulfillment status for partially shipped orders."""
        assert map_fulfillment_status(3, items_shipped=1, total_items=3) == FulfillmentStatus.PARTIAL

    def test_map_fulfillment_pending(self):
        """Test fulfillment status for pending orders."""
        assert map_fulfillment_status(1) == FulfillmentStatus.UNFULFILLED


class TestBigCommerceCarrierMapping:
    """Tests for carrier name and tracking URL mapping."""

    def test_get_carrier_name_ups(self):
        """Test UPS carrier name."""
        assert get_carrier_name("ups") == "UPS"

    def test_get_carrier_name_fedex(self):
        """Test FedEx carrier name."""
        assert get_carrier_name("fedex") == "FedEx"

    def test_get_carrier_name_unknown(self):
        """Test unknown carrier returns original."""
        assert get_carrier_name("custom_carrier") == "custom_carrier"

    def test_get_status_name(self):
        """Test status name lookup."""
        assert get_status_name(0) == "Incomplete"
        assert get_status_name(2) == "Shipped"
        assert get_status_name(10) == "Completed"

    def test_get_status_name_unknown(self):
        """Test unknown status ID."""
        assert "Unknown" in get_status_name(999)

    def test_get_tracking_url_ups(self):
        """Test UPS tracking URL."""
        url = get_tracking_url("ups", "1Z999AA10123456784")
        assert "1Z999AA10123456784" in url
        assert "ups.com" in url

    def test_get_tracking_url_unknown(self):
        """Test unknown carrier returns None."""
        assert get_tracking_url("unknown_carrier", "12345") is None


class TestBigCommerceConnector:
    """Tests for BigCommerceConnector class."""

    @pytest.fixture
    def connector(self):
        """Create a test connector."""
        return BigCommerceConnector(
            store_hash="abc123",
            access_token="test_token",
            return_window_days=30,
        )

    def test_platform_name(self, connector):
        """Test platform name property."""
        assert connector.platform_name == "bigcommerce"

    def test_base_url_v2(self, connector):
        """Test V2 base URL generation."""
        assert connector.base_url_v2 == "https://api.bigcommerce.com/stores/abc123/v2"

    def test_base_url_v3(self, connector):
        """Test V3 base URL generation."""
        assert connector.base_url_v3 == "https://api.bigcommerce.com/stores/abc123/v3"

    @pytest.mark.asyncio
    async def test_get_order(self, connector):
        """Test fetching an order."""
        mock_order = {
            "id": 123,
            "status_id": 9,
            "currency_code": "USD",
            "total_inc_tax": "99.99",
            "billing_address": {
                "email": "test@example.com",
                "first_name": "John",
                "last_name": "Doe",
            },
            "date_created": "Mon, 15 Jan 2024 10:30:00 +0000",
        }

        with patch.object(connector, "_request_v2", new_callable=AsyncMock) as mock_v2:
            mock_v2.side_effect = [
                mock_order,  # Order data
                [],          # Products
                [],          # Shipping addresses
                [],          # Shipments
            ]

            order = await connector.get_order("123")

            assert order is not None
            assert order.order_id == "123"
            assert order.platform == "bigcommerce"
            assert order.customer_email == "test@example.com"

    @pytest.mark.asyncio
    async def test_get_order_not_found(self, connector):
        """Test fetching non-existent order."""
        with patch.object(connector, "_request_v2", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = None
            order = await connector.get_order("999")
            assert order is None

    @pytest.mark.asyncio
    async def test_health_check(self, connector):
        """Test health check endpoint."""
        with patch.object(connector, "_request_v2", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"id": "abc123"}
            result = await connector.health_check()
            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, connector):
        """Test health check when API fails."""
        with patch.object(connector, "_request_v2", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = Exception("Connection failed")
            result = await connector.health_check()
            assert result is False

    def test_parse_datetime_rfc2822(self, connector):
        """Test datetime parsing with RFC 2822 format."""
        dt = connector._parse_datetime("Mon, 15 Jan 2024 10:30:00 +0000")
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15

    def test_parse_datetime_iso(self, connector):
        """Test datetime parsing with ISO format."""
        dt = connector._parse_datetime("2024-01-15T10:30:00Z")
        assert dt is not None
        assert dt.year == 2024

    def test_parse_datetime_none(self, connector):
        """Test datetime parsing with None."""
        assert connector._parse_datetime(None) is None

    @pytest.mark.asyncio
    async def test_get_tracking(self, connector):
        """Test tracking extraction from shipments."""
        mock_shipments = [
            {
                "id": 1,
                "tracking_number": "1Z999AA10123456784",
                "shipping_provider": "ups",
            }
        ]

        with patch.object(connector, "_request_v2", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_shipments
            tracking = await connector.get_tracking("123")

            assert len(tracking) == 1
            assert tracking[0].carrier == "UPS"
            assert tracking[0].tracking_number == "1Z999AA10123456784"
