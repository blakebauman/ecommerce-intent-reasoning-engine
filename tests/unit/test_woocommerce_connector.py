"""Unit tests for WooCommerce connector."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from intent_engine.integrations.woocommerce.connector import WooCommerceConnector
from intent_engine.integrations.woocommerce.mapping import (
    map_order_status,
    map_fulfillment_status,
    get_carrier_name,
    get_tracking_url,
)
from intent_engine.integrations.base import OrderStatus, FulfillmentStatus


class TestWooCommerceStatusMapping:
    """Tests for WooCommerce status mapping."""

    def test_map_pending_status(self):
        """Test mapping pending status."""
        assert map_order_status("pending") == OrderStatus.PENDING
        assert map_order_status("wc-pending") == OrderStatus.PENDING

    def test_map_processing_status(self):
        """Test mapping processing status."""
        assert map_order_status("processing") == OrderStatus.PROCESSING

    def test_map_completed_status_without_tracking(self):
        """Test mapping completed status without tracking."""
        assert map_order_status("completed") == OrderStatus.DELIVERED

    def test_map_completed_status_with_tracking(self):
        """Test mapping completed status with tracking."""
        assert map_order_status("completed", has_tracking=True) == OrderStatus.SHIPPED

    def test_map_completed_status_with_delivered(self):
        """Test mapping completed status with delivered tracking."""
        assert map_order_status(
            "completed", has_tracking=True, shipment_status="delivered"
        ) == OrderStatus.DELIVERED

    def test_map_cancelled_status(self):
        """Test mapping cancelled status."""
        assert map_order_status("cancelled") == OrderStatus.CANCELLED

    def test_map_refunded_status(self):
        """Test mapping refunded status."""
        assert map_order_status("refunded") == OrderStatus.REFUNDED

    def test_map_failed_status(self):
        """Test mapping failed status."""
        assert map_order_status("failed") == OrderStatus.FAILED

    def test_map_fulfillment_completed(self):
        """Test fulfillment status for completed orders."""
        assert map_fulfillment_status("completed") == FulfillmentStatus.FULFILLED

    def test_map_fulfillment_processing_with_tracking(self):
        """Test fulfillment status for processing with tracking."""
        assert map_fulfillment_status("processing", has_tracking=True) == FulfillmentStatus.PARTIAL

    def test_map_fulfillment_pending(self):
        """Test fulfillment status for pending orders."""
        assert map_fulfillment_status("pending") == FulfillmentStatus.UNFULFILLED


class TestWooCommerceCarrierMapping:
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

    def test_get_tracking_url_ups(self):
        """Test UPS tracking URL."""
        url = get_tracking_url("ups", "1Z999AA10123456784")
        assert "1Z999AA10123456784" in url
        assert "ups.com" in url

    def test_get_tracking_url_unknown(self):
        """Test unknown carrier returns None."""
        assert get_tracking_url("unknown_carrier", "12345") is None


class TestWooCommerceConnector:
    """Tests for WooCommerceConnector class."""

    @pytest.fixture
    def connector(self):
        """Create a test connector."""
        return WooCommerceConnector(
            store_url="https://test-store.com",
            consumer_key="ck_test",
            consumer_secret="cs_test",
            return_window_days=30,
        )

    def test_platform_name(self, connector):
        """Test platform name property."""
        assert connector.platform_name == "woocommerce"

    def test_base_url(self, connector):
        """Test base URL generation."""
        assert connector.base_url == "https://test-store.com/wp-json/wc/v3"

    @pytest.mark.asyncio
    async def test_get_order(self, connector):
        """Test fetching an order."""
        mock_order = {
            "id": 123,
            "number": "123",
            "status": "processing",
            "currency": "USD",
            "total": "99.99",
            "billing": {
                "email": "test@example.com",
                "first_name": "John",
                "last_name": "Doe",
            },
            "shipping": {},
            "line_items": [],
            "meta_data": [],
            "date_created": "2024-01-15T10:30:00",
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_order
            order = await connector.get_order("123")

            assert order is not None
            assert order.order_id == "123"
            assert order.platform == "woocommerce"
            assert order.customer_email == "test@example.com"

    @pytest.mark.asyncio
    async def test_get_order_not_found(self, connector):
        """Test fetching non-existent order."""
        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = None
            order = await connector.get_order("999")
            assert order is None

    @pytest.mark.asyncio
    async def test_health_check(self, connector):
        """Test health check endpoint."""
        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"status": "ok"}
            result = await connector.health_check()
            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, connector):
        """Test health check when API fails."""
        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = Exception("Connection failed")
            result = await connector.health_check()
            assert result is False

    def test_parse_datetime(self, connector):
        """Test datetime parsing."""
        dt = connector._parse_datetime("2024-01-15T10:30:00")
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15

    def test_parse_datetime_none(self, connector):
        """Test datetime parsing with None."""
        assert connector._parse_datetime(None) is None

    def test_parse_datetime_invalid(self, connector):
        """Test datetime parsing with invalid string."""
        assert connector._parse_datetime("invalid") is None

    def test_extract_tracking(self, connector):
        """Test tracking extraction from meta_data."""
        order_data = {
            "meta_data": [
                {
                    "key": "_wc_shipment_tracking_items",
                    "value": [
                        {
                            "tracking_provider": "ups",
                            "tracking_number": "1Z999AA10123456784",
                            "tracking_link": "https://ups.com/track/1Z999AA10123456784",
                        }
                    ],
                }
            ]
        }

        tracking = connector._extract_tracking(order_data)
        assert len(tracking) == 1
        assert tracking[0].carrier == "UPS"
        assert tracking[0].tracking_number == "1Z999AA10123456784"
