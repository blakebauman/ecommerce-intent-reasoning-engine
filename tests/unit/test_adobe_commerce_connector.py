"""Tests for Adobe Commerce connector and webhook handler."""

import hashlib
import hmac
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from intent_engine.integrations.adobe_commerce import (
    AdobeCommerceConnector,
    AdobeCommerceWebhookHandler,
    IMSOAuthAuth,
    IntegrationTokenAuth,
)
from intent_engine.integrations.adobe_commerce.mapping import (
    build_tracking_url,
    get_carrier_name,
    map_fulfillment_status,
    map_order_status,
)
from intent_engine.integrations.base import FulfillmentStatus, OrderStatus


class TestStatusMapping:
    """Tests for status mapping utilities."""

    def test_map_order_status_new(self) -> None:
        """Test mapping 'new' state to PENDING."""
        assert map_order_status("new") == OrderStatus.PENDING

    def test_map_order_status_processing(self) -> None:
        """Test mapping 'processing' state to PROCESSING."""
        assert map_order_status("processing") == OrderStatus.PROCESSING

    def test_map_order_status_processing_with_shipments(self) -> None:
        """Test mapping 'processing' with shipments to SHIPPED."""
        assert map_order_status("processing", shipment_count=1) == OrderStatus.SHIPPED

    def test_map_order_status_complete(self) -> None:
        """Test mapping 'complete' state to DELIVERED."""
        assert map_order_status("complete") == OrderStatus.DELIVERED

    def test_map_order_status_closed(self) -> None:
        """Test mapping 'closed' state to REFUNDED."""
        assert map_order_status("closed") == OrderStatus.REFUNDED

    def test_map_order_status_canceled(self) -> None:
        """Test mapping 'canceled' state to CANCELLED."""
        assert map_order_status("canceled") == OrderStatus.CANCELLED

    def test_map_order_status_holded(self) -> None:
        """Test mapping 'holded' state to PROCESSING."""
        assert map_order_status("holded") == OrderStatus.PROCESSING

    def test_map_order_status_case_insensitive(self) -> None:
        """Test that status mapping is case insensitive."""
        assert map_order_status("NEW") == OrderStatus.PENDING
        assert map_order_status("Complete") == OrderStatus.DELIVERED

    def test_map_order_status_unknown(self) -> None:
        """Test that unknown states default to PROCESSING."""
        assert map_order_status("unknown_state") == OrderStatus.PROCESSING

    def test_map_fulfillment_status_unfulfilled(self) -> None:
        """Test mapping to UNFULFILLED when nothing shipped."""
        assert map_fulfillment_status(10, 0) == FulfillmentStatus.UNFULFILLED

    def test_map_fulfillment_status_partial(self) -> None:
        """Test mapping to PARTIAL when some items shipped."""
        assert map_fulfillment_status(10, 5) == FulfillmentStatus.PARTIAL

    def test_map_fulfillment_status_fulfilled(self) -> None:
        """Test mapping to FULFILLED when all items shipped."""
        assert map_fulfillment_status(10, 10) == FulfillmentStatus.FULFILLED
        assert map_fulfillment_status(10, 15) == FulfillmentStatus.FULFILLED  # Over-shipped


class TestCarrierMapping:
    """Tests for carrier name and tracking URL utilities."""

    def test_get_carrier_name_known(self) -> None:
        """Test getting display name for known carriers."""
        assert get_carrier_name("ups") == "UPS"
        assert get_carrier_name("usps") == "USPS"
        assert get_carrier_name("fedex") == "FedEx"
        assert get_carrier_name("dhl") == "DHL"

    def test_get_carrier_name_case_insensitive(self) -> None:
        """Test that carrier lookup is case insensitive."""
        assert get_carrier_name("UPS") == "UPS"
        assert get_carrier_name("Fedex") == "FedEx"

    def test_get_carrier_name_unknown(self) -> None:
        """Test that unknown carriers return uppercased code."""
        assert get_carrier_name("custom_carrier") == "CUSTOM_CARRIER"

    def test_build_tracking_url_ups(self) -> None:
        """Test building UPS tracking URL."""
        url = build_tracking_url("ups", "1Z999AA10123456784")
        assert url == "https://www.ups.com/track?tracknum=1Z999AA10123456784"

    def test_build_tracking_url_usps(self) -> None:
        """Test building USPS tracking URL."""
        url = build_tracking_url("usps", "9400111899223033005011")
        assert url == "https://tools.usps.com/go/TrackConfirmAction?tLabels=9400111899223033005011"

    def test_build_tracking_url_fedex(self) -> None:
        """Test building FedEx tracking URL."""
        url = build_tracking_url("fedex", "123456789012")
        assert url == "https://www.fedex.com/fedextrack/?trknbr=123456789012"

    def test_build_tracking_url_unknown(self) -> None:
        """Test that unknown carriers return None."""
        url = build_tracking_url("custom_carrier", "12345")
        assert url is None


class TestIntegrationTokenAuth:
    """Tests for IntegrationTokenAuth strategy."""

    @pytest.mark.asyncio
    async def test_get_auth_headers(self) -> None:
        """Test that correct headers are returned."""
        auth = IntegrationTokenAuth(access_token="test-token-12345")
        headers = await auth.get_auth_headers()

        assert headers["Authorization"] == "Bearer test-token-12345"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_refresh_if_needed_is_noop(self) -> None:
        """Test that refresh is a no-op for integration tokens."""
        auth = IntegrationTokenAuth(access_token="test-token")
        # Should not raise
        await auth.refresh_if_needed()


class TestIMSOAuthAuth:
    """Tests for IMSOAuthAuth strategy."""

    @pytest.mark.asyncio
    async def test_initial_token_fetch(self) -> None:
        """Test that token is fetched on first header request."""
        auth = IMSOAuthAuth(
            client_id="test-client-id",
            client_secret="test-secret",
            org_id="test-org-id",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "fetched-token",
            "expires_in": 86400,
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        # Inject mock client
        auth._http_client = mock_client

        headers = await auth.get_auth_headers()

        assert headers["Authorization"] == "Bearer fetched-token"
        assert headers["x-gw-ims-org-id"] == "test-org-id"
        assert headers["x-api-key"] == "test-client-id"

    @pytest.mark.asyncio
    async def test_token_reuse_when_valid(self) -> None:
        """Test that valid token is reused without refresh."""
        auth = IMSOAuthAuth(
            client_id="test-client-id",
            client_secret="test-secret",
            org_id="test-org-id",
        )

        # Manually set a valid token
        auth._access_token = "cached-token"
        auth._token_expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

        headers = await auth.get_auth_headers()
        assert headers["Authorization"] == "Bearer cached-token"

    @pytest.mark.asyncio
    async def test_custom_scopes(self) -> None:
        """Test that custom scopes are used in token request."""
        auth = IMSOAuthAuth(
            client_id="test-client-id",
            client_secret="test-secret",
            org_id="test-org-id",
            scopes=["commerce.accs", "custom.scope"],
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "token",
            "expires_in": 86400,
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        # Inject mock client
        auth._http_client = mock_client

        await auth.get_auth_headers()

        # Verify scope was passed correctly
        call_args = mock_client.post.call_args
        assert "commerce.accs,custom.scope" in str(call_args)


class TestWebhookSignatureVerification:
    """Tests for webhook signature verification."""

    def test_verify_valid_signature(self) -> None:
        """Test verification of valid HMAC-SHA256 signature."""
        secret = "test-webhook-secret"
        handler = AdobeCommerceWebhookHandler(webhook_secret=secret)

        payload = b'{"event_type": "test", "data": {}}'
        expected_sig = hmac.new(
            secret.encode("utf-8"),
            payload,
            hashlib.sha256,
        ).hexdigest()

        assert handler.verify_signature(payload, expected_sig) is True

    def test_verify_invalid_signature(self) -> None:
        """Test rejection of invalid signature."""
        handler = AdobeCommerceWebhookHandler(webhook_secret="secret")

        payload = b'{"event_type": "test"}'
        invalid_sig = "invalid-signature"

        assert handler.verify_signature(payload, invalid_sig) is False

    def test_verify_empty_signature(self) -> None:
        """Test rejection of empty signature."""
        handler = AdobeCommerceWebhookHandler(webhook_secret="secret")

        payload = b'{"event_type": "test"}'

        assert handler.verify_signature(payload, "") is False


class TestWebhookEventHandling:
    """Tests for webhook event handling."""

    @pytest.mark.asyncio
    async def test_handle_order_event(self) -> None:
        """Test handling of order update event."""
        callback_called = False
        callback_args = None

        async def on_order_update(order_id, increment_id, status, order):
            nonlocal callback_called, callback_args
            callback_called = True
            callback_args = (order_id, increment_id, status, order)

        handler = AdobeCommerceWebhookHandler(
            webhook_secret="secret",
            on_order_update=on_order_update,
        )

        event = {
            "event_type": "observer.sales_order_save_after",
            "data": {
                "order": {
                    "entity_id": "12345",
                    "increment_id": "000000001",
                    "state": "processing",
                },
            },
        }

        result = await handler.handle_event(event)

        assert result["status"] == "processed"
        assert result["event"] == "order_update"
        assert result["order_id"] == "12345"
        assert result["order_number"] == "000000001"
        assert callback_called is True
        assert callback_args[0] == "12345"

    @pytest.mark.asyncio
    async def test_handle_shipment_event(self) -> None:
        """Test handling of shipment event."""
        handler = AdobeCommerceWebhookHandler(webhook_secret="secret")

        event = {
            "event_type": "observer.sales_order_shipment_save_after",
            "data": {
                "shipment": {
                    "entity_id": "1",
                    "order_id": "12345",
                    "tracks": [
                        {
                            "carrier_code": "ups",
                            "track_number": "1Z999AA10123456784",
                            "title": "UPS Ground",
                        },
                    ],
                },
            },
        }

        result = await handler.handle_event(event)

        assert result["status"] == "processed"
        assert result["event"] == "shipment"
        assert result["order_id"] == "12345"
        assert len(result["tracking"]) == 1
        assert result["tracking"][0]["carrier"] == "UPS"

    @pytest.mark.asyncio
    async def test_handle_cancel_event(self) -> None:
        """Test handling of cancellation event."""
        handler = AdobeCommerceWebhookHandler(webhook_secret="secret")

        event = {
            "event_type": "plugin.magento.sales.api.order_management.cancel",
            "data": {
                "order_id": "12345",
            },
        }

        result = await handler.handle_event(event)

        assert result["status"] == "processed"
        assert result["event"] == "cancellation"
        assert result["order_id"] == "12345"
        assert result["order_status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_handle_refund_event(self) -> None:
        """Test handling of refund event."""
        handler = AdobeCommerceWebhookHandler(webhook_secret="secret")

        event = {
            "event_type": "observer.sales_order_creditmemo_save_after",
            "data": {
                "creditmemo": {
                    "order_id": "12345",
                    "grand_total": 99.99,
                },
            },
        }

        result = await handler.handle_event(event)

        assert result["status"] == "processed"
        assert result["event"] == "refund"
        assert result["order_id"] == "12345"
        assert result["refund_amount"] == 99.99

    @pytest.mark.asyncio
    async def test_handle_unknown_event(self) -> None:
        """Test handling of unknown event type."""
        handler = AdobeCommerceWebhookHandler(webhook_secret="secret")

        event = {
            "event_type": "unknown.event.type",
            "data": {},
        }

        result = await handler.handle_event(event)

        assert result["status"] == "ignored"
        assert "Unknown event type" in result["reason"]


class TestConnectorOrderParsing:
    """Tests for order parsing in the connector."""

    def test_parse_shipping_address(self) -> None:
        """Test parsing shipping address from extension_attributes."""
        connector = AdobeCommerceConnector(
            base_url="https://test.com",
            auth_strategy=IntegrationTokenAuth("token"),
        )

        order_data = {
            "entity_id": "1",
            "extension_attributes": {
                "shipping_assignments": [
                    {
                        "shipping": {
                            "address": {
                                "firstname": "John",
                                "lastname": "Doe",
                                "street": ["123 Main St", "Apt 4"],
                                "city": "New York",
                                "region": "NY",
                                "country_id": "US",
                                "postcode": "10001",
                                "telephone": "555-1234",
                            },
                        },
                    },
                ],
            },
        }

        address = connector._parse_shipping_address(order_data)

        assert address is not None
        assert address.name == "John Doe"
        assert address.address1 == "123 Main St"
        assert address.address2 == "Apt 4"
        assert address.city == "New York"
        assert address.province == "NY"
        assert address.country == "US"
        assert address.zip_code == "10001"
        assert address.phone == "555-1234"

    def test_parse_shipping_address_missing(self) -> None:
        """Test parsing when shipping address is missing."""
        connector = AdobeCommerceConnector(
            base_url="https://test.com",
            auth_strategy=IntegrationTokenAuth("token"),
        )

        order_data = {"entity_id": "1"}
        address = connector._parse_shipping_address(order_data)
        assert address is None

    def test_parse_datetime(self) -> None:
        """Test parsing Adobe Commerce datetime strings."""
        connector = AdobeCommerceConnector(
            base_url="https://test.com",
            auth_strategy=IntegrationTokenAuth("token"),
        )

        # Standard format
        dt = connector._parse_datetime("2024-01-15 10:30:00")
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15

        # None input
        assert connector._parse_datetime(None) is None

        # Invalid format
        assert connector._parse_datetime("invalid") is None

    def test_build_search_criteria(self) -> None:
        """Test building searchCriteria query params."""
        connector = AdobeCommerceConnector(
            base_url="https://test.com",
            auth_strategy=IntegrationTokenAuth("token"),
        )

        params = connector._build_search_criteria(
            filters=[
                {"field": "customer_email", "value": "test@example.com", "condition_type": "eq"},
                {"field": "status", "value": "processing"},
            ],
            page_size=20,
            current_page=2,
            sort_field="created_at",
            sort_direction="DESC",
        )

        assert params["searchCriteria[pageSize]"] == "20"
        assert params["searchCriteria[currentPage]"] == "2"
        assert params["searchCriteria[filterGroups][0][filters][0][field]"] == "customer_email"
        assert params["searchCriteria[filterGroups][0][filters][0][value]"] == "test@example.com"
        assert params["searchCriteria[filterGroups][0][filters][0][conditionType]"] == "eq"
        assert params["searchCriteria[filterGroups][1][filters][0][field]"] == "status"
        assert params["searchCriteria[sortOrders][0][field]"] == "created_at"
        assert params["searchCriteria[sortOrders][0][direction]"] == "DESC"


class TestConnectorOrderInfo:
    """Tests for full order parsing."""

    def test_parse_order_full(self) -> None:
        """Test parsing a complete order response."""
        connector = AdobeCommerceConnector(
            base_url="https://test.com",
            auth_strategy=IntegrationTokenAuth("token"),
            return_window_days=30,
        )

        order_data = {
            "entity_id": "12345",
            "increment_id": "000000001",
            "state": "processing",
            "customer_email": "customer@example.com",
            "customer_firstname": "John",
            "customer_lastname": "Doe",
            "subtotal": 100.00,
            "shipping_amount": 10.00,
            "tax_amount": 8.00,
            "grand_total": 118.00,
            "order_currency_code": "USD",
            "created_at": "2024-01-15 10:00:00",
            "updated_at": "2024-01-16 12:00:00",
            "items": [
                {
                    "product_id": "101",
                    "sku": "SKU-101",
                    "name": "Test Product",
                    "qty_ordered": 2,
                    "qty_shipped": 0,
                    "price": 50.00,
                },
            ],
            "extension_attributes": {
                "shipping_assignments": [
                    {
                        "shipping": {
                            "address": {
                                "firstname": "John",
                                "lastname": "Doe",
                                "street": ["123 Main St"],
                                "city": "New York",
                                "region": "NY",
                                "country_id": "US",
                                "postcode": "10001",
                            },
                        },
                    },
                ],
            },
        }

        order = connector._parse_order(order_data, shipments=[])

        assert order.order_id == "12345"
        assert order.order_number == "000000001"
        assert order.platform == "adobe_commerce"
        assert order.status == OrderStatus.PROCESSING
        assert order.fulfillment_status == FulfillmentStatus.UNFULFILLED
        assert order.customer_email == "customer@example.com"
        assert order.customer_name == "John Doe"
        assert order.subtotal == 100.00
        assert order.shipping_cost == 10.00
        assert order.tax == 8.00
        assert order.total == 118.00
        assert order.currency == "USD"
        assert len(order.line_items) == 1
        assert order.line_items[0].sku == "SKU-101"
        assert order.shipping_address is not None
        assert order.shipping_address.city == "New York"

    def test_parse_order_with_shipments(self) -> None:
        """Test parsing order with shipment data."""
        connector = AdobeCommerceConnector(
            base_url="https://test.com",
            auth_strategy=IntegrationTokenAuth("token"),
        )

        order_data = {
            "entity_id": "12345",
            "increment_id": "000000001",
            "state": "processing",
            "customer_email": "customer@example.com",
            "items": [
                {"product_id": "101", "name": "Test", "qty_ordered": 1, "qty_shipped": 1},
            ],
        }

        shipments = [
            {
                "entity_id": "1",
                "tracks": [
                    {
                        "carrier_code": "ups",
                        "track_number": "1Z999AA10123456784",
                    },
                ],
            },
        ]

        order = connector._parse_order(order_data, shipments=shipments)

        assert order.status == OrderStatus.SHIPPED
        assert order.fulfillment_status == FulfillmentStatus.FULFILLED
        assert len(order.tracking) == 1
        assert order.tracking[0].carrier == "UPS"
        assert order.tracking[0].tracking_number == "1Z999AA10123456784"

    def test_parse_order_with_refund(self) -> None:
        """Test parsing order with refund amount."""
        connector = AdobeCommerceConnector(
            base_url="https://test.com",
            auth_strategy=IntegrationTokenAuth("token"),
        )

        order_data = {
            "entity_id": "12345",
            "increment_id": "000000001",
            "state": "closed",
            "customer_email": "customer@example.com",
            "total_refunded": 50.00,
            "items": [],
        }

        order = connector._parse_order(order_data)

        assert order.status == OrderStatus.REFUNDED
        assert order.refund_amount == 50.00
