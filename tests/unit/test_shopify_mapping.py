"""Unit tests for Shopify mapping module."""

from datetime import timezone

from intent_engine.integrations.base import FulfillmentStatus, OrderStatus
from intent_engine.integrations.shopify.mapping import (
    map_fulfillment_status,
    map_order_status,
    parse_datetime,
)


class TestShopifyOrderStatusMapping:
    """Tests for map_order_status."""

    def test_cancelled(self):
        """cancelled_at set returns CANCELLED."""
        assert (
            map_order_status({"cancelled_at": "2024-01-15T12:00:00Z"})
            == OrderStatus.CANCELLED
        )
        assert (
            map_order_status(
                {
                    "financial_status": "paid",
                    "fulfillment_status": "fulfilled",
                    "cancelled_at": "2024-01-15T12:00:00Z",
                }
            )
            == OrderStatus.CANCELLED
        )

    def test_refunded(self):
        """financial_status refunded returns REFUNDED."""
        assert (
            map_order_status({"financial_status": "refunded"})
            == OrderStatus.REFUNDED
        )

    def test_partially_refunded(self):
        """financial_status partially_refunded returns PARTIALLY_REFUNDED."""
        assert (
            map_order_status({"financial_status": "partially_refunded"})
            == OrderStatus.PARTIALLY_REFUNDED
        )

    def test_fulfilled_no_fulfillments(self):
        """fulfillment_status fulfilled with no fulfillments -> SHIPPED."""
        assert (
            map_order_status(
                {"financial_status": "paid", "fulfillment_status": "fulfilled"}
            )
            == OrderStatus.SHIPPED
        )

    def test_fulfilled_no_shipment_status(self):
        """fulfilled with fulfillments but no shipment_status -> SHIPPED."""
        assert (
            map_order_status(
                {
                    "financial_status": "paid",
                    "fulfillment_status": "fulfilled",
                    "fulfillments": [{"tracking_number": "1Z123"}],
                }
            )
            == OrderStatus.SHIPPED
        )

    def test_fulfilled_delivered(self):
        """fulfilled with shipment_status delivered -> DELIVERED."""
        assert (
            map_order_status(
                {
                    "financial_status": "paid",
                    "fulfillment_status": "fulfilled",
                    "fulfillments": [{"shipment_status": "delivered"}],
                }
            )
            == OrderStatus.DELIVERED
        )

    def test_fulfilled_out_for_delivery(self):
        """fulfilled with shipment_status out_for_delivery -> OUT_FOR_DELIVERY."""
        assert (
            map_order_status(
                {
                    "financial_status": "paid",
                    "fulfillment_status": "fulfilled",
                    "fulfillments": [{"shipment_status": "out_for_delivery"}],
                }
            )
            == OrderStatus.OUT_FOR_DELIVERY
        )

    def test_fulfilled_in_transit(self):
        """fulfilled with shipment_status in_transit -> IN_TRANSIT."""
        assert (
            map_order_status(
                {
                    "financial_status": "paid",
                    "fulfillment_status": "fulfilled",
                    "fulfillments": [{"shipment_status": "in_transit"}],
                }
            )
            == OrderStatus.IN_TRANSIT
        )

    def test_fulfilled_first_fulfillment_wins(self):
        """First fulfillment with a shipment_status is used."""
        assert (
            map_order_status(
                {
                    "financial_status": "paid",
                    "fulfillment_status": "fulfilled",
                    "fulfillments": [
                        {"shipment_status": "in_transit"},
                        {"shipment_status": "delivered"},
                    ],
                }
            )
            == OrderStatus.IN_TRANSIT
        )

    def test_partial_fulfillment(self):
        """fulfillment_status partial -> PROCESSING."""
        assert (
            map_order_status(
                {"financial_status": "paid", "fulfillment_status": "partial"}
            )
            == OrderStatus.PROCESSING
        )

    def test_paid(self):
        """financial_status paid (and not fulfilled) -> CONFIRMED."""
        assert (
            map_order_status({"financial_status": "paid"})
            == OrderStatus.CONFIRMED
        )

    def test_pending(self):
        """financial_status pending -> PENDING."""
        assert (
            map_order_status({"financial_status": "pending"})
            == OrderStatus.PENDING
        )

    def test_empty_defaults_to_processing(self):
        """Missing status fields default to PROCESSING."""
        assert map_order_status({}) == OrderStatus.PROCESSING
        assert map_order_status({"financial_status": "unknown"}) == OrderStatus.PROCESSING


class TestShopifyFulfillmentStatusMapping:
    """Tests for map_fulfillment_status."""

    def test_fulfilled(self):
        """fulfillment_status fulfilled -> FULFILLED."""
        assert (
            map_fulfillment_status({"fulfillment_status": "fulfilled"})
            == FulfillmentStatus.FULFILLED
        )

    def test_partial(self):
        """fulfillment_status partial -> PARTIAL."""
        assert (
            map_fulfillment_status({"fulfillment_status": "partial"})
            == FulfillmentStatus.PARTIAL
        )

    def test_unfulfilled(self):
        """Missing or other fulfillment_status -> UNFULFILLED."""
        assert (
            map_fulfillment_status({}) == FulfillmentStatus.UNFULFILLED
        )
        assert (
            map_fulfillment_status({"fulfillment_status": None})
            == FulfillmentStatus.UNFULFILLED
        )
        assert (
            map_fulfillment_status({"fulfillment_status": "unknown"})
            == FulfillmentStatus.UNFULFILLED
        )


class TestShopifyParseDatetime:
    """Tests for parse_datetime."""

    def test_valid_iso_with_z(self):
        """ISO string with Z is parsed and normalized to UTC."""
        result = parse_datetime("2024-01-15T12:30:00Z")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 12
        assert result.minute == 30
        assert result.tzinfo == timezone.utc

    def test_valid_iso_with_offset(self):
        """ISO string with +00:00 is parsed."""
        result = parse_datetime("2024-01-15T12:30:00+00:00")
        assert result is not None
        assert result.year == 2024
        assert result.tzinfo is not None

    def test_none_returns_none(self):
        """None input returns None."""
        assert parse_datetime(None) is None

    def test_empty_string_returns_none(self):
        """Empty string returns None."""
        assert parse_datetime("") is None

    def test_invalid_returns_none(self):
        """Invalid datetime string returns None."""
        assert parse_datetime("not-a-date") is None
        assert parse_datetime("2024-13-45T99:99:99Z") is None
