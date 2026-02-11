"""Status and datetime mapping for Shopify orders."""

from datetime import datetime
from typing import Any

from intent_engine.integrations.base import FulfillmentStatus, OrderStatus


def map_order_status(order_data: dict[str, Any]) -> OrderStatus:
    """Map Shopify order status to OrderStatus enum."""
    financial_status = order_data.get("financial_status", "")
    fulfillment_status = order_data.get("fulfillment_status")
    cancelled_at = order_data.get("cancelled_at")

    if cancelled_at:
        return OrderStatus.CANCELLED

    if financial_status == "refunded":
        return OrderStatus.REFUNDED

    if financial_status == "partially_refunded":
        return OrderStatus.PARTIALLY_REFUNDED

    if fulfillment_status == "fulfilled":
        for f in order_data.get("fulfillments", []):
            shipment_status = f.get("shipment_status")
            if shipment_status == "delivered":
                return OrderStatus.DELIVERED
            if shipment_status == "out_for_delivery":
                return OrderStatus.OUT_FOR_DELIVERY
            if shipment_status == "in_transit":
                return OrderStatus.IN_TRANSIT
        return OrderStatus.SHIPPED

    if fulfillment_status == "partial":
        return OrderStatus.PROCESSING

    if financial_status == "paid":
        return OrderStatus.CONFIRMED

    if financial_status == "pending":
        return OrderStatus.PENDING

    return OrderStatus.PROCESSING


def map_fulfillment_status(order_data: dict[str, Any]) -> FulfillmentStatus:
    """Map Shopify fulfillment status."""
    status = order_data.get("fulfillment_status")
    if status == "fulfilled":
        return FulfillmentStatus.FULFILLED
    if status == "partial":
        return FulfillmentStatus.PARTIAL
    return FulfillmentStatus.UNFULFILLED


def parse_datetime(dt_str: str | None) -> datetime | None:
    """Parse Shopify datetime string (ISO 8601)."""
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except ValueError:
        return None
