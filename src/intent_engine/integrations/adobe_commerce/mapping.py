"""Status mapping utilities for Adobe Commerce."""

from intent_engine.integrations.base import FulfillmentStatus, OrderStatus

# Map Adobe Commerce order states to unified OrderStatus
ADOBE_COMMERCE_STATUS_MAP: dict[str, OrderStatus] = {
    # Standard states
    "new": OrderStatus.PENDING,
    "pending_payment": OrderStatus.PENDING,
    "payment_review": OrderStatus.PENDING,
    "processing": OrderStatus.PROCESSING,
    "complete": OrderStatus.DELIVERED,
    "closed": OrderStatus.REFUNDED,
    "canceled": OrderStatus.CANCELLED,
    "holded": OrderStatus.PROCESSING,  # On hold, mapped to processing
    # Fraud detection states
    "fraud": OrderStatus.FAILED,
    # Custom states that may exist
    "pending": OrderStatus.PENDING,
    "shipped": OrderStatus.SHIPPED,
}


def map_order_status(state: str, shipment_count: int = 0) -> OrderStatus:
    """
    Map Adobe Commerce order state to unified OrderStatus.

    Args:
        state: Adobe Commerce order state (e.g., "processing", "complete").
        shipment_count: Number of shipments for the order.

    Returns:
        Unified OrderStatus enum value.
    """
    state_lower = state.lower()

    # Check if shipped (processing with shipments)
    if state_lower == "processing" and shipment_count > 0:
        return OrderStatus.SHIPPED

    return ADOBE_COMMERCE_STATUS_MAP.get(state_lower, OrderStatus.PROCESSING)


def map_fulfillment_status(
    total_items: int,
    shipped_items: int,
) -> FulfillmentStatus:
    """
    Determine fulfillment status based on shipped vs total items.

    Args:
        total_items: Total number of items in the order.
        shipped_items: Number of items that have been shipped.

    Returns:
        FulfillmentStatus enum value.
    """
    if shipped_items == 0:
        return FulfillmentStatus.UNFULFILLED
    elif shipped_items >= total_items:
        return FulfillmentStatus.FULFILLED
    else:
        return FulfillmentStatus.PARTIAL


# Map carrier codes to display names
CARRIER_NAMES: dict[str, str] = {
    "ups": "UPS",
    "usps": "USPS",
    "fedex": "FedEx",
    "dhl": "DHL",
    "dhlint": "DHL International",
    "custom": "Custom Carrier",
}


def get_carrier_name(carrier_code: str) -> str:
    """
    Get display name for a carrier code.

    Args:
        carrier_code: Adobe Commerce carrier code.

    Returns:
        Human-readable carrier name.
    """
    return CARRIER_NAMES.get(carrier_code.lower(), carrier_code.upper())


def build_tracking_url(carrier_code: str, tracking_number: str) -> str | None:
    """
    Build tracking URL for common carriers.

    Args:
        carrier_code: Adobe Commerce carrier code.
        tracking_number: The tracking number.

    Returns:
        Tracking URL if carrier is known, None otherwise.
    """
    carrier_lower = carrier_code.lower()

    tracking_urls: dict[str, str] = {
        "ups": f"https://www.ups.com/track?tracknum={tracking_number}",
        "usps": f"https://tools.usps.com/go/TrackConfirmAction?tLabels={tracking_number}",
        "fedex": f"https://www.fedex.com/fedextrack/?trknbr={tracking_number}",
        "dhl": f"https://www.dhl.com/en/express/tracking.html?AWB={tracking_number}",
        "dhlint": f"https://www.dhl.com/en/express/tracking.html?AWB={tracking_number}",
    }

    return tracking_urls.get(carrier_lower)
