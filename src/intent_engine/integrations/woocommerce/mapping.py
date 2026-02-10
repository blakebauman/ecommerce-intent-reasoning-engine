"""Status mapping for WooCommerce orders."""

from intent_engine.integrations.base import FulfillmentStatus, OrderStatus

# WooCommerce uses string-based statuses
# Core statuses: pending, processing, on-hold, completed, cancelled, refunded, failed
# Custom statuses may be prefixed with "wc-" in the database

WOOCOMMERCE_STATUS_MAP: dict[str, OrderStatus] = {
    # Payment pending
    "pending": OrderStatus.PENDING,
    "wc-pending": OrderStatus.PENDING,
    # Payment received, awaiting fulfillment
    "processing": OrderStatus.PROCESSING,
    "wc-processing": OrderStatus.PROCESSING,
    # Awaiting action (stock, payment confirmation, etc.)
    "on-hold": OrderStatus.PENDING,
    "wc-on-hold": OrderStatus.PENDING,
    # Order fulfilled and complete
    "completed": OrderStatus.DELIVERED,
    "wc-completed": OrderStatus.DELIVERED,
    # Cancelled by admin or customer
    "cancelled": OrderStatus.CANCELLED,
    "wc-cancelled": OrderStatus.CANCELLED,
    # Fully refunded
    "refunded": OrderStatus.REFUNDED,
    "wc-refunded": OrderStatus.REFUNDED,
    # Payment failed or declined
    "failed": OrderStatus.FAILED,
    "wc-failed": OrderStatus.FAILED,
    # Draft order (Checkout block)
    "checkout-draft": OrderStatus.PENDING,
    "wc-checkout-draft": OrderStatus.PENDING,
}

# Common carrier codes in WooCommerce Shipment Tracking plugin
CARRIER_MAP: dict[str, str] = {
    "ups": "UPS",
    "usps": "USPS",
    "fedex": "FedEx",
    "dhl": "DHL",
    "dhl_express": "DHL Express",
    "royal_mail": "Royal Mail",
    "canada_post": "Canada Post",
    "australia_post": "Australia Post",
    "dpd": "DPD",
    "gls": "GLS",
    "ontrac": "OnTrac",
    "lasership": "LaserShip",
    "amazon": "Amazon Logistics",
    "custom": "Custom",
}

# Tracking URL templates for common carriers
TRACKING_URL_TEMPLATES: dict[str, str] = {
    "ups": "https://www.ups.com/track?tracknum={tracking_number}",
    "usps": "https://tools.usps.com/go/TrackConfirmAction?tLabels={tracking_number}",
    "fedex": "https://www.fedex.com/fedextrack/?trknbr={tracking_number}",
    "dhl": "https://www.dhl.com/en/express/tracking.html?AWB={tracking_number}",
    "dhl_express": "https://www.dhl.com/en/express/tracking.html?AWB={tracking_number}",
    "royal_mail": "https://www.royalmail.com/track-your-item#/tracking-results/{tracking_number}",
    "canada_post": "https://www.canadapost-postescanada.ca/track-reperage/en#/search?searchFor={tracking_number}",
    "australia_post": "https://auspost.com.au/mypost/track/#/details/{tracking_number}",
}


def map_order_status(
    wc_status: str,
    has_tracking: bool = False,
    shipment_status: str | None = None,
) -> OrderStatus:
    """
    Map WooCommerce order status to OrderStatus enum.

    Args:
        wc_status: WooCommerce order status string.
        has_tracking: Whether tracking info exists for the order.
        shipment_status: Shipment tracking status if available.

    Returns:
        Mapped OrderStatus enum value.
    """
    # Normalize status (remove wc- prefix if present)
    normalized = wc_status.lower().strip()
    if normalized.startswith("wc-"):
        normalized = normalized[3:]

    # Check for shipped orders (completed with tracking)
    if normalized == "completed" and has_tracking:
        # If we have shipment status, map it
        if shipment_status:
            status_lower = shipment_status.lower()
            if status_lower in ("delivered", "complete"):
                return OrderStatus.DELIVERED
            elif status_lower in ("out_for_delivery", "out for delivery"):
                return OrderStatus.OUT_FOR_DELIVERY
            elif status_lower in ("in_transit", "in transit"):
                return OrderStatus.IN_TRANSIT
        return OrderStatus.SHIPPED

    # Look up in map
    if normalized in WOOCOMMERCE_STATUS_MAP:
        return WOOCOMMERCE_STATUS_MAP[normalized]

    # Check with wc- prefix
    if f"wc-{normalized}" in WOOCOMMERCE_STATUS_MAP:
        return WOOCOMMERCE_STATUS_MAP[f"wc-{normalized}"]

    # Default to processing for unknown statuses
    return OrderStatus.PROCESSING


def map_fulfillment_status(wc_status: str, has_tracking: bool = False) -> FulfillmentStatus:
    """
    Map WooCommerce status to FulfillmentStatus.

    WooCommerce doesn't have a separate fulfillment status - it's derived
    from order status and tracking presence.

    Args:
        wc_status: WooCommerce order status.
        has_tracking: Whether tracking info exists.

    Returns:
        Mapped FulfillmentStatus enum value.
    """
    normalized = wc_status.lower().strip()
    if normalized.startswith("wc-"):
        normalized = normalized[3:]

    if normalized in ("completed", "refunded"):
        return FulfillmentStatus.FULFILLED
    elif normalized == "processing" and has_tracking:
        return FulfillmentStatus.PARTIAL
    elif normalized in ("pending", "on-hold", "failed", "cancelled"):
        return FulfillmentStatus.UNFULFILLED

    # If processing without tracking, still unfulfilled
    if normalized == "processing":
        return FulfillmentStatus.UNFULFILLED

    return FulfillmentStatus.UNFULFILLED


def get_carrier_name(carrier_code: str) -> str:
    """
    Get human-readable carrier name from carrier code.

    Args:
        carrier_code: Carrier code from WooCommerce Shipment Tracking.

    Returns:
        Human-readable carrier name.
    """
    code_lower = carrier_code.lower().strip()
    return CARRIER_MAP.get(code_lower, carrier_code)


def get_tracking_url(carrier_code: str, tracking_number: str) -> str | None:
    """
    Build tracking URL for a carrier.

    Args:
        carrier_code: Carrier code from WooCommerce Shipment Tracking.
        tracking_number: The tracking number.

    Returns:
        Tracking URL if template exists, None otherwise.
    """
    code_lower = carrier_code.lower().strip()
    template = TRACKING_URL_TEMPLATES.get(code_lower)
    if template:
        return template.format(tracking_number=tracking_number)
    return None
