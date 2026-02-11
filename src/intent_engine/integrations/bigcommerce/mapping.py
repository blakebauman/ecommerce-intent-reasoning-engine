"""Status mapping for BigCommerce orders."""

from intent_engine.integrations.base import FulfillmentStatus, OrderStatus

# BigCommerce uses numeric status IDs
# See: https://developer.bigcommerce.com/docs/rest-management/orders#order-status-codes

BIGCOMMERCE_STATUS_MAP: dict[int, OrderStatus] = {
    0: OrderStatus.PENDING,  # Incomplete
    1: OrderStatus.PENDING,  # Pending
    2: OrderStatus.SHIPPED,  # Shipped
    3: OrderStatus.PROCESSING,  # Partially Shipped
    4: OrderStatus.REFUNDED,  # Refunded
    5: OrderStatus.CANCELLED,  # Cancelled
    6: OrderStatus.CANCELLED,  # Declined
    7: OrderStatus.PROCESSING,  # Awaiting Payment
    8: OrderStatus.PROCESSING,  # Awaiting Pickup
    9: OrderStatus.PROCESSING,  # Awaiting Shipment
    10: OrderStatus.DELIVERED,  # Completed
    11: OrderStatus.PROCESSING,  # Awaiting Fulfillment
    12: OrderStatus.PROCESSING,  # Manual Verification Required
    13: OrderStatus.PROCESSING,  # Disputed
    14: OrderStatus.PARTIALLY_REFUNDED,  # Partially Refunded
}

# Human-readable status names
BIGCOMMERCE_STATUS_NAMES: dict[int, str] = {
    0: "Incomplete",
    1: "Pending",
    2: "Shipped",
    3: "Partially Shipped",
    4: "Refunded",
    5: "Cancelled",
    6: "Declined",
    7: "Awaiting Payment",
    8: "Awaiting Pickup",
    9: "Awaiting Shipment",
    10: "Completed",
    11: "Awaiting Fulfillment",
    12: "Manual Verification Required",
    13: "Disputed",
    14: "Partially Refunded",
}

# Common carrier codes in BigCommerce
CARRIER_MAP: dict[str, str] = {
    "ups": "UPS",
    "usps": "USPS",
    "fedex": "FedEx",
    "dhl": "DHL",
    "dhl-express": "DHL Express",
    "royal-mail": "Royal Mail",
    "canada-post": "Canada Post",
    "auspost": "Australia Post",
    "dpd": "DPD",
    "gls": "GLS",
    "ontrac": "OnTrac",
    "lasership": "LaserShip",
    "amazon-logistics-us": "Amazon Logistics",
    "purolator": "Purolator",
    "sendle": "Sendle",
}

# Tracking URL templates for common carriers
TRACKING_URL_TEMPLATES: dict[str, str] = {
    "ups": "https://www.ups.com/track?tracknum={tracking_number}",
    "usps": "https://tools.usps.com/go/TrackConfirmAction?tLabels={tracking_number}",
    "fedex": "https://www.fedex.com/fedextrack/?trknbr={tracking_number}",
    "dhl": "https://www.dhl.com/en/express/tracking.html?AWB={tracking_number}",
    "dhl-express": "https://www.dhl.com/en/express/tracking.html?AWB={tracking_number}",
    "royal-mail": "https://www.royalmail.com/track-your-item#/tracking-results/{tracking_number}",
    "canada-post": "https://www.canadapost-postescanada.ca/track-reperage/en#/search?searchFor={tracking_number}",
    "auspost": "https://auspost.com.au/mypost/track/#/details/{tracking_number}",
}


def map_order_status(
    status_id: int,
    shipment_count: int = 0,
    items_shipped: int = 0,
    total_items: int = 0,
) -> OrderStatus:
    """
    Map BigCommerce order status ID to OrderStatus enum.

    Args:
        status_id: BigCommerce numeric status ID.
        shipment_count: Number of shipments created.
        items_shipped: Number of items shipped.
        total_items: Total number of items in order.

    Returns:
        Mapped OrderStatus enum value.
    """
    # Handle partially shipped more precisely
    if status_id == 3:  # Partially Shipped
        if items_shipped > 0 and total_items > 0:
            if items_shipped >= total_items:
                return OrderStatus.SHIPPED
        return OrderStatus.PROCESSING

    # Check if shipped order might be delivered (status 2)
    # BigCommerce doesn't have a built-in delivered status, so we use Completed (10)
    if status_id == 2 and shipment_count > 0:
        return OrderStatus.SHIPPED

    # Look up in map
    if status_id in BIGCOMMERCE_STATUS_MAP:
        return BIGCOMMERCE_STATUS_MAP[status_id]

    # Default to processing for unknown statuses
    return OrderStatus.PROCESSING


def map_fulfillment_status(
    status_id: int,
    items_shipped: int = 0,
    total_items: int = 0,
) -> FulfillmentStatus:
    """
    Map BigCommerce status to FulfillmentStatus.

    Args:
        status_id: BigCommerce numeric status ID.
        items_shipped: Number of items shipped.
        total_items: Total number of items in order.

    Returns:
        Mapped FulfillmentStatus enum value.
    """
    # Shipped or Completed
    if status_id in (2, 10):
        return FulfillmentStatus.FULFILLED

    # Partially Shipped
    if status_id == 3:
        if items_shipped > 0:
            if total_items > 0 and items_shipped >= total_items:
                return FulfillmentStatus.FULFILLED
            return FulfillmentStatus.PARTIAL
        return FulfillmentStatus.UNFULFILLED

    # Refunded, Cancelled, Declined - treat as fulfilled (complete)
    if status_id in (4, 5, 6, 14):
        return FulfillmentStatus.FULFILLED

    # All other statuses are unfulfilled
    return FulfillmentStatus.UNFULFILLED


def get_carrier_name(carrier_code: str) -> str:
    """
    Get human-readable carrier name from carrier code.

    Args:
        carrier_code: Carrier code from BigCommerce.

    Returns:
        Human-readable carrier name.
    """
    code_lower = carrier_code.lower().strip()
    return CARRIER_MAP.get(code_lower, carrier_code)


def get_status_name(status_id: int) -> str:
    """
    Get human-readable status name from status ID.

    Args:
        status_id: BigCommerce numeric status ID.

    Returns:
        Human-readable status name.
    """
    return BIGCOMMERCE_STATUS_NAMES.get(status_id, f"Unknown ({status_id})")


def get_tracking_url(carrier_code: str, tracking_number: str) -> str | None:
    """
    Build tracking URL for a carrier.

    Args:
        carrier_code: Carrier code from BigCommerce.
        tracking_number: The tracking number.

    Returns:
        Tracking URL if template exists, None otherwise.
    """
    code_lower = carrier_code.lower().strip()
    template = TRACKING_URL_TEMPLATES.get(code_lower)
    if template:
        return template.format(tracking_number=tracking_number)
    return None
