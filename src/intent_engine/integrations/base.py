"""Base platform connector interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class OrderStatus(str, Enum):
    """Standard order statuses across platforms."""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    IN_TRANSIT = "in_transit"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"
    RETURNED = "returned"
    FAILED = "failed"


class FulfillmentStatus(str, Enum):
    """Fulfillment statuses."""

    UNFULFILLED = "unfulfilled"
    PARTIAL = "partial"
    FULFILLED = "fulfilled"


@dataclass
class LineItem:
    """Order line item."""

    product_id: str
    variant_id: str | None
    sku: str | None
    name: str
    quantity: int
    price: float
    currency: str = "USD"


@dataclass
class ShippingAddress:
    """Shipping address."""

    name: str
    address1: str
    address2: str | None
    city: str
    province: str
    country: str
    zip_code: str
    phone: str | None = None


@dataclass
class TrackingInfo:
    """Shipment tracking information."""

    carrier: str
    tracking_number: str
    tracking_url: str | None = None
    estimated_delivery: datetime | None = None
    status: str | None = None


@dataclass
class OrderInfo:
    """
    Unified order information across platforms.

    This is the common model that all platform connectors return.
    """

    order_id: str
    platform: str
    order_number: str  # Customer-facing order number
    status: OrderStatus
    fulfillment_status: FulfillmentStatus

    # Customer info
    customer_email: str
    customer_name: str

    # Order details
    line_items: list[LineItem] = field(default_factory=list)
    subtotal: float = 0.0
    shipping_cost: float = 0.0
    tax: float = 0.0
    total: float = 0.0
    currency: str = "USD"

    # Shipping
    shipping_address: ShippingAddress | None = None
    tracking: list[TrackingInfo] = field(default_factory=list)

    # Timestamps
    created_at: datetime | None = None
    updated_at: datetime | None = None
    shipped_at: datetime | None = None
    delivered_at: datetime | None = None

    # Return/refund info
    is_returnable: bool = True
    return_window_ends: datetime | None = None
    refund_amount: float | None = None

    # Raw platform data for debugging
    raw_data: dict | None = None


class PlatformConnector(ABC):
    """
    Abstract base class for platform integrations.

    Implementations provide read-only access to order data
    from eCommerce platforms like Shopify, WooCommerce, etc.
    """

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return the platform name (e.g., 'shopify')."""
        ...

    @abstractmethod
    async def get_order(self, order_id: str) -> OrderInfo | None:
        """
        Fetch order details by ID.

        Args:
            order_id: The platform order ID.

        Returns:
            OrderInfo if found, None otherwise.
        """
        ...

    @abstractmethod
    async def get_order_by_number(self, order_number: str) -> OrderInfo | None:
        """
        Fetch order details by customer-facing order number.

        Args:
            order_number: The order number (e.g., "#1234").

        Returns:
            OrderInfo if found, None otherwise.
        """
        ...

    @abstractmethod
    async def get_customer_orders(
        self,
        customer_email: str,
        limit: int = 10,
    ) -> list[OrderInfo]:
        """
        Fetch recent orders for a customer.

        Args:
            customer_email: Customer email address.
            limit: Maximum number of orders to return.

        Returns:
            List of OrderInfo objects.
        """
        ...

    @abstractmethod
    async def get_tracking(self, order_id: str) -> list[TrackingInfo]:
        """
        Get tracking information for an order.

        Args:
            order_id: The platform order ID.

        Returns:
            List of TrackingInfo objects.
        """
        ...

    async def health_check(self) -> bool:
        """
        Check if the platform connection is healthy.

        Returns:
            True if connection is working.
        """
        return True
