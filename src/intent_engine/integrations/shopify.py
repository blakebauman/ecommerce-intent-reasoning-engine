"""Shopify Admin API connector (read-only)."""

from datetime import datetime, timedelta
from typing import Any

import httpx

from intent_engine.integrations.base import (
    FulfillmentStatus,
    LineItem,
    OrderInfo,
    OrderStatus,
    PlatformConnector,
    ShippingAddress,
    TrackingInfo,
)


class ShopifyConnector(PlatformConnector):
    """
    Shopify Admin API connector for order data.

    Provides read-only access to order information via the
    Shopify Admin REST API (version 2024-01).
    """

    API_VERSION = "2024-01"

    def __init__(
        self,
        store_domain: str,
        access_token: str,
        return_window_days: int = 30,
    ) -> None:
        """
        Initialize the Shopify connector.

        Args:
            store_domain: The Shopify store domain (e.g., "my-store.myshopify.com").
            access_token: Shopify Admin API access token.
            return_window_days: Default return window in days.
        """
        self.store_domain = store_domain.rstrip("/")
        self.access_token = access_token
        self.return_window_days = return_window_days
        self._client: httpx.AsyncClient | None = None

    @property
    def platform_name(self) -> str:
        return "shopify"

    @property
    def base_url(self) -> str:
        return f"https://{self.store_domain}/admin/api/{self.API_VERSION}"

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "X-Shopify-Access-Token": self.access_token,
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict | None:
        """Make an API request to Shopify."""
        try:
            response = await self.client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def get_order(self, order_id: str) -> OrderInfo | None:
        """Fetch order by Shopify order ID."""
        data = await self._request("GET", f"/orders/{order_id}.json")
        if not data or "order" not in data:
            return None
        return self._parse_order(data["order"])

    async def get_order_by_number(self, order_number: str) -> OrderInfo | None:
        """Fetch order by customer-facing order number."""
        # Clean the order number (remove # prefix if present)
        clean_number = order_number.lstrip("#")

        # Search by order number
        data = await self._request(
            "GET",
            "/orders.json",
            params={"name": order_number, "status": "any"},
        )

        if not data or "orders" not in data or not data["orders"]:
            # Try without the # prefix
            data = await self._request(
                "GET",
                "/orders.json",
                params={"name": f"#{clean_number}", "status": "any"},
            )

        if not data or "orders" not in data or not data["orders"]:
            return None

        return self._parse_order(data["orders"][0])

    async def get_customer_orders(
        self,
        customer_email: str,
        limit: int = 10,
    ) -> list[OrderInfo]:
        """Fetch recent orders for a customer by email."""
        data = await self._request(
            "GET",
            "/orders.json",
            params={
                "email": customer_email,
                "status": "any",
                "limit": limit,
            },
        )

        if not data or "orders" not in data:
            return []

        return [self._parse_order(order) for order in data["orders"]]

    async def get_tracking(self, order_id: str) -> list[TrackingInfo]:
        """Get tracking information for an order."""
        data = await self._request("GET", f"/orders/{order_id}/fulfillments.json")

        if not data or "fulfillments" not in data:
            return []

        tracking_list: list[TrackingInfo] = []

        for fulfillment in data["fulfillments"]:
            if fulfillment.get("tracking_number"):
                tracking_list.append(
                    TrackingInfo(
                        carrier=fulfillment.get("tracking_company", "Unknown"),
                        tracking_number=fulfillment["tracking_number"],
                        tracking_url=fulfillment.get("tracking_url"),
                        status=fulfillment.get("shipment_status"),
                    )
                )

        return tracking_list

    async def health_check(self) -> bool:
        """Check if the Shopify connection is working."""
        try:
            data = await self._request("GET", "/shop.json")
            return data is not None and "shop" in data
        except Exception:
            return False

    def _parse_order(self, order_data: dict) -> OrderInfo:
        """Parse Shopify order data into OrderInfo model."""
        # Parse line items
        line_items = [
            LineItem(
                product_id=str(item.get("product_id", "")),
                variant_id=str(item.get("variant_id", "")) if item.get("variant_id") else None,
                sku=item.get("sku"),
                name=item.get("name", ""),
                quantity=item.get("quantity", 1),
                price=float(item.get("price", 0)),
                currency=order_data.get("currency", "USD"),
            )
            for item in order_data.get("line_items", [])
        ]

        # Parse shipping address
        shipping_address = None
        if addr := order_data.get("shipping_address"):
            shipping_address = ShippingAddress(
                name=f"{addr.get('first_name', '')} {addr.get('last_name', '')}".strip(),
                address1=addr.get("address1", ""),
                address2=addr.get("address2"),
                city=addr.get("city", ""),
                province=addr.get("province", ""),
                country=addr.get("country", ""),
                zip_code=addr.get("zip", ""),
                phone=addr.get("phone"),
            )

        # Parse tracking from fulfillments
        tracking = []
        for fulfillment in order_data.get("fulfillments", []):
            if fulfillment.get("tracking_number"):
                tracking.append(
                    TrackingInfo(
                        carrier=fulfillment.get("tracking_company", "Unknown"),
                        tracking_number=fulfillment["tracking_number"],
                        tracking_url=fulfillment.get("tracking_url"),
                        status=fulfillment.get("shipment_status"),
                    )
                )

        # Parse timestamps
        created_at = self._parse_datetime(order_data.get("created_at"))
        updated_at = self._parse_datetime(order_data.get("updated_at"))

        # Calculate return window
        return_window_ends = None
        if created_at:
            return_window_ends = created_at + timedelta(days=self.return_window_days)

        # Map Shopify status to our status enum
        status = self._map_status(order_data)
        fulfillment_status = self._map_fulfillment_status(order_data)

        # Parse refund info
        refund_amount = None
        if refunds := order_data.get("refunds"):
            refund_amount = sum(
                float(r.get("total_refund_set", {}).get("shop_money", {}).get("amount", 0))
                for r in refunds
            )

        return OrderInfo(
            order_id=str(order_data["id"]),
            platform="shopify",
            order_number=order_data.get("name", str(order_data["id"])),
            status=status,
            fulfillment_status=fulfillment_status,
            customer_email=order_data.get("email", ""),
            customer_name=f"{order_data.get('customer', {}).get('first_name', '')} "
                         f"{order_data.get('customer', {}).get('last_name', '')}".strip(),
            line_items=line_items,
            subtotal=float(order_data.get("subtotal_price", 0)),
            shipping_cost=float(
                order_data.get("total_shipping_price_set", {})
                .get("shop_money", {})
                .get("amount", 0)
            ),
            tax=float(order_data.get("total_tax", 0)),
            total=float(order_data.get("total_price", 0)),
            currency=order_data.get("currency", "USD"),
            shipping_address=shipping_address,
            tracking=tracking,
            created_at=created_at,
            updated_at=updated_at,
            is_returnable=return_window_ends is not None and return_window_ends > datetime.utcnow(),
            return_window_ends=return_window_ends,
            refund_amount=refund_amount,
            raw_data=order_data,
        )

    def _map_status(self, order_data: dict) -> OrderStatus:
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
            # Check shipment status from fulfillments
            for f in order_data.get("fulfillments", []):
                shipment_status = f.get("shipment_status")
                if shipment_status == "delivered":
                    return OrderStatus.DELIVERED
                elif shipment_status == "out_for_delivery":
                    return OrderStatus.OUT_FOR_DELIVERY
                elif shipment_status == "in_transit":
                    return OrderStatus.IN_TRANSIT
            return OrderStatus.SHIPPED

        if fulfillment_status == "partial":
            return OrderStatus.PROCESSING

        if financial_status == "paid":
            return OrderStatus.CONFIRMED

        if financial_status == "pending":
            return OrderStatus.PENDING

        return OrderStatus.PROCESSING

    def _map_fulfillment_status(self, order_data: dict) -> FulfillmentStatus:
        """Map Shopify fulfillment status."""
        status = order_data.get("fulfillment_status")
        if status == "fulfilled":
            return FulfillmentStatus.FULFILLED
        elif status == "partial":
            return FulfillmentStatus.PARTIAL
        return FulfillmentStatus.UNFULFILLED

    def _parse_datetime(self, dt_str: str | None) -> datetime | None:
        """Parse Shopify datetime string."""
        if not dt_str:
            return None
        try:
            # Shopify uses ISO 8601 format
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except ValueError:
            return None
