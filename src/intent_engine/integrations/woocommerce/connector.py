"""WooCommerce REST API connector (read-only) with customer profile support."""

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from intent_engine.integrations.base import (
    LineItem,
    OrderInfo,
    OrderStatus,
    PlatformConnector,
    ShippingAddress,
    TrackingInfo,
)
from intent_engine.integrations.woocommerce.mapping import (
    get_carrier_name,
    get_tracking_url,
    map_fulfillment_status,
    map_order_status,
)
from intent_engine.models.context import (
    CustomerProfile,
    CustomerTier,
    OrderContext,
    ProductContext,
    ReturnEligibility,
)


class WooCommerceConnector(PlatformConnector):
    """
    WooCommerce REST API connector for order data.

    Provides read-only access to order information via the
    WooCommerce REST API (v3).

    Authentication uses Basic Auth with consumer key/secret.
    """

    API_VERSION = "wc/v3"

    def __init__(
        self,
        store_url: str,
        consumer_key: str,
        consumer_secret: str,
        return_window_days: int = 30,
    ) -> None:
        """
        Initialize the WooCommerce connector.

        Args:
            store_url: The WooCommerce store URL (e.g., "https://mystore.com").
            consumer_key: WooCommerce REST API consumer key.
            consumer_secret: WooCommerce REST API consumer secret.
            return_window_days: Default return window in days.
        """
        self.store_url = store_url.rstrip("/")
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.return_window_days = return_window_days
        self._client: httpx.AsyncClient | None = None

    @property
    def platform_name(self) -> str:
        return "woocommerce"

    @property
    def base_url(self) -> str:
        return f"{self.store_url}/wp-json/{self.API_VERSION}"

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            # WooCommerce uses Basic Auth with consumer key/secret
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                auth=(self.consumer_key, self.consumer_secret),
                headers={
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

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict | list | None:
        """Make an API request to WooCommerce."""
        try:
            response = await self.client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def get_order(self, order_id: str) -> OrderInfo | None:
        """Fetch order by WooCommerce order ID."""
        data = await self._request("GET", f"/orders/{order_id}")
        if not data or not isinstance(data, dict):
            return None
        return await self._parse_order(data)

    async def get_order_by_number(self, order_number: str) -> OrderInfo | None:
        """
        Fetch order by customer-facing order number.

        In WooCommerce, the order number is typically the order ID,
        but can be customized with plugins like Sequential Order Numbers.
        """
        # Clean the order number (remove # prefix if present)
        clean_number = order_number.lstrip("#")

        # First try direct ID lookup (most common case)
        if clean_number.isdigit():
            order = await self.get_order(clean_number)
            if order:
                return order

        # Search by order number meta field (for Sequential Order Numbers plugin)
        data = await self._request(
            "GET",
            "/orders",
            params={"search": clean_number, "per_page": 5},
        )

        if data and isinstance(data, list) and len(data) > 0:
            # Check each result for matching order number
            for order_data in data:
                # Check meta_data for custom order number
                for meta in order_data.get("meta_data", []):
                    if meta.get("key") in ("_order_number", "order_number"):
                        if meta.get("value") == clean_number:
                            return await self._parse_order(order_data)
                # If order number matches ID
                if str(order_data.get("id")) == clean_number:
                    return await self._parse_order(order_data)
                # Check the number field (some configs)
                if str(order_data.get("number")) == clean_number:
                    return await self._parse_order(order_data)

            # Return first result as fallback
            return await self._parse_order(data[0])

        return None

    async def get_customer_orders(
        self,
        customer_email: str,
        limit: int = 10,
    ) -> list[OrderInfo]:
        """Fetch recent orders for a customer by email."""
        # First get customer ID by email
        customers = await self._request(
            "GET",
            "/customers",
            params={"email": customer_email, "per_page": 1},
        )

        if customers and isinstance(customers, list) and len(customers) > 0:
            customer_id = customers[0].get("id")
            data = await self._request(
                "GET",
                "/orders",
                params={
                    "customer": customer_id,
                    "per_page": limit,
                    "orderby": "date",
                    "order": "desc",
                },
            )
        else:
            # Fallback to search by billing email
            data = await self._request(
                "GET",
                "/orders",
                params={
                    "search": customer_email,
                    "per_page": limit,
                    "orderby": "date",
                    "order": "desc",
                },
            )

        if not data or not isinstance(data, list):
            return []

        orders = []
        for order_data in data:
            # Verify email matches (search might return partial matches)
            billing = order_data.get("billing", {})
            if billing.get("email", "").lower() == customer_email.lower():
                order = await self._parse_order(order_data)
                orders.append(order)

        return orders

    async def get_tracking(self, order_id: str) -> list[TrackingInfo]:
        """
        Get tracking information for an order.

        WooCommerce requires the Shipment Tracking plugin for tracking.
        Tracking data is stored in order meta_data.
        """
        order = await self._request("GET", f"/orders/{order_id}")
        if not order or not isinstance(order, dict):
            return []

        return self._extract_tracking(order)

    def _extract_tracking(self, order_data: dict) -> list[TrackingInfo]:
        """Extract tracking info from order meta_data."""
        tracking_list: list[TrackingInfo] = []

        for meta in order_data.get("meta_data", []):
            # WooCommerce Shipment Tracking plugin stores data here
            if meta.get("key") == "_wc_shipment_tracking_items":
                items = meta.get("value", [])
                if isinstance(items, list):
                    for item in items:
                        carrier = get_carrier_name(item.get("tracking_provider", ""))
                        tracking_number = item.get("tracking_number", "")

                        if tracking_number:
                            tracking_url = item.get("tracking_link") or get_tracking_url(
                                item.get("tracking_provider", ""),
                                tracking_number,
                            )
                            tracking_list.append(
                                TrackingInfo(
                                    carrier=carrier,
                                    tracking_number=tracking_number,
                                    tracking_url=tracking_url,
                                    status=item.get("status"),
                                )
                            )

        return tracking_list

    async def health_check(self) -> bool:
        """Check if the WooCommerce connection is working."""
        try:
            # System status endpoint requires authentication
            data = await self._request("GET", "/system_status")
            return data is not None
        except (httpx.HTTPError, OSError):
            return False

    async def _parse_order(self, order_data: dict) -> OrderInfo:
        """Parse WooCommerce order data into OrderInfo model."""
        # Parse line items
        line_items = [
            LineItem(
                product_id=str(item.get("product_id", "")),
                variant_id=str(item.get("variation_id", "")) if item.get("variation_id") else None,
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
        if addr := order_data.get("shipping"):
            name = f"{addr.get('first_name', '')} {addr.get('last_name', '')}".strip()
            if name or addr.get("address_1"):
                shipping_address = ShippingAddress(
                    name=name or addr.get("company", ""),
                    address1=addr.get("address_1", ""),
                    address2=addr.get("address_2") or None,
                    city=addr.get("city", ""),
                    province=addr.get("state", ""),
                    country=addr.get("country", ""),
                    zip_code=addr.get("postcode", ""),
                    phone=order_data.get("billing", {}).get("phone"),
                )

        # Extract tracking from meta_data
        tracking = self._extract_tracking(order_data)
        has_tracking = len(tracking) > 0

        # Parse timestamps
        created_at = self._parse_datetime(order_data.get("date_created"))
        updated_at = self._parse_datetime(order_data.get("date_modified"))
        completed_at = self._parse_datetime(order_data.get("date_completed"))

        # Calculate return window
        return_window_ends = None
        if created_at:
            return_window_ends = created_at + timedelta(days=self.return_window_days)

        # Map WooCommerce status to our status enum
        wc_status = order_data.get("status", "pending")
        status = map_order_status(wc_status, has_tracking=has_tracking)
        fulfillment_status = map_fulfillment_status(wc_status, has_tracking=has_tracking)

        # Parse refund info
        refund_amount = None
        refunds = order_data.get("refunds", [])
        if refunds:
            refund_amount = sum(abs(float(r.get("total", 0))) for r in refunds)

        # Get customer name from billing
        billing = order_data.get("billing", {})
        customer_name = f"{billing.get('first_name', '')} {billing.get('last_name', '')}".strip()

        # Order number (may be customized by plugins)
        order_number = str(order_data.get("number", order_data.get("id", "")))
        # Check for custom order number in meta
        for meta in order_data.get("meta_data", []):
            if meta.get("key") in ("_order_number", "order_number"):
                order_number = str(meta.get("value", order_number))
                break

        return OrderInfo(
            order_id=str(order_data["id"]),
            platform="woocommerce",
            order_number=order_number,
            status=status,
            fulfillment_status=fulfillment_status,
            customer_email=billing.get("email", ""),
            customer_name=customer_name,
            line_items=line_items,
            subtotal=float(order_data.get("total", 0))
            - float(order_data.get("shipping_total", 0))
            - float(order_data.get("total_tax", 0)),
            shipping_cost=float(order_data.get("shipping_total", 0)),
            tax=float(order_data.get("total_tax", 0)),
            total=float(order_data.get("total", 0)),
            currency=order_data.get("currency", "USD"),
            shipping_address=shipping_address,
            tracking=tracking,
            created_at=created_at,
            updated_at=updated_at,
            shipped_at=completed_at if has_tracking else None,
            delivered_at=completed_at if status == OrderStatus.DELIVERED else None,
            is_returnable=return_window_ends is not None and return_window_ends > datetime.now(UTC),
            return_window_ends=return_window_ends,
            refund_amount=refund_amount,
            raw_data=order_data,
        )

    def _parse_datetime(self, dt_str: str | None) -> datetime | None:
        """Parse WooCommerce datetime string."""
        if not dt_str:
            return None
        try:
            # WooCommerce uses ISO 8601 format
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            # Ensure timezone-aware (assume UTC if naive)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except ValueError:
            return None

    # =========================================================================
    # Customer Profile and Context Enrichment Methods
    # =========================================================================

    async def get_customer_by_email(self, email: str) -> CustomerProfile | None:
        """
        Fetch customer profile by email address.

        Args:
            email: Customer email address.

        Returns:
            CustomerProfile if found, None otherwise.
        """
        data = await self._request(
            "GET",
            "/customers",
            params={"email": email, "per_page": 1},
        )

        if not data or not isinstance(data, list) or len(data) == 0:
            return None

        customer_data = data[0]
        return await self._build_customer_profile(customer_data)

    async def get_customer_by_id(self, customer_id: str) -> CustomerProfile | None:
        """
        Fetch customer profile by WooCommerce customer ID.

        Args:
            customer_id: WooCommerce customer ID.

        Returns:
            CustomerProfile if found, None otherwise.
        """
        data = await self._request("GET", f"/customers/{customer_id}")

        if not data or not isinstance(data, dict):
            return None

        return await self._build_customer_profile(data)

    async def _build_customer_profile(self, customer_data: dict) -> CustomerProfile:
        """Build CustomerProfile from WooCommerce customer data."""
        customer_id = str(customer_data["id"])
        email = customer_data.get("email", "")

        # Calculate lifetime value and order count
        total_spent = float(customer_data.get("total_spent", "0"))
        orders_count = customer_data.get("orders_count", 0)

        # Parse first order date
        first_order_date = self._parse_datetime(customer_data.get("date_created"))

        # Determine tier based on spending and order history
        tier = self._determine_customer_tier(total_spent, orders_count, customer_data)

        # Build name from billing
        billing = customer_data.get("billing", {})
        name = f"{billing.get('first_name', '')} {billing.get('last_name', '')}".strip()
        if not name:
            name = f"{customer_data.get('first_name', '')} {customer_data.get('last_name', '')}".strip()

        return CustomerProfile(
            customer_id=customer_id,
            email=email,
            name=name,
            tier=tier,
            lifetime_value=total_spent,
            total_orders=orders_count,
            first_order_date=first_order_date,
            support_tickets_30d=0,
            complaints_90d=0,
            returns_90d=0,
            preferred_contact="email",
            language="en",
            is_vip=tier == CustomerTier.VIP,
            is_at_risk=customer_data.get("is_paying_customer", True) is False and orders_count > 0,
        )

    def _determine_customer_tier(
        self,
        total_spent: float,
        orders_count: int,
        customer_data: dict,
    ) -> CustomerTier:
        """Determine customer tier based on value and history."""
        role = customer_data.get("role", "customer").lower()

        # Check for explicit roles
        if "vip" in role:
            return CustomerTier.VIP
        if "wholesale" in role or "premium" in role:
            return CustomerTier.PREMIUM

        # Check meta_data for tier tags
        for meta in customer_data.get("meta_data", []):
            key = meta.get("key", "").lower()
            value = str(meta.get("value", "")).lower()
            if key in ("customer_tier", "_customer_tier", "tier"):
                if value == "vip":
                    return CustomerTier.VIP
                elif value == "premium":
                    return CustomerTier.PREMIUM
                elif value == "flagged":
                    return CustomerTier.FLAGGED
                elif value == "at_risk":
                    return CustomerTier.AT_RISK

        # Tier by value
        if total_spent >= 1000 or orders_count >= 20:
            return CustomerTier.VIP
        if total_spent >= 500 or orders_count >= 10:
            return CustomerTier.PREMIUM
        if orders_count == 0:
            return CustomerTier.NEW

        return CustomerTier.STANDARD

    async def get_order_context(self, order_id: str) -> OrderContext | None:
        """
        Get enriched order context for policy decisions.

        Args:
            order_id: WooCommerce order ID.

        Returns:
            OrderContext with policy-relevant data.
        """
        order_info = await self.get_order(order_id)
        if not order_info:
            return None

        return self._build_order_context(order_info)

    async def get_order_context_by_number(self, order_number: str) -> OrderContext | None:
        """
        Get enriched order context by order number.

        Args:
            order_number: Customer-facing order number.

        Returns:
            OrderContext with policy-relevant data.
        """
        order_info = await self.get_order_by_number(order_number)
        if not order_info:
            return None

        return self._build_order_context(order_info)

    def _build_order_context(self, order_info: OrderInfo) -> OrderContext:
        """Build OrderContext from OrderInfo."""
        now = datetime.now(UTC)

        # Build product contexts
        items = [
            ProductContext(
                product_id=item.product_id,
                sku=item.sku,
                name=item.name,
                price=item.price,
                currency=item.currency,
                is_returnable=True,
                return_window_days=self.return_window_days,
            )
            for item in order_info.line_items
        ]

        # Calculate return window status
        days_until_return_expires = None
        is_within_return_window = True
        return_eligibility = ReturnEligibility.ELIGIBLE

        if order_info.return_window_ends:
            delta = order_info.return_window_ends - now
            days_until_return_expires = delta.days
            is_within_return_window = delta.total_seconds() > 0

            if not is_within_return_window:
                return_eligibility = ReturnEligibility.EXPIRED
        elif order_info.created_at:
            window_end = order_info.created_at + timedelta(days=self.return_window_days)
            delta = window_end - now
            days_until_return_expires = delta.days
            is_within_return_window = delta.total_seconds() > 0

            if not is_within_return_window:
                return_eligibility = ReturnEligibility.EXPIRED

        # Extract tracking info
        tracking_number = None
        carrier = None
        tracking_url = None
        if order_info.tracking:
            first_tracking = order_info.tracking[0]
            tracking_number = first_tracking.tracking_number
            carrier = first_tracking.carrier
            tracking_url = first_tracking.tracking_url

        # Shipping address as dict
        shipping_address = None
        if order_info.shipping_address:
            addr = order_info.shipping_address
            shipping_address = {
                "name": addr.name,
                "address1": addr.address1,
                "address2": addr.address2,
                "city": addr.city,
                "province": addr.province,
                "country": addr.country,
                "zip": addr.zip_code,
            }

        return OrderContext(
            order_id=order_info.order_id,
            order_number=order_info.order_number,
            platform=order_info.platform,
            status=order_info.status.value,
            fulfillment_status=order_info.fulfillment_status.value,
            is_cancelled=order_info.status == OrderStatus.CANCELLED,
            customer_email=order_info.customer_email,
            customer_name=order_info.customer_name,
            items=items,
            subtotal=order_info.subtotal,
            shipping_cost=order_info.shipping_cost,
            tax=order_info.tax,
            total=order_info.total,
            currency=order_info.currency,
            created_at=order_info.created_at,
            shipped_at=order_info.shipped_at,
            delivered_at=order_info.delivered_at,
            shipping_address=shipping_address,
            tracking_number=tracking_number,
            carrier=carrier,
            tracking_url=tracking_url,
            return_eligibility=return_eligibility,
            return_window_ends=order_info.return_window_ends,
            days_until_return_expires=days_until_return_expires,
            is_within_return_window=is_within_return_window,
            refund_amount=order_info.refund_amount,
            is_fully_refunded=order_info.status == OrderStatus.REFUNDED,
            is_partially_refunded=order_info.status == OrderStatus.PARTIALLY_REFUNDED,
        )

    async def get_customer_order_history(
        self,
        customer_email: str,
        limit: int = 10,
    ) -> list[OrderContext]:
        """
        Get order history as OrderContext objects.

        Args:
            customer_email: Customer email address.
            limit: Maximum number of orders to return.

        Returns:
            List of OrderContext objects.
        """
        orders = await self.get_customer_orders(customer_email, limit)
        return [self._build_order_context(order) for order in orders]
