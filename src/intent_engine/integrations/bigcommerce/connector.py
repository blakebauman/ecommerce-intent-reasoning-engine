"""BigCommerce REST API connector (read-only) with customer profile support."""

from datetime import datetime, timedelta, timezone
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
from intent_engine.integrations.bigcommerce.mapping import (
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


class BigCommerceConnector(PlatformConnector):
    """
    BigCommerce REST API connector for order data.

    Provides read-only access to order information via the
    BigCommerce REST API (mix of V2 for orders and V3 for customers).

    Authentication uses X-Auth-Token header (similar to Shopify pattern).
    """

    def __init__(
        self,
        store_hash: str,
        access_token: str,
        return_window_days: int = 30,
    ) -> None:
        """
        Initialize the BigCommerce connector.

        Args:
            store_hash: The BigCommerce store hash (e.g., "abc123").
            access_token: BigCommerce API access token.
            return_window_days: Default return window in days.
        """
        self.store_hash = store_hash
        self.access_token = access_token
        self.return_window_days = return_window_days
        self._client: httpx.AsyncClient | None = None

    @property
    def platform_name(self) -> str:
        return "bigcommerce"

    @property
    def base_url_v2(self) -> str:
        return f"https://api.bigcommerce.com/stores/{self.store_hash}/v2"

    @property
    def base_url_v3(self) -> str:
        return f"https://api.bigcommerce.com/stores/{self.store_hash}/v3"

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={
                    "X-Auth-Token": self.access_token,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=30.0,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request_v2(self, method: str, path: str, **kwargs: Any) -> dict | list | None:
        """Make an API request to BigCommerce V2 API."""
        url = f"{self.base_url_v2}{path}"
        try:
            response = await self.client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def _request_v3(self, method: str, path: str, **kwargs: Any) -> dict | None:
        """Make an API request to BigCommerce V3 API."""
        url = f"{self.base_url_v3}{path}"
        try:
            response = await self.client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def get_order(self, order_id: str) -> OrderInfo | None:
        """Fetch order by BigCommerce order ID."""
        data = await self._request_v2("GET", f"/orders/{order_id}")
        if not data or not isinstance(data, dict):
            return None

        # Fetch additional data for complete order info
        products = await self._request_v2("GET", f"/orders/{order_id}/products")
        shipping = await self._request_v2("GET", f"/orders/{order_id}/shipping_addresses")
        shipments = await self._request_v2("GET", f"/orders/{order_id}/shipments")

        return self._parse_order(
            data,
            products=products if isinstance(products, list) else [],
            shipping=shipping if isinstance(shipping, list) else [],
            shipments=shipments if isinstance(shipments, list) else [],
        )

    async def get_order_by_number(self, order_number: str) -> OrderInfo | None:
        """
        Fetch order by customer-facing order number.

        In BigCommerce, the order ID is the order number by default.
        """
        # Clean the order number (remove # prefix if present)
        clean_number = order_number.lstrip("#")

        # BigCommerce order IDs are the order numbers
        if clean_number.isdigit():
            return await self.get_order(clean_number)

        # If not numeric, search by various fields
        # Note: V2 API doesn't support direct search, so we try ID first
        return None

    async def get_customer_orders(
        self,
        customer_email: str,
        limit: int = 10,
    ) -> list[OrderInfo]:
        """Fetch recent orders for a customer by email."""
        # First get customer ID by email
        customers = await self._request_v3(
            "GET",
            "/customers",
            params={"email:in": customer_email, "limit": 1},
        )

        orders_data = []

        if customers and isinstance(customers, dict):
            customer_list = customers.get("data", [])
            if customer_list:
                customer_id = customer_list[0].get("id")
                data = await self._request_v2(
                    "GET",
                    "/orders",
                    params={
                        "customer_id": customer_id,
                        "limit": limit,
                        "sort": "date_created:desc",
                    },
                )
                if data and isinstance(data, list):
                    orders_data = data

        if not orders_data:
            # Fallback: search all orders and filter by email
            # Note: This is less efficient but works for guest orders
            data = await self._request_v2(
                "GET",
                "/orders",
                params={
                    "limit": 50,
                    "sort": "date_created:desc",
                },
            )
            if data and isinstance(data, list):
                for order in data:
                    billing = order.get("billing_address", {})
                    if billing.get("email", "").lower() == customer_email.lower():
                        orders_data.append(order)
                        if len(orders_data) >= limit:
                            break

        orders = []
        for order_data in orders_data:
            order_id = str(order_data.get("id", ""))
            products = await self._request_v2("GET", f"/orders/{order_id}/products")
            shipping = await self._request_v2("GET", f"/orders/{order_id}/shipping_addresses")
            shipments = await self._request_v2("GET", f"/orders/{order_id}/shipments")

            order = self._parse_order(
                order_data,
                products=products if isinstance(products, list) else [],
                shipping=shipping if isinstance(shipping, list) else [],
                shipments=shipments if isinstance(shipments, list) else [],
            )
            orders.append(order)

        return orders

    async def get_tracking(self, order_id: str) -> list[TrackingInfo]:
        """Get tracking information for an order."""
        shipments = await self._request_v2("GET", f"/orders/{order_id}/shipments")

        if not shipments or not isinstance(shipments, list):
            return []

        tracking_list: list[TrackingInfo] = []

        for shipment in shipments:
            tracking_number = shipment.get("tracking_number", "")
            if tracking_number:
                carrier_code = shipment.get("shipping_provider", "")
                tracking_url = shipment.get("tracking_link") or get_tracking_url(
                    carrier_code, tracking_number
                )
                tracking_list.append(
                    TrackingInfo(
                        carrier=get_carrier_name(carrier_code),
                        tracking_number=tracking_number,
                        tracking_url=tracking_url,
                        status=None,  # BigCommerce doesn't track shipment status
                    )
                )

        return tracking_list

    async def health_check(self) -> bool:
        """Check if the BigCommerce connection is working."""
        try:
            # Store info endpoint
            data = await self._request_v2("GET", "/store")
            return data is not None
        except Exception:
            return False

    def _parse_order(
        self,
        order_data: dict,
        products: list | None = None,
        shipping: list | None = None,
        shipments: list | None = None,
    ) -> OrderInfo:
        """Parse BigCommerce order data into OrderInfo model."""
        products = products or []
        shipping = shipping or []
        shipments = shipments or []

        # Parse line items
        line_items = [
            LineItem(
                product_id=str(item.get("product_id", "")),
                variant_id=str(item.get("variant_id", "")) if item.get("variant_id") else None,
                sku=item.get("sku"),
                name=item.get("name", ""),
                quantity=item.get("quantity", 1),
                price=float(item.get("base_price", 0)),
                currency=order_data.get("currency_code", "USD"),
            )
            for item in products
        ]

        # Parse shipping address (first one)
        shipping_address = None
        if shipping:
            addr = shipping[0]
            name = f"{addr.get('first_name', '')} {addr.get('last_name', '')}".strip()
            shipping_address = ShippingAddress(
                name=name or addr.get("company", ""),
                address1=addr.get("street_1", ""),
                address2=addr.get("street_2") or None,
                city=addr.get("city", ""),
                province=addr.get("state", ""),
                country=addr.get("country", ""),
                zip_code=addr.get("zip", ""),
                phone=addr.get("phone"),
            )

        # Parse tracking from shipments
        tracking = []
        for shipment in shipments:
            tracking_number = shipment.get("tracking_number", "")
            if tracking_number:
                carrier_code = shipment.get("shipping_provider", "")
                tracking_url = shipment.get("tracking_link") or get_tracking_url(
                    carrier_code, tracking_number
                )
                tracking.append(
                    TrackingInfo(
                        carrier=get_carrier_name(carrier_code),
                        tracking_number=tracking_number,
                        tracking_url=tracking_url,
                        status=None,
                    )
                )

        # Parse timestamps
        created_at = self._parse_datetime(order_data.get("date_created"))
        updated_at = self._parse_datetime(order_data.get("date_modified"))
        shipped_at = self._parse_datetime(order_data.get("date_shipped"))

        # Calculate return window
        return_window_ends = None
        if created_at:
            return_window_ends = created_at + timedelta(days=self.return_window_days)

        # Map BigCommerce status to our status enum
        status_id = order_data.get("status_id", 1)
        items_shipped = sum(
            int(item.get("quantity_shipped", 0))
            for item in products
        )
        total_items = sum(int(item.get("quantity", 1)) for item in products)

        status = map_order_status(
            status_id,
            shipment_count=len(shipments),
            items_shipped=items_shipped,
            total_items=total_items,
        )
        fulfillment_status = map_fulfillment_status(
            status_id,
            items_shipped=items_shipped,
            total_items=total_items,
        )

        # Parse refund info
        refund_amount = float(order_data.get("refunded_amount", 0))

        # Get customer info from billing address
        billing = order_data.get("billing_address", {})
        customer_name = f"{billing.get('first_name', '')} {billing.get('last_name', '')}".strip()
        customer_email = billing.get("email", "")

        # Calculate financials
        subtotal = float(order_data.get("subtotal_ex_tax", 0))
        shipping_cost = float(order_data.get("shipping_cost_ex_tax", 0))
        tax = float(order_data.get("total_tax", 0))
        total = float(order_data.get("total_inc_tax", 0))

        return OrderInfo(
            order_id=str(order_data["id"]),
            platform="bigcommerce",
            order_number=str(order_data.get("id", "")),
            status=status,
            fulfillment_status=fulfillment_status,
            customer_email=customer_email,
            customer_name=customer_name,
            line_items=line_items,
            subtotal=subtotal,
            shipping_cost=shipping_cost,
            tax=tax,
            total=total,
            currency=order_data.get("currency_code", "USD"),
            shipping_address=shipping_address,
            tracking=tracking,
            created_at=created_at,
            updated_at=updated_at,
            shipped_at=shipped_at,
            delivered_at=None,  # BigCommerce doesn't track delivery
            is_returnable=return_window_ends is not None and return_window_ends > datetime.now(timezone.utc),
            return_window_ends=return_window_ends,
            refund_amount=refund_amount if refund_amount > 0 else None,
            raw_data=order_data,
        )

    def _parse_datetime(self, dt_str: str | None) -> datetime | None:
        """Parse BigCommerce datetime string."""
        if not dt_str:
            return None
        try:
            # BigCommerce uses RFC 2822 format
            # Example: "Tue, 20 Nov 2024 18:15:30 +0000"
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(dt_str)
            # Ensure timezone-aware (assume UTC if naive)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            try:
                # Fallback to ISO 8601
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
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
        data = await self._request_v3(
            "GET",
            "/customers",
            params={"email:in": email, "limit": 1, "include": "addresses"},
        )

        if not data or not isinstance(data, dict):
            return None

        customers = data.get("data", [])
        if not customers:
            return None

        customer_data = customers[0]
        return await self._build_customer_profile(customer_data)

    async def get_customer_by_id(self, customer_id: str) -> CustomerProfile | None:
        """
        Fetch customer profile by BigCommerce customer ID.

        Args:
            customer_id: BigCommerce customer ID.

        Returns:
            CustomerProfile if found, None otherwise.
        """
        data = await self._request_v3(
            "GET",
            f"/customers",
            params={"id:in": customer_id, "include": "addresses"},
        )

        if not data or not isinstance(data, dict):
            return None

        customers = data.get("data", [])
        if not customers:
            return None

        return await self._build_customer_profile(customers[0])

    async def _build_customer_profile(self, customer_data: dict) -> CustomerProfile:
        """Build CustomerProfile from BigCommerce customer data."""
        customer_id = str(customer_data["id"])
        email = customer_data.get("email", "")

        # Get order stats - need to query orders
        orders_data = await self._request_v2(
            "GET",
            "/orders",
            params={"customer_id": customer_id, "limit": 100},
        )

        orders_count = 0
        total_spent = 0.0
        first_order_date = None

        if orders_data and isinstance(orders_data, list):
            orders_count = len(orders_data)
            total_spent = sum(float(o.get("total_inc_tax", 0)) for o in orders_data)
            if orders_data:
                # Last in list is oldest (sorted desc by default)
                oldest = orders_data[-1]
                first_order_date = self._parse_datetime(oldest.get("date_created"))

        # Determine tier based on spending and order history
        tier = self._determine_customer_tier(total_spent, orders_count, customer_data)

        # Build name
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
            is_at_risk=False,
        )

    def _determine_customer_tier(
        self,
        total_spent: float,
        orders_count: int,
        customer_data: dict,
    ) -> CustomerTier:
        """Determine customer tier based on value and history."""
        customer_group_id = customer_data.get("customer_group_id")

        # Check for VIP/wholesale groups (common BigCommerce pattern)
        # Group IDs are store-specific, but we can check for common patterns
        if customer_group_id:
            # This would need to be configured per-store
            pass

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
            order_id: BigCommerce order ID.

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
        now = datetime.utcnow()

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
