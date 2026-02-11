"""Adobe Commerce (Magento) REST API connector."""

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from intent_engine.integrations.adobe_commerce.auth import AdobeCommerceAuthStrategy
from intent_engine.integrations.adobe_commerce.mapping import (
    build_tracking_url,
    get_carrier_name,
    map_fulfillment_status,
    map_order_status,
)
from intent_engine.integrations.base import (
    LineItem,
    OrderInfo,
    OrderStatus,
    PlatformConnector,
    ShippingAddress,
    TrackingInfo,
)
from intent_engine.models.context import (
    CustomerProfile,
    CustomerTier,
    OrderContext,
    ProductContext,
    ReturnEligibility,
)


class AdobeCommerceConnector(PlatformConnector):
    """
    Adobe Commerce (Magento) REST API connector.

    Supports both deployment models:
    - PaaS: Self-hosted on cloud infrastructure (uses IntegrationTokenAuth)
    - SaaS: Adobe Commerce Cloud Service (uses IMSOAuthAuth)

    Provides read-only access to order, customer, and shipment data.
    """

    def __init__(
        self,
        base_url: str,
        auth_strategy: AdobeCommerceAuthStrategy,
        store_code: str = "default",
        return_window_days: int = 30,
    ) -> None:
        """
        Initialize the Adobe Commerce connector.

        Args:
            base_url: Adobe Commerce store base URL (e.g., "https://your-store.com").
            auth_strategy: Authentication strategy (IntegrationTokenAuth or IMSOAuthAuth).
            store_code: Store view code (default: "default").
            return_window_days: Default return window in days.
        """
        self.base_url = base_url.rstrip("/")
        self.auth_strategy = auth_strategy
        self.store_code = store_code
        self.return_window_days = return_window_days
        self._client: httpx.AsyncClient | None = None

    @property
    def platform_name(self) -> str:
        return "adobe_commerce"

    @property
    def api_base_url(self) -> str:
        """Get the REST API base URL with store code."""
        return f"{self.base_url}/rest/{self.store_code}/V1"

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.api_base_url,
                timeout=30.0,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

        # Close auth client if it has one (IMSOAuthAuth)
        if hasattr(self.auth_strategy, "close"):
            await self.auth_strategy.close()

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
    ) -> dict | list | None:
        """
        Make an authenticated API request to Adobe Commerce.

        Args:
            method: HTTP method.
            path: API path (relative to api_base_url).
            params: Query parameters.
            json_data: JSON body data.

        Returns:
            Parsed JSON response or None for 404.
        """
        headers = await self.auth_strategy.get_auth_headers()

        try:
            response = await self.client.request(
                method,
                path,
                headers=headers,
                params=params,
                json=json_data,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    def _build_search_criteria(
        self,
        filters: list[dict[str, Any]],
        page_size: int = 10,
        current_page: int = 1,
        sort_field: str | None = None,
        sort_direction: str = "DESC",
    ) -> dict[str, str]:
        """
        Build Adobe Commerce searchCriteria query params.

        Adobe Commerce uses a complex nested query parameter structure:
        searchCriteria[filterGroups][0][filters][0][field]=email
        searchCriteria[filterGroups][0][filters][0][value]=test@example.com
        searchCriteria[filterGroups][0][filters][0][conditionType]=eq

        Args:
            filters: List of filter dicts with field, value, condition_type.
            page_size: Number of results per page.
            current_page: Page number (1-indexed).
            sort_field: Field to sort by.
            sort_direction: Sort direction (ASC or DESC).

        Returns:
            Dictionary of query parameters.
        """
        params: dict[str, str] = {
            "searchCriteria[pageSize]": str(page_size),
            "searchCriteria[currentPage]": str(current_page),
        }

        # Add filters
        for i, f in enumerate(filters):
            prefix = f"searchCriteria[filterGroups][{i}][filters][0]"
            params[f"{prefix}[field]"] = f["field"]
            params[f"{prefix}[value]"] = f["value"]
            params[f"{prefix}[conditionType]"] = f.get("condition_type", "eq")

        # Add sorting
        if sort_field:
            params["searchCriteria[sortOrders][0][field]"] = sort_field
            params["searchCriteria[sortOrders][0][direction]"] = sort_direction

        return params

    async def get_order(self, order_id: str) -> OrderInfo | None:
        """
        Fetch order by Adobe Commerce entity_id.

        Args:
            order_id: The order entity_id.

        Returns:
            OrderInfo if found, None otherwise.
        """
        data = await self._request("GET", f"/orders/{order_id}")
        if not data or not isinstance(data, dict):
            return None

        # Fetch shipments for this order
        shipments = await self._get_order_shipments(order_id)

        return self._parse_order(data, shipments)

    async def get_order_by_number(self, order_number: str) -> OrderInfo | None:
        """
        Fetch order by increment_id (customer-facing order number).

        Args:
            order_number: The increment_id (e.g., "000000001" or "#000000001").

        Returns:
            OrderInfo if found, None otherwise.
        """
        # Clean the order number (remove # prefix if present)
        clean_number = order_number.lstrip("#")

        params = self._build_search_criteria(
            [
                {"field": "increment_id", "value": clean_number, "condition_type": "eq"},
            ]
        )

        data = await self._request("GET", "/orders", params=params)
        if not data or not isinstance(data, dict):
            return None

        items = data.get("items", [])
        if not items:
            return None

        order = items[0]
        order_id = str(order.get("entity_id", ""))

        # Fetch shipments
        shipments = await self._get_order_shipments(order_id)

        return self._parse_order(order, shipments)

    async def get_customer_orders(
        self,
        customer_email: str,
        limit: int = 10,
    ) -> list[OrderInfo]:
        """
        Fetch recent orders for a customer by email.

        Args:
            customer_email: Customer email address.
            limit: Maximum number of orders to return.

        Returns:
            List of OrderInfo objects.
        """
        params = self._build_search_criteria(
            [
                {"field": "customer_email", "value": customer_email, "condition_type": "eq"},
            ],
            page_size=limit,
            sort_field="created_at",
            sort_direction="DESC",
        )

        data = await self._request("GET", "/orders", params=params)
        if not data or not isinstance(data, dict):
            return []

        orders = []
        for order_data in data.get("items", []):
            order_id = str(order_data.get("entity_id", ""))
            shipments = await self._get_order_shipments(order_id)
            orders.append(self._parse_order(order_data, shipments))

        return orders

    async def get_tracking(self, order_id: str) -> list[TrackingInfo]:
        """
        Get tracking information for an order.

        Args:
            order_id: The order entity_id.

        Returns:
            List of TrackingInfo objects.
        """
        shipments = await self._get_order_shipments(order_id)
        return self._extract_tracking_from_shipments(shipments)

    async def _get_order_shipments(self, order_id: str) -> list[dict]:
        """
        Fetch shipments for an order.

        Args:
            order_id: The order entity_id.

        Returns:
            List of shipment data dictionaries.
        """
        params = self._build_search_criteria(
            [
                {"field": "order_id", "value": order_id, "condition_type": "eq"},
            ]
        )

        data = await self._request("GET", "/shipments", params=params)
        if not data or not isinstance(data, dict):
            return []

        return data.get("items", [])

    def _extract_tracking_from_shipments(
        self,
        shipments: list[dict],
    ) -> list[TrackingInfo]:
        """Extract tracking info from shipment data."""
        tracking_list: list[TrackingInfo] = []

        for shipment in shipments:
            for track in shipment.get("tracks", []):
                tracking_number = track.get("track_number", "")
                carrier_code = track.get("carrier_code", "")
                carrier = get_carrier_name(carrier_code)
                tracking_url = build_tracking_url(carrier_code, tracking_number)

                tracking_list.append(
                    TrackingInfo(
                        carrier=carrier,
                        tracking_number=tracking_number,
                        tracking_url=tracking_url,
                    )
                )

        return tracking_list

    async def health_check(self) -> bool:
        """
        Check if the Adobe Commerce connection is working.

        Uses the store config endpoint which requires minimal permissions.

        Returns:
            True if connection is healthy.
        """
        try:
            data = await self._request("GET", "/store/storeConfigs")
            return data is not None and isinstance(data, list) and len(data) > 0
        except (httpx.HTTPError, OSError):
            return False

    def _parse_order(
        self,
        order_data: dict,
        shipments: list[dict] | None = None,
    ) -> OrderInfo:
        """Parse Adobe Commerce order data into OrderInfo model."""
        shipments = shipments or []

        # Parse line items
        line_items = [
            LineItem(
                product_id=str(item.get("product_id", "")),
                variant_id=None,  # Adobe Commerce doesn't use variants like Shopify
                sku=item.get("sku"),
                name=item.get("name", ""),
                quantity=int(item.get("qty_ordered", 1)),
                price=float(item.get("price", 0)),
                currency=order_data.get("order_currency_code", "USD"),
            )
            for item in order_data.get("items", [])
        ]

        # Parse shipping address from extension_attributes
        shipping_address = self._parse_shipping_address(order_data)

        # Extract tracking from shipments
        tracking = self._extract_tracking_from_shipments(shipments)

        # Parse timestamps
        created_at = self._parse_datetime(order_data.get("created_at"))
        updated_at = self._parse_datetime(order_data.get("updated_at"))

        # Calculate return window
        return_window_ends = None
        if created_at:
            return_window_ends = created_at + timedelta(days=self.return_window_days)

        # Map status
        state = order_data.get("state", "")
        status = map_order_status(state, len(shipments))

        # Calculate fulfillment status
        total_items = sum(int(item.get("qty_ordered", 0)) for item in order_data.get("items", []))
        shipped_items = sum(int(item.get("qty_shipped", 0)) for item in order_data.get("items", []))
        fulfillment_status = map_fulfillment_status(total_items, shipped_items)

        # Parse refund info
        refund_amount = None
        total_refunded = float(order_data.get("total_refunded", 0) or 0)
        if total_refunded > 0:
            refund_amount = total_refunded

        # Customer info
        customer_name = ""
        if order_data.get("customer_firstname"):
            customer_name = f"{order_data.get('customer_firstname', '')} {order_data.get('customer_lastname', '')}".strip()

        return OrderInfo(
            order_id=str(order_data.get("entity_id", "")),
            platform="adobe_commerce",
            order_number=order_data.get("increment_id", str(order_data.get("entity_id", ""))),
            status=status,
            fulfillment_status=fulfillment_status,
            customer_email=order_data.get("customer_email", ""),
            customer_name=customer_name,
            line_items=line_items,
            subtotal=float(order_data.get("subtotal", 0)),
            shipping_cost=float(order_data.get("shipping_amount", 0)),
            tax=float(order_data.get("tax_amount", 0)),
            total=float(order_data.get("grand_total", 0)),
            currency=order_data.get("order_currency_code", "USD"),
            shipping_address=shipping_address,
            tracking=tracking,
            created_at=created_at,
            updated_at=updated_at,
            is_returnable=return_window_ends is not None and return_window_ends > datetime.now(UTC),
            return_window_ends=return_window_ends,
            refund_amount=refund_amount,
            raw_data=order_data,
        )

    def _parse_shipping_address(self, order_data: dict) -> ShippingAddress | None:
        """
        Parse shipping address from Adobe Commerce order data.

        Address is in extension_attributes.shipping_assignments[0].shipping.address
        """
        ext_attrs = order_data.get("extension_attributes", {})
        shipping_assignments = ext_attrs.get("shipping_assignments", [])

        if not shipping_assignments:
            return None

        shipping = shipping_assignments[0].get("shipping", {})
        addr = shipping.get("address")

        if not addr:
            return None

        # Street is an array in Adobe Commerce
        street = addr.get("street", [])
        address1 = street[0] if street else ""
        address2 = street[1] if len(street) > 1 else None

        return ShippingAddress(
            name=f"{addr.get('firstname', '')} {addr.get('lastname', '')}".strip(),
            address1=address1,
            address2=address2,
            city=addr.get("city", ""),
            province=addr.get("region", ""),
            country=addr.get("country_id", ""),
            zip_code=addr.get("postcode", ""),
            phone=addr.get("telephone"),
        )

    def _parse_datetime(self, dt_str: str | None) -> datetime | None:
        """Parse Adobe Commerce datetime string."""
        if not dt_str:
            return None
        try:
            # Adobe Commerce uses ISO 8601 format (YYYY-MM-DD HH:MM:SS)
            # Parse and make timezone-aware (assume UTC if no timezone)
            dt = datetime.fromisoformat(dt_str.replace(" ", "T"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except ValueError:
            return None

    # =========================================================================
    # Customer Profile Methods
    # =========================================================================

    async def get_customer_by_email(self, email: str) -> CustomerProfile | None:
        """
        Fetch customer profile by email address.

        Args:
            email: Customer email address.

        Returns:
            CustomerProfile if found, None otherwise.
        """
        # URL-encode the email for the search
        params = self._build_search_criteria(
            [
                {"field": "email", "value": email, "condition_type": "eq"},
            ]
        )

        data = await self._request("GET", "/customers/search", params=params)
        if not data or not isinstance(data, dict):
            return None

        items = data.get("items", [])
        if not items:
            return None

        return await self._build_customer_profile(items[0])

    async def get_customer_by_id(self, customer_id: str) -> CustomerProfile | None:
        """
        Fetch customer profile by Adobe Commerce customer ID.

        Args:
            customer_id: Customer entity_id.

        Returns:
            CustomerProfile if found, None otherwise.
        """
        data = await self._request("GET", f"/customers/{customer_id}")
        if not data or not isinstance(data, dict):
            return None

        return await self._build_customer_profile(data)

    async def _build_customer_profile(self, customer_data: dict) -> CustomerProfile:
        """Build CustomerProfile from Adobe Commerce customer data."""
        customer_id = str(customer_data.get("id", ""))
        email = customer_data.get("email", "")

        # Get order history to calculate lifetime value
        orders = await self.get_customer_orders(email, limit=100)
        total_orders = len(orders)
        lifetime_value = sum(order.total for order in orders)

        # Get first order date
        first_order_date = None
        if orders:
            oldest_order = min(orders, key=lambda o: o.created_at or datetime.max)
            first_order_date = oldest_order.created_at

        # Determine tier
        tier = self._determine_customer_tier(
            lifetime_value,
            total_orders,
            customer_data,
        )

        return CustomerProfile(
            customer_id=customer_id,
            email=email,
            name=f"{customer_data.get('firstname', '')} {customer_data.get('lastname', '')}".strip(),
            tier=tier,
            lifetime_value=lifetime_value,
            total_orders=total_orders,
            first_order_date=first_order_date,
            support_tickets_30d=0,  # Would come from support system
            complaints_90d=0,
            returns_90d=0,
            preferred_contact="email",
            language="en",  # Would come from store view
            is_vip=tier == CustomerTier.VIP,
            is_at_risk=False,
        )

    def _determine_customer_tier(
        self,
        lifetime_value: float,
        total_orders: int,
        customer_data: dict,
    ) -> CustomerTier:
        """Determine customer tier based on value and history."""
        # Check customer group (Adobe Commerce customer groups)
        group_id = customer_data.get("group_id", 0)
        # Common group IDs: 0=NOT LOGGED IN, 1=General, 2=Wholesale, 3=Retailer
        # VIP groups are typically custom (higher IDs)
        if group_id >= 4:  # Assume custom groups are VIP/Premium
            return CustomerTier.VIP

        # Tier by value
        if lifetime_value >= 1000 or total_orders >= 20:
            return CustomerTier.VIP
        if lifetime_value >= 500 or total_orders >= 10:
            return CustomerTier.PREMIUM
        if total_orders == 0:
            return CustomerTier.NEW

        return CustomerTier.STANDARD

    # =========================================================================
    # Order Context Methods
    # =========================================================================

    async def get_order_context(self, order_id: str) -> OrderContext | None:
        """
        Get enriched order context for policy decisions.

        Args:
            order_id: Adobe Commerce order entity_id.

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
            order_number: Customer-facing increment_id.

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
