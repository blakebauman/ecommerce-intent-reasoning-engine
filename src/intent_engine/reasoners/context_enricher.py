"""Context enricher for pulling live order and customer data."""

import logging
from datetime import datetime

import redis.asyncio as redis

from intent_engine.integrations.base import PlatformConnector
from intent_engine.integrations.shopify import ShopifyConnector
from intent_engine.models.context import (
    CustomerProfile,
    EnrichedContext,
    OrderContext,
)
from intent_engine.models.request import IntentRequest

logger = logging.getLogger(__name__)


class ContextEnricher:
    """
    Enrich intent requests with live order and customer data.

    Pulls data from platform connectors (Shopify, etc.) and caches
    results in Redis to avoid repeated API calls.

    Cache TTL: 5 minutes (configurable)
    """

    DEFAULT_CACHE_TTL = 300  # 5 minutes

    def __init__(
        self,
        connector: PlatformConnector | None = None,
        redis_client: redis.Redis | None = None,
        cache_ttl: int = DEFAULT_CACHE_TTL,
    ) -> None:
        """
        Initialize the context enricher.

        Args:
            connector: Platform connector (Shopify, etc.).
            redis_client: Redis client for caching.
            cache_ttl: Cache TTL in seconds.
        """
        self.connector = connector
        self.redis_client = redis_client
        self.cache_ttl = cache_ttl

    async def enrich(self, request: IntentRequest) -> EnrichedContext:
        """
        Enrich an intent request with order and customer context.

        Args:
            request: The intent request to enrich.

        Returns:
            EnrichedContext with customer and order data.
        """
        data_sources: list[str] = []
        customer: CustomerProfile | None = None
        order: OrderContext | None = None
        recent_orders: list[OrderContext] = []

        # Skip enrichment if no connector
        if not self.connector:
            logger.debug("No connector configured, skipping enrichment")
            return EnrichedContext(data_sources=["none"])

        # Try to get customer profile
        customer_email = self._extract_customer_email(request)
        if customer_email:
            customer = await self._get_customer_profile(customer_email)
            if customer:
                data_sources.append(f"{self.connector.platform_name}:customer")

        # Try to get order context
        if request.order_ids:
            order = await self._get_order_context(request.order_ids[0])
            if order:
                data_sources.append(f"{self.connector.platform_name}:order")

                # If we got an order, use its customer email for profile
                if not customer and order.customer_email:
                    customer = await self._get_customer_profile(order.customer_email)
                    if customer:
                        data_sources.append(f"{self.connector.platform_name}:customer")

        # Get recent order history for context
        if customer_email or (customer and customer.email):
            email = customer_email or customer.email
            recent_orders = await self._get_recent_orders(email, limit=5)
            if recent_orders:
                data_sources.append(f"{self.connector.platform_name}:order_history")

        return EnrichedContext(
            customer=customer,
            order=order,
            recent_orders=recent_orders,
            enriched_at=datetime.utcnow(),
            data_sources=data_sources,
        )

    async def enrich_with_order(
        self,
        order_id: str,
        customer_email: str | None = None,
    ) -> EnrichedContext:
        """
        Enrich context for a specific order.

        Args:
            order_id: The order ID to look up.
            customer_email: Optional customer email.

        Returns:
            EnrichedContext with order and customer data.
        """
        data_sources: list[str] = []
        customer: CustomerProfile | None = None
        order: OrderContext | None = None
        recent_orders: list[OrderContext] = []

        if not self.connector:
            return EnrichedContext(data_sources=["none"])

        # Get order
        order = await self._get_order_context(order_id)
        if order:
            data_sources.append(f"{self.connector.platform_name}:order")

            # Get customer from order
            email = customer_email or order.customer_email
            if email:
                customer = await self._get_customer_profile(email)
                if customer:
                    data_sources.append(f"{self.connector.platform_name}:customer")

                recent_orders = await self._get_recent_orders(email, limit=5)
                if recent_orders:
                    data_sources.append(f"{self.connector.platform_name}:order_history")

        return EnrichedContext(
            customer=customer,
            order=order,
            recent_orders=recent_orders,
            enriched_at=datetime.utcnow(),
            data_sources=data_sources,
        )

    def _extract_customer_email(self, request: IntentRequest) -> str | None:
        """Extract customer email from request metadata."""
        # Check raw_metadata
        if email := request.raw_metadata.get("from_email"):
            return email
        if email := request.raw_metadata.get("customer_email"):
            return email

        # Check for email in the customer_id if it looks like an email
        if request.customer_id and "@" in request.customer_id:
            return request.customer_id

        return None

    async def _get_customer_profile(self, email: str) -> CustomerProfile | None:
        """Get customer profile with caching."""
        cache_key = f"customer:{email}"

        # Try cache first
        if self.redis_client:
            try:
                cached = await self.redis_client.get(cache_key)
                if cached:
                    logger.debug(f"Customer profile cache hit: {email}")
                    return CustomerProfile.model_validate_json(cached)
            except Exception as e:
                logger.warning(f"Redis cache read failed: {e}")

        # Fetch from connector
        if not self.connector:
            return None

        try:
            if isinstance(self.connector, ShopifyConnector):
                profile = await self.connector.get_customer_by_email(email)
            else:
                # Generic connector doesn't have customer methods
                return None

            if profile and self.redis_client:
                try:
                    await self.redis_client.setex(
                        cache_key,
                        self.cache_ttl,
                        profile.model_dump_json(),
                    )
                except Exception as e:
                    logger.warning(f"Redis cache write failed: {e}")

            return profile

        except Exception as e:
            logger.error(f"Failed to fetch customer profile: {e}")
            return None

    async def _get_order_context(self, order_id: str) -> OrderContext | None:
        """Get order context with caching."""
        cache_key = f"order:{order_id}"

        # Try cache first
        if self.redis_client:
            try:
                cached = await self.redis_client.get(cache_key)
                if cached:
                    logger.debug(f"Order context cache hit: {order_id}")
                    return OrderContext.model_validate_json(cached)
            except Exception as e:
                logger.warning(f"Redis cache read failed: {e}")

        # Fetch from connector
        if not self.connector:
            return None

        try:
            if isinstance(self.connector, ShopifyConnector):
                # Try by order number first (customer-facing format)
                order = await self.connector.get_order_context_by_number(order_id)
                if not order:
                    # Try by internal ID
                    order = await self.connector.get_order_context(order_id)
            else:
                # Generic connector
                order_info = await self.connector.get_order_by_number(order_id)
                if not order_info:
                    order_info = await self.connector.get_order(order_id)

                if order_info:
                    # Build basic OrderContext from OrderInfo
                    order = self._order_info_to_context(order_info)
                else:
                    order = None

            if order and self.redis_client:
                try:
                    await self.redis_client.setex(
                        cache_key,
                        self.cache_ttl,
                        order.model_dump_json(),
                    )
                except Exception as e:
                    logger.warning(f"Redis cache write failed: {e}")

            return order

        except Exception as e:
            logger.error(f"Failed to fetch order context: {e}")
            return None

    async def _get_recent_orders(
        self,
        email: str,
        limit: int = 5,
    ) -> list[OrderContext]:
        """Get recent orders for a customer."""
        cache_key = f"orders:{email}:{limit}"

        # Try cache first
        if self.redis_client:
            try:
                cached = await self.redis_client.get(cache_key)
                if cached:
                    import json
                    orders_data = json.loads(cached)
                    return [OrderContext.model_validate(o) for o in orders_data]
            except Exception as e:
                logger.warning(f"Redis cache read failed: {e}")

        # Fetch from connector
        if not self.connector:
            return []

        try:
            if isinstance(self.connector, ShopifyConnector):
                orders = await self.connector.get_customer_order_history(email, limit)
            else:
                order_infos = await self.connector.get_customer_orders(email, limit)
                orders = [self._order_info_to_context(o) for o in order_infos]

            if orders and self.redis_client:
                try:
                    import json
                    orders_json = json.dumps([o.model_dump() for o in orders], default=str)
                    await self.redis_client.setex(
                        cache_key,
                        self.cache_ttl,
                        orders_json,
                    )
                except Exception as e:
                    logger.warning(f"Redis cache write failed: {e}")

            return orders

        except Exception as e:
            logger.error(f"Failed to fetch order history: {e}")
            return []

    def _order_info_to_context(self, order_info) -> OrderContext:
        """Convert OrderInfo to OrderContext for non-Shopify connectors."""
        from intent_engine.models.context import ProductContext, ReturnEligibility

        items = [
            ProductContext(
                product_id=item.product_id,
                sku=item.sku,
                name=item.name,
                price=item.price,
                currency=item.currency,
            )
            for item in order_info.line_items
        ]

        return_eligibility = (
            ReturnEligibility.ELIGIBLE
            if order_info.is_returnable
            else ReturnEligibility.EXPIRED
        )

        tracking_number = None
        carrier = None
        tracking_url = None
        if order_info.tracking:
            tracking_number = order_info.tracking[0].tracking_number
            carrier = order_info.tracking[0].carrier
            tracking_url = order_info.tracking[0].tracking_url

        return OrderContext(
            order_id=order_info.order_id,
            order_number=order_info.order_number,
            platform=order_info.platform,
            status=order_info.status.value,
            fulfillment_status=order_info.fulfillment_status.value,
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
            tracking_number=tracking_number,
            carrier=carrier,
            tracking_url=tracking_url,
            return_eligibility=return_eligibility,
            return_window_ends=order_info.return_window_ends,
            is_within_return_window=order_info.is_returnable,
        )
