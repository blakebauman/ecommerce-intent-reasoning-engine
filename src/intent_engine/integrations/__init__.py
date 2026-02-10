"""Platform integrations for order data."""

from intent_engine.integrations.adobe_commerce import (
    AdobeCommerceConnector,
    AdobeCommerceWebhookHandler,
    IMSOAuthAuth,
    IntegrationTokenAuth,
)
from intent_engine.integrations.base import OrderInfo, PlatformConnector
from intent_engine.integrations.bigcommerce import (
    BigCommerceConnector,
    BigCommerceWebhookHandler,
)
from intent_engine.integrations.shopify import ShopifyConnector
from intent_engine.integrations.woocommerce import (
    WooCommerceConnector,
    WooCommerceWebhookHandler,
)

__all__ = [
    # Base
    "OrderInfo",
    "PlatformConnector",
    # Shopify
    "ShopifyConnector",
    # Adobe Commerce
    "AdobeCommerceConnector",
    "AdobeCommerceWebhookHandler",
    "IntegrationTokenAuth",
    "IMSOAuthAuth",
    # WooCommerce
    "WooCommerceConnector",
    "WooCommerceWebhookHandler",
    # BigCommerce
    "BigCommerceConnector",
    "BigCommerceWebhookHandler",
]
