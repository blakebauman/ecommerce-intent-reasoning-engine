"""Platform integrations for order data."""

from intent_engine.integrations.adobe_commerce import (
    AdobeCommerceConnector,
    AdobeCommerceWebhookHandler,
    IMSOAuthAuth,
    IntegrationTokenAuth,
)
from intent_engine.integrations.base import OrderInfo, PlatformConnector
from intent_engine.integrations.shopify import ShopifyConnector

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
]
