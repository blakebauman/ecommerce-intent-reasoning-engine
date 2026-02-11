"""Platform integrations for order data and catalog."""

from intent_engine.integrations.adobe_commerce import (
    AdobeCommerceConnector,
    AdobeCommerceOptimizerCatalogProvider,
    AdobeCommerceWebhookHandler,
    IMSOAuthAuth,
    IntegrationTokenAuth,
)
from intent_engine.integrations.base import (
    CatalogProvider,
    OrderInfo,
    PlatformConnector,
)
from intent_engine.integrations.bigcommerce import (
    BigCommerceConnector,
    BigCommerceWebhookHandler,
)
from intent_engine.integrations.shopify import ShopifyCatalogProvider, ShopifyConnector
from intent_engine.integrations.woocommerce import (
    WooCommerceConnector,
    WooCommerceWebhookHandler,
)

__all__ = [
    # Base
    "CatalogProvider",
    "OrderInfo",
    "PlatformConnector",
    # Shopify
    "ShopifyCatalogProvider",
    "ShopifyConnector",
    # Adobe Commerce
    "AdobeCommerceConnector",
    "AdobeCommerceOptimizerCatalogProvider",
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
