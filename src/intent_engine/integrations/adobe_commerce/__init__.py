"""Adobe Commerce (Magento) integration module."""

from intent_engine.integrations.adobe_commerce.auth import (
    AdobeCommerceAuthStrategy,
    IMSOAuthAuth,
    IntegrationTokenAuth,
)
from intent_engine.integrations.adobe_commerce.catalog import (
    AdobeCommerceOptimizerCatalogProvider,
)
from intent_engine.integrations.adobe_commerce.connector import AdobeCommerceConnector
from intent_engine.integrations.adobe_commerce.webhooks import AdobeCommerceWebhookHandler

__all__ = [
    "AdobeCommerceConnector",
    "AdobeCommerceOptimizerCatalogProvider",
    "AdobeCommerceAuthStrategy",
    "IntegrationTokenAuth",
    "IMSOAuthAuth",
    "AdobeCommerceWebhookHandler",
]
