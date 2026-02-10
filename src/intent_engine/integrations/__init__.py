"""Platform integrations for order data."""

from intent_engine.integrations.base import OrderInfo, PlatformConnector
from intent_engine.integrations.shopify import ShopifyConnector

__all__ = ["OrderInfo", "PlatformConnector", "ShopifyConnector"]
