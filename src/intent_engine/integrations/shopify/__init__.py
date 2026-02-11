"""Shopify integration: connector, catalog, mapping, and webhooks."""

from intent_engine.integrations.shopify.catalog import ShopifyCatalogProvider
from intent_engine.integrations.shopify.connector import ShopifyConnector

__all__ = ["ShopifyCatalogProvider", "ShopifyConnector"]
