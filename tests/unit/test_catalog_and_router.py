"""Unit tests for catalog providers, lifecycle router, and catalog agent wiring."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "spaCy/import chain not compatible with Python 3.14+",
        allow_module_level=True,
    )

from intent_engine.agents.catalog_agent import get_catalog_provider_from_settings
from intent_engine.agents.models import CustomerMessage
from intent_engine.agents.router import LifecycleRouter, PRE_PURCHASE_CATEGORIES
from intent_engine.models.catalog import CatalogProduct, InventoryInfo


class TestCatalogModels:
    """Tests for catalog Pydantic models."""

    def test_catalog_product_minimal(self) -> None:
        """CatalogProduct with required fields only."""
        p = CatalogProduct(
            product_id="p1",
            name="Widget",
            price=29.99,
        )
        assert p.product_id == "p1"
        assert p.name == "Widget"
        assert p.price == 29.99
        assert p.currency == "USD"
        assert p.is_in_stock is True
        assert p.sku is None

    def test_catalog_product_full(self) -> None:
        """CatalogProduct with optional fields."""
        p = CatalogProduct(
            product_id="p1",
            name="Widget",
            price=29.99,
            sku="WIDGET-01",
            category="Gadgets",
            is_in_stock=True,
            inventory_quantity=42,
            image_url="https://example.com/img.jpg",
        )
        assert p.sku == "WIDGET-01"
        assert p.category == "Gadgets"
        assert p.inventory_quantity == 42

    def test_inventory_info(self) -> None:
        """InventoryInfo model."""
        inv = InventoryInfo(
            product_id="p1",
            sku="SKU-01",
            quantity_available=10,
            is_in_stock=True,
        )
        assert inv.product_id == "p1"
        assert inv.quantity_available == 10
        assert inv.is_in_stock is True


class TestGetCatalogProviderFromSettings:
    """Tests for get_catalog_provider_from_settings."""

    def test_returns_none_when_no_config(self) -> None:
        """When neither Shopify nor Adobe Optimizer is configured, returns None."""
        settings = MagicMock()
        settings.shopify_store_domain = ""
        settings.shopify_access_token = ""
        settings.adobe_commerce_optimizer_tenant_id = ""
        settings.adobe_commerce_optimizer_catalog_view_id = ""
        provider = get_catalog_provider_from_settings(settings=settings)
        assert provider is None

    def test_returns_shopify_when_configured(self) -> None:
        """When Shopify is configured, returns ShopifyCatalogProvider."""
        settings = MagicMock()
        settings.shopify_store_domain = "store.myshopify.com"
        settings.shopify_access_token = "token"
        settings.adobe_commerce_optimizer_tenant_id = ""
        settings.adobe_commerce_optimizer_catalog_view_id = ""
        provider = get_catalog_provider_from_settings(settings=settings)
        assert provider is not None
        assert provider.platform_name == "shopify"

    def test_returns_adobe_optimizer_when_configured_no_shopify(self) -> None:
        """When only Adobe Optimizer is configured, returns Adobe provider."""
        settings = MagicMock()
        settings.shopify_store_domain = ""
        settings.shopify_access_token = ""
        settings.adobe_commerce_optimizer_tenant_id = "tenant-123"
        settings.adobe_commerce_optimizer_catalog_view_id = "view-456"
        settings.adobe_commerce_optimizer_locale = "en_US"
        settings.adobe_commerce_optimizer_region = "na1"
        settings.adobe_commerce_optimizer_environment = "sandbox"
        settings.adobe_commerce_optimizer_price_book_id = ""
        provider = get_catalog_provider_from_settings(settings=settings)
        assert provider is not None
        assert provider.platform_name == "adobe_commerce"


class TestShopifyCatalogProvider:
    """Tests for ShopifyCatalogProvider with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_search_products_returns_mapped_catalog_products(self) -> None:
        """search_products maps Shopify API response to CatalogProduct list."""
        from intent_engine.integrations.shopify.catalog import ShopifyCatalogProvider

        provider = ShopifyCatalogProvider(
            store_domain="store.myshopify.com",
            access_token="token",
        )
        mock_response = {
            "products": [
                {
                    "id": 12345,
                    "title": "Blue Widget",
                    "body_html": "<p>Description</p>",
                    "product_type": "Gadgets",
                    "variants": [
                        {
                            "id": 67890,
                            "price": "29.99",
                            "inventory_quantity": 5,
                            "sku": "WID-BLU",
                        }
                    ],
                }
            ]
        }
        with patch.object(provider, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            products = await provider.search_products("widget", limit=20)
        assert len(products) == 1
        p = products[0]
        assert p.product_id == "12345"
        assert p.name == "Blue Widget"
        assert p.price == 29.99
        assert p.sku == "WID-BLU"
        assert p.category == "Gadgets"
        assert p.is_in_stock is True
        assert p.inventory_quantity == 5
        mock_req.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_products_empty_when_no_results(self) -> None:
        """search_products returns empty list when API returns no products."""
        from intent_engine.integrations.shopify.catalog import ShopifyCatalogProvider

        provider = ShopifyCatalogProvider(
            store_domain="store.myshopify.com",
            access_token="token",
        )
        with patch.object(provider, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {"products": []}
            products = await provider.search_products("nonexistent")
        assert products == []

    @pytest.mark.asyncio
    async def test_get_product_by_id(self) -> None:
        """get_product by product_id returns mapped CatalogProduct."""
        from intent_engine.integrations.shopify.catalog import ShopifyCatalogProvider

        provider = ShopifyCatalogProvider(
            store_domain="store.myshopify.com",
            access_token="token",
        )
        mock_product = {
            "id": 999,
            "title": "Single Item",
            "body_html": None,
            "variants": [{"price": "19.99", "inventory_quantity": 0, "sku": "SINGLE"}],
        }
        with patch.object(provider, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {"product": mock_product}
            p = await provider.get_product(product_id="999")
        assert p is not None
        assert p.product_id == "999"
        assert p.name == "Single Item"
        assert p.price == 19.99
        assert p.is_in_stock is False

    @pytest.mark.asyncio
    async def test_get_inventory_from_product(self) -> None:
        """get_inventory returns InventoryInfo from product data."""
        from intent_engine.integrations.shopify.catalog import ShopifyCatalogProvider

        provider = ShopifyCatalogProvider(
            store_domain="store.myshopify.com",
            access_token="token",
        )
        mock_product = {
            "id": 100,
            "title": "Item",
            "body_html": None,
            "variants": [
                {"id": 200, "price": "10", "inventory_quantity": 3, "sku": "SKU-X"}
            ],
        }
        with patch.object(provider, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {"product": mock_product}
            inv = await provider.get_inventory(product_id="100")
        assert inv is not None
        assert inv.product_id == "100"
        assert inv.sku == "SKU-X"
        assert inv.quantity_available == 3
        assert inv.is_in_stock is True


class TestAdobeCommerceOptimizerCatalogProvider:
    """Tests for AdobeCommerceOptimizerCatalogProvider with mocked GraphQL."""

    @pytest.mark.asyncio
    async def test_search_products_returns_mapped_catalog_products(self) -> None:
        """search_products maps GraphQL productSearch to CatalogProduct list."""
        from intent_engine.integrations.adobe_commerce.catalog import (
            AdobeCommerceOptimizerCatalogProvider,
        )

        provider = AdobeCommerceOptimizerCatalogProvider(
            tenant_id="tenant-1",
            catalog_view_id="view-1",
        )
        mock_data = {
            "productSearch": {
                "items": [
                    {
                        "productView": {
                            "id": "p-abc",
                            "sku": "ADOBE-SKU-1",
                            "name": "Adobe Widget",
                            "inStock": True,
                            "url": "https://store.com/widget",
                            "price": {
                                "final": {"amount": {"value": 39.99, "currency": "USD"}},
                                "regular": {"amount": {"value": 49.99, "currency": "USD"}},
                            },
                        }
                    }
                ]
            }
        }
        with patch.object(provider, "_graphql", new_callable=AsyncMock) as mock_graphql:
            mock_graphql.return_value = mock_data
            products = await provider.search_products("widget", limit=20)
        assert len(products) == 1
        p = products[0]
        assert p.product_id == "p-abc"
        assert p.name == "Adobe Widget"
        assert p.sku == "ADOBE-SKU-1"
        assert p.price == 39.99
        assert p.compare_at_price == 49.99
        assert p.is_in_stock is True
        assert p.url == "https://store.com/widget"

    @pytest.mark.asyncio
    async def test_get_product_by_sku(self) -> None:
        """get_product by SKU returns mapped CatalogProduct."""
        from intent_engine.integrations.adobe_commerce.catalog import (
            AdobeCommerceOptimizerCatalogProvider,
        )

        provider = AdobeCommerceOptimizerCatalogProvider(
            tenant_id="tenant-1",
            catalog_view_id="view-1",
        )
        mock_product = {
            "id": "p-xyz",
            "sku": "LOOKUP-SKU",
            "name": "Lookup Product",
            "inStock": False,
            "price": {
                "final": {"amount": {"value": 15.0, "currency": "USD"}},
            },
        }
        with patch.object(provider, "_graphql", new_callable=AsyncMock) as mock_graphql:
            mock_graphql.return_value = {"products": [mock_product]}
            p = await provider.get_product(sku="LOOKUP-SKU")
        assert p is not None
        assert p.product_id == "p-xyz"
        assert p.sku == "LOOKUP-SKU"
        assert p.price == 15.0
        assert p.is_in_stock is False


class TestLifecycleRouterCategories:
    """Tests for router pre-purchase category set."""

    def test_pre_purchase_categories_include_product_and_discovery(self) -> None:
        """PRODUCT_INQUIRY and DISCOVERY route to pre-purchase."""
        assert "PRODUCT_INQUIRY" in PRE_PURCHASE_CATEGORIES
        assert "DISCOVERY" in PRE_PURCHASE_CATEGORIES
        assert "ORDER_STATUS" not in PRE_PURCHASE_CATEGORIES


class TestLifecycleRouter:
    """Tests for LifecycleRouter routing behavior."""

    @pytest.fixture
    def mock_engine(self) -> MagicMock:
        engine = MagicMock()
        engine.resolve = AsyncMock()
        return engine

    @pytest.fixture
    def mock_customer_service_agent(self) -> MagicMock:
        agent = MagicMock()
        agent.process_message = AsyncMock()
        return agent

    @pytest.mark.asyncio
    async def test_routes_to_post_purchase_for_order_status(
        self,
        mock_engine: MagicMock,
        mock_customer_service_agent: MagicMock,
    ) -> None:
        """When primary intent is ORDER_STATUS, router calls customer service agent."""
        from intent_engine.models.intent import IntentConfidence, ResolvedIntent
        from intent_engine.models.response import ReasoningResult

        mock_engine.resolve.return_value = ReasoningResult(
            request_id="req-1",
            resolved_intents=[
                ResolvedIntent(
                    category="ORDER_STATUS",
                    intent="WISMO",
                    confidence=0.92,
                    confidence_tier=IntentConfidence.HIGH,
                    evidence=["where is my order"],
                )
            ],
            is_compound=False,
            entities=[],
            confidence_summary=0.92,
            path_taken="fast_path",
        )
        mock_customer_service_agent.process_message.return_value = MagicMock(
            message_id="msg-1",
            conversation_id=None,
            intents=[],
            is_compound=False,
            entities=[],
            response_text="Your order is in transit.",
            response_tone="helpful",
            actions=[],
            primary_action=None,
            order_context=None,
            customer_context=None,
            requires_human=False,
            human_handoff_reason=None,
            confidence=0.92,
            processing_time_ms=100,
        )

        mock_pre_purchase_agent = MagicMock()
        mock_pre_purchase_agent.run = AsyncMock()
        with patch(
            "intent_engine.agents.router.get_pre_purchase_agent",
            return_value=mock_pre_purchase_agent,
        ):
            router = LifecycleRouter(
                intent_engine=mock_engine,
                customer_service_agent=mock_customer_service_agent,
                catalog_provider=None,
            )
        message = CustomerMessage(message_id="msg-1", text="Where is my order?")
        response = await router.process_message(message)

        assert response.response_text == "Your order is in transit."
        mock_customer_service_agent.process_message.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_routes_to_pre_purchase_for_product_inquiry(
        self,
        mock_engine: MagicMock,
        mock_customer_service_agent: MagicMock,
    ) -> None:
        """When primary intent is PRODUCT_INQUIRY, router calls pre-purchase agent."""
        from intent_engine.agents.pre_purchase_agent import PrePurchaseOutput
        from intent_engine.models.intent import IntentConfidence, ResolvedIntent
        from intent_engine.models.response import ReasoningResult

        mock_engine.resolve.return_value = ReasoningResult(
            request_id="req-1",
            resolved_intents=[
                ResolvedIntent(
                    category="PRODUCT_INQUIRY",
                    intent="STOCK",
                    confidence=0.88,
                    confidence_tier=IntentConfidence.HIGH,
                    evidence=["in stock"],
                )
            ],
            is_compound=False,
            entities=[],
            confidence_summary=0.88,
            path_taken="fast_path",
        )

        mock_pre_purchase_output = PrePurchaseOutput(
            response_text="We have that item in stock.",
            products=[],
            primary_intent="PRODUCT_INQUIRY.STOCK",
            confidence=0.88,
        )
        mock_pre_purchase_agent = MagicMock()
        mock_pre_purchase_agent.run = AsyncMock(
            return_value=MagicMock(output=mock_pre_purchase_output)
        )

        with patch(
            "intent_engine.agents.router.get_pre_purchase_agent",
            return_value=mock_pre_purchase_agent,
        ):
            router = LifecycleRouter(
                intent_engine=mock_engine,
                customer_service_agent=mock_customer_service_agent,
                catalog_provider=None,
            )
        message = CustomerMessage(message_id="msg-1", text="Is the blue widget in stock?")
        response = await router.process_message(message)

        assert response.response_text == "We have that item in stock."
        assert response.confidence == 0.88
        mock_pre_purchase_agent.run.assert_called_once()
        mock_customer_service_agent.process_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_to_post_purchase_if_pre_purchase_raises(
        self,
        mock_engine: MagicMock,
        mock_customer_service_agent: MagicMock,
    ) -> None:
        """When pre-purchase agent raises, router falls back to customer service agent."""
        from intent_engine.models.intent import IntentConfidence, ResolvedIntent
        from intent_engine.models.response import ReasoningResult

        mock_engine.resolve.return_value = ReasoningResult(
            request_id="req-1",
            resolved_intents=[
                ResolvedIntent(
                    category="PRODUCT_INQUIRY",
                    intent="STOCK",
                    confidence=0.88,
                    confidence_tier=IntentConfidence.HIGH,
                    evidence=[],
                )
            ],
            is_compound=False,
            entities=[],
            confidence_summary=0.88,
            path_taken="fast_path",
        )
        mock_customer_service_agent.process_message.return_value = MagicMock(
            message_id="msg-1",
            conversation_id=None,
            intents=[],
            is_compound=False,
            entities=[],
            response_text="Let me help with that.",
            response_tone="helpful",
            actions=[],
            primary_action=None,
            order_context=None,
            customer_context=None,
            requires_human=False,
            human_handoff_reason=None,
            confidence=0.5,
            processing_time_ms=50,
        )

        mock_pre_purchase_agent = MagicMock()
        mock_pre_purchase_agent.run = AsyncMock(side_effect=RuntimeError("Catalog unavailable"))

        with patch(
            "intent_engine.agents.router.get_pre_purchase_agent",
            return_value=mock_pre_purchase_agent,
        ):
            router = LifecycleRouter(
                intent_engine=mock_engine,
                customer_service_agent=mock_customer_service_agent,
                catalog_provider=None,
            )
        message = CustomerMessage(message_id="msg-1", text="Is the widget in stock?")
        response = await router.process_message(message)

        assert response.response_text == "Let me help with that."
        mock_customer_service_agent.process_message.assert_called_once_with(message)
