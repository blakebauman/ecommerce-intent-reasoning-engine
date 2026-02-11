"""Product catalog agent: tools over CatalogProvider for search and product details."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings

from intent_engine.integrations.base import CatalogProvider
from intent_engine.models.catalog import CatalogProduct, InventoryInfo

if TYPE_CHECKING:
    from intent_engine.config import Settings


@dataclass
class CatalogAgentDeps:
    """Dependencies for the catalog agent."""

    catalog: CatalogProvider


class CatalogAgentOutput(BaseModel):
    """Structured output from the catalog agent."""

    products: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of product summaries (id, name, price, is_in_stock, etc.)",
    )
    reply_snippet: str = Field(
        default="",
        description="Short customer-facing summary or answer based on catalog results",
    )
    total_found: int = Field(default=0, ge=0, description="Total number of products found")


CATALOG_AGENT_SYSTEM_PROMPT = """You are a product catalog assistant for an eCommerce store.
You have tools to search products, get product details, and check inventory.
Use the tools to answer customer questions about products, availability, and details.
When returning results, provide a brief, helpful reply_snippet that a customer would see (1-3 sentences).
Include the most relevant product info in the reply_snippet (e.g., name, price, in-stock status)."""


def get_catalog_agent(
    model_name: str = "claude-sonnet-4-5",
    model: Any = None,
) -> Agent[CatalogAgentDeps, CatalogAgentOutput]:
    """
    Create the ProductCatalogAgent with catalog tools.

    Args:
        model_name: Anthropic model for the agent.
        model: Optional model instance for testing.

    Returns:
        Configured Agent with search_products, get_product, get_inventory tools.
    """
    model_or_id = model if model is not None else f"anthropic:{model_name}"

    agent: Agent[CatalogAgentDeps, CatalogAgentOutput] = Agent(
        model_or_id,
        deps_type=CatalogAgentDeps,
        output_type=CatalogAgentOutput,
        system_prompt=CATALOG_AGENT_SYSTEM_PROMPT,
        model_settings=ModelSettings(temperature=0.2, max_tokens=1024),
    )

    @agent.tool
    async def search_products(
        ctx: RunContext[CatalogAgentDeps],
        query: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search the product catalog by text query and optional category. Returns product summaries."""
        catalog = ctx.deps.catalog
        products = await catalog.search_products(query, category=category, limit=limit)
        return [_product_to_dict(p) for p in products]

    @agent.tool
    async def get_product(
        ctx: RunContext[CatalogAgentDeps],
        product_id: str | None = None,
        sku: str | None = None,
    ) -> dict[str, Any] | None:
        """Get full details for a single product by product_id or SKU. Returns None if not found."""
        if not product_id and not sku:
            return None
        catalog = ctx.deps.catalog
        product = await catalog.get_product(product_id=product_id, sku=sku)
        return _product_to_dict(product) if product else None

    @agent.tool
    async def get_inventory(
        ctx: RunContext[CatalogAgentDeps],
        product_id: str | None = None,
        sku: str | None = None,
    ) -> dict[str, Any] | None:
        """Check inventory for a product or variant by product_id or SKU. Returns quantity and in-stock status."""
        if not product_id and not sku:
            return None
        catalog = ctx.deps.catalog
        inv = await catalog.get_inventory(product_id=product_id, sku=sku)
        if not inv:
            return None
        return {
            "product_id": inv.product_id,
            "variant_id": inv.variant_id,
            "sku": inv.sku,
            "quantity_available": inv.quantity_available,
            "is_in_stock": inv.is_in_stock,
            "restock_date": inv.restock_date.isoformat() if inv.restock_date else None,
        }

    @agent.tool
    async def get_products_by_category(
        ctx: RunContext[CatalogAgentDeps],
        category_name: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get products in a category (e.g., 'Electronics', 'Best Sellers')."""
        catalog = ctx.deps.catalog
        products = await catalog.get_products_by_category(
            category_name=category_name,
            limit=limit,
        )
        return [_product_to_dict(p) for p in products]

    @agent.tool
    async def get_categories(ctx: RunContext[CatalogAgentDeps]) -> list[dict[str, Any]]:
        """List available product categories/collections for discovery."""
        catalog = ctx.deps.catalog
        categories = await catalog.get_categories()
        return [
            {
                "category_id": c.category_id,
                "name": c.name,
                "handle": c.handle,
                "product_count": c.product_count,
            }
            for c in categories
        ]

    return agent


def _product_to_dict(p: CatalogProduct) -> dict[str, Any]:
    """Convert CatalogProduct to a JSON-serializable dict for tool return."""
    return {
        "product_id": p.product_id,
        "name": p.name,
        "description_plain": p.description_plain,
        "category": p.category,
        "sku": p.sku,
        "price": p.price,
        "compare_at_price": p.compare_at_price,
        "currency": p.currency,
        "is_in_stock": p.is_in_stock,
        "inventory_quantity": p.inventory_quantity,
        "image_url": p.image_url,
        "url": p.url,
    }


def get_catalog_provider_from_settings(settings: "Settings | None" = None) -> CatalogProvider | None:
    """
    Build a CatalogProvider from application settings.

    Returns:
        ShopifyCatalogProvider if Shopify credentials are set;
        AdobeCommerceOptimizerCatalogProvider if Adobe Commerce Optimizer is configured;
        else None.
    """
    if settings is None:
        from intent_engine.config import get_settings
        settings = get_settings()
    if settings.shopify_store_domain and settings.shopify_access_token:
        from intent_engine.integrations.shopify import ShopifyCatalogProvider
        return ShopifyCatalogProvider(
            store_domain=settings.shopify_store_domain,
            access_token=settings.shopify_access_token,
        )
    if (
        settings.adobe_commerce_optimizer_tenant_id
        and settings.adobe_commerce_optimizer_catalog_view_id
    ):
        from intent_engine.integrations.adobe_commerce.catalog import (
            AdobeCommerceOptimizerCatalogProvider,
        )
        return AdobeCommerceOptimizerCatalogProvider(
            tenant_id=settings.adobe_commerce_optimizer_tenant_id,
            catalog_view_id=settings.adobe_commerce_optimizer_catalog_view_id,
            locale=settings.adobe_commerce_optimizer_locale,
            region=settings.adobe_commerce_optimizer_region,
            environment=settings.adobe_commerce_optimizer_environment,
            price_book_id=settings.adobe_commerce_optimizer_price_book_id or None,
        )
    return None
