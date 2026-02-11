"""Catalog models for product and inventory data."""

from datetime import datetime

from pydantic import BaseModel, Field


class InventoryInfo(BaseModel):
    """Inventory availability for a product or variant."""

    product_id: str = Field(description="Platform product ID")
    variant_id: str | None = Field(default=None, description="Variant ID if variant-level")
    sku: str | None = Field(default=None, description="SKU")
    quantity_available: int = Field(default=0, ge=0, description="Available quantity")
    is_in_stock: bool = Field(default=True, description="Whether item is in stock")
    restock_date: datetime | None = Field(default=None, description="Expected restock date if known")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "product_id": "gid://shopify/Product/123",
                    "variant_id": "gid://shopify/ProductVariant/456",
                    "sku": "SKU-001",
                    "quantity_available": 42,
                    "is_in_stock": True,
                }
            ]
        }
    }


class CatalogProduct(BaseModel):
    """
    Unified product information from catalog (not from order line items).

    Returned by CatalogProvider search_products, get_product, get_products_by_category.
    """

    product_id: str = Field(description="Platform product ID")
    name: str = Field(description="Product title")
    description: str | None = Field(default=None, description="HTML or plain description")
    description_plain: str | None = Field(default=None, description="Plain-text description snippet")

    # Categorization
    category: str | None = Field(default=None, description="Primary category or product type")
    categories: list[str] = Field(default_factory=list, description="All categories/collections")
    vendor: str | None = Field(default=None)

    # Identifiers
    sku: str | None = Field(default=None, description="Primary or first variant SKU")
    variant_ids: list[str] = Field(default_factory=list, description="Variant IDs")

    # Pricing (primary or first variant)
    price: float = Field(ge=0, description="Price in currency units")
    compare_at_price: float | None = Field(default=None, ge=0, description="Compare-at price if on sale")
    currency: str = Field(default="USD")

    # Availability
    is_in_stock: bool = Field(default=True)
    inventory_quantity: int | None = Field(default=None, ge=0)
    restock_date: datetime | None = Field(default=None)

    # Display
    image_url: str | None = Field(default=None)
    url: str | None = Field(default=None, description="Storefront URL if available")

    # Timestamps
    created_at: datetime | None = None
    updated_at: datetime | None = None

    # Raw platform data for debugging
    raw_data: dict[str, object] | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "product_id": "12345",
                    "name": "Example Widget",
                    "category": "Widgets",
                    "sku": "WIDGET-01",
                    "price": 29.99,
                    "currency": "USD",
                    "is_in_stock": True,
                }
            ]
        }
    }


class CatalogCategory(BaseModel):
    """Category or collection for discovery."""

    category_id: str = Field(description="Platform category/collection ID")
    name: str = Field(description="Category name")
    handle: str | None = Field(default=None, description="URL handle")
    product_count: int | None = Field(default=None, ge=0)
    parent_id: str | None = Field(default=None)

    model_config = {
        "json_schema_extra": {
            "examples": [{"category_id": "cat-1", "name": "Electronics", "handle": "electronics"}]
        }
    }
