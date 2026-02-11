"""Adobe Commerce Optimizer (Merchandising Services) catalog provider.

Uses the Merchandising API GraphQL endpoint for product search and details.
See https://developer.adobe.com/commerce/services/optimizer/
"""

import re
from typing import Any

import httpx

from intent_engine.integrations.base import CatalogProvider
from intent_engine.models.catalog import (
    CatalogCategory,
    CatalogProduct,
    InventoryInfo,
)


# GraphQL: productSearch for search; products(skus) for get by SKU
QUERY_PRODUCT_SEARCH = """
query ProductSearch($phrase: String!, $pageSize: Int) {
  productSearch(phrase: $phrase, page_size: $pageSize) {
    items {
      productView {
        id
        sku
        name
        description
        shortDescription
        url
        inStock
        lowStock
        images(roles: []) {
          url
          roles
        }
        attributes(roles: []) {
          name
          label
          value
        }
        ... on SimpleProductView {
          price {
            regular { amount { value currency } }
            final { amount { value currency } }
          }
        }
        ... on ComplexProductView {
          priceRange {
            minimum { final { amount { value currency } } }
            maximum { final { amount { value currency } } }
          }
        }
      }
    }
  }
}
"""

QUERY_PRODUCTS_BY_SKU = """
query ProductsBySku($skus: [String!]!) {
  products(skus: $skus) {
    id
    sku
    name
    description
    shortDescription
    url
    inStock
    lowStock
    images(roles: []) {
      url
      roles
    }
    attributes(roles: []) {
      name
      label
      value
    }
    ... on SimpleProductView {
      price {
        regular { amount { value currency } }
        final { amount { value currency } }
      }
    }
    ... on ComplexProductView {
      priceRange {
        minimum { final { amount { value currency } } }
        maximum { final { amount { value currency } } }
      }
    }
  }
}
"""


def _extract_price(product_view: dict[str, Any]) -> tuple[float, float | None, str]:
    """Extract price, compare_at (optional), and currency from ProductView."""
    currency = "USD"
    price = 0.0
    compare_at: float | None = None

    # SimpleProductView
    if "price" in product_view and product_view["price"]:
        p = product_view["price"]
        if p.get("final", {}).get("amount"):
            amt = p["final"]["amount"]
            price = float(amt.get("value", 0))
            currency = (amt.get("currency") or "USD").strip()
        if p.get("regular", {}).get("amount"):
            reg = float(p["regular"]["amount"].get("value", 0))
            if reg > price and price > 0:
                compare_at = reg
    # ComplexProductView: use minimum of price range
    elif "priceRange" in product_view and product_view["priceRange"]:
        pr = product_view["priceRange"]
        min_amt = (pr.get("minimum") or {}).get("final") or (pr.get("minimum") or {}).get("regular") or {}
        if min_amt.get("amount"):
            amt = min_amt["amount"]
            price = float(amt.get("value", 0))
            currency = (amt.get("currency") or "USD").strip()
        max_amt = (pr.get("maximum") or {}).get("final") or {}
        if max_amt.get("amount") and price > 0:
            max_val = float(max_amt["amount"].get("value", 0))
            if max_val > price:
                compare_at = max_val

    return price, compare_at, currency


def _first_image_url(images: list[dict[str, Any]] | None) -> str | None:
    """Get first image URL, preferring image/small_image/thumbnail role."""
    if not images:
        return None
    for img in images:
        if img.get("url"):
            return img["url"]
    return None


def _category_from_attributes(attributes: list[dict[str, Any]] | None) -> str | None:
    """Derive primary category from attributes (e.g. category_gear)."""
    if not attributes:
        return None
    for attr in attributes:
        name = (attr.get("name") or "").lower()
        if "category" in name or "product_type" in name:
            val = attr.get("value")
            if isinstance(val, list) and val:
                return str(val[0])
            if val:
                return str(val)
    return None


def _product_view_to_catalog(product_view: dict[str, Any]) -> CatalogProduct:
    """Map Merchandising API ProductView to CatalogProduct."""
    price, compare_at, currency = _extract_price(product_view)
    images = product_view.get("images") or []
    image_url = _first_image_url(images)
    attrs = product_view.get("attributes") or []
    category = _category_from_attributes(attrs)
    categories = []
    for a in attrs:
        if "category" in (a.get("name") or "").lower():
            v = a.get("value")
            if isinstance(v, list):
                categories.extend(str(x) for x in v)
            elif v:
                categories.append(str(v))
    if category and category not in categories:
        categories.insert(0, category)
    desc = product_view.get("shortDescription") or product_view.get("description") or ""
    description_plain = (
        re.sub(r"<[^>]+>", "", desc).replace("\n", " ").strip()[:500] or None
    )
    sku = product_view.get("sku") or ""
    return CatalogProduct(
        product_id=product_view.get("id") or sku,
        name=product_view.get("name") or "",
        description=product_view.get("description"),
        description_plain=description_plain,
        category=category,
        categories=categories,
        vendor=None,
        sku=sku or None,
        variant_ids=[],
        price=price,
        compare_at_price=compare_at,
        currency=currency,
        is_in_stock=product_view.get("inStock", True),
        inventory_quantity=None,
        restock_date=None,
        image_url=image_url,
        url=product_view.get("url"),
        created_at=None,
        updated_at=None,
        raw_data=None,
    )


class AdobeCommerceOptimizerCatalogProvider(CatalogProvider):
    """
    Product catalog via Adobe Commerce Optimizer Merchandising API (GraphQL).

    Requires Optimizer subscription and catalog data ingested via Data Ingestion API.
    Configure: tenant ID, catalog view ID, locale. No auth token required per docs.
    """

    def __init__(
        self,
        tenant_id: str,
        catalog_view_id: str,
        locale: str = "en_US",
        *,
        region: str = "na1",
        environment: str = "sandbox",
        price_book_id: str | None = None,
    ) -> None:
        """
        Initialize the Optimizer catalog provider.

        Args:
            tenant_id: Adobe Commerce Optimizer tenant/instance ID (from Cloud Manager).
            catalog_view_id: Catalog view ID (AC-View-ID).
            locale: Catalog source locale, e.g. en_US (AC-Source-Locale).
            region: Cloud region (na1, etc.).
            environment: 'sandbox' or '' for production.
            price_book_id: Optional price book ID (AC-Price-Book-ID).
        """
        self.tenant_id = tenant_id.strip()
        self.catalog_view_id = catalog_view_id.strip()
        self.locale = locale.strip() or "en_US"
        self.region = region.strip() or "na1"
        self.environment = (environment or "").strip().lower()
        self.price_book_id = (price_book_id or "").strip() or None
        self._client: httpx.AsyncClient | None = None

    @property
    def platform_name(self) -> str:
        return "adobe_commerce"

    @property
    def graphql_url(self) -> str:
        """Merchandising API GraphQL endpoint (SaaS)."""
        # Sandbox: na1-sandbox.api.commerce.adobe.com; production: na1.api.commerce.adobe.com
        host = f"{self.region}-{self.environment}" if self.environment else self.region
        return f"https://{host}.api.commerce.adobe.com/{self.tenant_id}/graphql"

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {
            "Content-Type": "application/json",
            "AC-View-ID": self.catalog_view_id,
            "AC-Source-Locale": self.locale,
        }
        if self.price_book_id:
            h["AC-Price-Book-ID"] = self.price_book_id
        return h

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers=self._headers(),
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a GraphQL request."""
        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables
        response = await self.client.post(self.graphql_url, json=payload)
        response.raise_for_status()
        data = response.json()
        if "errors" in data and data["errors"]:
            raise RuntimeError(f"GraphQL errors: {data['errors']}")
        return data.get("data") or {}

    async def search_products(
        self,
        query: str,
        *,
        category: str | None = None,
        limit: int = 20,
    ) -> list[CatalogProduct]:
        page_size = min(limit, 50)
        variables: dict[str, Any] = {"phrase": query.strip() or "*", "pageSize": page_size}
        data = await self._graphql(QUERY_PRODUCT_SEARCH, variables)
        items = (data.get("productSearch") or {}).get("items") or []
        result: list[CatalogProduct] = []
        for item in items:
            pv = item.get("productView")
            if not pv:
                continue
            cp = _product_view_to_catalog(pv)
            if category and (not cp.category or category.lower() not in (cp.category or "").lower()):
                if not any(category.lower() in (c or "").lower() for c in cp.categories):
                    continue
            result.append(cp)
            if len(result) >= limit:
                break
        return result

    async def get_product(
        self,
        product_id: str | None = None,
        sku: str | None = None,
    ) -> CatalogProduct | None:
        # Merchandising API products() accepts SKUs only; treat product_id as sku if sku not given
        use_sku = sku or product_id
        if not use_sku:
            return None
        skus = [use_sku.strip()]
        data = await self._graphql(QUERY_PRODUCTS_BY_SKU, {"skus": skus})
        products = data.get("products") or []
        if not products:
            return None
        return _product_view_to_catalog(products[0])

    async def get_inventory(
        self,
        product_id: str | None = None,
        sku: str | None = None,
    ) -> InventoryInfo | None:
        product = await self.get_product(product_id=product_id, sku=sku)
        if not product:
            return None
        return InventoryInfo(
            product_id=product.product_id,
            variant_id=None,
            sku=product.sku,
            quantity_available=product.inventory_quantity or 0,
            is_in_stock=product.is_in_stock,
            restock_date=product.restock_date,
        )

    async def get_categories(self) -> list[CatalogCategory]:
        # Merchandising API does not expose a top-level categories query in the snippet;
        # categories are derived from product attributes. Return empty for now.
        return []

    async def get_products_by_category(
        self,
        category_id: str | None = None,
        category_name: str | None = None,
        limit: int = 20,
    ) -> list[CatalogProduct]:
        # Use search with category as phrase to approximate category filter
        name = category_name or category_id
        if not name:
            return []
        return await self.search_products(name, limit=limit)
