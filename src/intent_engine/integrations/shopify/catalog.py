"""Shopify catalog provider (products and inventory) via Admin REST API."""

from datetime import datetime, timezone
from typing import Any

import httpx

from intent_engine.integrations.base import CatalogProvider
from intent_engine.models.catalog import (
    CatalogCategory,
    CatalogProduct,
    InventoryInfo,
)


class ShopifyCatalogProvider(CatalogProvider):
    """
    Shopify product catalog via Admin REST API (2024-01).

    Uses the same store domain and access token as ShopifyConnector.
    """

    API_VERSION = "2024-01"

    def __init__(self, store_domain: str, access_token: str) -> None:
        self.store_domain = store_domain.rstrip("/")
        self.access_token = access_token
        self._client: httpx.AsyncClient | None = None

    @property
    def platform_name(self) -> str:
        return "shopify"

    @property
    def base_url(self) -> str:
        return f"https://{self.store_domain}/admin/api/{self.API_VERSION}"

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "X-Shopify-Access-Token": self.access_token,
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict | None:
        try:
            response = await self.client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    def _parse_product(self, p: dict[str, Any]) -> CatalogProduct:
        """Map Shopify product JSON to CatalogProduct."""
        variants = p.get("variants") or []
        first = variants[0] if variants else {}
        total_qty = sum(int(v.get("inventory_quantity", 0)) for v in variants)
        price_str = first.get("price", "0")
        try:
            price = float(price_str)
        except (TypeError, ValueError):
            price = 0.0
        compare_at = first.get("compare_at_price")
        compare_at_price = float(compare_at) if compare_at else None
        image = None
        if p.get("image"):
            image = p["image"].get("src")
        elif variants and variants[0].get("image_id"):
            for img in p.get("images") or []:
                if str(img.get("id")) == str(variants[0].get("image_id")):
                    image = img.get("src")
                    break
        if not image and (p.get("images") or []):
            image = p["images"][0].get("src")

        return CatalogProduct(
            product_id=str(p.get("id", "")),
            name=p.get("title", ""),
            description=p.get("body_html"),
            description_plain=(p.get("body_html") or "")[:500].replace("\n", " ").strip() or None,
            category=p.get("product_type") or None,
            categories=[c for c in [p.get("product_type")] if c],
            vendor=p.get("vendor"),
            sku=first.get("sku") if first else None,
            variant_ids=[str(v.get("id", "")) for v in variants if v.get("id")],
            price=price,
            compare_at_price=compare_at_price,
            currency="USD",
            is_in_stock=total_qty > 0,
            inventory_quantity=total_qty if total_qty >= 0 else None,
            restock_date=None,
            image_url=image,
            url=None,
            created_at=self._parse_optional_datetime(p.get("created_at")),
            updated_at=self._parse_optional_datetime(p.get("updated_at")),
            raw_data=None,
        )

    @staticmethod
    def _parse_optional_datetime(s: str | None) -> datetime | None:
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    async def search_products(
        self,
        query: str,
        *,
        category: str | None = None,
        limit: int = 20,
    ) -> list[CatalogProduct]:
        params: dict[str, Any] = {"limit": min(limit, 250)}
        if query.strip():
            params["title"] = query.strip()
        if category:
            params["product_type"] = category

        data = await self._request("GET", "/products.json", params=params)
        if not data or "products" not in data:
            return []
        return [self._parse_product(p) for p in data["products"][:limit]]

    async def get_product(
        self,
        product_id: str | None = None,
        sku: str | None = None,
    ) -> CatalogProduct | None:
        if product_id:
            # Numeric ID; strip gid if present
            pid = product_id.split("/")[-1] if "/" in product_id else product_id
            data = await self._request("GET", f"/products/{pid}.json")
            if data and "product" in data:
                return self._parse_product(data["product"])
            return None
        if sku:
            data = await self._request(
                "GET",
                "/products.json",
                params={"limit": 250},
            )
            if not data or "products" not in data:
                return None
            for p in data["products"]:
                for v in p.get("variants") or []:
                    if (v.get("sku") or "").strip() == sku.strip():
                        return self._parse_product(p)
            return None
        return None

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
            variant_id=product.variant_ids[0] if product.variant_ids else None,
            sku=product.sku,
            quantity_available=product.inventory_quantity or 0,
            is_in_stock=product.is_in_stock,
            restock_date=product.restock_date,
        )

    async def get_categories(self) -> list[CatalogCategory]:
        # Shopify REST: product_type is a string on product, not a first-class collection.
        # Fetch products and collect unique product_type values.
        data = await self._request("GET", "/products.json", params={"limit": 250})
        if not data or "products" not in data:
            return []
        seen: set[str] = set()
        categories: list[CatalogCategory] = []
        for p in data["products"]:
            pt = (p.get("product_type") or "").strip()
            if pt and pt not in seen:
                seen.add(pt)
                categories.append(
                    CatalogCategory(
                        category_id=pt,
                        name=pt,
                        handle=pt.lower().replace(" ", "-"),
                        product_count=None,
                    )
                )
        return categories

    async def get_products_by_category(
        self,
        category_id: str | None = None,
        category_name: str | None = None,
        limit: int = 20,
    ) -> list[CatalogProduct]:
        name = category_name or category_id
        if not name:
            return []
        return await self.search_products("", category=name, limit=limit)
