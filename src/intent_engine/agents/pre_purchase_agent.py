"""Pre-purchase agent: product and discovery intents with catalog delegation."""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings

from intent_engine.agents.catalog_agent import (
    CatalogAgentDeps,
    get_catalog_agent,
)
from intent_engine.engine import IntentEngine
from intent_engine.models.request import InputChannel, IntentRequest


@dataclass
class PrePurchaseDeps:
    """Dependencies for the pre-purchase agent."""

    intent_engine: IntentEngine
    catalog_provider: Any  # CatalogProvider | None


class PrePurchaseOutput(BaseModel):
    """Structured response from the pre-purchase agent."""

    response_text: str = Field(description="Customer-facing reply")
    products: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Product summaries if catalog was queried",
    )
    primary_intent: str | None = Field(default=None, description="Resolved intent code if classified")
    confidence: float = Field(default=0.0, ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "response_text": "We have several widgets in stock. The Classic Widget is $29.99 and available now.",
                    "products": [{"name": "Classic Widget", "price": 29.99, "is_in_stock": True}],
                    "primary_intent": "PRODUCT_INQUIRY.STOCK",
                    "confidence": 0.9,
                }
            ]
        }
    }


PRE_PURCHASE_SYSTEM_PROMPT = """You are a friendly pre-purchase assistant for an eCommerce store.
You help customers find products, check availability, compare options, and get recommendations.

You have two tools:
1. classify_intent - Use this first to understand what the customer wants (product search, stock check, recommendations, comparison, etc.).
2. query_catalog - Use this to search products, get product details, or get recommendations. Call it with the customer's question or a short search query.

Flow: Classify the customer's intent, then use query_catalog when they are asking about products, availability, recommendations, or discovery. 
Respond with a helpful, concise message (2-4 sentences) that includes relevant product info from the catalog when available.
If the catalog returns no results, say so politely and suggest they try different keywords or browse categories."""


def get_pre_purchase_agent(
    model_name: str = "claude-sonnet-4-5",
    model: Any = None,
) -> Agent[PrePurchaseDeps, PrePurchaseOutput]:
    """
    Create the PrePurchaseAgent with intent classification and catalog delegation tools.

    Returns:
        Agent that classifies intent and delegates to ProductCatalogAgent for product/discovery intents.
    """
    model_or_id = model if model is not None else f"anthropic:{model_name}"

    agent: Agent[PrePurchaseDeps, PrePurchaseOutput] = Agent(
        model_or_id,
        deps_type=PrePurchaseDeps,
        output_type=PrePurchaseOutput,
        system_prompt=PRE_PURCHASE_SYSTEM_PROMPT,
        model_settings=ModelSettings(temperature=0.3, max_tokens=1024),
    )

    @agent.tool
    async def classify_intent(ctx: RunContext[PrePurchaseDeps], raw_text: str) -> dict[str, Any]:
        """Classify the customer's intent (e.g. product search, stock check, recommendations). Returns intent codes and confidence."""
        engine = ctx.deps.intent_engine
        request = IntentRequest(
            request_id=f"pre-{id(raw_text)}",
            tenant_id="pre-purchase",
            channel=InputChannel.CHAT,
            raw_text=raw_text,
        )
        result = await engine.resolve(request)
        primary = result.resolved_intents[0] if result.resolved_intents else None
        return {
            "resolved_intents": [
                {
                    "category": i.category,
                    "intent": i.intent,
                    "confidence": i.confidence,
                    "intent_code": i.intent_code,
                }
                for i in result.resolved_intents
            ],
            "primary_intent": primary.intent_code if primary else None,
            "primary_confidence": primary.confidence if primary else 0.0,
            "is_compound": result.is_compound,
        }

    @agent.tool
    async def query_catalog(ctx: RunContext[PrePurchaseDeps], message: str) -> dict[str, Any]:
        """Query the product catalog with the customer's question or search. Returns product summaries and a reply snippet."""
        catalog_provider = ctx.deps.catalog_provider
        if catalog_provider is None:
            return {
                "products": [],
                "reply_snippet": "Product catalog is not available right now. Please try again later or contact support.",
                "total_found": 0,
            }
        catalog_agent = get_catalog_agent()
        deps = CatalogAgentDeps(catalog=catalog_provider)
        result = await catalog_agent.run(message, deps=deps, usage=ctx.usage)
        out = result.output
        return {
            "products": out.products,
            "reply_snippet": out.reply_snippet,
            "total_found": out.total_found,
        }

    return agent
