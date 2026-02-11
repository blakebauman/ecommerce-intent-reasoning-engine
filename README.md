# eCommerce Intent Reasoning Engine

A reasoning engine for eCommerce intent classification and resolution. This is not a simple classifier — it decomposes compound intents, resolves conflicts, and outputs structured action plans.

## Features

- **62 intents** across 10 categories
- **Multi-channel support**: Chat, Email, Form
- **Platform integrations**: Shopify, Adobe Commerce, WooCommerce, BigCommerce (order connectors)
- **Fast path matching** (embedding similarity) — handles ~50% of requests
- **LLM reasoning** for compound/ambiguous intents
- **Sentiment analysis** with sarcasm detection
- **Tier-aware escalation** (VIP, AT_RISK, Standard)
- **Entity extraction** (orders, tracking, damage severity, defect categories)

### Intent Categories

| Category | Intents | Examples |
|----------|---------|----------|
| `ORDER_STATUS` | WISMO, DELIVERY_ESTIMATE, CONFIRMATION, TRACKING_ISSUE | "Where is my order?" |
| `ORDER_MODIFY` | CANCEL_ORDER, CHANGE_ADDRESS, CHANGE_ITEMS, EXPEDITE | "Cancel my order" |
| `RETURN_EXCHANGE` | RETURN_INITIATE, EXCHANGE_REQUEST, REFUND_STATUS, WARRANTY_CLAIM, STORE_CREDIT | "I want to return this" |
| `COMPLAINT` | DAMAGED_ITEM, WRONG_ITEM, MISSING_ITEM, QUALITY_ISSUE, LATE_DELIVERY, LOST_PACKAGE | "Item arrived broken" |
| `PRODUCT_INQUIRY` | STOCK, RESTOCK, SIZE_FIT, FEATURES, COMPATIBILITY, MATERIALS, USAGE | "Is this in stock?" |
| `ACCOUNT_BILLING` | PAYMENT_ISSUE, UPDATE_PAYMENT, INVOICE, PROMO_CODE, SUBSCRIPTION, ACCOUNT_ACCESS | "My payment failed" |
| `DISCOVERY` | RECOMMENDATION, COMPARISON, BEST_SELLER, NEW_ARRIVALS, GIFT_SUGGESTION | "What do you recommend?" |
| `LOYALTY_REWARDS` | POINTS_BALANCE, EARN_POINTS, REDEEM_POINTS, MEMBER_STATUS, POINTS_VALUE | "How many points do I have?" |
| `SHIPPING` | SHIPPING_OPTIONS, INTERNATIONAL, SHIPPING_COST, DELIVERY_TIME | "Do you ship internationally?" |
| `BULK_WHOLESALE` | BULK_DISCOUNT, BUSINESS_ACCOUNT, MINIMUM_ORDER, WHOLESALE_PRICING | "Volume discount available?" |
| `META` | GREETING, FAREWELL, HUMAN_HANDOFF, FEEDBACK, UNCLEAR, OFF_TOPIC | "I want to speak to a manager" |

## Quick Start

### Prerequisites

- Python 3.12-3.13 (spaCy not yet compatible with 3.14)
- Docker and Docker Compose
- [just](https://github.com/casey/just) command runner (recommended)
- Anthropic API key (optional for fast-path only mode)

### Setup

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env with your API keys (ANTHROPIC_API_KEY for LLM reasoning)

# 2. Start infrastructure and API via Docker (recommended)
just up          # Start postgres, redis, and API
just seed        # Seed intent catalog (1010 examples)

# Or for local development:
just setup       # Create venv and install dependencies
# Alternatively: uv sync --extra dev && uv run python -m spacy download en_core_web_sm
just infra       # Start postgres and redis only
just seed        # Seed intent catalog
just dev         # Run API locally with hot reload
```

### Verify It Works

```bash
just health                           # Check API health
just resolve "Where is my order?"     # Test intent resolution
```

### Usage

```bash
# Resolve an intent
curl -X POST http://localhost:8000/v1/intent/resolve \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "req-123",
    "tenant_id": "merchant-1",
    "channel": "chat",
    "raw_text": "Where is my order #12345?"
  }'
```

### Response

```json
{
  "request_id": "req-123",
  "resolved_intents": [
    {
      "category": "ORDER_STATUS",
      "intent": "WISMO",
      "confidence": 0.92,
      "confidence_tier": "high",
      "evidence": ["where is my order"]
    }
  ],
  "is_compound": false,
  "entities": [
    {
      "entity_type": "order_id",
      "value": "12345",
      "confidence": 0.99
    }
  ],
  "path_taken": "fast_path",
  "processing_time_ms": 45
}
```

## Orchestration Agent

The orchestration agent provides a complete customer service flow by combining intent classification, order data fetching, and response generation.

### Lifecycle routing

The **`/v1/agent/chat`** endpoint uses a **lifecycle router** that classifies each message and routes it to the right agent:

- **Pre-purchase** (product search, recommendations, discovery): messages with primary intent `PRODUCT_INQUIRY` or `DISCOVERY` are handled by the **pre-purchase agent**, which uses the **product catalog** (Shopify or Adobe Commerce Optimizer) to answer questions about products, stock, and recommendations.
- **Post-purchase** (orders, returns, complaints, loyalty, shipping): all other intents are handled by the **customer service agent**, which fetches order/customer data from platform connectors and generates responses with actions.

You do not need to choose an agent yourself; the router decides from the classified intent.

### Chat Endpoint

```bash
curl -X POST http://localhost:8000/v1/agent/chat \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "message_id": "msg-123",
    "customer_email": "john@example.com",
    "text": "Where is my order #12345?",
    "platform": "shopify"
  }'
```

### Response

```json
{
  "message_id": "msg-123",
  "response_text": "Your order #12345 is currently in transit via UPS. Track it here: https://...",
  "intents": [{"category": "ORDER_STATUS", "intent": "WISMO", "confidence": 0.92}],
  "actions": [{"action_type": "provide_order_status", "description": "Provide order status"}],
  "confidence": 0.92,
  "processing_time_ms": 250
}
```

### How It Works

```
Customer Message
       │
       ▼
┌──────────────────┐
│  Intent Engine   │  → Classify intent (WISMO, Return, Product search, etc.)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Lifecycle Router │  → PRODUCT_INQUIRY / DISCOVERY? → Pre-purchase agent
│                  │  → Otherwise → Customer service agent
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
Pre-purchase   Post-purchase
(Catalog)      (Order + Response Generator)
    │         │
    └────┬────┘
         ▼
   Agent Response
   (text + actions)
```

### Programmatic Usage

The `/v1/agent/chat` endpoint uses `LifecycleRouter` to route messages. For direct agent access:

```python
from intent_engine.agents import CustomerServiceAgent, CustomerMessage, LifecycleRouter
from intent_engine.agents.catalog_agent import get_catalog_provider_from_settings
from intent_engine.config import get_settings

# Full lifecycle (pre-purchase + post-purchase) via router
settings = get_settings()
agent = CustomerServiceAgent(settings=settings)
await agent.initialize()
router = LifecycleRouter(
    intent_engine=agent.intent_engine,
    customer_service_agent=agent,
    catalog_provider=get_catalog_provider_from_settings(),
)
response = await router.process_message(CustomerMessage(
    message_id="msg-123",
    customer_email="john@example.com",
    text="Where is my order #12345?",
    platform="shopify",
))
print(response.response_text)
```

### Catalog and pre-purchase

Product search and discovery use a **catalog provider**. Configure one of the following so that pre-purchase messages (e.g. “Do you have this in stock?”, “What do you recommend?”) are answered from your catalog.

| Provider | Env vars | Notes |
|----------|----------|--------|
| **Shopify** | `SHOPIFY_STORE_DOMAIN`, `SHOPIFY_ACCESS_TOKEN` | Uses Admin REST API for products. |
| **Adobe Commerce Optimizer** | `ADOBE_COMMERCE_OPTIMIZER_TENANT_ID`, `ADOBE_COMMERCE_OPTIMIZER_CATALOG_VIEW_ID`, `ADOBE_COMMERCE_OPTIMIZER_LOCALE`; optional `ADOBE_COMMERCE_OPTIMIZER_REGION`, `ADOBE_COMMERCE_OPTIMIZER_ENVIRONMENT`, `ADOBE_COMMERCE_OPTIMIZER_PRICE_BOOK_ID` | [Merchandising Services](https://developer.adobe.com/commerce/services/optimizer/) GraphQL API (SaaS). |

If both Shopify and Adobe Optimizer are set, **Shopify** is used. If no catalog is configured, the router still classifies intents; pre-purchase intents are handled by the pre-purchase agent but catalog tools will return “catalog not available.”

A2A and MCP expose **catalog actions**: `search_catalog`, `get_product_details`, `get_inventory`, `pre_purchase_chat`. See `/.well-known/agent.json` and the MCP tool list for schemas.

## Development

This project uses [just](https://github.com/casey/just) as a command runner. Run `just` to see all available commands.

### Available Commands

| Command | Description |
|---------|-------------|
| **Setup** | |
| `just setup` | Create virtual environment and install dependencies |
| `just install` | Install dependencies only |
| **Docker** | |
| `just up` | Start all services (postgres, redis, api) |
| `just infra` | Start infrastructure only (postgres, redis) |
| `just down` | Stop all services |
| `just logs [service]` | View logs (default: api) |
| `just restart [service]` | Restart a service |
| **Database** | |
| `just seed` | Seed the intent catalog |
| `just seed-refresh` | Clear and re-seed catalog |
| `just psql` | Connect to PostgreSQL |
| **Development** | |
| `just dev` | Run API locally with hot reload |
| `just api` | Run API via Docker |
| **Testing** | |
| `just test` | Run all tests |
| `just test-cov` | Run tests with coverage |
| `just test-unit` | Run unit tests only |
| `just test-int` | Run integration tests only |
| **Evaluation** | |
| `just eval` | Run evaluation on golden set |
| `just eval-save` | Run evaluation and save results |
| **Code Quality** | |
| `just lint` | Run linter |
| `just lint-fix` | Run linter and fix issues |
| `just fmt` | Format code |
| `just typecheck` | Run type checker |
| `just check` | Run all checks |
| **API Testing** | |
| `just health` | Health check |
| `just resolve "text"` | Test intent resolution |
| `just intents` | List available intents |
| `just catalog` | Get catalog stats |
| **Multi-agent** | |
| `just mcp` | Run MCP server for agent access |
| `just a2a-test` | Test A2A agent card and task submission |
| **Cleanup** | |
| `just clean` | Remove Docker volumes and containers |
| `just clean-cache` | Remove Python cache files |

### Run Tests

```bash
just test           # All tests
just test-unit      # Unit tests only
just test-cov       # With coverage report
```

### Run Evaluation

```bash
just eval           # Run against golden set
just eval-save      # Save results to file
```

### Lint and Format

```bash
just lint           # Check for issues
just lint-fix       # Auto-fix issues
just fmt            # Format code
just check          # Run all checks
```

## Production

Before deploying, configure security, CORS, secrets, and infrastructure. See **[docs/PRODUCTION.md](docs/PRODUCTION.md)** for a full checklist and **[docs/DEPLOY.md](docs/DEPLOY.md)** for a step-by-step runbook.

**Commands:** `just build-prod` (build production image), `just up-prod` (run with production Compose override). Set `TENANT_DEV_MODE=false`, a strong `API_KEY`, and restrict `CORS_ORIGINS`; use managed PostgreSQL (with pgvector) and Redis, run `just seed` once, and put TLS behind a reverse proxy with `/ready` as the readiness probe.

## Platform Integrations

### Shopify

```python
from intent_engine.integrations import ShopifyConnector

connector = ShopifyConnector(
    store_domain="your-store.myshopify.com",
    access_token="your-access-token",
)
order = await connector.get_order_by_number("#1234")
```

### Adobe Commerce

Supports both deployment models:
- **PaaS** (self-hosted): Uses integration access token
- **SaaS** (Adobe Commerce Cloud): Uses Adobe IMS OAuth 2.0

```python
from intent_engine.integrations import (
    AdobeCommerceConnector,
    IntegrationTokenAuth,  # For PaaS
    IMSOAuthAuth,          # For SaaS
)

# PaaS (self-hosted)
connector = AdobeCommerceConnector(
    base_url="https://your-store.com",
    auth_strategy=IntegrationTokenAuth(access_token="your-token"),
    store_code="default",
)

# SaaS (Adobe Commerce Cloud)
connector = AdobeCommerceConnector(
    base_url="https://your-store.com",
    auth_strategy=IMSOAuthAuth(
        client_id="your-client-id",
        client_secret="your-client-secret",
        org_id="your-org-id",
    ),
)

order = await connector.get_order_by_number("000000001")
```

#### Webhook Support

Adobe Commerce webhooks are received at `/webhooks/adobe-commerce` with HMAC-SHA256 signature verification.

Supported events:
- `observer.sales_order_save_after` - Order created/updated
- `observer.sales_order_shipment_save_after` - Shipment created
- `plugin.magento.sales.api.order_management.cancel` - Order cancelled
- `observer.sales_order_creditmemo_save_after` - Refund created

### Environment Variables

```bash
# Shopify
SHOPIFY_STORE_DOMAIN=your-store.myshopify.com
SHOPIFY_ACCESS_TOKEN=your-access-token

# Adobe Commerce - PaaS
ADOBE_COMMERCE_BASE_URL=https://your-store.com
ADOBE_COMMERCE_ACCESS_TOKEN=your-integration-token
ADOBE_COMMERCE_STORE_CODE=default

# Adobe Commerce - SaaS
ADOBE_COMMERCE_IMS_CLIENT_ID=your-client-id
ADOBE_COMMERCE_IMS_CLIENT_SECRET=your-client-secret
ADOBE_COMMERCE_IMS_ORG_ID=your-org-id

# Adobe Commerce Webhooks
ADOBE_COMMERCE_WEBHOOK_SECRET=your-webhook-secret
ADOBE_COMMERCE_WEBHOOK_ENABLED=true

# Adobe Commerce Optimizer (catalog for product search / pre-purchase)
# ADOBE_COMMERCE_OPTIMIZER_TENANT_ID=your-tenant-id
# ADOBE_COMMERCE_OPTIMIZER_CATALOG_VIEW_ID=your-catalog-view-id
# ADOBE_COMMERCE_OPTIMIZER_LOCALE=en_US
# ADOBE_COMMERCE_OPTIMIZER_REGION=na1
# ADOBE_COMMERCE_OPTIMIZER_ENVIRONMENT=sandbox
```

## Architecture

The intent engine pipeline classifies and decomposes customer messages. The orchestration layer (chat endpoint) routes responses to pre-purchase or post-purchase agents based on intent.

```
┌─────────────────────────────────────────────────────┐
│                   INPUT (Chat)                       │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│              EXTRACTION LAYER                         │
│  Entity extraction │ Semantic embeddings              │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│            MATCHING LAYER (Fast Path)                 │
│  Embedding similarity │ Threshold: ≥0.85             │
│  Handles ~50% of requests                             │
└──────────────────────────┬───────────────────────────┘
                           │ (low confidence or compound)
                           ▼
┌──────────────────────────────────────────────────────┐
│            REASONING LAYER (Deep Path)                │
│  LLM-powered decomposition (Claude)                  │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│              RESOLUTION LAYER                         │
│  Intent → Action mapping │ Response generation       │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼ (feeds orchestration agent)
┌──────────────────────────────────────────────────────┐
│           ORCHESTRATION (Lifecycle Router)            │
│  Pre-purchase: Catalog agent │ Post-purchase: Orders │
└──────────────────────────────────────────────────────┘
```

## Success Criteria

1. **F1 > 85%** on golden eval set (575 examples)
2. **Fast path latency < 200ms** (p99)
3. **Compound detection recall > 85%**

### Current Metrics

| Metric | Value |
|--------|-------|
| Macro F1 | 81.0% |
| Fast Path Rate | 51% |
| Training Examples | 1010 |
| Intent Coverage | 62 intents |

## License

MIT
