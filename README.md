# eCommerce Intent Reasoning Engine

A reasoning engine for eCommerce intent classification and resolution. This is not a simple classifier — it decomposes compound intents, resolves conflicts, and outputs structured action plans.

## Features

- **62 intents** across 10 categories
- **Multi-channel support**: Chat, Email, Form
- **Platform integrations**: Shopify, Adobe Commerce (read-only)
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
│  Intent Engine   │  → Classify intent (WISMO, Return, etc.)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Platform Connector│  → Fetch order/customer data
│ (Shopify/Adobe)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Response Generator│  → Generate customer response
└────────┬─────────┘
         │
         ▼
   Agent Response
   (text + actions)
```

### Programmatic Usage

```python
from intent_engine.agents import CustomerServiceAgent, CustomerMessage
from intent_engine.config import get_settings

agent = CustomerServiceAgent(settings=get_settings())
await agent.initialize()

response = await agent.process_message(CustomerMessage(
    message_id="msg-123",
    customer_email="john@example.com",
    text="Where is my order #12345?",
    platform="shopify",
))

print(response.response_text)
# "Your order #12345 is currently in transit..."

print(response.actions[0].action_type)
# ActionType.PROVIDE_ORDER_STATUS
```

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
```

## Architecture

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
│  Handles ~70-80% of requests                          │
└──────────────────────────┬───────────────────────────┘
                           │ (low confidence or compound)
                           ▼
┌──────────────────────────────────────────────────────┐
│            REASONING LAYER (Deep Path)                │
│  LLM-powered decomposition (Claude)                   │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│              RESOLUTION LAYER                         │
│  Intent → Action mapping │ Response generation       │
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
