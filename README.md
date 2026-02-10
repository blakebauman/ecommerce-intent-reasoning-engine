# eCommerce Intent Reasoning Engine

A reasoning engine for eCommerce intent classification and resolution. This is not a simple classifier — it decomposes compound intents, resolves conflicts, and outputs structured action plans.

## MVP Scope (Phase 1)

- **8 core order lifecycle intents**
- **Chat channel only**
- **Platform integrations**: Shopify, Adobe Commerce (read-only)
- **Fast path matching** (embedding similarity)
- **LLM reasoning** for compound intents

### Core Intents

| Intent | Description |
|--------|-------------|
| `ORDER_STATUS.WISMO` | Where is my order |
| `ORDER_STATUS.DELIVERY_ESTIMATE` | When will it arrive |
| `ORDER_MODIFY.CANCEL_ORDER` | Cancel my order |
| `ORDER_MODIFY.CHANGE_ADDRESS` | Change shipping address |
| `RETURN_EXCHANGE.RETURN_INITIATE` | Start a return |
| `RETURN_EXCHANGE.EXCHANGE_REQUEST` | Exchange for different item |
| `RETURN_EXCHANGE.REFUND_STATUS` | Check refund status |
| `COMPLAINT.DAMAGED_ITEM` | Item arrived damaged |

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
just seed        # Seed intent catalog (279 examples)

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

1. **F1 > 85%** on golden eval set (200 examples)
2. **Fast path latency < 200ms** (p99)
3. **Compound detection recall > 85%**

## License

MIT
