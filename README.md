# eCommerce Intent Reasoning Engine

A reasoning engine for eCommerce intent classification and resolution. This is not a simple classifier — it decomposes compound intents, resolves conflicts, and outputs structured action plans.

## MVP Scope (Phase 1)

- **8 core order lifecycle intents**
- **Chat channel only**
- **Shopify integration (read-only)**
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

- Python 3.12+
- Docker and Docker Compose
- Anthropic API key

### Setup

1. Clone the repository and set up environment:

```bash
cp .env.example .env
# Edit .env with your API keys
```

2. Start infrastructure:

```bash
docker-compose up -d postgres redis
```

3. Install dependencies:

```bash
pip install uv
uv pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

4. Seed the intent catalog:

```bash
python scripts/seed_catalog.py
```

5. Run the API:

```bash
uvicorn intent_engine.api.server:app --reload
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

### Run Tests

```bash
pytest tests/ -v
```

### Run Evaluation

```bash
python scripts/run_eval.py
```

### Lint and Format

```bash
ruff check src/ tests/
ruff format src/ tests/
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
