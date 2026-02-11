# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

eCommerce Intent Reasoning Engine - a multi-layer NLP system that classifies customer intents and generates action plans for post-purchase customer service. Not a simple classifier - it decomposes compound intents, resolves conflicts, and outputs structured actions.

**Key differentiators:**
- Two-path architecture: fast path (embedding similarity ~50% of requests) and reasoning path (LLM decomposition for complex cases)
- 62 intents across 10 categories (ORDER_STATUS, RETURN_EXCHANGE, COMPLAINT, etc.)
- Multi-platform support: Shopify, Adobe Commerce, WooCommerce, BigCommerce
- Sentiment-aware routing with sarcasm detection

## Commands

```bash
# Setup
just setup           # Create venv, install deps, download spaCy model
just infra           # Start postgres + redis (Docker)
just seed            # Seed intent catalog (1010 examples)

# Development
just dev             # Run API locally with hot reload (port 8000)
just up              # Start all services via Docker

# Testing
just test            # Run all tests
just test-unit       # Run unit tests only
just test-int        # Run integration tests only
just test-cov        # Tests with coverage
pytest tests/unit/test_foo.py -v  # Run single test file

# Code Quality
just check           # Lint + typecheck + format check
just lint-fix        # Auto-fix linting issues
just fmt             # Format code with ruff

# Evaluation
just eval            # Run against golden eval set (575 examples)
just eval-save       # Save results to evals/results/

# API Testing
just resolve "Where is my order?"  # Test intent resolution
just health          # Health check

# Multi-Agent
just mcp             # Run MCP server for multi-agent access
just a2a-test        # Test A2A agent card endpoint
```

## Architecture

### Processing Pipeline (src/intent_engine/engine.py)

```
Request → Entity Extraction → Sentiment Analysis → Embedding Generation
       → Context Enrichment → Similarity Matching → Compound Detection
       → Policy Evaluation → [Fast Path | LLM Reasoning] → Result
```

**Fast Path** (≥0.85 similarity, no compound signals): Direct resolution from embedding match
**Reasoning Path**: LLM decomposition via pydantic-ai for compound/ambiguous intents

### Core Modules

- `engine.py` - Main orchestrator, coordinates all pipeline stages
- `extractors/` - Entity extraction (spaCy NER), sentiment analysis, embeddings (sentence-transformers)
- `matchers/` - Similarity matching against intent catalog, compound intent detection
- `reasoners/` - LLM decomposition, context enrichment, policy engine
- `storage/` - pgvector for embeddings, intent catalog management
- `integrations/` - Platform connectors (Shopify, Adobe Commerce, WooCommerce, BigCommerce)
- `agents/` - CustomerServiceAgent orchestrator for full chat flow

### Intent Resolution Flow

1. `IntentRequest` → unified input model from any channel (chat/email/form)
2. Entity extraction finds order IDs, tracking numbers, dates
3. Embedding generated via sentence-transformers (all-MiniLM-L6-v2)
4. Similarity search against intent catalog (pgvector)
5. If high confidence single match → fast path resolution
6. If compound/ambiguous → LLM reasoning via `IntentDecomposer`
7. `ReasoningResult` returned with intents, entities, actions, confidence

### Orchestration Agent (src/intent_engine/agents/orchestrator.py)

`CustomerServiceAgent` coordinates the full customer service flow:
- Intent classification via `IntentEngine`
- Order/customer data fetching via platform connectors
- Response generation via templates or LLM
- Action recommendations based on intent + context

## Key Patterns

### Settings Management
All config via pydantic-settings in `config.py`. Environment variables or `.env` file.

### Testing
- pytest-asyncio with `asyncio_mode = "auto"`
- Fixtures in `tests/conftest.py`
- Integration tests require running infra (`just infra`)

### Platform Integrations
All connectors inherit from `PlatformConnector` base class (`integrations/base.py`).
Adobe Commerce supports both PaaS (integration token) and SaaS (IMS OAuth) auth.

### Evaluation
Golden eval set at `evals/datasets/`. Metrics tracked: macro F1, fast path rate, compound detection recall.
Target: F1 > 85%, fast path latency < 200ms p99.

## Environment Variables

Required:
- `DATABASE_URL` - PostgreSQL with pgvector
- `REDIS_URL` - For caching/rate limiting

Optional:
- `ANTHROPIC_API_KEY` - Enables LLM reasoning path (without it, fast-path only)
- `SHOPIFY_STORE_DOMAIN`, `SHOPIFY_ACCESS_TOKEN` - Shopify integration
- `ADOBE_COMMERCE_BASE_URL`, `ADOBE_COMMERCE_ACCESS_TOKEN` - Adobe Commerce
- `WOOCOMMERCE_STORE_URL`, `WOOCOMMERCE_CONSUMER_KEY`, `WOOCOMMERCE_CONSUMER_SECRET` - WooCommerce
- `BIGCOMMERCE_STORE_HASH`, `BIGCOMMERCE_ACCESS_TOKEN` - BigCommerce
- `ENABLE_TRACING` - OpenTelemetry tracing (default: true)
- `ENABLE_METRICS` - Prometheus metrics (default: true)

## Multi-Agent Access

- **MCP Server** (`mcp/server.py`): Exposes intent resolution as MCP tools for external agents
- **A2A Protocol** (`a2a/`): Agent-to-Agent communication via `/.well-known/agent.json`

## Python Version

Requires Python 3.12-3.13 (spaCy not compatible with 3.14). Use `just test-docker` if running Python 3.14 locally.
