# Intent Engine - Development Commands
# Run `just` to see all available commands

# Default recipe - show help
default:
    @just --list

# === Setup ===

# Create virtual environment and install dependencies
setup:
    python3 -m venv .venv
    .venv/bin/pip install -e ".[dev]"
    .venv/bin/python -m spacy download en_core_web_sm

# Install dependencies only
install:
    .venv/bin/pip install -e ".[dev]"

# === Docker Services ===

# Start all Docker services (postgres, redis, api)
up:
    docker-compose up -d

# Start infrastructure only (postgres, redis)
infra:
    docker-compose up -d postgres redis

# Stop all Docker services
down:
    docker-compose down

# View Docker logs
logs service="api":
    docker-compose logs -f {{ service }}

# Restart a service
restart service="api":
    docker-compose restart {{ service }}

# === Database ===

# Seed the intent catalog
seed:
    .venv/bin/python scripts/seed_catalog.py

# Seed catalog (clear existing first)
seed-refresh:
    .venv/bin/python scripts/seed_catalog.py --refresh

# Connect to PostgreSQL
psql:
    docker-compose exec postgres psql -U intent_engine -d intent_engine

# === Development ===

# Run the API locally (requires Python 3.12-3.13)
dev:
    .venv/bin/uvicorn intent_engine.api.server:app --reload --host 0.0.0.0 --port 8000

# Run the API via Docker
api:
    docker-compose up -d api
    docker-compose logs -f api

# Run the MCP server for multi-agent access
mcp:
    .venv/bin/python -m intent_engine.mcp.server

# Test A2A agent card endpoint
a2a-test:
    @echo "Testing A2A agent card..."
    curl -s http://localhost:8000/.well-known/agent.json | python3 -m json.tool
    @echo "\nTesting A2A task submission..."
    curl -s -X POST http://localhost:8000/a2a/tasks \
        -H "Content-Type: application/json" \
        -d '{"action": "list_intent_taxonomy", "input": {}}' \
        | python3 -m json.tool

# === Testing ===

# Run all tests
test:
    .venv/bin/pytest tests/ -v

# Run tests with coverage
test-cov:
    .venv/bin/pytest tests/ -v --cov=intent_engine --cov-report=term-missing

# Run only unit tests
test-unit:
    .venv/bin/pytest tests/unit/ -v

# Run only integration tests
test-int:
    .venv/bin/pytest tests/integration/ -v

# Run tests in Docker container (avoids Python 3.14 spaCy issues)
test-docker:
    docker-compose run --rm api pytest tests/ -v

# Run integration tests in Docker container
test-int-docker:
    docker-compose run --rm api pytest tests/integration/ -v

# Run tests using dedicated test container (good practice for isolation)
test-container:
    docker-compose --profile test run --rm test pytest tests/ -v

# Run integration tests using dedicated test container
test-int-container:
    docker-compose --profile test run --rm test pytest tests/integration/ -v

# === Evaluation ===

# Run evaluation on golden set
eval:
    .venv/bin/python scripts/run_eval.py

# Run evaluation and save results
eval-save:
    .venv/bin/python scripts/run_eval.py --output evals/results/latest.json

# === Code Quality ===

# Run linter
lint:
    .venv/bin/ruff check src/ tests/

# Run linter and fix issues
lint-fix:
    .venv/bin/ruff check src/ tests/ --fix

# Format code
fmt:
    .venv/bin/ruff format src/ tests/

# Run type checker
typecheck:
    .venv/bin/mypy src/

# Run all checks (lint, format check, typecheck)
check: lint typecheck
    .venv/bin/ruff format src/ tests/ --check

# === API Testing ===

# Health check
health:
    curl -s http://localhost:8000/health | python3 -m json.tool

# Test intent resolution (requires API_KEY env var or uses default)
resolve text:
    curl -s -X POST http://localhost:8000/v1/intent/resolve \
        -H "Authorization: Bearer ${API_KEY:-dev-api-key}" \
        -H "Content-Type: application/json" \
        -d '{"request_id": "test", "tenant_id": "test", "channel": "chat", "raw_text": "{{ text }}"}' \
        | python3 -m json.tool

# List available intents
intents:
    curl -s http://localhost:8000/v1/intent/intents \
        -H "Authorization: Bearer ${API_KEY:-dev-api-key}" \
        | python3 -m json.tool

# Get catalog stats
catalog:
    curl -s http://localhost:8000/v1/intent/catalog \
        -H "Authorization: Bearer ${API_KEY:-dev-api-key}" \
        | python3 -m json.tool

# === Cleanup ===

# Remove all Docker volumes and containers
clean:
    docker-compose down -v

# Remove Python cache files
clean-cache:
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
