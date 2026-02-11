# Production Readiness Checklist

This document outlines what to configure and verify before deploying the Intent Reasoning Engine to production.

## 1. Security & Authentication

### API keys and tenant mode
- [ ] **Disable dev mode**: Set `TENANT_DEV_MODE=false`. In dev mode the app accepts any API key and uses a single dev tenant.
- [ ] **Use a strong API key**: Set `API_KEY` to a cryptographically random value (e.g. 32+ bytes, from a secrets manager). Do not use `dev-api-key`.
- [ ] **Single-tenant production**: With `TENANT_DEV_MODE=false` and `API_KEY` set, use `tenant_store_backend=memory` (default). The app registers one tenant with `tenant_id="default"`. Use `tenant_id: "default"` in API request bodies.
- [ ] **Multi-tenant production**: Set `TENANT_STORE_BACKEND=db`. Tenants are read from the `tenants` table (see `scripts/init_db.sql`). Seed tenants with `python scripts/seed_tenant.py` or the Admin API (when `ADMIN_API_KEY` is set): `GET/POST /v1/admin/tenants` with `Authorization: Bearer <ADMIN_API_KEY>`.

### CORS
- [ ] **Restrict CORS**: Set `CORS_ORIGINS` to your real front-end origins. Either comma-separated or JSON array, e.g.  
  `CORS_ORIGINS=https://app.example.com,https://admin.example.com`  
  or  
  `CORS_ORIGINS='["https://app.example.com","https://admin.example.com"]'`  
  Do not use `*` in production.

### Secrets
- [ ] **No secrets in code or image**: Use a secrets manager (e.g. AWS Secrets Manager, Vault) or platform env (e.g. Cloud Run env, Kubernetes secrets) for `DATABASE_URL`, `REDIS_URL`, `ANTHROPIC_API_KEY`, `API_KEY`, and any platform API keys (Shopify, Adobe Commerce, WooCommerce, BigCommerce).
- [ ] **Database**: Use a dedicated DB user with minimal required privileges; use SSL/TLS for `DATABASE_URL` in production (`?sslmode=require` or equivalent).

---

## 2. Environment & Config

### Required env in production
- `DATABASE_URL` – PostgreSQL with pgvector (use managed DB; enable SSL).
- `REDIS_URL` – Redis (use managed Redis or secure internal endpoint).
- `API_KEY` – Strong secret; required when `TENANT_DEV_MODE=false`.
- `SERVICE_ENVIRONMENT=production` – Drives telemetry and logging behavior.

### Recommended env
- `CORS_ORIGINS` – JSON array of allowed origins (see above).
- `TENANT_DEV_MODE=false`.
- `LOG_LEVEL=INFO` (or `WARNING` to reduce volume).
- `LOG_JSON=true` – Keep for structured logging in production.
- `OTLP_ENDPOINT` – Your OTLP collector (e.g. Grafana Cloud, Datadog, self-hosted). Set to empty or disable tracing if not used: `ENABLE_TRACING=false`, `ENABLE_METRICS=false`.

### Startup validation (production only)
- When `SERVICE_ENVIRONMENT=production`, the app logs **warnings** (does not fail) for: `TENANT_DEV_MODE=true`, default/empty `API_KEY` in single-tenant mode, or `CORS_ORIGINS` including `*`. Check startup logs after deploy.

### Optional (features)
- `ANTHROPIC_API_KEY` – Required for LLM reasoning path; without it only fast-path resolution works.
- Platform keys (Shopify, Adobe Commerce, WooCommerce, BigCommerce) – Only if you use those integrations.
- `ENABLE_TRACING`, `ENABLE_METRICS` – Set to `true` if you run an OTLP/Prometheus stack.

---

## 3. Infrastructure

### Database
- [ ] **PostgreSQL with pgvector**: Use a managed instance or run migrations from `scripts/init_db.sql` (or equivalent) so the schema and extension exist. The init script creates the `tenants` table for DB-backed multi-tenancy. If the DB already existed before that was added, run `just migrate-tenants` (or `psql $DATABASE_URL -f scripts/migrate_tenants_table.sql`) to create the table.
- [ ] **Seed catalog**: Run `just seed` (or `python scripts/seed_catalog.py`) once per environment so the intent catalog and embeddings are populated.
- [ ] **Seed tenants (DB backend)**: When using `TENANT_STORE_BACKEND=db`, add tenants with `python scripts/seed_tenant.py --tenant-id <id> --name "<Name>" --api-key <key> [--tier professional]`.
- [ ] **Backups**: Enable automated backups and point-in-time recovery on the DB.

### Redis
- [ ] **Persistence**: Use Redis with persistence (RDB/AOF) if you rely on it for rate limiting or batch queue state.
- [ ] **High availability**: For production, consider a managed Redis cluster or replica set.

### Compute
- [ ] **No `--reload`**: Run the app without hot reload. The default `Dockerfile` CMD is correct:  
  `uvicorn intent_engine.api.server:app --host 0.0.0.0 --port 8000`
- [ ] **Workers**: For more throughput, run multiple Uvicorn workers (e.g. `--workers 4`). Ensure your embedding/LLM usage is safe with concurrency (no global mutable state).
- [ ] **Resource limits**: Set CPU/memory limits and requests (e.g. in Kubernetes or Docker); the engine loads spaCy and sentence-transformers at startup.

### TLS & ingress
- [ ] **Terminate TLS at load balancer/reverse proxy**: Do not expose the app directly on the internet. Use nginx, Cloud Load Balancer, or your platform’s HTTPS termination.
- [ ] **Security headers**: The API adds `X-Content-Type-Options: nosniff` and `X-Frame-Options: DENY` to all responses.
- [ ] **Health checks**: Use `/health` for liveness and `/ready` for readiness (DB + Redis + tenant store when backend=db). Configure your orchestrator to use `/ready` for traffic.

---

## 4. Observability

- [ ] **Structured logs**: Keep `LOG_JSON=true` and ship logs to your log aggregator (e.g. Cloud Logging, Datadog, ELK).
- [ ] **Metrics**: If enabled, scrape `/metrics` (Prometheus) or export via OTLP. Dashboards are under `etc/grafana/provisioning/`.
- [ ] **Tracing**: If enabled, set `OTLP_ENDPOINT` to your collector. Jaeger/Prometheus/Grafana can be run with `docker-compose --profile observability up -d`.
- [ ] **Alerts**: Define alerts on readiness failures, error rate, latency (e.g. p99), and rate-limit hits.

---

## 5. Application Behavior

- [ ] **Rate limiting**: With Redis and multi-tenant enabled, rate limits are applied per tenant. Tune `DEFAULT_RATE_LIMIT_RPM` and `DEFAULT_RATE_LIMIT_BURST` as needed.
- [ ] **Batch worker**: If you use batch processing, ensure the batch worker runs (e.g. same container or a dedicated worker process) and that `BATCH_WORKER_ENABLED=true` where appropriate. In production, tenant discovery may need to be driven by your tenant store rather than a hardcoded list.
- [ ] **WebSockets**: If used, ensure `WS_ENABLED=true` and that your proxy supports WebSocket upgrade; tune `WS_MAX_CONNECTIONS_PER_TENANT` and `WS_PING_INTERVAL` if needed.

---

## 6. CI/CD and Releases

- [ ] **CI**: The existing GitHub Actions workflow runs lint, typecheck, and tests (including integration tests with Postgres/Redis). Keep these green on main.
- [ ] **Build**: Use the project `Dockerfile` to build a production image; avoid mounting local `src` in production (the Dockerfile already copies `src/`, `data/`, `scripts/`, `evals/`). Use the production Compose override: `docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build` (or `just up-prod`). See `docker-compose.prod.yml` for no-reload, no source mounts, and production env defaults.
- [ ] **Secrets in CI**: Do not put production secrets in GitHub; use environment secrets or a secrets manager for any production deploy step.
- [ ] **Migrations**: If you add DB migrations later, run them as a separate step or init container before starting the API.

---

## 7. Quick production env example

```bash
# Required
export DATABASE_URL="postgresql://user:pass@dbhost:5432/intent_engine?sslmode=require"
export REDIS_URL="redis://redishost:6379/0"
export API_KEY="<strong-random-key>"
export SERVICE_ENVIRONMENT=production
export TENANT_DEV_MODE=false

# CORS (comma-separated or JSON array)
export CORS_ORIGINS="https://your-app.example.com"

# Optional: LLM and observability
export ANTHROPIC_API_KEY="sk-ant-..."
export OTLP_ENDPOINT="https://your-otel-collector"
export ENABLE_TRACING=true
export ENABLE_METRICS=true
```

---

## Deploy runbook

For an ordered step-by-step deploy, see **[docs/DEPLOY.md](DEPLOY.md)**.

## Summary

| Area            | Action |
|-----------------|--------|
| Security        | `TENANT_DEV_MODE=false`, strong `API_KEY`, restrict `CORS_ORIGINS`, secrets from vault/env |
| Database        | Managed Postgres + pgvector, run init script, seed catalog, backups |
| Redis           | Managed or persistent Redis; used for rate limiting and batch queue |
| App process     | Run uvicorn without `--reload`; optional multi-worker |
| TLS / exposure  | TLS at proxy; use `/ready` for readiness |
| Observability   | JSON logs, `/metrics`, OTLP if used; alerts on health and latency |
| Tenancy         | Set `TENANT_STORE_BACKEND=db` and seed tenants with `scripts/seed_tenant.py` |
