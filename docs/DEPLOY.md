# Deploy to Production – Runbook

Ordered steps to deploy the Intent Reasoning Engine to production. See [PRODUCTION.md](PRODUCTION.md) for the full checklist.

## Prerequisites

- PostgreSQL (with pgvector) and Redis available
- Docker (optional; can run process directly)
- Secrets for `DATABASE_URL`, `REDIS_URL`, `API_KEY` (and optionally `ADMIN_API_KEY`, `ANTHROPIC_API_KEY`)

---

## 1. Database

1. **Create or use existing PostgreSQL** with pgvector (`CREATE EXTENSION vector;`).
2. **Run schema**:  
   - New DB: run full `scripts/init_db.sql` (e.g. `psql $DATABASE_URL -f scripts/init_db.sql`).  
   - Existing DB that already has other tables: run `just migrate-tenants` or `psql $DATABASE_URL -f scripts/migrate_tenants_table.sql` so the `tenants` table exists.
3. **Seed intent catalog** (once per environment):  
   `just seed` or `python scripts/seed_catalog.py`  
   (Or in Docker: `just seed-prod` after the API stack is up.)

---

## 2. Environment

Set (env file or host):

- `DATABASE_URL` – Postgres connection string (SSL in prod).
- `REDIS_URL` – Redis connection string.
- `API_KEY` – Strong secret (required when not in dev mode).
- `SERVICE_ENVIRONMENT=production`
- `TENANT_DEV_MODE=false`
- `CORS_ORIGINS` – Your front-end origins (comma-separated or JSON array).

**Single-tenant:**  
Leave `TENANT_STORE_BACKEND=memory` (default). The app will register one tenant with `tenant_id="default"`.

**Multi-tenant:**  
- `TENANT_STORE_BACKEND=db`
- Seed tenants: `just seed-tenant <tenant_id> "<Name>" <api_key> [tier]` or use the Admin API (see below).
- Optional: `ADMIN_API_KEY` – enables `GET/POST /v1/admin/tenants` for listing and creating/updating tenants.

---

## 3. Build context

The repo includes a **`.dockerignore`** so that `docker build` does not send `.git`, `.env`, `.venv`, `tests/`, or caches into the build context. This keeps builds faster and avoids leaking secrets. Do not remove it for production builds.

For production, the **Dockerfile** is built with **`INSTALL_DEV=false`** (no pytest, ruff, mypy, etc.), so the image is smaller. Use `just build-prod` or `just up-prod`; the prod Compose override passes this build arg automatically.

---

## 4. Run the app

**Docker (production override):**

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
# Or: just up-prod
```

**Or run the process directly** (no `--reload`):

```bash
uvicorn intent_engine.api.server:app --host 0.0.0.0 --port 8000
```

Ensure TLS is terminated at a reverse proxy or load balancer; do not expose the app directly to the internet.

---

## 5. Verify

- **Liveness:** `GET /health` → 200, `"status": "healthy"`.
- **Readiness:** `GET /ready` → 200, `"status": "ready"`, `database`, `redis` (and `tenant_store` when backend=db) all `"ok"`.  
  Point your orchestrator’s readiness probe at `/ready`.
- **Auth:**  
  `POST /v1/intent/resolve` with `Authorization: Bearer <API_KEY>` and body `{"request_id":"...","tenant_id":"default", ...}` → not 401.

---

## 6. Multi-tenant: add tenants

- **CLI:** `just seed-tenant <tenant_id> "<Name>" <api_key> [tier]`
- **Admin API** (when `TENANT_STORE_BACKEND=db` and `ADMIN_API_KEY` is set):  
  - List: `GET /v1/admin/tenants` with `Authorization: Bearer <ADMIN_API_KEY>`  
  - Create/update: `POST /v1/admin/tenants` with same header and JSON body (tenant_id, name, api_key, tier, is_active, etc.).

---

## 7. Optional

- **Observability:** Set `OTLP_ENDPOINT`, `ENABLE_TRACING=true`, `ENABLE_METRICS=true`; scrape `/metrics` or use OTLP.
- **Workers:** Run multiple Uvicorn workers (e.g. `--workers 2`) for more throughput.

---

## 8. Backups

- **PostgreSQL:** Back up the database regularly (daily or per-deploy). Include the `intent_catalog`, `tenants`, `conversations`, and `audit_log` tables. Use your provider’s point-in-time recovery or `pg_dump`; restore to the same or new instance and run migrations if needed.
- **Redis:** If you rely on rate limiting or batch queue state across restarts, use Redis persistence (RDB/AOF) and/or back up Redis. For cache-only use, backups are optional.
- **Secrets:** Store `API_KEY`, `ADMIN_API_KEY`, `DATABASE_URL`, and other secrets in a vault or secrets manager; avoid keeping production secrets only in env files on disk.
