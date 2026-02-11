-- Migration: add tenants table for DB-backed multi-tenancy.
-- Run on existing DBs that were created before this table was added.
-- Usage: psql "$DATABASE_URL" -f scripts/migrate_tenants_table.sql

-- Tenants table for multi-tenant production (API key → tenant lookup)
CREATE TABLE IF NOT EXISTS tenants (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    api_key VARCHAR(255) UNIQUE NOT NULL,
    tier VARCHAR(50) NOT NULL DEFAULT 'starter',
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    settings JSONB DEFAULT '{}'::jsonb
);

CREATE UNIQUE INDEX IF NOT EXISTS tenants_api_key_idx ON tenants (api_key) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS tenants_tenant_id_idx ON tenants (tenant_id);
CREATE INDEX IF NOT EXISTS tenants_is_active_idx ON tenants (is_active);
