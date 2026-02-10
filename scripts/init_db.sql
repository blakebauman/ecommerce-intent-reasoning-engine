-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Intent catalog table for storing intent examples with embeddings
CREATE TABLE IF NOT EXISTS intent_catalog (
    id SERIAL PRIMARY KEY,
    intent_code VARCHAR(100) NOT NULL,
    category VARCHAR(50) NOT NULL,
    example_text TEXT NOT NULL,
    embedding vector(384),  -- all-MiniLM-L6-v2 produces 384-dim vectors
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create HNSW index for fast similarity search
CREATE INDEX IF NOT EXISTS intent_catalog_embedding_idx
ON intent_catalog
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Index for intent lookups
CREATE INDEX IF NOT EXISTS intent_catalog_intent_code_idx
ON intent_catalog (intent_code);

-- Conversation history table for multi-turn context
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(100) UNIQUE NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    customer_id VARCHAR(100),
    messages JSONB DEFAULT '[]'::jsonb,
    resolved_intents JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS conversations_tenant_idx ON conversations (tenant_id);
CREATE INDEX IF NOT EXISTS conversations_customer_idx ON conversations (customer_id);

-- Audit log for decision tracking
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(100) UNIQUE NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    input_text TEXT,
    extracted_entities JSONB,
    resolved_intents JSONB,
    path_taken VARCHAR(20),  -- 'fast_path' or 'reasoning_path'
    confidence FLOAT,
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS audit_log_tenant_idx ON audit_log (tenant_id);
CREATE INDEX IF NOT EXISTS audit_log_created_at_idx ON audit_log (created_at DESC);
