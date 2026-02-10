"""Configuration management using pydantic-settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    database_url: str = Field(
        default="postgresql://intent_engine:intent_engine_dev@localhost:5432/intent_engine",
        description="PostgreSQL connection string",
    )

    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection string",
    )

    # Anthropic / Pydantic AI
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key for Claude",
    )

    # LLM Settings
    llm_model: str = Field(
        default="claude-sonnet-4-5",
        description="Default LLM model for reasoning (e.g., claude-sonnet-4-5, claude-opus-4-5)",
    )
    llm_max_tokens: int = Field(
        default=2048,
        description="Max tokens for LLM responses",
    )
    llm_temperature: float = Field(
        default=0.1,
        description="LLM temperature (lower = more deterministic)",
    )

    # Embedding Model
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )

    # Intent Matching Thresholds
    fast_path_threshold: float = Field(
        default=0.85,
        description="Minimum similarity for fast path resolution",
    )
    ambiguity_gap_threshold: float = Field(
        default=0.10,
        description="Minimum gap between top-2 matches to avoid ambiguity",
    )
    compound_detection_threshold: float = Field(
        default=0.60,
        description="Confidence threshold for compound intent detection",
    )
    low_confidence_threshold: float = Field(
        default=0.60,
        description="Below this, may need clarification",
    )

    # Shopify Integration
    shopify_api_key: str = Field(
        default="",
        description="Shopify API key",
    )
    shopify_api_secret: str = Field(
        default="",
        description="Shopify API secret",
    )
    shopify_store_domain: str = Field(
        default="",
        description="Shopify store domain",
    )
    shopify_access_token: str = Field(
        default="",
        description="Shopify Admin API access token",
    )

    # Adobe Commerce Integration - PaaS (self-hosted)
    adobe_commerce_base_url: str = Field(
        default="",
        description="Adobe Commerce store base URL",
    )
    adobe_commerce_access_token: str = Field(
        default="",
        description="Adobe Commerce integration access token (PaaS)",
    )
    adobe_commerce_store_code: str = Field(
        default="default",
        description="Adobe Commerce store view code",
    )

    # Adobe Commerce Integration - SaaS (Cloud Service)
    adobe_commerce_ims_client_id: str = Field(
        default="",
        description="Adobe IMS client ID for OAuth (SaaS)",
    )
    adobe_commerce_ims_client_secret: str = Field(
        default="",
        description="Adobe IMS client secret for OAuth (SaaS)",
    )
    adobe_commerce_ims_org_id: str = Field(
        default="",
        description="Adobe organization ID (SaaS)",
    )

    # Adobe Commerce Webhooks
    adobe_commerce_webhook_secret: str = Field(
        default="",
        description="Adobe Commerce webhook HMAC secret",
    )
    adobe_commerce_webhook_enabled: bool = Field(
        default=True,
        description="Enable Adobe Commerce webhook processing",
    )

    # API Settings
    api_key: str = Field(
        default="dev-api-key",
        description="API key for authentication",
    )
    api_title: str = Field(
        default="Intent Reasoning Engine",
        description="API title",
    )
    api_version: str = Field(
        default="0.1.0",
        description="API version",
    )

    # Performance
    embedding_cache_ttl: int = Field(
        default=3600,
        description="TTL for embedding cache in seconds",
    )
    max_concurrent_requests: int = Field(
        default=100,
        description="Maximum concurrent API requests",
    )

    # spaCy
    spacy_model: str = Field(
        default="en_core_web_sm",
        description="spaCy model for NER",
    )

    # =========================================================================
    # Phase 3: Production Hardening Settings
    # =========================================================================

    # Observability - OpenTelemetry
    otlp_endpoint: str = Field(
        default="http://localhost:4317",
        description="OTLP gRPC endpoint for traces",
    )
    log_level: str = Field(
        default="INFO",
        description="Log level (DEBUG, INFO, WARNING, ERROR)",
    )
    log_json: bool = Field(
        default=True,
        description="Enable JSON structured logging",
    )
    enable_tracing: bool = Field(
        default=True,
        description="Enable OpenTelemetry tracing",
    )
    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus metrics",
    )
    service_name: str = Field(
        default="intent-engine",
        description="Service name for telemetry",
    )
    service_environment: str = Field(
        default="development",
        description="Deployment environment (development, staging, production)",
    )

    # Multi-Tenancy
    enable_multi_tenant: bool = Field(
        default=True,
        description="Enable multi-tenant mode",
    )
    default_rate_limit_rpm: int = Field(
        default=100,
        description="Default rate limit (requests per minute)",
    )
    default_rate_limit_burst: int = Field(
        default=20,
        description="Default burst size for rate limiting",
    )
    tenant_dev_mode: bool = Field(
        default=True,
        description="Enable dev mode (accept any API key)",
    )

    # Batch Processing
    batch_worker_concurrency: int = Field(
        default=5,
        description="Number of concurrent item processors",
    )
    batch_max_items: int = Field(
        default=1000,
        description="Maximum items per batch job",
    )
    batch_job_ttl_hours: int = Field(
        default=24,
        description="Hours to keep completed batch jobs",
    )
    batch_poll_interval: float = Field(
        default=1.0,
        description="Seconds between queue polls",
    )
    batch_worker_enabled: bool = Field(
        default=True,
        description="Enable background batch worker",
    )

    # WebSocket
    ws_ping_interval: int = Field(
        default=30,
        description="WebSocket ping interval in seconds",
    )
    ws_max_connections_per_tenant: int = Field(
        default=100,
        description="Maximum WebSocket connections per tenant",
    )
    ws_enabled: bool = Field(
        default=True,
        description="Enable WebSocket API",
    )

    # WooCommerce Integration
    woocommerce_store_url: str = Field(
        default="",
        description="WooCommerce store URL (e.g., https://mystore.com)",
    )
    woocommerce_consumer_key: str = Field(
        default="",
        description="WooCommerce REST API consumer key",
    )
    woocommerce_consumer_secret: str = Field(
        default="",
        description="WooCommerce REST API consumer secret",
    )
    woocommerce_webhook_secret: str = Field(
        default="",
        description="WooCommerce webhook HMAC secret",
    )

    # BigCommerce Integration
    bigcommerce_store_hash: str = Field(
        default="",
        description="BigCommerce store hash",
    )
    bigcommerce_access_token: str = Field(
        default="",
        description="BigCommerce API access token",
    )
    bigcommerce_client_secret: str = Field(
        default="",
        description="BigCommerce app client secret (for webhooks)",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
