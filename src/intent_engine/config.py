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


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
