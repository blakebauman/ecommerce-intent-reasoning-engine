"""Tenant models and configuration."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class TenantTier(str, Enum):
    """Tenant subscription tiers with different rate limits and features."""

    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


# Default rate limits by tier (requests per minute)
TIER_RATE_LIMITS: dict[TenantTier, dict[str, int]] = {
    TenantTier.FREE: {
        "requests_per_minute": 20,
        "burst_size": 5,
        "max_batch_size": 10,
        "max_websocket_connections": 2,
    },
    TenantTier.STARTER: {
        "requests_per_minute": 60,
        "burst_size": 15,
        "max_batch_size": 100,
        "max_websocket_connections": 10,
    },
    TenantTier.PROFESSIONAL: {
        "requests_per_minute": 200,
        "burst_size": 50,
        "max_batch_size": 500,
        "max_websocket_connections": 50,
    },
    TenantTier.ENTERPRISE: {
        "requests_per_minute": 1000,
        "burst_size": 200,
        "max_batch_size": 2000,
        "max_websocket_connections": 500,
    },
}


class TenantConfig(BaseModel):
    """Configuration for a tenant."""

    tenant_id: str
    name: str
    tier: TenantTier = TenantTier.STARTER

    # API key for authentication
    api_key: str

    # Rate limiting (overrides tier defaults if set)
    requests_per_minute: int | None = None
    burst_size: int | None = None
    max_batch_size: int | None = None
    max_websocket_connections: int | None = None

    # Feature flags
    fast_path_enabled: bool = True
    reasoning_path_enabled: bool = True
    batch_processing_enabled: bool = True
    websocket_enabled: bool = True

    # Platform integrations (which platforms are configured)
    shopify_enabled: bool = False
    woocommerce_enabled: bool = False
    bigcommerce_enabled: bool = False
    adobe_commerce_enabled: bool = False

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

    def get_rate_limit(self) -> int:
        """Get the rate limit for this tenant."""
        if self.requests_per_minute is not None:
            return self.requests_per_minute
        return TIER_RATE_LIMITS[self.tier]["requests_per_minute"]

    def get_burst_size(self) -> int:
        """Get the burst size for this tenant."""
        if self.burst_size is not None:
            return self.burst_size
        return TIER_RATE_LIMITS[self.tier]["burst_size"]

    def get_max_batch_size(self) -> int:
        """Get the maximum batch size for this tenant."""
        if self.max_batch_size is not None:
            return self.max_batch_size
        return TIER_RATE_LIMITS[self.tier]["max_batch_size"]

    def get_max_websocket_connections(self) -> int:
        """Get the maximum WebSocket connections for this tenant."""
        if self.max_websocket_connections is not None:
            return self.max_websocket_connections
        return TIER_RATE_LIMITS[self.tier]["max_websocket_connections"]

    model_config = {"json_schema_extra": {"examples": [
        {
            "tenant_id": "tenant-001",
            "name": "Acme Corp",
            "tier": "professional",
            "api_key": "ak_live_xxx",
            "shopify_enabled": True,
        }
    ]}}


class TenantUsageStats(BaseModel):
    """Usage statistics for a tenant."""

    tenant_id: str
    period_start: datetime
    period_end: datetime

    # Request counts
    total_requests: int = 0
    fast_path_requests: int = 0
    reasoning_path_requests: int = 0
    batch_requests: int = 0
    websocket_messages: int = 0

    # Rate limiting
    rate_limit_exceeded_count: int = 0

    # Latency
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # LLM usage
    llm_calls: int = 0
    llm_tokens_input: int = 0
    llm_tokens_output: int = 0

    model_config = {"json_schema_extra": {"examples": [
        {
            "tenant_id": "tenant-001",
            "period_start": "2024-01-01T00:00:00Z",
            "period_end": "2024-01-31T23:59:59Z",
            "total_requests": 50000,
            "fast_path_requests": 35000,
            "reasoning_path_requests": 15000,
        }
    ]}}
