"""Redis-based token bucket rate limiter."""

import logging
import time
from typing import Any

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(
        self,
        tenant_id: str,
        limit: int,
        retry_after: float,
    ) -> None:
        self.tenant_id = tenant_id
        self.limit = limit
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded for tenant {tenant_id}. "
            f"Limit: {limit}/min. Retry after: {retry_after:.2f}s"
        )


class RateLimiter:
    """
    Redis-based token bucket rate limiter.

    Uses a sliding window token bucket algorithm:
    - Tokens are added at a constant rate (rate/60 per second)
    - Requests consume tokens
    - If no tokens available, request is rejected
    - Supports burst by allowing bucket to fill up to burst_size

    Redis keys:
    - rate_limit:{tenant_id}:tokens - Current token count
    - rate_limit:{tenant_id}:last_update - Last update timestamp
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        default_rate: int = 100,
        default_burst: int = 20,
    ) -> None:
        """
        Initialize the rate limiter.

        Args:
            redis_client: Async Redis client.
            default_rate: Default requests per minute.
            default_burst: Default burst size (max tokens).
        """
        self.redis = redis_client
        self.default_rate = default_rate
        self.default_burst = default_burst

    def _key_tokens(self, tenant_id: str) -> str:
        """Get Redis key for token count."""
        return f"rate_limit:{tenant_id}:tokens"

    def _key_last_update(self, tenant_id: str) -> str:
        """Get Redis key for last update timestamp."""
        return f"rate_limit:{tenant_id}:last_update"

    async def check_rate_limit(
        self,
        tenant_id: str,
        rate_limit: int | None = None,
        burst_size: int | None = None,
        tokens_required: int = 1,
    ) -> dict[str, Any]:
        """
        Check if a request is allowed under rate limiting.

        Uses atomic Redis operations to ensure thread safety.

        Args:
            tenant_id: The tenant ID to check.
            rate_limit: Rate limit in requests per minute (uses default if not provided).
            burst_size: Maximum burst size (uses default if not provided).
            tokens_required: Number of tokens this request consumes.

        Returns:
            Dict with rate limit info:
            - allowed: Whether the request is allowed
            - remaining: Remaining tokens
            - limit: The rate limit
            - reset_after: Seconds until tokens refill

        Raises:
            RateLimitExceeded: If rate limit is exceeded.
        """
        rate = rate_limit or self.default_rate
        burst = burst_size or self.default_burst

        # Calculate tokens per second
        tokens_per_second = rate / 60.0

        key_tokens = self._key_tokens(tenant_id)
        key_last_update = self._key_last_update(tenant_id)

        # Use Lua script for atomic operation
        lua_script = """
        local key_tokens = KEYS[1]
        local key_last_update = KEYS[2]
        local rate_per_sec = tonumber(ARGV[1])
        local burst = tonumber(ARGV[2])
        local tokens_required = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])

        -- Get current state
        local tokens = tonumber(redis.call('GET', key_tokens)) or burst
        local last_update = tonumber(redis.call('GET', key_last_update)) or now

        -- Calculate tokens to add based on time elapsed
        local elapsed = now - last_update
        local tokens_to_add = elapsed * rate_per_sec
        tokens = math.min(burst, tokens + tokens_to_add)

        -- Check if we have enough tokens
        if tokens >= tokens_required then
            tokens = tokens - tokens_required
            redis.call('SET', key_tokens, tokens)
            redis.call('SET', key_last_update, now)
            redis.call('EXPIRE', key_tokens, 120)
            redis.call('EXPIRE', key_last_update, 120)
            return {1, tokens, 0}
        else
            -- Calculate time until we have enough tokens
            local tokens_needed = tokens_required - tokens
            local wait_time = tokens_needed / rate_per_sec
            return {0, tokens, wait_time}
        end
        """

        now = time.time()

        result = await self.redis.eval(
            lua_script,
            2,
            key_tokens,
            key_last_update,
            str(tokens_per_second),
            str(burst),
            str(tokens_required),
            str(now),
        )

        allowed = bool(result[0])
        remaining = float(result[1])
        retry_after = float(result[2])

        if not allowed:
            logger.warning(
                f"Rate limit exceeded for tenant {tenant_id}. "
                f"Remaining: {remaining:.2f}, Retry after: {retry_after:.2f}s"
            )
            raise RateLimitExceeded(
                tenant_id=tenant_id,
                limit=rate,
                retry_after=retry_after,
            )

        return {
            "allowed": True,
            "remaining": int(remaining),
            "limit": rate,
            "reset_after": 60.0 / rate if remaining < 1 else 0,
        }

    async def get_usage(self, tenant_id: str) -> dict[str, Any]:
        """
        Get current rate limit usage for a tenant.

        Args:
            tenant_id: The tenant ID to check.

        Returns:
            Dict with current usage info.
        """
        key_tokens = self._key_tokens(tenant_id)

        tokens = await self.redis.get(key_tokens)
        tokens = float(tokens) if tokens else self.default_burst

        return {
            "tenant_id": tenant_id,
            "remaining_tokens": int(tokens),
            "max_tokens": self.default_burst,
        }

    async def reset(self, tenant_id: str) -> None:
        """
        Reset rate limit for a tenant.

        Args:
            tenant_id: The tenant ID to reset.
        """
        key_tokens = self._key_tokens(tenant_id)
        key_last_update = self._key_last_update(tenant_id)

        await self.redis.delete(key_tokens, key_last_update)
        logger.info(f"Reset rate limit for tenant {tenant_id}")
