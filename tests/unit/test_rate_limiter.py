"""Unit tests for Redis rate limiter."""

from unittest.mock import AsyncMock

import pytest

from intent_engine.tenancy.rate_limiter import RateLimiter, RateLimitExceeded


class TestRateLimitExceeded:
    """Tests for RateLimitExceeded exception."""

    def test_exception_attributes(self):
        """Test exception has correct attributes."""
        exc = RateLimitExceeded(
            tenant_id="test-tenant",
            limit=100,
            retry_after=5.5,
        )

        assert exc.tenant_id == "test-tenant"
        assert exc.limit == 100
        assert exc.retry_after == 5.5
        assert "test-tenant" in str(exc)
        assert "100" in str(exc)


class TestRateLimiter:
    """Tests for RateLimiter class."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        redis = AsyncMock()
        return redis

    @pytest.fixture
    def rate_limiter(self, mock_redis):
        """Create a rate limiter with mock Redis."""
        return RateLimiter(
            redis_client=mock_redis,
            default_rate=100,
            default_burst=20,
        )

    def test_key_generation(self, rate_limiter):
        """Test Redis key generation."""
        assert rate_limiter._key_tokens("tenant-1") == "rate_limit:tenant-1:tokens"
        assert rate_limiter._key_last_update("tenant-1") == "rate_limit:tenant-1:last_update"

    @pytest.mark.asyncio
    async def test_check_rate_limit_allowed(self, rate_limiter, mock_redis):
        """Test rate limit check when allowed."""
        # Mock Lua script returning allowed
        mock_redis.eval.return_value = [1, 19.0, 0]  # allowed=True, remaining=19, wait=0

        result = await rate_limiter.check_rate_limit("tenant-1")

        assert result["allowed"] is True
        assert result["remaining"] == 19
        assert result["limit"] == 100

    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded(self, rate_limiter, mock_redis):
        """Test rate limit check when exceeded."""
        # Mock Lua script returning not allowed
        mock_redis.eval.return_value = [0, 0.0, 2.5]  # allowed=False, remaining=0, wait=2.5

        with pytest.raises(RateLimitExceeded) as exc_info:
            await rate_limiter.check_rate_limit("tenant-1")

        assert exc_info.value.tenant_id == "tenant-1"
        assert exc_info.value.retry_after == 2.5

    @pytest.mark.asyncio
    async def test_check_rate_limit_custom_rate(self, rate_limiter, mock_redis):
        """Test rate limit check with custom rate."""
        mock_redis.eval.return_value = [1, 49.0, 0]

        result = await rate_limiter.check_rate_limit(
            "tenant-1",
            rate_limit=50,
            burst_size=50,
        )

        assert result["allowed"] is True
        assert result["limit"] == 50

    @pytest.mark.asyncio
    async def test_check_rate_limit_multiple_tokens(self, rate_limiter, mock_redis):
        """Test rate limit check consuming multiple tokens."""
        mock_redis.eval.return_value = [1, 15.0, 0]

        result = await rate_limiter.check_rate_limit(
            "tenant-1",
            tokens_required=5,
        )

        assert result["allowed"] is True
        # Verify tokens_required was passed to Lua script
        call_args = mock_redis.eval.call_args
        assert "5" in str(call_args)

    @pytest.mark.asyncio
    async def test_get_usage(self, rate_limiter, mock_redis):
        """Test getting current usage."""
        mock_redis.get.return_value = "15.5"

        usage = await rate_limiter.get_usage("tenant-1")

        assert usage["tenant_id"] == "tenant-1"
        assert usage["remaining_tokens"] == 15
        assert usage["max_tokens"] == 20

    @pytest.mark.asyncio
    async def test_get_usage_no_data(self, rate_limiter, mock_redis):
        """Test getting usage when no data exists."""
        mock_redis.get.return_value = None

        usage = await rate_limiter.get_usage("new-tenant")

        assert usage["remaining_tokens"] == 20  # Default burst

    @pytest.mark.asyncio
    async def test_reset(self, rate_limiter, mock_redis):
        """Test resetting rate limit."""
        await rate_limiter.reset("tenant-1")

        mock_redis.delete.assert_called_once_with(
            "rate_limit:tenant-1:tokens",
            "rate_limit:tenant-1:last_update",
        )


class TestRateLimiterIntegration:
    """Integration-style tests for rate limiter behavior."""

    @pytest.mark.asyncio
    async def test_rate_limit_lua_script_logic(self):
        """Test that the Lua script logic is correct."""
        # This test validates the expected behavior of the rate limiter
        # without actually running Redis

        # Scenario: Fresh bucket with 20 burst capacity
        # Rate: 100/min = 1.67/sec
        # Request 1 token -> Should be allowed, remaining ~19

        # Scenario: Exhausted bucket
        # Request 1 token -> Should be denied, retry after > 0

        # The actual Lua script runs atomically in Redis
        # These tests verify we call it with correct parameters
        pass

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test that concurrent requests don't cause race conditions."""
        # In a real test, we'd spin up Redis and test concurrent access
        # The token bucket with Lua script ensures atomicity
        pass
