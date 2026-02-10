"""Pytest configuration and fixtures."""

import os
import sys
from pathlib import Path

import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Set test environment variables
os.environ.setdefault("API_KEY", "test-api-key")
os.environ.setdefault("DATABASE_URL", "postgresql://intent_engine:intent_engine_dev@localhost:5432/intent_engine")


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_chat_input() -> dict:
    """Sample chat input for testing."""
    return {
        "message": "Where is my order #12345?",
        "session_id": "sess-123",
        "tenant_id": "merchant-1",
        "customer_email": "test@example.com",
        "customer_id": "cust-456",
    }


@pytest.fixture
def sample_intent_request():
    """Sample IntentRequest for testing."""
    from intent_engine.models.request import InputChannel, IntentRequest

    return IntentRequest(
        request_id="test-req-1",
        tenant_id="test-merchant",
        channel=InputChannel.CHAT,
        raw_text="Where is my order #12345?",
    )
