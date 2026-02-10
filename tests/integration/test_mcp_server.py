"""Integration tests for the MCP server."""

import json
import sys

import pytest

# Skip entire module on Python 3.14+ due to spaCy incompatibility
if sys.version_info >= (3, 14):
    pytest.skip(
        "spaCy not compatible with Python 3.14+ (uses Pydantic v1)",
        allow_module_level=True,
    )

from mcp.types import TextContent

from intent_engine.mcp.server import (
    _handle_classify_fast,
    _handle_list_taxonomy,
    _handle_resolve_intent,
    create_mcp_server,
)


class TestMCPServer:
    """Tests for the MCP server implementation."""

    @pytest.fixture
    def server(self):
        """Create MCP server instance."""
        return create_mcp_server()

    def test_server_creation(self, server):
        """Test that server is created with correct name."""
        assert server.name == "intent-engine"

    def test_server_has_list_tools_handler(self, server):
        """Test that server has list_tools handler registered."""
        from mcp.types import ListToolsRequest
        assert ListToolsRequest in server.request_handlers

    def test_server_has_call_tool_handler(self, server):
        """Test that server has call_tool handler registered."""
        from mcp.types import CallToolRequest
        assert CallToolRequest in server.request_handlers


class TestMCPToolExecution:
    """Tests for MCP tool execution."""

    @pytest.mark.asyncio
    async def test_list_taxonomy_execution(self):
        """Test list_intent_taxonomy returns valid data."""
        result = await _handle_list_taxonomy({})

        assert len(result) == 1
        assert isinstance(result[0], TextContent)

        data = json.loads(result[0].text)
        assert "intent_count" in data
        assert "intents" in data
        assert "categories" in data
        assert data["intent_count"] == 8  # 8 core MVP intents

    @pytest.mark.asyncio
    async def test_list_taxonomy_returns_categories(self):
        """Test list_intent_taxonomy returns expected categories."""
        result = await _handle_list_taxonomy({})
        data = json.loads(result[0].text)

        expected_categories = ["ORDER_STATUS", "ORDER_MODIFY", "RETURN_EXCHANGE", "COMPLAINT"]
        for category in expected_categories:
            assert category in data["categories"]

    @pytest.mark.asyncio
    async def test_list_taxonomy_returns_intents(self):
        """Test list_intent_taxonomy returns intent details."""
        result = await _handle_list_taxonomy({})
        data = json.loads(result[0].text)

        assert len(data["intents"]) == 8
        for intent in data["intents"]:
            assert "intent_code" in intent
            assert "category" in intent
            assert "description" in intent

    @pytest.mark.asyncio
    async def test_resolve_intent_missing_text(self):
        """Test resolve_intent returns error without raw_text."""
        result = await _handle_resolve_intent({})

        assert len(result) == 1
        assert "error" in result[0].text.lower()
        assert "raw_text" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_resolve_intent_empty_text(self):
        """Test resolve_intent returns error with empty raw_text."""
        result = await _handle_resolve_intent({"raw_text": ""})

        assert len(result) == 1
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_classify_fast_missing_text(self):
        """Test classify_intent_fast returns error without raw_text."""
        result = await _handle_classify_fast({})

        assert len(result) == 1
        assert "error" in result[0].text.lower()
        assert "raw_text" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_classify_fast_empty_text(self):
        """Test classify_intent_fast returns error with empty raw_text."""
        result = await _handle_classify_fast({"raw_text": ""})

        assert len(result) == 1
        assert "error" in result[0].text.lower()


class TestMCPWithDatabase:
    """Tests requiring database connection."""

    @pytest.mark.skip(reason="Requires database connection with seeded catalog")
    @pytest.mark.asyncio
    async def test_resolve_intent_with_text(self):
        """Test resolve_intent with valid text."""
        result = await _handle_resolve_intent({"raw_text": "Where is my order?"})

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "resolved_intents" in data
        assert "confidence_summary" in data

    @pytest.mark.skip(reason="Requires database connection with seeded catalog")
    @pytest.mark.asyncio
    async def test_classify_fast_with_text(self):
        """Test classify_intent_fast with valid text."""
        result = await _handle_classify_fast({"raw_text": "Where is my order?"})

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "decision" in data
        assert "top_matches" in data
