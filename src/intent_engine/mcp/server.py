"""MCP server implementation with SSE transport for remote agent access."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool
from starlette.applications import Starlette
from starlette.routing import Route

from intent_engine.config import Settings, get_settings
from intent_engine.engine import IntentEngine
from intent_engine.models.request import InputChannel, IntentRequest
from intent_engine.storage.intent_catalog import IntentCatalogStore

logger = logging.getLogger(__name__)

# Global engine instance for MCP server
_mcp_engine: IntentEngine | None = None


async def get_engine() -> IntentEngine:
    """Get or initialize the intent engine."""
    global _mcp_engine
    if _mcp_engine is None:
        settings = get_settings()
        _mcp_engine = IntentEngine(settings=settings)
        await _mcp_engine.initialize()
    return _mcp_engine


def create_mcp_server() -> Server:
    """Create and configure the MCP server with intent resolution tools."""
    server = Server("intent-engine")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List all available MCP tools."""
        return [
            Tool(
                name="resolve_intent",
                description=(
                    "Resolve customer intent with full reasoning. "
                    "Takes a customer message and returns classified intents with "
                    "confidence scores, extracted entities, and reasoning trace."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "raw_text": {
                            "type": "string",
                            "description": "The customer message to classify",
                        },
                        "customer_tier": {
                            "type": "string",
                            "enum": ["standard", "VIP"],
                            "description": "Customer tier for priority routing",
                        },
                        "order_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Known order IDs for context",
                        },
                        "previous_intents": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Previously resolved intents in conversation",
                        },
                    },
                    "required": ["raw_text"],
                },
            ),
            Tool(
                name="classify_intent_fast",
                description=(
                    "Quick intent classification using embeddings only (no LLM). "
                    "Best for simple, single-intent messages. Returns top matches "
                    "with similarity scores."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "raw_text": {
                            "type": "string",
                            "description": "The customer message to classify",
                        },
                    },
                    "required": ["raw_text"],
                },
            ),
            Tool(
                name="list_intent_taxonomy",
                description=(
                    "Return all supported intent categories for agent planning. "
                    "Use this to understand what kinds of customer intents can be "
                    "resolved by this engine."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle tool invocations."""
        try:
            if name == "resolve_intent":
                return await _handle_resolve_intent(arguments)
            elif name == "classify_intent_fast":
                return await _handle_classify_fast(arguments)
            elif name == "list_intent_taxonomy":
                return await _handle_list_taxonomy(arguments)
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
        except Exception as e:
            logger.exception(f"Error executing tool {name}")
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    return server


async def _handle_resolve_intent(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle resolve_intent tool call."""
    raw_text = arguments.get("raw_text", "")
    if not raw_text:
        return [TextContent(type="text", text="Error: raw_text is required")]

    engine = await get_engine()

    request = IntentRequest(
        request_id="mcp-request",
        tenant_id="mcp-agent",
        channel=InputChannel.CHAT,
        raw_text=raw_text,
        customer_tier=arguments.get("customer_tier"),
        order_ids=arguments.get("order_ids", []),
        previous_intents=arguments.get("previous_intents", []),
    )

    result = await engine.resolve(request)

    # Convert to JSON-serializable format
    response = {
        "request_id": result.request_id,
        "resolved_intents": [
            {
                "category": intent.category,
                "intent": intent.intent,
                "confidence": intent.confidence,
                "confidence_tier": intent.confidence_tier.value,
                "evidence": intent.evidence,
            }
            for intent in result.resolved_intents
        ],
        "is_compound": result.is_compound,
        "entities": [
            {
                "entity_type": entity.entity_type,
                "value": entity.value,
                "confidence": entity.confidence,
            }
            for entity in result.entities
        ],
        "confidence_summary": result.confidence_summary,
        "requires_human": result.requires_human,
        "human_handoff_reason": result.human_handoff_reason,
        "path_taken": result.path_taken,
        "processing_time_ms": result.processing_time_ms,
    }

    return [TextContent(type="text", text=json.dumps(response, indent=2))]


async def _handle_classify_fast(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle classify_intent_fast tool call - embedding similarity only."""
    raw_text = arguments.get("raw_text", "")
    if not raw_text:
        return [TextContent(type="text", text="Error: raw_text is required")]

    engine = await get_engine()

    # Generate embedding and match
    embedding = engine.components.embedding_extractor.embed(raw_text)
    match_result = await engine.components.intent_matcher.match(
        text=raw_text,
        embedding=embedding,
    )

    response = {
        "decision": match_result.decision.value,
        "top_matches": [
            {
                "intent_code": match.intent_code,
                "similarity": round(match.similarity, 4),
                "matched_example": match.matched_example[:100],
            }
            for match in match_result.top_matches[:5]
        ],
    }

    if match_result.resolved_intent:
        response["resolved_intent"] = {
            "category": match_result.resolved_intent.category,
            "intent": match_result.resolved_intent.intent,
            "confidence": match_result.resolved_intent.confidence,
        }

    return [TextContent(type="text", text=json.dumps(response, indent=2))]


async def _handle_list_taxonomy(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle list_intent_taxonomy tool call."""
    catalog_store = IntentCatalogStore(
        vector_store=None,  # type: ignore - not needed for static data
        embedding_extractor=None,
    )
    intents = catalog_store.get_core_intents()

    response = {
        "intent_count": len(intents),
        "intents": intents,
        "categories": list(set(i["category"] for i in intents)),
    }

    return [TextContent(type="text", text=json.dumps(response, indent=2))]


def create_sse_app(server: Server) -> Starlette:
    """Create a Starlette app with SSE transport for the MCP server."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(
                streams[0], streams[1], server.create_initialization_options()
            )

    async def handle_messages(request):
        """Wrap the SSE transport's handle_post_message for Starlette."""
        await sse.handle_post_message(request.scope, request.receive, request._send)

    return Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Route("/messages/", endpoint=handle_messages, methods=["POST"]),
        ],
    )


@asynccontextmanager
async def mcp_lifespan(settings: Settings) -> AsyncIterator[None]:
    """Lifespan manager for MCP server."""
    global _mcp_engine
    logger.info("Starting MCP server...")

    # Initialize engine
    _mcp_engine = IntentEngine(settings=settings)
    await _mcp_engine.initialize()
    logger.info("MCP Intent Engine initialized")

    yield

    # Cleanup
    if _mcp_engine:
        await _mcp_engine.shutdown()
        _mcp_engine = None
    logger.info("MCP server shutdown complete")


async def run_mcp_server(host: str = "0.0.0.0", port: int = 8001) -> None:
    """Run the MCP server with SSE transport."""
    import uvicorn

    settings = get_settings()

    # Initialize engine before starting server
    global _mcp_engine
    _mcp_engine = IntentEngine(settings=settings)
    await _mcp_engine.initialize()

    server = create_mcp_server()
    app = create_sse_app(server)

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server_instance = uvicorn.Server(config)

    logger.info(f"MCP server starting on http://{host}:{port}")
    logger.info("SSE endpoint: /sse")
    logger.info("Message endpoint: /messages/")

    try:
        await server_instance.serve()
    finally:
        if _mcp_engine:
            await _mcp_engine.shutdown()


def main() -> None:
    """Entry point for running MCP server standalone."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(run_mcp_server())


if __name__ == "__main__":
    main()
