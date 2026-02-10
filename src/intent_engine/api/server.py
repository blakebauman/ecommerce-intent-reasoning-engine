"""FastAPI application server."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from intent_engine.api.a2a_routes import a2a_router
from intent_engine.api.middleware import RequestLoggingMiddleware
from intent_engine.api.routes import router, set_engine
from intent_engine.config import get_settings
from intent_engine.engine import IntentEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.

    Initializes and shuts down the intent engine.
    """
    logger.info("Starting Intent Engine...")

    # Initialize engine
    settings = get_settings()
    engine = IntentEngine(settings=settings)
    await engine.initialize()
    set_engine(engine)

    logger.info("Intent Engine initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down Intent Engine...")
    await engine.shutdown()
    logger.info("Intent Engine shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=(
            "eCommerce Intent Reasoning Engine API. "
            "Classifies and decomposes customer intents from text messages."
        ),
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add request logging middleware
    app.add_middleware(RequestLoggingMiddleware)

    # Include routes
    app.include_router(router)

    # Include A2A protocol routes
    app.include_router(a2a_router)

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "intent_engine.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
