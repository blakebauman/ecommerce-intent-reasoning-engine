"""Webhook receiver endpoints for platform integrations."""

import logging
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, status

from intent_engine.config import get_settings
from intent_engine.integrations.adobe_commerce import AdobeCommerceWebhookHandler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# Webhook handler instances (initialized lazily)
_adobe_commerce_handler: AdobeCommerceWebhookHandler | None = None


def get_adobe_commerce_handler() -> AdobeCommerceWebhookHandler | None:
    """Get the Adobe Commerce webhook handler."""
    global _adobe_commerce_handler

    settings = get_settings()

    if not settings.adobe_commerce_webhook_enabled:
        return None

    if _adobe_commerce_handler is None and settings.adobe_commerce_webhook_secret:
        _adobe_commerce_handler = AdobeCommerceWebhookHandler(
            webhook_secret=settings.adobe_commerce_webhook_secret,
        )

    return _adobe_commerce_handler


def set_adobe_commerce_handler(handler: AdobeCommerceWebhookHandler | None) -> None:
    """Set the Adobe Commerce webhook handler (for testing)."""
    global _adobe_commerce_handler
    _adobe_commerce_handler = handler


@router.post(
    "/adobe-commerce",
    summary="Receive Adobe Commerce webhooks",
    description="Endpoint for Adobe Commerce webhook events (orders, shipments, etc.).",
    response_model=dict[str, Any],
)
async def receive_adobe_commerce_webhook(
    request: Request,
    x_adobe_signature: str = Header(
        ...,
        alias="X-Adobe-Signature",
        description="HMAC-SHA256 signature for payload verification",
    ),
) -> dict[str, Any]:
    """
    Receive and process Adobe Commerce webhook events.

    Verifies the webhook signature and routes to the appropriate handler.

    Supported events:
    - observer.sales_order_save_after - Order created/updated
    - observer.sales_order_shipment_save_after - Shipment created
    - plugin.magento.sales.api.order_management.cancel - Order cancelled
    - observer.sales_order_creditmemo_save_after - Refund created

    Returns:
        Processing result with status and event details.
    """
    settings = get_settings()

    # Check if webhooks are enabled
    if not settings.adobe_commerce_webhook_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Adobe Commerce webhooks are disabled",
        )

    handler = get_adobe_commerce_handler()
    if not handler:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Adobe Commerce webhook handler not configured",
        )

    # Read raw body for signature verification
    body = await request.body()

    # Verify signature
    if not handler.verify_signature(body, x_adobe_signature):
        logger.warning("Invalid Adobe Commerce webhook signature")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook signature",
        )

    # Parse the event
    try:
        event = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse webhook payload: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON payload",
        )

    # Process the event
    try:
        result = await handler.handle_event(event)
        return result
    except Exception as e:
        logger.exception(f"Error processing Adobe Commerce webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing webhook event",
        )


@router.get(
    "/health",
    summary="Webhook health check",
    description="Check webhook endpoint health and configuration status.",
)
async def webhook_health() -> dict[str, Any]:
    """
    Check webhook endpoint health.

    Returns configuration status for all platform webhooks.
    """
    settings = get_settings()

    return {
        "status": "healthy",
        "platforms": {
            "adobe_commerce": {
                "enabled": settings.adobe_commerce_webhook_enabled,
                "configured": bool(settings.adobe_commerce_webhook_secret),
            },
        },
    }
