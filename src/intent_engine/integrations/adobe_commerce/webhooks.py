"""Webhook handler for Adobe Commerce events."""

import hashlib
import hmac
import logging
from typing import Any

from intent_engine.integrations.adobe_commerce.mapping import (
    get_carrier_name,
    map_order_status,
)
from intent_engine.integrations.base import OrderStatus

logger = logging.getLogger(__name__)


class AdobeCommerceWebhookHandler:
    """
    Handler for Adobe Commerce webhook events.

    Supports Adobe Commerce Extensibility webhooks:
    - observer.sales_order_save_after - Order created/updated
    - observer.sales_order_shipment_save_after - Shipment created
    - plugin.magento.sales.api.order_management.cancel - Order cancelled

    Webhook signature verification uses HMAC-SHA256.
    """

    # Supported webhook event types
    ORDER_EVENTS = {
        "observer.sales_order_save_after",
        "plugin.magento.sales.api.order_management.place",
        "plugin.magento.sales.api.order_management.notify",
    }
    SHIPMENT_EVENTS = {
        "observer.sales_order_shipment_save_after",
        "plugin.magento.sales.api.shipment_repository.save",
    }
    CANCEL_EVENTS = {
        "plugin.magento.sales.api.order_management.cancel",
    }
    REFUND_EVENTS = {
        "observer.sales_order_creditmemo_save_after",
        "plugin.magento.sales.api.creditmemo_management.refund",
    }

    def __init__(
        self,
        webhook_secret: str,
        on_order_update: Any = None,
        on_shipment: Any = None,
        on_cancellation: Any = None,
        on_refund: Any = None,
    ) -> None:
        """
        Initialize webhook handler.

        Args:
            webhook_secret: HMAC secret for signature verification.
            on_order_update: Callback for order updates (async callable).
            on_shipment: Callback for shipment events (async callable).
            on_cancellation: Callback for cancellation events (async callable).
            on_refund: Callback for refund events (async callable).
        """
        self.webhook_secret = webhook_secret
        self.on_order_update = on_order_update
        self.on_shipment = on_shipment
        self.on_cancellation = on_cancellation
        self.on_refund = on_refund

    def verify_signature(
        self,
        payload: bytes,
        signature: str,
    ) -> bool:
        """
        Verify webhook signature using HMAC-SHA256.

        Adobe Commerce webhooks sign the payload with HMAC-SHA256
        and send the signature in the X-Adobe-Signature header.

        Args:
            payload: Raw request body bytes.
            signature: Signature from X-Adobe-Signature header.

        Returns:
            True if signature is valid, False otherwise.
        """
        if not signature:
            return False

        expected_signature = hmac.new(
            self.webhook_secret.encode("utf-8"),
            payload,
            hashlib.sha256,
        ).hexdigest()

        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(expected_signature, signature)

    async def handle_event(self, event: dict) -> dict[str, Any]:
        """
        Handle a webhook event.

        Routes the event to the appropriate handler based on event type.

        Args:
            event: Parsed webhook event payload.

        Returns:
            Dictionary with handling result.
        """
        event_type = event.get("event_type", "")
        data = event.get("data", {})

        logger.info(f"Handling Adobe Commerce webhook event: {event_type}")

        if event_type in self.ORDER_EVENTS:
            return await self.handle_order_event(data)
        elif event_type in self.SHIPMENT_EVENTS:
            return await self.handle_shipment_event(data)
        elif event_type in self.CANCEL_EVENTS:
            return await self.handle_cancel_event(data)
        elif event_type in self.REFUND_EVENTS:
            return await self.handle_refund_event(data)
        else:
            logger.warning(f"Unknown webhook event type: {event_type}")
            return {"status": "ignored", "reason": f"Unknown event type: {event_type}"}

    async def handle_order_event(self, data: dict) -> dict[str, Any]:
        """
        Handle order created/updated events.

        Args:
            data: Order data from webhook.

        Returns:
            Handling result.
        """
        order = data.get("order", data)
        order_id = str(order.get("entity_id", order.get("order_id", "")))
        increment_id = order.get("increment_id", "")
        state = order.get("state", "")
        status = map_order_status(state)

        result = {
            "status": "processed",
            "event": "order_update",
            "order_id": order_id,
            "order_number": increment_id,
            "order_status": status.value,
        }

        if self.on_order_update:
            await self.on_order_update(order_id, increment_id, status, order)

        logger.info(f"Processed order event: {increment_id} -> {status.value}")
        return result

    async def handle_shipment_event(self, data: dict) -> dict[str, Any]:
        """
        Handle shipment created events.

        Args:
            data: Shipment data from webhook.

        Returns:
            Handling result.
        """
        shipment = data.get("shipment", data)
        order_id = str(shipment.get("order_id", ""))
        shipment_id = str(shipment.get("entity_id", ""))

        # Extract tracking info
        tracks = shipment.get("tracks", [])
        tracking_info = []
        for track in tracks:
            tracking_info.append({
                "carrier": get_carrier_name(track.get("carrier_code", "")),
                "tracking_number": track.get("track_number", ""),
                "title": track.get("title", ""),
            })

        result = {
            "status": "processed",
            "event": "shipment",
            "order_id": order_id,
            "shipment_id": shipment_id,
            "tracking": tracking_info,
        }

        if self.on_shipment:
            await self.on_shipment(order_id, shipment_id, tracking_info, shipment)

        logger.info(f"Processed shipment event for order: {order_id}")
        return result

    async def handle_cancel_event(self, data: dict) -> dict[str, Any]:
        """
        Handle order cancellation events.

        Args:
            data: Cancellation data from webhook.

        Returns:
            Handling result.
        """
        order_id = str(data.get("order_id", data.get("entity_id", "")))

        result = {
            "status": "processed",
            "event": "cancellation",
            "order_id": order_id,
            "order_status": OrderStatus.CANCELLED.value,
        }

        if self.on_cancellation:
            await self.on_cancellation(order_id, data)

        logger.info(f"Processed cancellation event for order: {order_id}")
        return result

    async def handle_refund_event(self, data: dict) -> dict[str, Any]:
        """
        Handle refund/credit memo events.

        Args:
            data: Refund data from webhook.

        Returns:
            Handling result.
        """
        creditmemo = data.get("creditmemo", data)
        order_id = str(creditmemo.get("order_id", ""))
        refund_amount = float(creditmemo.get("grand_total", 0))

        result = {
            "status": "processed",
            "event": "refund",
            "order_id": order_id,
            "refund_amount": refund_amount,
        }

        if self.on_refund:
            await self.on_refund(order_id, refund_amount, creditmemo)

        logger.info(f"Processed refund event for order: {order_id}, amount: {refund_amount}")
        return result
