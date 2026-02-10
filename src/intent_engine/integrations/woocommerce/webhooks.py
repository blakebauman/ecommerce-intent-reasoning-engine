"""Webhook handler for WooCommerce events."""

import hashlib
import hmac
import logging
from typing import Any

from intent_engine.integrations.base import OrderStatus
from intent_engine.integrations.woocommerce.mapping import (
    get_carrier_name,
    map_order_status,
)

logger = logging.getLogger(__name__)


class WooCommerceWebhookHandler:
    """
    Handler for WooCommerce webhook events.

    WooCommerce webhooks can be configured for various events:
    - order.created - New order placed
    - order.updated - Order details changed
    - order.deleted - Order deleted/trashed
    - order.restored - Order restored from trash

    Webhook signature verification uses HMAC-SHA256 via
    the X-WC-Webhook-Signature header.
    """

    # Supported webhook topics
    ORDER_TOPICS = {
        "order.created",
        "order.updated",
    }
    CUSTOMER_TOPICS = {
        "customer.created",
        "customer.updated",
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

        WooCommerce webhooks sign the payload with HMAC-SHA256
        and send the signature in the X-WC-Webhook-Signature header
        as a base64-encoded string.

        Args:
            payload: Raw request body bytes.
            signature: Signature from X-WC-Webhook-Signature header.

        Returns:
            True if signature is valid, False otherwise.
        """
        if not signature:
            return False

        import base64

        expected_signature = base64.b64encode(
            hmac.new(
                self.webhook_secret.encode("utf-8"),
                payload,
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")

        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(expected_signature, signature)

    async def handle_event(
        self,
        topic: str,
        payload: dict,
    ) -> dict[str, Any]:
        """
        Handle a webhook event.

        Routes the event to the appropriate handler based on topic.

        Args:
            topic: Webhook topic from X-WC-Webhook-Topic header.
            payload: Parsed webhook payload (order/customer data).

        Returns:
            Dictionary with handling result.
        """
        logger.info(f"Handling WooCommerce webhook: {topic}")

        if topic in self.ORDER_TOPICS:
            return await self.handle_order_event(topic, payload)
        elif topic in self.CUSTOMER_TOPICS:
            logger.info(f"Customer event {topic} - no action needed")
            return {"status": "acknowledged", "event": topic}
        elif topic == "order.deleted":
            return await self.handle_delete_event(payload)
        else:
            logger.warning(f"Unknown webhook topic: {topic}")
            return {"status": "ignored", "reason": f"Unknown topic: {topic}"}

    async def handle_order_event(
        self,
        topic: str,
        payload: dict,
    ) -> dict[str, Any]:
        """
        Handle order created/updated events.

        WooCommerce sends the full order object as the payload.

        Args:
            topic: The webhook topic.
            payload: Full order data from webhook.

        Returns:
            Handling result.
        """
        order_id = str(payload.get("id", ""))
        order_number = str(payload.get("number", order_id))
        wc_status = payload.get("status", "pending")

        # Check for refunds
        refunds = payload.get("refunds", [])
        has_refunds = len(refunds) > 0
        refund_total = sum(abs(float(r.get("total", 0))) for r in refunds) if has_refunds else 0

        # Check for tracking (shipment tracking plugin)
        tracking_items = []
        for meta in payload.get("meta_data", []):
            if meta.get("key") == "_wc_shipment_tracking_items":
                items = meta.get("value", [])
                if isinstance(items, list):
                    for item in items:
                        tracking_items.append({
                            "carrier": get_carrier_name(item.get("tracking_provider", "")),
                            "tracking_number": item.get("tracking_number", ""),
                            "tracking_link": item.get("tracking_link", ""),
                        })

        has_tracking = len(tracking_items) > 0
        status = map_order_status(wc_status, has_tracking=has_tracking)

        result: dict[str, Any] = {
            "status": "processed",
            "event": topic,
            "order_id": order_id,
            "order_number": order_number,
            "order_status": status.value,
            "wc_status": wc_status,
        }

        # Handle different status scenarios
        if wc_status == "cancelled":
            result["event_type"] = "cancellation"
            if self.on_cancellation:
                await self.on_cancellation(order_id, payload)
            logger.info(f"Processed cancellation for order: {order_number}")

        elif wc_status == "refunded" or (has_refunds and refund_total > 0):
            result["event_type"] = "refund"
            result["refund_amount"] = refund_total
            if self.on_refund:
                await self.on_refund(order_id, refund_total, payload)
            logger.info(f"Processed refund for order: {order_number}, amount: {refund_total}")

        elif has_tracking and topic == "order.updated":
            result["event_type"] = "shipment"
            result["tracking"] = tracking_items
            if self.on_shipment:
                await self.on_shipment(order_id, tracking_items, payload)
            logger.info(f"Processed shipment for order: {order_number}")

        else:
            result["event_type"] = "order_update"
            if self.on_order_update:
                await self.on_order_update(order_id, order_number, status, payload)
            logger.info(f"Processed order event: {order_number} -> {status.value}")

        return result

    async def handle_delete_event(self, payload: dict) -> dict[str, Any]:
        """
        Handle order deletion events.

        Args:
            payload: Order data from webhook.

        Returns:
            Handling result.
        """
        order_id = str(payload.get("id", ""))

        result = {
            "status": "processed",
            "event": "order.deleted",
            "order_id": order_id,
            "order_status": OrderStatus.CANCELLED.value,
        }

        if self.on_cancellation:
            await self.on_cancellation(order_id, payload)

        logger.info(f"Processed delete event for order: {order_id}")
        return result
