"""Webhook handler for BigCommerce events."""

import hashlib
import hmac
import logging
from typing import Any

from intent_engine.integrations.base import OrderStatus
from intent_engine.integrations.bigcommerce.mapping import (
    get_carrier_name,
    map_order_status,
)

logger = logging.getLogger(__name__)


class BigCommerceWebhookHandler:
    """
    Handler for BigCommerce webhook events.

    BigCommerce webhooks are configured per-store and support various scopes:
    - store/order/created
    - store/order/updated
    - store/order/statusUpdated
    - store/shipment/created
    - store/shipment/updated

    Webhook signature verification uses HMAC-SHA256 via
    the X-Bc-Webhook-Hmac-Sha256 header.

    Note: BigCommerce webhook payloads contain minimal data.
    The webhook sends the resource ID and you must fetch full data via API.
    """

    # Supported webhook scopes
    ORDER_SCOPES = {
        "store/order/created",
        "store/order/updated",
        "store/order/statusUpdated",
    }
    SHIPMENT_SCOPES = {
        "store/shipment/created",
        "store/shipment/updated",
    }

    def __init__(
        self,
        client_secret: str,
        on_order_update: Any = None,
        on_shipment: Any = None,
        on_cancellation: Any = None,
        on_refund: Any = None,
    ) -> None:
        """
        Initialize webhook handler.

        Args:
            client_secret: BigCommerce app client secret for signature verification.
            on_order_update: Callback for order updates (async callable).
            on_shipment: Callback for shipment events (async callable).
            on_cancellation: Callback for cancellation events (async callable).
            on_refund: Callback for refund events (async callable).
        """
        self.client_secret = client_secret
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

        BigCommerce webhooks sign the payload with HMAC-SHA256
        and send the signature in the X-Bc-Webhook-Hmac-Sha256 header.

        Args:
            payload: Raw request body bytes.
            signature: Signature from X-Bc-Webhook-Hmac-Sha256 header.

        Returns:
            True if signature is valid, False otherwise.
        """
        if not signature:
            return False

        import base64

        expected_signature = base64.b64encode(
            hmac.new(
                self.client_secret.encode("utf-8"),
                payload,
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")

        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(expected_signature, signature)

    async def handle_event(
        self,
        scope: str,
        payload: dict,
    ) -> dict[str, Any]:
        """
        Handle a webhook event.

        Routes the event to the appropriate handler based on scope.

        Note: BigCommerce webhook payloads are minimal. They contain:
        - scope: The webhook scope
        - store_id: The store ID
        - data: Contains the resource type and ID
        - hash: Verification hash
        - created_at: Timestamp
        - producer: Source of the event

        Args:
            scope: Webhook scope (e.g., "store/order/created").
            payload: Parsed webhook payload.

        Returns:
            Dictionary with handling result.
        """
        logger.info(f"Handling BigCommerce webhook: {scope}")

        if scope in self.ORDER_SCOPES:
            return await self.handle_order_event(scope, payload)
        elif scope in self.SHIPMENT_SCOPES:
            return await self.handle_shipment_event(scope, payload)
        else:
            logger.warning(f"Unknown webhook scope: {scope}")
            return {"status": "ignored", "reason": f"Unknown scope: {scope}"}

    async def handle_order_event(
        self,
        scope: str,
        payload: dict,
    ) -> dict[str, Any]:
        """
        Handle order events.

        BigCommerce order webhooks include minimal data:
        {
            "scope": "store/order/created",
            "store_id": "12345",
            "data": {
                "type": "order",
                "id": 123
            },
            "hash": "...",
            "created_at": 1234567890,
            "producer": "stores/12345"
        }

        Full order data must be fetched via the API.

        Args:
            scope: The webhook scope.
            payload: Webhook payload.

        Returns:
            Handling result.
        """
        data = payload.get("data", {})
        order_id = str(data.get("id", ""))
        store_id = str(payload.get("store_id", ""))

        # For status updates, we get status info
        status_info = data.get("status", {})
        previous_status_id = status_info.get("previous_status_id")
        new_status_id = status_info.get("new_status_id")

        result: dict[str, Any] = {
            "status": "processed",
            "event": scope,
            "order_id": order_id,
            "store_id": store_id,
        }

        # Determine event type from status change
        if new_status_id is not None:
            new_status = map_order_status(new_status_id)
            result["order_status"] = new_status.value
            result["previous_status_id"] = previous_status_id
            result["new_status_id"] = new_status_id

            # Check for cancellation (status 5 or 6)
            if new_status_id in (5, 6):
                result["event_type"] = "cancellation"
                if self.on_cancellation:
                    await self.on_cancellation(order_id, payload)
                logger.info(f"Processed cancellation for order: {order_id}")

            # Check for refund (status 4 or 14)
            elif new_status_id in (4, 14):
                result["event_type"] = "refund"
                # Note: Actual refund amount must be fetched via API
                if self.on_refund:
                    await self.on_refund(order_id, None, payload)
                logger.info(f"Processed refund for order: {order_id}")

            else:
                result["event_type"] = "order_update"
                if self.on_order_update:
                    await self.on_order_update(order_id, order_id, new_status, payload)
                logger.info(f"Processed order event: {order_id} -> {new_status.value}")

        else:
            # Created or generic update
            result["event_type"] = "order_update"
            if self.on_order_update:
                await self.on_order_update(order_id, order_id, OrderStatus.PENDING, payload)
            logger.info(f"Processed order event: {order_id}")

        return result

    async def handle_shipment_event(
        self,
        scope: str,
        payload: dict,
    ) -> dict[str, Any]:
        """
        Handle shipment events.

        BigCommerce shipment webhooks include:
        {
            "scope": "store/shipment/created",
            "store_id": "12345",
            "data": {
                "type": "shipment",
                "id": 456,
                "order_id": 123
            },
            "hash": "...",
            "created_at": 1234567890,
            "producer": "stores/12345"
        }

        Full shipment data (tracking) must be fetched via API.

        Args:
            scope: The webhook scope.
            payload: Webhook payload.

        Returns:
            Handling result.
        """
        data = payload.get("data", {})
        shipment_id = str(data.get("id", ""))
        order_id = str(data.get("order_id", ""))
        store_id = str(payload.get("store_id", ""))

        result = {
            "status": "processed",
            "event": scope,
            "event_type": "shipment",
            "shipment_id": shipment_id,
            "order_id": order_id,
            "store_id": store_id,
        }

        # Note: Tracking details must be fetched via API
        # The webhook only notifies that a shipment was created/updated
        if self.on_shipment:
            await self.on_shipment(order_id, [], payload)

        logger.info(f"Processed shipment event for order: {order_id}")
        return result
