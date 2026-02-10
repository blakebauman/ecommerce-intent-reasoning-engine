"""WebSocket endpoint for real-time intent resolution."""

import asyncio
import json
import logging
import uuid
from typing import Any, Callable

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from intent_engine.api.ws_auth import WebSocketAuthenticator, get_connection_limit_for_tenant
from intent_engine.api.ws_models import (
    STEP_DESCRIPTIONS,
    ConnectedPayload,
    ErrorPayload,
    JobUpdatePayload,
    ReasoningStep,
    ReasoningStepPayload,
    WSMessage,
    WSMessageType,
)
from intent_engine.observability.metrics import (
    record_websocket_connection,
    record_websocket_message,
)
from intent_engine.tenancy.context import set_tenant_context, tenant_context
from intent_engine.tenancy.models import TenantConfig

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


class ConnectionManager:
    """
    Manages WebSocket connections across tenants.

    Features:
    - Multi-client support
    - Per-tenant connection limits
    - Job subscription tracking
    - Broadcast capabilities
    """

    def __init__(self) -> None:
        # Active connections: tenant_id -> {connection_id -> WebSocket}
        self._connections: dict[str, dict[str, WebSocket]] = {}
        # Connection info: connection_id -> TenantConfig
        self._tenants: dict[str, TenantConfig] = {}
        # Job subscriptions: job_id -> {connection_id}
        self._job_subscriptions: dict[str, set[str]] = {}
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        tenant: TenantConfig,
    ) -> str:
        """
        Accept a new WebSocket connection.

        Args:
            websocket: The WebSocket connection.
            tenant: Authenticated tenant.

        Returns:
            Connection ID.

        Raises:
            Exception: If connection limit exceeded.
        """
        async with self._lock:
            tenant_id = tenant.tenant_id

            # Initialize tenant connections dict if needed
            if tenant_id not in self._connections:
                self._connections[tenant_id] = {}

            # Check connection limit
            limit = get_connection_limit_for_tenant(tenant)
            current = len(self._connections[tenant_id])
            if current >= limit:
                raise Exception(f"Connection limit exceeded: {current}/{limit}")

            # Accept the connection
            await websocket.accept()

            # Generate connection ID and store
            connection_id = str(uuid.uuid4())
            self._connections[tenant_id][connection_id] = websocket
            self._tenants[connection_id] = tenant

            # Record metrics
            record_websocket_connection(tenant_id, delta=1)

            logger.info(
                f"WebSocket connected: {connection_id} (tenant: {tenant_id}, "
                f"connections: {current + 1}/{limit})"
            )

            return connection_id

    async def disconnect(self, connection_id: str) -> None:
        """
        Handle WebSocket disconnection.

        Args:
            connection_id: The connection ID.
        """
        async with self._lock:
            tenant = self._tenants.pop(connection_id, None)
            if tenant:
                tenant_id = tenant.tenant_id
                if tenant_id in self._connections:
                    self._connections[tenant_id].pop(connection_id, None)
                    if not self._connections[tenant_id]:
                        del self._connections[tenant_id]

                # Remove from all job subscriptions
                for job_id in list(self._job_subscriptions.keys()):
                    self._job_subscriptions[job_id].discard(connection_id)
                    if not self._job_subscriptions[job_id]:
                        del self._job_subscriptions[job_id]

                # Record metrics
                record_websocket_connection(tenant_id, delta=-1)

                logger.info(f"WebSocket disconnected: {connection_id}")

    async def subscribe_to_job(self, connection_id: str, job_id: str) -> None:
        """Subscribe a connection to job updates."""
        async with self._lock:
            if job_id not in self._job_subscriptions:
                self._job_subscriptions[job_id] = set()
            self._job_subscriptions[job_id].add(connection_id)
            logger.debug(f"Connection {connection_id} subscribed to job {job_id}")

    async def unsubscribe_from_job(self, connection_id: str, job_id: str) -> None:
        """Unsubscribe a connection from job updates."""
        async with self._lock:
            if job_id in self._job_subscriptions:
                self._job_subscriptions[job_id].discard(connection_id)
                if not self._job_subscriptions[job_id]:
                    del self._job_subscriptions[job_id]

    async def send_to_connection(
        self,
        connection_id: str,
        message: WSMessage,
    ) -> bool:
        """
        Send a message to a specific connection.

        Args:
            connection_id: The connection ID.
            message: The message to send.

        Returns:
            True if sent successfully, False otherwise.
        """
        tenant = self._tenants.get(connection_id)
        if not tenant:
            return False

        tenant_id = tenant.tenant_id
        connections = self._connections.get(tenant_id, {})
        websocket = connections.get(connection_id)

        if websocket:
            try:
                await websocket.send_text(message.model_dump_json())
                record_websocket_message(tenant_id, "outbound", message.type.value)
                return True
            except Exception as e:
                logger.warning(f"Failed to send to {connection_id}: {e}")
                return False
        return False

    async def broadcast_to_tenant(
        self,
        tenant_id: str,
        message: WSMessage,
    ) -> int:
        """
        Broadcast a message to all connections for a tenant.

        Args:
            tenant_id: The tenant ID.
            message: The message to broadcast.

        Returns:
            Number of connections that received the message.
        """
        connections = self._connections.get(tenant_id, {})
        sent = 0

        for connection_id, websocket in list(connections.items()):
            try:
                await websocket.send_text(message.model_dump_json())
                sent += 1
            except Exception as e:
                logger.warning(f"Failed to broadcast to {connection_id}: {e}")

        if sent > 0:
            record_websocket_message(tenant_id, "outbound", message.type.value)

        return sent

    async def notify_job_subscribers(
        self,
        job_id: str,
        status: str,
        progress: float | None = None,
        message: str | None = None,
    ) -> int:
        """
        Notify all subscribers of a job update.

        Args:
            job_id: The job ID.
            status: Job status.
            progress: Optional progress (0.0 to 1.0).
            message: Optional status message.

        Returns:
            Number of notifications sent.
        """
        subscribers = self._job_subscriptions.get(job_id, set())
        if not subscribers:
            return 0

        ws_message = WSMessage(
            type=WSMessageType.JOB_UPDATE,
            payload=JobUpdatePayload(
                job_id=job_id,
                status=status,
                progress=progress,
                message=message,
            ).model_dump(),
        )

        sent = 0
        for connection_id in list(subscribers):
            if await self.send_to_connection(connection_id, ws_message):
                sent += 1

        return sent

    def get_connection_count(self, tenant_id: str | None = None) -> int:
        """Get current connection count."""
        if tenant_id:
            return len(self._connections.get(tenant_id, {}))
        return sum(len(conns) for conns in self._connections.values())


# Global connection manager
connection_manager = ConnectionManager()


class StreamingReasoningCallback:
    """
    Callback handler for streaming reasoning steps.

    Sends reasoning step updates to the WebSocket client
    as the intent resolution pipeline progresses.
    """

    def __init__(
        self,
        connection_id: str,
        request_id: str,
        manager: ConnectionManager,
    ) -> None:
        self.connection_id = connection_id
        self.request_id = request_id
        self.manager = manager

    async def on_step(
        self,
        step: ReasoningStep,
        duration_ms: int | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """
        Called when a reasoning step completes.

        Args:
            step: The reasoning step.
            duration_ms: Step duration in milliseconds.
            data: Optional step data.
        """
        message = WSMessage(
            type=WSMessageType.REASONING_STEP,
            request_id=self.request_id,
            payload=ReasoningStepPayload(
                step_name=step.value,
                description=STEP_DESCRIPTIONS.get(step, step.value),
                duration_ms=duration_ms,
                data=data or {},
            ).model_dump(),
        )

        await self.manager.send_to_connection(self.connection_id, message)


def create_websocket_endpoint(
    engine_getter: Callable,
    authenticator: WebSocketAuthenticator | None = None,
) -> APIRouter:
    """
    Create WebSocket router with engine access.

    Args:
        engine_getter: Function to get the IntentEngine instance.
        authenticator: WebSocket authenticator.

    Returns:
        Configured APIRouter.
    """

    ws_router = APIRouter(tags=["websocket"])

    @ws_router.websocket("/ws")
    async def websocket_endpoint(
        websocket: WebSocket,
        tenant: TenantConfig = Depends(authenticator) if authenticator else None,
    ):
        """
        WebSocket endpoint for real-time intent resolution.

        Protocol:
        1. Client connects with authentication
        2. Server sends 'connected' message
        3. Client sends 'resolve' messages
        4. Server streams 'reasoning_step' messages
        5. Server sends 'result' message

        Query Parameters:
            token: API key for authentication
        """
        # Handle case where no authenticator is configured
        if tenant is None:
            await websocket.close(code=1008, reason="Authentication not configured")
            return

        connection_id = None

        try:
            # Accept and register connection
            connection_id = await connection_manager.connect(websocket, tenant)

            # Set tenant context
            set_tenant_context(tenant)

            # Send connected message
            await connection_manager.send_to_connection(
                connection_id,
                WSMessage(
                    type=WSMessageType.CONNECTED,
                    payload=ConnectedPayload(
                        connection_id=connection_id,
                        tenant_id=tenant.tenant_id,
                    ).model_dump(),
                ),
            )

            # Message loop
            while True:
                data = await websocket.receive_text()
                record_websocket_message(tenant.tenant_id, "inbound", "unknown")

                try:
                    message = json.loads(data)
                    msg_type = message.get("type")
                    request_id = message.get("request_id", str(uuid.uuid4()))
                    payload = message.get("payload", {})

                    if msg_type == WSMessageType.PING.value:
                        await connection_manager.send_to_connection(
                            connection_id,
                            WSMessage(type=WSMessageType.PONG, request_id=request_id),
                        )

                    elif msg_type == WSMessageType.RESOLVE.value:
                        await handle_resolve(
                            connection_id=connection_id,
                            request_id=request_id,
                            payload=payload,
                            tenant=tenant,
                            engine_getter=engine_getter,
                        )

                    elif msg_type == WSMessageType.SUBSCRIBE.value:
                        job_id = payload.get("job_id")
                        if job_id:
                            await connection_manager.subscribe_to_job(connection_id, job_id)
                            await connection_manager.send_to_connection(
                                connection_id,
                                WSMessage(
                                    type=WSMessageType.SUBSCRIBED,
                                    request_id=request_id,
                                    payload={"job_id": job_id},
                                ),
                            )

                    elif msg_type == WSMessageType.UNSUBSCRIBE.value:
                        job_id = payload.get("job_id")
                        if job_id:
                            await connection_manager.unsubscribe_from_job(connection_id, job_id)
                            await connection_manager.send_to_connection(
                                connection_id,
                                WSMessage(
                                    type=WSMessageType.UNSUBSCRIBED,
                                    request_id=request_id,
                                    payload={"job_id": job_id},
                                ),
                            )

                    else:
                        await connection_manager.send_to_connection(
                            connection_id,
                            WSMessage(
                                type=WSMessageType.ERROR,
                                request_id=request_id,
                                payload=ErrorPayload(
                                    code="unknown_message_type",
                                    message=f"Unknown message type: {msg_type}",
                                ).model_dump(),
                            ),
                        )

                except json.JSONDecodeError:
                    await connection_manager.send_to_connection(
                        connection_id,
                        WSMessage(
                            type=WSMessageType.ERROR,
                            payload=ErrorPayload(
                                code="invalid_json",
                                message="Invalid JSON message",
                            ).model_dump(),
                        ),
                    )

        except WebSocketDisconnect:
            logger.debug(f"WebSocket disconnected: {connection_id}")
        except Exception as e:
            logger.exception(f"WebSocket error: {e}")
            if connection_id:
                try:
                    await connection_manager.send_to_connection(
                        connection_id,
                        WSMessage(
                            type=WSMessageType.ERROR,
                            payload=ErrorPayload(
                                code="internal_error",
                                message=str(e),
                            ).model_dump(),
                        ),
                    )
                except Exception:
                    pass
        finally:
            if connection_id:
                await connection_manager.disconnect(connection_id)

    return ws_router


async def handle_resolve(
    connection_id: str,
    request_id: str,
    payload: dict,
    tenant: TenantConfig,
    engine_getter: Callable,
) -> None:
    """
    Handle a resolve request over WebSocket.

    Streams reasoning steps as they complete.

    Args:
        connection_id: The connection ID.
        request_id: The request ID.
        payload: Request payload with raw_text.
        tenant: Tenant configuration.
        engine_getter: Function to get the engine.
    """
    raw_text = payload.get("raw_text", "")

    if not raw_text:
        await connection_manager.send_to_connection(
            connection_id,
            WSMessage(
                type=WSMessageType.ERROR,
                request_id=request_id,
                payload=ErrorPayload(
                    code="missing_text",
                    message="raw_text is required",
                ).model_dump(),
            ),
        )
        return

    try:
        # Get engine and resolve
        engine = engine_getter()
        if engine is None:
            raise Exception("Engine not initialized")

        # Create streaming callback
        callback = StreamingReasoningCallback(
            connection_id=connection_id,
            request_id=request_id,
            manager=connection_manager,
        )

        # Run resolution with tenant context
        with tenant_context(tenant):
            # Note: In a full implementation, the engine would accept
            # a callback for streaming. For now, we just run the resolution.
            result = await engine.resolve_text(
                text=raw_text,
                request_id=request_id,
                tenant_id=tenant.tenant_id,
            )

        # Send the complete step
        await callback.on_step(ReasoningStep.COMPLETE)

        # Send result
        await connection_manager.send_to_connection(
            connection_id,
            WSMessage(
                type=WSMessageType.RESULT,
                request_id=request_id,
                payload=result.model_dump(),
            ),
        )

    except Exception as e:
        logger.exception(f"Error resolving intent: {e}")
        await connection_manager.send_to_connection(
            connection_id,
            WSMessage(
                type=WSMessageType.ERROR,
                request_id=request_id,
                payload=ErrorPayload(
                    code="resolution_error",
                    message=str(e),
                ).model_dump(),
            ),
        )


def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager."""
    return connection_manager
