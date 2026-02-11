"""WebSocket authentication utilities."""

import logging
from collections.abc import Callable

from fastapi import Query, WebSocket, WebSocketException, status

from intent_engine.tenancy.models import TenantConfig

logger = logging.getLogger(__name__)


class WebSocketAuthenticator:
    """
    Authenticator for WebSocket connections.

    Supports authentication via:
    1. Query parameter: ?token=<api_key>
    2. Subprotocol: Sec-WebSocket-Protocol header

    Usage:
        authenticator = WebSocketAuthenticator(tenant_lookup)

        @app.websocket("/ws")
        async def websocket_endpoint(
            websocket: WebSocket,
            tenant: TenantConfig = Depends(authenticator),
        ):
            ...
    """

    def __init__(
        self,
        tenant_lookup: Callable[[str], TenantConfig | None] | None = None,
        dev_mode: bool = False,
        dev_tenant: TenantConfig | None = None,
    ) -> None:
        """
        Initialize the authenticator.

        Args:
            tenant_lookup: Async or sync function to look up tenant by API key.
            dev_mode: If True, accept any token and use dev tenant.
            dev_tenant: Tenant config to use in dev mode.
        """
        self.tenant_lookup = tenant_lookup
        self.dev_mode = dev_mode
        self.dev_tenant = dev_tenant

    async def __call__(
        self,
        websocket: WebSocket,
        token: str | None = Query(default=None),
    ) -> TenantConfig:
        """
        Authenticate a WebSocket connection.

        Args:
            websocket: The WebSocket connection.
            token: API key from query parameter.

        Returns:
            TenantConfig for the authenticated tenant.

        Raises:
            WebSocketException: If authentication fails.
        """
        api_key = self._extract_api_key(websocket, token)

        if not api_key:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Authentication required",
            )

        tenant = await self._authenticate(api_key)

        if tenant is None:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Invalid API key",
            )

        if not tenant.is_active:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Tenant account is inactive",
            )

        if not tenant.websocket_enabled:
            raise WebSocketException(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="WebSocket access not enabled for this tenant",
            )

        logger.info(f"WebSocket authenticated for tenant: {tenant.tenant_id}")
        return tenant

    def _extract_api_key(
        self,
        websocket: WebSocket,
        token: str | None,
    ) -> str | None:
        """
        Extract API key from WebSocket request.

        Checks:
        1. Query parameter 'token'
        2. Sec-WebSocket-Protocol header (format: "bearer.{api_key}")

        Args:
            websocket: The WebSocket connection.
            token: Token from query parameter.

        Returns:
            API key if found, None otherwise.
        """
        # Check query parameter first
        if token:
            return token

        # Check subprotocol header
        subprotocols = websocket.scope.get("subprotocols", [])
        for proto in subprotocols:
            if proto.startswith("bearer."):
                return proto[7:]  # Remove "bearer." prefix

        # Check authorization header (some clients support this)
        headers = dict(websocket.scope.get("headers", []))
        auth_header = headers.get(b"authorization", b"").decode()
        if auth_header.startswith("Bearer "):
            return auth_header[7:]

        return None

    async def _authenticate(self, api_key: str) -> TenantConfig | None:
        """
        Authenticate using API key.

        Args:
            api_key: The API key to authenticate.

        Returns:
            TenantConfig if valid, None otherwise.
        """
        # Dev mode: return dev tenant
        if self.dev_mode and self.dev_tenant:
            return self.dev_tenant

        # Use tenant lookup if available
        if self.tenant_lookup:
            try:
                result = self.tenant_lookup(api_key)
                if hasattr(result, "__await__"):
                    return await result
                return result
            except Exception as e:
                logger.error(f"Error during WebSocket authentication: {e}")
                return None

        return None


def get_connection_limit_for_tenant(tenant: TenantConfig) -> int:
    """
    Get the WebSocket connection limit for a tenant.

    Args:
        tenant: The tenant configuration.

    Returns:
        Maximum allowed connections.
    """
    return tenant.get_max_websocket_connections()
