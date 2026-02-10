"""Authentication strategies for Adobe Commerce API."""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

import httpx


class AdobeCommerceAuthStrategy(ABC):
    """Abstract base class for Adobe Commerce authentication strategies."""

    @abstractmethod
    async def get_auth_headers(self) -> dict[str, str]:
        """
        Get authentication headers for API requests.

        Returns:
            Dictionary of headers to include in requests.
        """
        ...

    @abstractmethod
    async def refresh_if_needed(self) -> None:
        """Refresh credentials if they are expired or about to expire."""
        ...


class IntegrationTokenAuth(AdobeCommerceAuthStrategy):
    """
    PaaS authentication using integration access token.

    For self-hosted Adobe Commerce (Magento Open Source or Commerce).
    Token is created in Admin > System > Integrations.
    """

    def __init__(self, access_token: str) -> None:
        """
        Initialize with integration access token.

        Args:
            access_token: Long-lived integration access token from Admin panel.
        """
        self.access_token = access_token

    async def get_auth_headers(self) -> dict[str, str]:
        """Get Bearer token authorization header."""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    async def refresh_if_needed(self) -> None:
        """No refresh needed for integration tokens (long-lived)."""
        pass


class IMSOAuthAuth(AdobeCommerceAuthStrategy):
    """
    SaaS authentication using Adobe IMS server-to-server OAuth 2.0.

    For Adobe Commerce Cloud Service with Adobe IMS integration.
    Uses client credentials grant flow.
    """

    IMS_TOKEN_URL = "https://ims-na1.adobelogin.com/ims/token/v3"
    TOKEN_EXPIRY_BUFFER = timedelta(minutes=5)

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        org_id: str,
        scopes: list[str] | None = None,
    ) -> None:
        """
        Initialize with Adobe IMS credentials.

        Args:
            client_id: Adobe IMS client ID (API key).
            client_secret: Adobe IMS client secret.
            org_id: Adobe organization ID.
            scopes: OAuth scopes (default: ["commerce.accs"]).
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.org_id = org_id
        self.scopes = scopes or ["commerce.accs"]

        self._access_token: str | None = None
        self._token_expires_at: datetime | None = None
        self._http_client: httpx.AsyncClient | None = None

    @property
    def _client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def get_auth_headers(self) -> dict[str, str]:
        """Get Bearer token and IMS headers."""
        await self.refresh_if_needed()

        if not self._access_token:
            raise RuntimeError("Failed to obtain access token")

        return {
            "Authorization": f"Bearer {self._access_token}",
            "x-gw-ims-org-id": self.org_id,
            "x-api-key": self.client_id,
            "Content-Type": "application/json",
        }

    async def refresh_if_needed(self) -> None:
        """Refresh the access token if expired or about to expire."""
        now = datetime.now(timezone.utc)

        if (
            self._access_token
            and self._token_expires_at
            and now < (self._token_expires_at - self.TOKEN_EXPIRY_BUFFER)
        ):
            # Token is still valid
            return

        await self._fetch_access_token()

    async def _fetch_access_token(self) -> None:
        """Fetch a new access token from Adobe IMS."""
        scope_str = ",".join(self.scopes)

        response = await self._client.post(
            self.IMS_TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": scope_str,
            },
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        response.raise_for_status()

        data = response.json()
        self._access_token = data["access_token"]

        # Calculate expiry (IMS returns expires_in in seconds)
        expires_in = data.get("expires_in", 86400)  # Default 24 hours
        self._token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
