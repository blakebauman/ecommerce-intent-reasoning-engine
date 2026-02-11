"""Domain exceptions for the intent engine API.

These map to consistent HTTP responses when handled by the global exception handler.
"""


class IntentEngineError(Exception):
    """Base exception for intent engine domain errors."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 500,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail or message


class EngineNotReadyError(IntentEngineError):
    """Raised when the engine or a dependency is not initialized."""

    def __init__(self, message: str = "Engine not initialized", detail: str | None = None) -> None:
        super().__init__(message, status_code=503, detail=detail or message)


class IntentValidationError(IntentEngineError):
    """Raised when request input is invalid."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message, status_code=400, detail=detail or message)


class ResourceNotFoundError(IntentEngineError):
    """Raised when a requested resource does not exist."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message, status_code=404, detail=detail or message)
