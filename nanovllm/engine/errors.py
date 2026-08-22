class EngineError(RuntimeError):
    """Base class for online-engine errors."""


class DuplicateRequestError(EngineError):
    """Raised when a request ID has already been used by this engine."""


class RequestQueueFullError(EngineError):
    """Raised when online admission reaches the configured queue limit."""


class RequestTooLongError(ValueError):
    """Raised when a request cannot fit in the model or KV-cache capacity."""


class EngineClosedError(EngineError):
    """Raised when work is submitted to a closed online engine."""
