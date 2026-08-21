from __future__ import annotations


class FictionMasterError(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class ModelNotConfiguredError(FictionMasterError):
    def __init__(self, capability: str) -> None:
        super().__init__(
            "MODEL_NOT_CONFIGURED",
            f"{capability} model API key is not configured",
            retryable=False,
        )


class ProviderError(FictionMasterError):
    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__("PROVIDER_ERROR", message, retryable=retryable)


class NotFoundError(FictionMasterError):
    def __init__(self, resource: str) -> None:
        super().__init__("NOT_FOUND", f"{resource} not found", retryable=False)
