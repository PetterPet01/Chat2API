from __future__ import annotations

from typing import Any


class DeepSeekProxyError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 502,
        code: str = "upstream_error",
        error_type: str = "upstream_error",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.error_type = error_type
        self.details = details or {}


class AuthenticationError(DeepSeekProxyError):
    def __init__(
        self, message: str = "DeepSeek authentication token is invalid or expired"
    ) -> None:
        super().__init__(
            message,
            status_code=401,
            code="invalid_deepseek_token",
            error_type="authentication_error",
        )


class ConfigurationError(DeepSeekProxyError):
    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            status_code=503,
            code="configuration_error",
            error_type="configuration_error",
        )


class ImageInputError(DeepSeekProxyError):
    def __init__(self, message: str, *, code: str = "invalid_image") -> None:
        super().__init__(
            message,
            status_code=400,
            code=code,
            error_type="invalid_request_error",
        )


class UpstreamProtocolError(DeepSeekProxyError):
    def __init__(
        self, message: str, *, status_code: int = 502, code: str = "upstream_protocol_error"
    ) -> None:
        super().__init__(message, status_code=status_code, code=code)
