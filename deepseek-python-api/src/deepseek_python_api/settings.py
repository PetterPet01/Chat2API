from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    deepseek_auth_token: SecretStr | None = Field(default=None)
    allow_request_token: bool = False
    api_key: SecretStr | None = Field(default=None)
    management_api_key: SecretStr | None = Field(default=None)

    tokens_file: str = "deepseek_tokens.json"
    proxies_file: str = "proxies.txt"
    rotation_strategy: Literal["fill_first", "round_robin"] = "fill_first"
    token_health_check_interval_seconds: int = Field(default=300, ge=0, le=86400)
    token_failure_cooldown_seconds: int = Field(default=300, ge=0, le=86400)

    # Rotating proxy behaviour
    # When True, a rate-limit (429) response automatically triggers IP rotation
    # on any rotating proxy bound to that token before the next retry.
    proxy_rotate_on_ratelimit: bool = True
    # Minimum seconds that must pass between automatic rotation attempts on the
    # *same* rotating proxy (provider hard limit is 60 s; we default to 61 s to
    # give a small safety margin).
    proxy_rotate_min_interval_seconds: float = Field(default=61.0, ge=0.0, le=3600.0)

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    deepseek_api_base: str = "https://chat.deepseek.com/api"
    access_token_ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    connect_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
    request_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    challenge_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    challenge_retry_attempts: int = Field(default=2, ge=1, le=5)
    pow_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    pow_workers: int = Field(default=2, ge=1, le=16)

    max_image_bytes: int = Field(default=100 * 1024 * 1024, ge=1, le=200 * 1024 * 1024)
    max_images: int = Field(default=10, ge=1, le=50)
    remote_image_max_redirects: int = Field(default=3, ge=0, le=10)
    allow_private_image_urls: bool = False
    file_poll_attempts: int = Field(default=20, ge=1, le=120)
    file_poll_interval_seconds: float = Field(default=1.0, ge=0, le=10)

    delete_sessions: bool = True

    @field_validator("deepseek_api_base")
    @classmethod
    def normalize_api_base(cls, value: str) -> str:
        normalized = value.rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("deepseek_api_base must be an absolute HTTP(S) URL")
        return normalized

    def configured_deepseek_token(self) -> str | None:
        return self.deepseek_auth_token.get_secret_value() if self.deepseek_auth_token else None

    def configured_api_key(self) -> str | None:
        return self.api_key.get_secret_value() if self.api_key else None

    def configured_management_api_key(self) -> str | None:
        return self.management_api_key.get_secret_value() if self.management_api_key else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
