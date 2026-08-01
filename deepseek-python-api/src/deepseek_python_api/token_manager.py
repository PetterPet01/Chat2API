from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

from .errors import AuthenticationError, ConfigurationError, DeepSeekProxyError
from .settings import Settings
from .upstream import DeepSeekClient

LOGGER = logging.getLogger(__name__)
MAX_CONCURRENT_HEALTH_CHECKS = 5

RotationStrategy = Literal["fill_first", "round_robin"]
TokenStatus = Literal["unchecked", "healthy", "unhealthy", "cooldown", "disabled"]


@dataclass(slots=True)
class ManagedToken:
    id: str
    name: str
    token: str
    enabled: bool = True
    status: TokenStatus = "unchecked"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_checked_at: float | None = None
    last_ok_at: float | None = None
    last_error: str | None = None
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    last_used_at: float | None = None
    cooldown_until: float | None = None
    in_flight: int = 0

    @property
    def fingerprint(self) -> str:
        return token_fingerprint(self.token)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ManagedToken:
        token = payload.get("token")
        if not isinstance(token, str) or not token:
            raise ConfigurationError("Managed token store contains an invalid token entry")
        token_id = payload.get("id")
        name = payload.get("name")
        return cls(
            id=token_id if isinstance(token_id, str) and token_id else f"tok_{uuid4().hex}",
            name=name if isinstance(name, str) and name else "DeepSeek token",
            token=token,
            enabled=bool(payload.get("enabled", True)),
            status=_status_or_default(payload.get("status")),
            created_at=_float_or_now(payload.get("created_at")),
            updated_at=_float_or_now(payload.get("updated_at")),
            last_checked_at=_optional_float(payload.get("last_checked_at")),
            last_ok_at=_optional_float(payload.get("last_ok_at")),
            last_error=payload.get("last_error")
            if isinstance(payload.get("last_error"), str)
            else None,
            success_count=max(0, int(payload.get("success_count") or 0)),
            failure_count=max(0, int(payload.get("failure_count") or 0)),
            consecutive_failures=max(0, int(payload.get("consecutive_failures") or 0)),
            last_used_at=_optional_float(payload.get("last_used_at")),
            cooldown_until=_optional_float(payload.get("cooldown_until")),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "token": self.token,
            "enabled": self.enabled,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_checked_at": self.last_checked_at,
            "last_ok_at": self.last_ok_at,
            "last_error": self.last_error,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "consecutive_failures": self.consecutive_failures,
            "last_used_at": self.last_used_at,
            "cooldown_until": self.cooldown_until,
        }

    def public_view(self, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        effective_status = self.status
        if not self.enabled:
            effective_status = "disabled"
        elif self.cooldown_until and self.cooldown_until > now:
            effective_status = "cooldown"
        return {
            "id": self.id,
            "name": self.name,
            "enabled": self.enabled,
            "status": effective_status,
            "redacted_token": redact_token(self.token),
            "fingerprint": self.fingerprint,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_checked_at": self.last_checked_at,
            "last_ok_at": self.last_ok_at,
            "last_error": self.last_error,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "consecutive_failures": self.consecutive_failures,
            "last_used_at": self.last_used_at,
            "cooldown_until": self.cooldown_until,
            "in_flight": self.in_flight,
        }


@dataclass(slots=True)
class TokenLease:
    token_id: str | None
    token: str
    manager: TokenManager | None
    _released: bool = False

    async def release(self, *, success: bool, error: BaseException | None = None) -> None:
        if self._released:
            return
        self._released = True
        if self.manager and self.token_id:
            await self.manager.release(self.token_id, success=success, error=error)


class TokenManager:
    def __init__(self, settings: Settings, client: DeepSeekClient) -> None:
        self.settings = settings
        self.client = client
        self.path = Path(settings.tokens_file).expanduser()
        self.strategy: RotationStrategy = settings.rotation_strategy
        self._tokens: list[ManagedToken] = []
        self._round_robin_index = 0
        self._lock = asyncio.Lock()
        self._supervisor_task: asyncio.Task[None] | None = None

    def load(self) -> None:
        if not self.path.exists():
            self._tokens = []
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError("Failed to load managed DeepSeek token store") from exc
        strategy = payload.get("rotation_strategy")
        if strategy in {"fill_first", "round_robin"}:
            self.strategy = strategy
        tokens = payload.get("tokens", [])
        if not isinstance(tokens, list):
            raise ConfigurationError("Managed DeepSeek token store has an invalid shape")
        self._tokens = [
            ManagedToken.from_payload(item) for item in tokens if isinstance(item, dict)
        ]
        self._round_robin_index = int(payload.get("round_robin_index") or 0)

    async def start_supervisor(self) -> None:
        if self.settings.token_health_check_interval_seconds <= 0 or self._supervisor_task:
            return
        self._supervisor_task = asyncio.create_task(self._supervise(), name="deepseek-token-health")

    async def stop_supervisor(self) -> None:
        if not self._supervisor_task:
            return
        self._supervisor_task.cancel()
        try:
            await self._supervisor_task
        except asyncio.CancelledError:
            pass
        self._supervisor_task = None

    @property
    def supervisor_running(self) -> bool:
        return self._supervisor_task is not None and not self._supervisor_task.done()

    async def acquire(self, excluded_ids: set[str] | None = None) -> TokenLease:
        excluded_ids = excluded_ids or set()
        async with self._lock:
            if self._tokens:
                token = self._select_token(excluded_ids)
                if token is None:
                    raise ConfigurationError("No enabled healthy DeepSeek tokens are available")
                token.in_flight += 1
                token.last_used_at = time.time()
                token.updated_at = token.last_used_at
                self._save_unlocked()
                return TokenLease(token.id, token.token, self)

            configured = self.settings.configured_deepseek_token()
            if configured:
                return TokenLease(None, configured, None)
            raise ConfigurationError("No DeepSeek tokens are configured")

    async def release(
        self, token_id: str, *, success: bool, error: BaseException | None = None
    ) -> None:
        async with self._lock:
            token = self._find_token(token_id)
            if not token:
                return
            token.in_flight = max(0, token.in_flight - 1)
            if success:
                self._mark_success(token, checked=False)
            else:
                self._mark_failure(token, error)
            self._save_unlocked()

    async def add_token(
        self, token: str, *, name: str | None = None, enabled: bool = True
    ) -> dict[str, Any]:
        if not token:
            raise DeepSeekProxyError(
                "DeepSeek token may not be empty",
                status_code=400,
                code="invalid_token",
                error_type="invalid_request_error",
            )
        async with self._lock:
            fingerprint = token_fingerprint(token)
            if any(item.fingerprint == fingerprint for item in self._tokens):
                raise DeepSeekProxyError(
                    "DeepSeek token already exists",
                    status_code=409,
                    code="duplicate_token",
                    error_type="invalid_request_error",
                )
            now = time.time()
            record = ManagedToken(
                id=f"tok_{uuid4().hex}",
                name=name or f"DeepSeek token {len(self._tokens) + 1}",
                token=token,
                enabled=enabled,
                created_at=now,
                updated_at=now,
            )
            self._tokens.append(record)
            self._save_unlocked()
            return record.public_view(now)

    async def update_token(
        self,
        token_id: str,
        *,
        name: str | None = None,
        token: str | None = None,
        enabled: bool | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            record = self._require_token(token_id)
            if name is not None:
                record.name = name or record.name
            if token is not None:
                if not token:
                    raise DeepSeekProxyError(
                        "DeepSeek token may not be empty",
                        status_code=400,
                        code="invalid_token",
                        error_type="invalid_request_error",
                    )
                fingerprint = token_fingerprint(token)
                if any(
                    item.id != token_id and item.fingerprint == fingerprint for item in self._tokens
                ):
                    raise DeepSeekProxyError(
                        "DeepSeek token already exists",
                        status_code=409,
                        code="duplicate_token",
                        error_type="invalid_request_error",
                    )
                if token != record.token:
                    self.client.forget_token(record.token)
                record.token = token
                record.status = "unchecked"
                record.last_error = None
                record.cooldown_until = None
                record.consecutive_failures = 0
            if enabled is not None:
                record.enabled = enabled
                if enabled and record.status == "disabled":
                    record.status = "unchecked"
            record.updated_at = time.time()
            self._save_unlocked()
            return record.public_view()

    async def delete_token(self, token_id: str) -> None:
        async with self._lock:
            record = self._require_token(token_id)
            self.client.forget_token(record.token)
            self._tokens = [item for item in self._tokens if item.id != token_id]
            self._round_robin_index = 0
            self._save_unlocked()

    async def list_tokens(self) -> list[dict[str, Any]]:
        async with self._lock:
            now = time.time()
            return [token.public_view(now) for token in self._tokens]

    async def get_token(self, token_id: str) -> dict[str, Any]:
        async with self._lock:
            return self._require_token(token_id).public_view()

    async def set_strategy(self, strategy: RotationStrategy) -> dict[str, Any]:
        async with self._lock:
            self.strategy = strategy
            self._round_robin_index = 0
            self._save_unlocked()
            return self.summary_unlocked()

    async def summary(self) -> dict[str, Any]:
        async with self._lock:
            return self.summary_unlocked()

    def summary_unlocked(self) -> dict[str, Any]:
        now = time.time()
        views = [token.public_view(now) for token in self._tokens]
        counts = {
            status: 0 for status in ("unchecked", "healthy", "unhealthy", "cooldown", "disabled")
        }
        for view in views:
            counts[str(view["status"])] = counts.get(str(view["status"]), 0) + 1
        return {
            "rotation_strategy": self.strategy,
            "supervisor_running": self.supervisor_running,
            "token_count": len(views),
            "counts": counts,
            "has_static_fallback": self.settings.configured_deepseek_token() is not None,
        }

    async def check_token(self, token_id: str) -> dict[str, Any]:
        async with self._lock:
            record = self._require_token(token_id)
            raw_token = record.token
        try:
            await self.client.acquire_token(raw_token, force_refresh=True)
        except Exception as exc:
            async with self._lock:
                record = self._require_token(token_id)
                self._mark_failure(record, exc, checked=True)
                self._save_unlocked()
                return record.public_view()
        async with self._lock:
            record = self._require_token(token_id)
            self._mark_success(record, checked=True)
            self._save_unlocked()
            return record.public_view()

    async def check_all(self) -> list[dict[str, Any]]:
        async with self._lock:
            token_ids = [token.id for token in self._tokens]
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_HEALTH_CHECKS)

        async def check_with_limit(token_id: str) -> dict[str, Any]:
            async with semaphore:
                return await self.check_token(token_id)

        return await asyncio.gather(*(check_with_limit(token_id) for token_id in token_ids))

    async def _supervise(self) -> None:
        interval = self.settings.token_health_check_interval_seconds
        while True:
            await asyncio.sleep(interval)
            try:
                await self.check_all()
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.warning("DeepSeek token health sweep failed", exc_info=True)
                continue

    def _select_token(self, excluded_ids: set[str]) -> ManagedToken | None:
        now = time.time()
        candidates = [
            token
            for token in self._tokens
            if self._is_usable(token, now) and token.id not in excluded_ids
        ]
        if not candidates:
            return None
        if self.strategy == "fill_first":
            return candidates[0]
        for offset in range(len(self._tokens)):
            index = (self._round_robin_index + offset) % len(self._tokens)
            token = self._tokens[index]
            if token in candidates:
                self._round_robin_index = (index + 1) % len(self._tokens)
                return token
        return None

    def _is_usable(self, token: ManagedToken, now: float) -> bool:
        if not token.enabled or token.status in {"disabled", "unhealthy"}:
            return False
        if token.cooldown_until and token.cooldown_until > now:
            return False
        if token.status == "cooldown" and token.cooldown_until and token.cooldown_until <= now:
            token.status = "unchecked"
            token.cooldown_until = None
        return True

    def _mark_success(self, token: ManagedToken, *, checked: bool) -> None:
        now = time.time()
        token.status = "healthy"
        token.updated_at = now
        token.last_ok_at = now
        token.last_error = None
        token.cooldown_until = None
        token.success_count += 1
        token.consecutive_failures = 0
        if checked:
            token.last_checked_at = now

    def _mark_failure(
        self, token: ManagedToken, error: BaseException | None, *, checked: bool = False
    ) -> None:
        now = time.time()
        token.updated_at = now
        token.failure_count += 1
        token.consecutive_failures += 1
        token.last_error = _safe_error(error)
        if checked:
            token.last_checked_at = now
        if isinstance(error, AuthenticationError):
            token.status = "unhealthy"
            token.cooldown_until = None
        else:
            token.status = "cooldown"
            token.cooldown_until = now + self.settings.token_failure_cooldown_seconds

    def _find_token(self, token_id: str) -> ManagedToken | None:
        return next((token for token in self._tokens if token.id == token_id), None)

    def _require_token(self, token_id: str) -> ManagedToken:
        token = self._find_token(token_id)
        if not token:
            raise _not_found(token_id)
        return token

    def _save_unlocked(self) -> None:
        payload = {
            "version": 1,
            "rotation_strategy": self.strategy,
            "round_robin_index": self._round_robin_index,
            "tokens": [token.to_payload() for token in self._tokens],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            os.chmod(temp_name, 0o600)
            os.replace(temp_name, self.path)
            os.chmod(self.path, 0o600)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)


def token_fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def redact_token(token: str) -> str:
    if len(token) <= 8:
        return "****"
    return f"{token[:4]}...{token[-4:]}"


def _safe_error(error: BaseException | None) -> str | None:
    if error is None:
        return None
    if isinstance(error, DeepSeekProxyError):
        return error.message
    return error.__class__.__name__


def _not_found(token_id: str) -> DeepSeekProxyError:
    return DeepSeekProxyError(
        f"Managed DeepSeek token not found: {token_id}",
        status_code=404,
        code="token_not_found",
        error_type="invalid_request_error",
    )


def _status_or_default(value: object) -> TokenStatus:
    if value in {"unchecked", "healthy", "unhealthy", "cooldown", "disabled"}:
        return cast("TokenStatus", value)
    return "unchecked"


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _float_or_now(value: object) -> float:
    return float(value) if isinstance(value, int | float) else time.time()
