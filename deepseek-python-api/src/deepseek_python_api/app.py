from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

import httpx
from fastapi import Body, Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from .dashboard import DASHBOARD_HTML
from .errors import (
    AuthenticationError,
    ConfigurationError,
    DeepSeekProxyError,
    UpstreamProtocolError,
)
from .models import MODELS
from .pow import DeepSeekHashSolver, PowSolverProtocol
from .proxies import normalize_proxy_url
from .schemas import ChatCompletionRequest, ModelList, ModelObject
from .service import CompletionContext, CompletionService
from .settings import Settings, get_settings
from .token_manager import RotationStrategy, TokenLease, TokenManager
from .upstream import DeepSeekClient

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AppState:
    settings: Settings
    http_client: httpx.AsyncClient
    pow_solver: PowSolverProtocol
    client: DeepSeekClient
    service: CompletionService
    token_manager: TokenManager


class AddManagedTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token: str
    name: str | None = None
    enabled: bool = True
    check: bool = False
    proxy: str | None = None


class UpdateManagedTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token: str | None = None
    name: str | None = None
    enabled: bool | None = None
    proxy: str | None = None


class UpdateRotationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: RotationStrategy


class UpdateProxyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool


def create_app(
    settings: Settings | None = None,
    *,
    http_client: httpx.AsyncClient | None = None,
    pow_solver: PowSolverProtocol | None = None,
) -> FastAPI:
    configured_settings = settings or get_settings()
    owns_http_client = http_client is None
    owns_pow_solver = pow_solver is None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        client = http_client or httpx.AsyncClient()
        solver = pow_solver or DeepSeekHashSolver(
            timeout_seconds=configured_settings.pow_timeout_seconds,
            workers=configured_settings.pow_workers,
        )
        upstream = DeepSeekClient(
            settings=configured_settings,
            http_client=client,
            pow_solver=solver,
        )
        token_manager = TokenManager(configured_settings, upstream)
        token_manager.load()
        await token_manager.start_supervisor()
        app.state.runtime = AppState(
            settings=configured_settings,
            http_client=client,
            pow_solver=solver,
            client=upstream,
            service=CompletionService(upstream),
            token_manager=token_manager,
        )
        yield
        await token_manager.stop_supervisor()
        await upstream.aclose()
        if owns_http_client:
            await client.aclose()
        if owns_pow_solver and isinstance(solver, DeepSeekHashSolver):
            solver.close()

    app = FastAPI(
        title="DeepSeek Python API",
        version="0.1.0",
        description="OpenAI-compatible proxy for DeepSeek's web chat API",
        lifespan=lifespan,
    )

    @app.exception_handler(DeepSeekProxyError)
    async def handle_proxy_error(_request: Request, exc: DeepSeekProxyError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": exc.message,
                    "type": exc.error_type,
                    "code": exc.code,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        message = str(first.get("msg") or "Invalid request")
        location = first.get("loc") or []
        param = ".".join(str(item) for item in location if item != "body") or None
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": param,
                    "code": "invalid_request",
                }
            },
        )

    async def require_client_api_key(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> None:
        runtime = _runtime(app)
        expected = runtime.settings.configured_api_key()
        if not expected:
            return
        provided = x_api_key
        if authorization and authorization.startswith("Bearer "):
            provided = authorization[7:].strip()
        if not provided or not secrets.compare_digest(provided, expected):
            raise DeepSeekProxyError(
                "Invalid API key",
                status_code=401,
                code="invalid_api_key",
                error_type="authentication_error",
            )

    def require_management_api_key(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> None:
        runtime = _runtime(app)
        expected = (
            runtime.settings.configured_management_api_key()
            or runtime.settings.configured_api_key()
        )
        if not expected:
            raise ConfigurationError(
                "MANAGEMENT_API_KEY or API_KEY must be configured for management access"
            )
        provided = x_api_key
        if authorization and authorization.startswith("Bearer "):
            provided = authorization[7:].strip()
        if not provided or not secrets.compare_digest(provided, expected):
            raise DeepSeekProxyError(
                "Invalid management API key",
                status_code=401,
                code="invalid_management_api_key",
                error_type="authentication_error",
            )

    async def resolve_token_lease(
        x_deepseek_token: str | None = Header(default=None),
    ) -> TokenLease:
        runtime = _runtime(app)
        if x_deepseek_token:
            if not runtime.settings.allow_request_token:
                raise ConfigurationError("Per-request DeepSeek tokens are disabled")
            return TokenLease(None, x_deepseek_token, None)
        return await runtime.token_manager.acquire()

    def normalize_request_proxy(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_proxy_url(value)
        if normalized is None:
            raise DeepSeekProxyError(
                "Invalid proxy URL",
                status_code=400,
                code="invalid_proxy",
                error_type="invalid_request_error",
            )
        return normalized

    async def start_with_token_failover(
        body: ChatCompletionRequest,
        initial_lease: TokenLease,
    ) -> CompletionContext:
        runtime = _runtime(app)
        lease = initial_lease
        excluded_ids: set[str] = set()
        while True:
            try:
                return await runtime.service.start(body, token_lease=lease)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Auto-rotate the rotating proxy on 429 rate-limit before failing over
                if (
                    runtime.settings.proxy_rotate_on_ratelimit
                    and _is_ratelimit_error(exc)
                    and lease.proxy_url
                ):
                    LOGGER.info(
                        "Rate-limit detected; attempting proxy IP rotation for %s",
                        lease.proxy_url,
                    )
                    await runtime.token_manager.proxy_pool.rotate_for_proxy_url(
                        lease.proxy_url, client=runtime.http_client
                    )
                if not _should_failover(lease, exc):
                    raise
                if lease.token_id:
                    excluded_ids.add(lease.token_id)
                try:
                    lease = await runtime.token_manager.acquire(excluded_ids=excluded_ids)
                except DeepSeekProxyError:
                    raise exc from None
                LOGGER.info("Retrying DeepSeek completion with another managed token")

    def _is_ratelimit_error(exc: BaseException) -> bool:
        """Return True when the upstream responded with HTTP 429."""
        if isinstance(exc, UpstreamProtocolError):
            return exc.status_code == 429
        if isinstance(exc, DeepSeekProxyError):
            return exc.status_code == 429
        return False

    def _should_failover(lease: TokenLease, exc: BaseException) -> bool:
        if lease.manager is None or lease.token_id is None:
            return False
        if isinstance(exc, AuthenticationError):
            return True
        if isinstance(exc, UpstreamProtocolError):
            return exc.status_code >= 500
        return isinstance(exc, httpx.TransportError | httpx.TimeoutException)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(
        "/v1/models",
        response_model=ModelList,
        dependencies=[Depends(require_client_api_key)],
    )
    async def list_models() -> ModelList:
        return ModelList(data=[ModelObject(id=model) for model in MODELS])

    @app.post(
        "/v1/chat/completions",
        dependencies=[Depends(require_client_api_key)],
        response_model=None,
    )
    async def chat_completions(
        body: ChatCompletionRequest,
        token_lease: TokenLease = Depends(resolve_token_lease),
    ) -> JSONResponse | StreamingResponse:
        context = await start_with_token_failover(body, token_lease)
        if body.stream:
            return StreamingResponse(
                _runtime(app).service.stream_openai(context),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        result = await _runtime(app).service.collect_openai(context)
        return JSONResponse(result)

    @app.get("/v0/management/status", dependencies=[Depends(require_management_api_key)])
    async def management_status() -> dict[str, object]:
        return await _runtime(app).token_manager.summary()

    @app.get("/v0/management/tokens", dependencies=[Depends(require_management_api_key)])
    async def list_managed_tokens() -> dict[str, object]:
        return {"data": await _runtime(app).token_manager.list_tokens()}

    @app.post("/v0/management/tokens", dependencies=[Depends(require_management_api_key)])
    async def add_managed_token(
        payload: Annotated[AddManagedTokenRequest, Body()],
    ) -> dict[str, object]:
        view = await _runtime(app).token_manager.add_token(
            payload.token,
            name=payload.name,
            enabled=payload.enabled,
            proxy_url=normalize_request_proxy(payload.proxy),
        )
        if payload.check:
            view = await _runtime(app).token_manager.check_token(str(view["id"]))
        return view

    @app.get("/v0/management/tokens/{token_id}", dependencies=[Depends(require_management_api_key)])
    async def get_managed_token(token_id: str) -> dict[str, object]:
        return await _runtime(app).token_manager.get_token(token_id)

    @app.patch(
        "/v0/management/tokens/{token_id}", dependencies=[Depends(require_management_api_key)]
    )
    async def update_managed_token(
        token_id: str,
        payload: Annotated[UpdateManagedTokenRequest, Body()],
    ) -> dict[str, object]:
        return await _runtime(app).token_manager.update_token(
            token_id,
            name=payload.name,
            token=payload.token,
            enabled=payload.enabled,
            proxy_url=normalize_request_proxy(payload.proxy),
        )

    @app.delete(
        "/v0/management/tokens/{token_id}", dependencies=[Depends(require_management_api_key)]
    )
    async def delete_managed_token(token_id: str) -> dict[str, bool]:
        await _runtime(app).token_manager.delete_token(token_id)
        return {"deleted": True}

    @app.post(
        "/v0/management/tokens/{token_id}/check",
        dependencies=[Depends(require_management_api_key)],
    )
    async def check_managed_token(token_id: str) -> dict[str, object]:
        return await _runtime(app).token_manager.check_token(token_id)

    @app.post("/v0/management/tokens/check", dependencies=[Depends(require_management_api_key)])
    async def check_all_managed_tokens() -> dict[str, object]:
        return {"data": await _runtime(app).token_manager.check_all()}

    @app.patch("/v0/management/rotation", dependencies=[Depends(require_management_api_key)])
    async def update_rotation_strategy(
        payload: Annotated[UpdateRotationRequest, Body()],
    ) -> dict[str, object]:
        return await _runtime(app).token_manager.set_strategy(payload.strategy)

    # -----------------------------------------------------------------------
    # Proxy-pool management
    # -----------------------------------------------------------------------

    @app.get("/v0/management/proxies", dependencies=[Depends(require_management_api_key)])
    async def proxy_pool_status() -> dict[str, object]:
        """Return status of the entire proxy pool (static + rotating entries)."""
        return _runtime(app).token_manager.proxy_pool.summary()

    @app.post("/v0/management/proxies/rotate", dependencies=[Depends(require_management_api_key)])
    async def rotate_all_proxies() -> dict[str, object]:
        """Trigger IP rotation for every rotating proxy whose cooldown has expired."""
        runtime = _runtime(app)
        results = await runtime.token_manager.proxy_pool.rotate_all(client=runtime.http_client)
        return {"results": results}

    @app.post(
        "/v0/management/tokens/{token_id}/rotate-proxy",
        dependencies=[Depends(require_management_api_key)],
    )
    async def rotate_token_proxy(token_id: str) -> dict[str, object]:
        """Rotate the IP of the rotating proxy bound to a specific token."""
        runtime = _runtime(app)
        return await runtime.token_manager.rotate_proxy_for_token(
            token_id, http_client=runtime.http_client
        )

    @app.patch(
        "/v0/management/proxies/{proxy_id}",
        dependencies=[Depends(require_management_api_key)],
    )
    async def update_proxy(
        proxy_id: str,
        payload: Annotated[UpdateProxyRequest, Body()],
    ) -> dict[str, object]:
        """Enable or disable a proxy by its stable proxy_id."""
        runtime = _runtime(app)
        if payload.enabled:
            return await runtime.token_manager.enable_proxy(proxy_id)
        return await runtime.token_manager.disable_proxy(proxy_id)

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        return HTMLResponse(DASHBOARD_HTML)

    return app


def _runtime(app: FastAPI) -> AppState:
    runtime = app.state.runtime
    if not isinstance(runtime, AppState):
        raise RuntimeError("Application runtime is not initialized")
    return runtime


app = create_app()
