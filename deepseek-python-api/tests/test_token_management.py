from __future__ import annotations

import stat
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pydantic import SecretStr

from conftest import FakePowSolver
from deepseek_python_api.app import create_app
from deepseek_python_api.errors import (
    AuthenticationError,
    ConfigurationError,
    DeepSeekProxyError,
    UpstreamProtocolError,
)
from deepseek_python_api.settings import Settings
from deepseek_python_api.token_manager import TokenManager
from deepseek_python_api.upstream import DeepSeekClient


class FakeDeepSeekClient:
    def __init__(self) -> None:
        self.failures: dict[str, Exception] = {}
        self.checked: list[str] = []
        self.forgotten: list[str] = []

    async def acquire_token(self, user_token: str, *, force_refresh: bool = False) -> str:
        self.checked.append(user_token)
        failure = self.failures.get(user_token)
        if failure:
            raise failure
        return f"access:{user_token}:{force_refresh}"

    def forget_token(self, user_token: str) -> None:
        self.forgotten.append(user_token)


def make_settings(path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "tokens_file": str(path),
        "token_health_check_interval_seconds": 0,
        "token_failure_cooldown_seconds": 60,
    }
    values.update(overrides)
    return Settings(**values)


def token_manager(settings: Settings, client: FakeDeepSeekClient) -> TokenManager:
    return TokenManager(settings, cast("DeepSeekClient", client))


@pytest.mark.asyncio
async def test_token_manager_persists_redacts_and_rotates(tmp_path: Path) -> None:
    client = FakeDeepSeekClient()
    manager = token_manager(make_settings(tmp_path / "tokens.json"), client)
    manager.load()

    first = await manager.add_token("first-managed-token", name="first")
    second = await manager.add_token("second-managed-token", name="second")

    assert first["redacted_token"] == "firs...oken"
    assert "token" not in first
    assert stat.S_IMODE((tmp_path / "tokens.json").stat().st_mode) == 0o600

    first_lease = await manager.acquire()
    assert first_lease.token == "first-managed-token"
    await first_lease.release(success=True)

    await manager.set_strategy("round_robin")
    lease_one = await manager.acquire()
    await lease_one.release(success=True)
    lease_two = await manager.acquire()
    await lease_two.release(success=True)
    assert [lease_one.token_id, lease_two.token_id] == [first["id"], second["id"]]

    replacement = await manager.update_token(str(first["id"]), token="replacement-managed-token")
    assert replacement["status"] == "unchecked"
    assert client.forgotten == ["first-managed-token"]

    reloaded = token_manager(make_settings(tmp_path / "tokens.json"), client)
    reloaded.load()
    assert (await reloaded.summary())["token_count"] == 2

    await reloaded.delete_token(str(first["id"]))
    assert "replacement-managed-token" in client.forgotten

    with pytest.raises(DeepSeekProxyError, match="already exists"):
        await reloaded.add_token("second-managed-token")


@pytest.mark.asyncio
async def test_token_manager_error_paths_supervisor_and_short_redaction(tmp_path: Path) -> None:
    client = FakeDeepSeekClient()
    manager = token_manager(make_settings(tmp_path / "tokens.json"), client)

    with pytest.raises(DeepSeekProxyError, match="may not be empty"):
        await manager.add_token("")
    with pytest.raises(DeepSeekProxyError, match="not found"):
        await manager.get_token("missing")

    short = await manager.add_token("short")
    assert short["redacted_token"] == "****"
    disabled = await manager.add_token("disabled-token", enabled=False)
    assert disabled["status"] == "disabled"
    enabled = await manager.update_token(str(disabled["id"]), enabled=True)
    assert enabled["status"] == "unchecked"
    same_token = await manager.update_token(str(short["id"]), token="short")
    assert same_token["status"] == "unchecked"
    assert client.forgotten == []

    lease = await manager.acquire()
    await manager.delete_token(str(lease.token_id))
    await lease.release(success=True)

    no_usable = token_manager(make_settings(tmp_path / "no_usable.json"), client)
    unhealthy = await no_usable.add_token("bad")
    client.failures["bad"] = AuthenticationError()
    await no_usable.check_token(str(unhealthy["id"]))
    with pytest.raises(ConfigurationError, match="No enabled healthy"):
        await no_usable.acquire()

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Failed to load"):
        token_manager(make_settings(malformed), client).load()

    invalid_shape = tmp_path / "invalid_shape.json"
    invalid_shape.write_text('{"tokens": {}}', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid shape"):
        token_manager(make_settings(invalid_shape), client).load()

    invalid_token = tmp_path / "invalid_token.json"
    invalid_token.write_text('{"tokens": [{"token": ""}]}', encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid token"):
        token_manager(make_settings(invalid_token), client).load()

    raw_payload = tmp_path / "payload.json"
    raw_payload.write_text(
        '{"rotation_strategy":"round_robin","round_robin_index":3,'
        '"tokens":[{"token":"loaded-token","status":"mystery","created_at":"bad"}]}',
        encoding="utf-8",
    )
    loaded = token_manager(make_settings(raw_payload), client)
    loaded.load()
    assert (await loaded.summary())["rotation_strategy"] == "round_robin"
    assert (await loaded.list_tokens())[0]["status"] == "unchecked"

    supervisor = token_manager(
        make_settings(tmp_path / "supervisor.json", token_health_check_interval_seconds=1), client
    )
    await supervisor.start_supervisor()
    await supervisor.start_supervisor()
    assert supervisor.supervisor_running is True
    await supervisor.stop_supervisor()
    assert supervisor.supervisor_running is False


@pytest.mark.asyncio
async def test_token_health_states_and_cooldown(tmp_path: Path) -> None:
    client = FakeDeepSeekClient()
    client.failures["expired-token"] = AuthenticationError()
    client.failures["temporary-token"] = UpstreamProtocolError("temporary outage")
    manager = token_manager(make_settings(tmp_path / "tokens.json"), client)

    expired = await manager.add_token("expired-token")
    temporary = await manager.add_token("temporary-token")
    healthy = await manager.add_token("healthy-token")

    assert (await manager.check_token(str(expired["id"])))["status"] == "unhealthy"
    temporary_view = await manager.check_token(str(temporary["id"]))
    assert temporary_view["status"] == "cooldown"
    assert temporary_view["cooldown_until"] is not None
    assert (await manager.check_token(str(healthy["id"])))["status"] == "healthy"

    summary = await manager.summary()
    assert summary["counts"]["unhealthy"] == 1
    assert summary["counts"]["cooldown"] == 1
    assert summary["counts"]["healthy"] == 1


@pytest.mark.asyncio
async def test_management_api_crud_dashboard_and_validation(
    tmp_path: Path, fake_pow: FakePowSolver
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("users/current"):
            assert request.headers["authorization"] == "Bearer managed-token"
            return httpx.Response(200, json={"data": {"biz_data": {"token": "access"}}})
        raise AssertionError(request.url)

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        Settings(
            management_api_key=SecretStr("management-key"),
            tokens_file=str(tmp_path / "tokens.json"),
            token_health_check_interval_seconds=0,
        ),
        http_client=upstream,
        pow_solver=fake_pow,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as api:
            unauthorized = await api.get("/v0/management/status")
            assert unauthorized.status_code == 401
            invalid_add = await api.post(
                "/v0/management/tokens",
                headers={"Authorization": "Bearer management-key"},
                json={"name": "missing-token"},
            )
            assert invalid_add.status_code == 400

            headers = {"Authorization": "Bearer management-key"}
            added = await api.post(
                "/v0/management/tokens",
                headers=headers,
                json={"token": "managed-token", "name": "primary", "check": True},
            )
            assert added.status_code == 200
            token_id = added.json()["id"]
            assert added.json()["status"] == "healthy"
            assert "managed-token" not in added.text

            listed = await api.get("/v0/management/tokens", headers=headers)
            assert listed.status_code == 200
            assert listed.json()["data"][0]["name"] == "primary"
            assert "managed-token" not in listed.text

            invalid_rotation = await api.patch(
                "/v0/management/rotation",
                headers=headers,
                json={"strategy": "random"},
            )
            assert invalid_rotation.status_code == 400
            rotated = await api.patch(
                "/v0/management/rotation",
                headers=headers,
                json={"strategy": "round_robin"},
            )
            assert rotated.json()["rotation_strategy"] == "round_robin"

            got = await api.get(f"/v0/management/tokens/{token_id}", headers=headers)
            assert got.json()["id"] == token_id
            checked_all = await api.post("/v0/management/tokens/check", headers=headers)
            assert checked_all.status_code == 200
            checked_one = await api.post(f"/v0/management/tokens/{token_id}/check", headers=headers)
            assert checked_one.status_code == 200

            updated = await api.patch(
                f"/v0/management/tokens/{token_id}",
                headers=headers,
                json={"enabled": False, "name": "disabled"},
            )
            assert updated.json()["status"] == "disabled"

            dashboard = await api.get("/dashboard")
            assert dashboard.status_code == 200
            assert "DeepSeek Python API Dashboard" in dashboard.text
            assert "managed-token" not in dashboard.text

            deleted = await api.delete(f"/v0/management/tokens/{token_id}", headers=headers)
            assert deleted.json() == {"deleted": True}
    await upstream.aclose()


@pytest.mark.asyncio
async def test_managed_chat_failover_uses_next_token(
    tmp_path: Path, fake_pow: FakePowSolver
) -> None:
    completion_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal completion_calls
        path = request.url.path
        auth = request.headers.get("authorization")
        if path.endswith("users/current"):
            if auth == "Bearer bad-token":
                return httpx.Response(401)
            if auth == "Bearer good-token":
                return httpx.Response(200, json={"data": {"biz_data": {"token": "access"}}})
        if path.endswith("chat_session/create"):
            assert auth == "Bearer access"
            return httpx.Response(200, json={"data": {"biz_data": {"id": "session"}}})
        if path.endswith("create_pow_challenge"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "biz_data": {
                            "challenge": {
                                "algorithm": "DeepSeekHashV1",
                                "challenge": "c",
                                "salt": "s",
                                "difficulty": 1,
                                "expire_at": 9999999999,
                                "signature": "sig",
                            }
                        }
                    }
                },
            )
        if path.endswith("chat/completion"):
            completion_calls += 1
            return httpx.Response(200, stream=ByteStream())
        if path.endswith("chat_session/delete"):
            return httpx.Response(200, json={"code": 0})
        raise AssertionError(request.url)

    class ByteStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b'data: {"v":"ok","response_message_id":"m"}\n\n'

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        Settings(
            management_api_key=SecretStr("management-key"),
            tokens_file=str(tmp_path / "tokens.json"),
            token_health_check_interval_seconds=0,
            deepseek_api_base="https://chat.deepseek.test/api",
        ),
        http_client=upstream,
        pow_solver=fake_pow,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as api:
            headers = {"Authorization": "Bearer management-key"}
            first = await api.post(
                "/v0/management/tokens",
                headers=headers,
                json={"token": "bad-token", "name": "bad"},
            )
            second = await api.post(
                "/v0/management/tokens",
                headers=headers,
                json={"token": "good-token", "name": "good"},
            )
            assert first.status_code == 200
            assert second.status_code == 200

            response = await api.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v4-flash",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            assert response.status_code == 200
            assert response.json()["choices"][0]["message"]["content"] == "ok"

            tokens = (await api.get("/v0/management/tokens", headers=headers)).json()["data"]
            by_name = {item["name"]: item for item in tokens}
            assert by_name["bad"]["status"] == "unhealthy"
            assert by_name["good"]["status"] == "healthy"
            assert completion_calls == 1
    await upstream.aclose()
