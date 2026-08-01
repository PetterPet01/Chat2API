from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest
from pydantic import SecretStr

from conftest import FakePowSolver
from deepseek_python_api.app import create_app
from deepseek_python_api.settings import Settings


class ByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


@pytest.mark.asyncio
async def test_health_models_and_validation(fake_pow: FakePowSolver) -> None:
    settings = Settings(
        deepseek_auth_token=SecretStr("token"),
        api_key=SecretStr("client-key"),
    )
    upstream = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(500))
    )
    app = create_app(settings, http_client=upstream, pow_solver=fake_pow)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.get("/health")).json() == {"status": "ok"}
            unauthorized = await client.get("/v1/models")
            assert unauthorized.status_code == 401
            models = await client.get("/v1/models", headers={"Authorization": "Bearer client-key"})
            assert models.status_code == 200
            assert any(item["id"] == "deepseek-v4-flash" for item in models.json()["data"])
            invalid = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer client-key"},
                json={"model": "deepseek-v4-flash", "messages": []},
            )
            assert invalid.status_code == 400
            assert invalid.json()["error"]["code"] == "invalid_request"
    await upstream.aclose()


@pytest.mark.asyncio
async def test_non_streaming_chat_completion(fake_pow: FakePowSolver) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("users/current"):
            return httpx.Response(200, json={"data": {"biz_data": {"token": "access"}}})
        if request.url.path.endswith("chat_session/create"):
            return httpx.Response(200, json={"data": {"biz_data": {"id": "session"}}})
        if request.url.path.endswith("create_pow_challenge"):
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
        if request.url.path.endswith("chat/completion"):
            assert json.loads(request.content)["prompt"] == "hello"
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=ByteStream(
                    [
                        b'data: {"v":{"response":{"thinking_enabled":true,"fragments":',
                        (
                            b'[{"type":"THINK","content":"reason "},'
                            b'{"type":"ANSWER","content":"hello FINISHED"}]}}'
                            b',"response_message_id":"m"}\n\n'
                        ),
                        b"data: [DONE]\n\n",
                    ]
                ),
            )
        if request.url.path.endswith("chat_session/delete"):
            return httpx.Response(200, json={"code": 0})
        raise AssertionError(request.url)

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        Settings(
            deepseek_auth_token=SecretStr("user-token"),
            deepseek_api_base="https://chat.deepseek.test/api",
        ),
        http_client=upstream,
        pow_solver=fake_pow,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v4-flash-think",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
    await upstream.aclose()
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "hello"
    assert body["choices"][0]["message"]["reasoning_content"] == "reason"
    assert body["id"].startswith("chatcmpl-")
    assert "session" not in body["id"]


@pytest.mark.asyncio
async def test_streaming_chat_completion(fake_pow: FakePowSolver) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("users/current"):
            return httpx.Response(200, json={"data": {"biz_data": {"token": "access"}}})
        if request.url.path.endswith("chat_session/create"):
            return httpx.Response(200, json={"biz_data": {"id": "session"}})
        if request.url.path.endswith("create_pow_challenge"):
            return httpx.Response(
                200,
                json={
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
                },
            )
        if request.url.path.endswith("chat/completion"):
            return httpx.Response(
                200,
                stream=ByteStream(
                    [
                        b'data: {"v":"hel","response_message_id":"m"}\n\n',
                        b'data: {"v":"lo"}\n\ndata: [DONE]\n\n',
                    ]
                ),
            )
        if request.url.path.endswith("chat_session/delete"):
            return httpx.Response(200, json={"code": 0})
        raise AssertionError(request.url)

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        Settings(
            deepseek_auth_token=SecretStr("user-token"),
            deepseek_api_base="https://chat.deepseek.test/api",
        ),
        http_client=upstream,
        pow_solver=fake_pow,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v4-flash",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )
    await upstream.aclose()
    assert response.status_code == 200
    assert response.text.endswith("data: [DONE]\n\n")
    assert '"role":"assistant"' in response.text
    assert '"content":"hel"' in response.text
    assert '"content":"lo"' in response.text
    assert '"finish_reason":"stop"' in response.text


@pytest.mark.asyncio
async def test_request_token_requires_explicit_enable(fake_pow: FakePowSolver) -> None:
    upstream = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(500))
    )
    app = create_app(Settings(), http_client=upstream, pow_solver=fake_pow)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"X-DeepSeek-Token": "token"},
                json={
                    "model": "deepseek-v4-flash",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
    await upstream.aclose()
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "configuration_error"
