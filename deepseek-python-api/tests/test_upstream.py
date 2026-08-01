from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest

from conftest import FakePowSolver
from deepseek_python_api.errors import AuthenticationError, UpstreamProtocolError
from deepseek_python_api.images import ImageFile
from deepseek_python_api.settings import Settings
from deepseek_python_api.upstream import (
    CHAT_COMPLETION_PATH,
    FILE_UPLOAD_PATH,
    DeepSeekClient,
)


def challenge_payload() -> dict[str, Any]:
    return {
        "code": 0,
        "data": {
            "biz_data": {
                "challenge": {
                    "algorithm": "DeepSeekHashV1",
                    "challenge": "challenge",
                    "salt": "salt",
                    "difficulty": 1,
                    "expire_at": 9999999999,
                    "signature": "signature",
                }
            }
        },
    }


@pytest.mark.asyncio
async def test_token_exchange_is_cached(settings: Settings, fake_pow: FakePowSolver) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers["authorization"] == "Bearer user-token"
        return httpx.Response(200, json={"data": {"biz_data": {"token": "access"}}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = DeepSeekClient(settings=settings, http_client=http, pow_solver=fake_pow)
        assert await client.acquire_token("user-token") == "access"
        assert await client.acquire_token("user-token") == "access"
    assert calls == 1


@pytest.mark.asyncio
async def test_invalid_token_maps_to_authentication_error(
    settings: Settings, fake_pow: FakePowSolver
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(401))
    ) as http:
        client = DeepSeekClient(settings=settings, http_client=http, pow_solver=fake_pow)
        with pytest.raises(AuthenticationError):
            await client.acquire_token("bad-token")


@pytest.mark.asyncio
async def test_session_accepts_nested_and_direct_shapes(
    settings: Settings, fake_pow: FakePowSolver
) -> None:
    responses = iter(
        [
            httpx.Response(200, json={"data": {"biz_data": {"chat_session": {"id": "one"}}}}),
            httpx.Response(200, json={"biz_data": {"id": "two"}}),
        ]
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: next(responses))
    ) as http:
        client = DeepSeekClient(settings=settings, http_client=http, pow_solver=fake_pow)
        assert await client.create_session("access") == "one"
        assert await client.create_session("access") == "two"


@pytest.mark.asyncio
async def test_upload_uses_target_specific_pow_and_polls(
    settings: Settings, fake_pow: FakePowSolver
) -> None:
    requests: list[httpx.Request] = []
    statuses = iter(["PENDING", "SUCCESS"])

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("create_pow_challenge"):
            assert json.loads(request.content)["target_path"] == FILE_UPLOAD_PATH
            return httpx.Response(200, json=challenge_payload())
        if request.url.path.endswith("upload_file"):
            assert request.headers["x-ds-pow-response"] == f"pow:{FILE_UPLOAD_PATH}"
            assert request.headers["x-file-size"] == "3"
            assert request.headers["x-model-type"] == "vision"
            assert b'name="file"' in request.content
            return httpx.Response(200, json={"data": {"biz_data": {"id": "file-1"}}})
        if request.url.path.endswith("fetch_files"):
            status = next(statuses)
            return httpx.Response(
                200,
                json={
                    "data": {
                        "biz_data": {
                            "files": [
                                {
                                    "id": "file-1",
                                    "status": status,
                                    "audit_result": "pass" if status == "SUCCESS" else "",
                                    "model_kind": "VISION",
                                    "is_image": True,
                                }
                            ]
                        }
                    }
                },
            )
        raise AssertionError(request.url)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = DeepSeekClient(settings=settings, http_client=http, pow_solver=fake_pow)
        file_id = await client.upload_image(
            ImageFile("data:", "image.png", "image/png", b"png"), "access"
        )
    assert file_id == "file-1"
    assert [path for _challenge, path in fake_pow.calls] == [FILE_UPLOAD_PATH]
    assert len(requests) == 4


@pytest.mark.asyncio
async def test_upload_rejects_failed_audit(settings: Settings, fake_pow: FakePowSolver) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("create_pow_challenge"):
            return httpx.Response(200, json=challenge_payload())
        if request.url.path.endswith("upload_file"):
            return httpx.Response(200, json={"biz_data": {"file": {"id": "file-1"}}})
        return httpx.Response(
            200,
            json={
                "biz_data": {
                    "files": [{"id": "file-1", "status": "PENDING", "audit_result": "rejected"}]
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = DeepSeekClient(settings=settings, http_client=http, pow_solver=fake_pow)
        with pytest.raises(UpstreamProtocolError, match="audit"):
            await client.upload_image(
                ImageFile("data:", "image.png", "image/png", b"png"), "access"
            )


@pytest.mark.asyncio
async def test_full_vision_completion_payload_and_cleanup(
    settings: Settings, fake_pow: FakePowSolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    completion_payload: dict[str, Any] = {}
    delete_calls = 0

    async def fake_upload_images(
        self: DeepSeekClient, image_urls: list[str], access_token: str
    ) -> list[str]:
        assert image_urls == ["data:image/png;base64," + base64.b64encode(b"png").decode()]
        assert access_token == "access"
        return ["file-1"]

    monkeypatch.setattr(DeepSeekClient, "upload_images", fake_upload_images)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal delete_calls, completion_payload
        if request.url.path.endswith("users/current"):
            return httpx.Response(200, json={"data": {"biz_data": {"token": "access"}}})
        if request.url.path.endswith("chat_session/create"):
            return httpx.Response(200, json={"data": {"biz_data": {"id": "session-1"}}})
        if request.url.path.endswith("create_pow_challenge"):
            assert json.loads(request.content)["target_path"] == CHAT_COMPLETION_PATH
            return httpx.Response(200, json=challenge_payload())
        if request.url.path.endswith("chat/completion"):
            completion_payload = json.loads(request.content)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    b'data: {"v":"answer","response_message_id":"message-1"}\n\ndata: [DONE]\n\n'
                ),
            )
        if request.url.path.endswith("chat_session/delete"):
            delete_calls += 1
            return httpx.Response(200, json={"code": 0})
        raise AssertionError(request.url)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = DeepSeekClient(settings=settings, http_client=http, pow_solver=fake_pow)
        stream = await client.start_completion(
            user_token="user-token",
            prompt="describe",
            model_type="default",
            search_enabled=False,
            thinking_enabled=False,
            image_urls=["data:image/png;base64," + base64.b64encode(b"png").decode()],
        )
        body = b"".join([chunk async for chunk in stream.iter_bytes()])
        await stream.close()
    assert b"answer" in body
    assert completion_payload["model_type"] == "vision"
    assert completion_payload["ref_file_ids"] == ["file-1"]
    assert completion_payload["chat_session_id"] == "session-1"
    assert delete_calls == 1
    assert [path for _challenge, path in fake_pow.calls] == [CHAT_COMPLETION_PATH]
