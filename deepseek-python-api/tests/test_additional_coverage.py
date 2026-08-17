from __future__ import annotations

import socket
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from conftest import FakePowSolver
from deepseek_python_api import __main__ as main_module
from deepseek_python_api.app import create_app
from deepseek_python_api.errors import DeepSeekProxyError, ImageInputError, UpstreamProtocolError
from deepseek_python_api.images import ImageFile, ImageLoader
from deepseek_python_api.prompt import messages_to_prompt
from deepseek_python_api.schemas import ChatCompletionRequest
from deepseek_python_api.settings import Settings
from deepseek_python_api.sse import DeepSeekEventState, iter_sse_data
from deepseek_python_api.upstream import CHAT_COMPLETION_PATH, DeepSeekClient


class ByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


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


def test_main_passes_settings_to_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake_get_settings() -> Settings:
        return Settings(host="127.0.0.2", port=9001, log_level="DEBUG")

    def fake_run(app: str, *, host: str, port: int, log_level: str) -> None:
        calls.update({"app": app, "host": host, "port": port, "log_level": log_level})

    monkeypatch.setattr(main_module, "get_settings", fake_get_settings)
    monkeypatch.setattr(main_module, "run", fake_run)

    main_module.main()

    assert calls == {
        "app": "deepseek_python_api.app:app",
        "host": "127.0.0.2",
        "port": 9001,
        "log_level": "debug",
    }


@pytest.mark.asyncio
async def test_default_lifespan_constructs_owned_resources() -> None:
    app = create_app(Settings(deepseek_auth_token=SecretStr("token")))

    async with app.router.lifespan_context(app):
        assert app.state.runtime.settings.configured_deepseek_token() == "token"


@pytest.mark.asyncio
async def test_per_request_token_enabled_and_missing_config_error(
    tmp_path, fake_pow: FakePowSolver
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("users/current"):
            assert request.headers["authorization"] == "Bearer request-token"
            return httpx.Response(200, json={"data": {"biz_data": {"token": "access"}}})
        if request.url.path.endswith("chat_session/create"):
            return httpx.Response(200, json={"data": {"biz_data": {"id": "session"}}})
        if request.url.path.endswith("create_pow_challenge"):
            return httpx.Response(200, json=challenge_payload())
        if request.url.path.endswith("chat/completion"):
            return httpx.Response(
                200,
                stream=ByteStream([b'data: {"v":"ok","response_message_id":"m"}\n\n']),
            )
        if request.url.path.endswith("chat_session/delete"):
            return httpx.Response(200, json={"code": 0})
        raise AssertionError(request.url)

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        Settings(
            api_key=None,
            allow_request_token=True,
            tokens_file=str(tmp_path / "tokens.json"),
            deepseek_api_base="https://chat.deepseek.test/api",
            proxies_file=str(tmp_path / "missing-proxies.txt"),
        ),
        http_client=upstream,
        pow_solver=fake_pow,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            missing = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v4-flash",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert missing.status_code == 503
            response = await client.post(
                "/v1/chat/completions",
                headers={"X-DeepSeek-Token": "request-token"},
                json={
                    "model": "deepseek-v4-flash",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
    await upstream.aclose()

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_image_loader_rejects_invalid_remote_url(settings: Settings) -> None:
    async with httpx.AsyncClient() as client:
        with pytest.raises(ImageInputError, match="absolute HTTP"):
            await ImageLoader(client, settings).load("ftp://example.com/image.png")


@pytest.mark.asyncio
async def test_image_loader_redirect_missing_location(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    async def fake_getaddrinfo(*_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    monkeypatch.setattr("asyncio.BaseEventLoop.getaddrinfo", fake_getaddrinfo)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(302))
    ) as client:
        with pytest.raises(ImageInputError, match="location"):
            await ImageLoader(client, settings).load("https://example.com/image.png")


@pytest.mark.asyncio
async def test_image_loader_redirect_limit(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    async def fake_getaddrinfo(*_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    monkeypatch.setattr("asyncio.BaseEventLoop.getaddrinfo", fake_getaddrinfo)
    limited = settings.model_copy(update={"remote_image_max_redirects": 0})
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(302, headers={"location": "/again.png"})
        )
    ) as client:
        with pytest.raises(ImageInputError, match="redirect limit"):
            await ImageLoader(client, limited).load("https://example.com/image.png")


@pytest.mark.asyncio
async def test_image_loader_content_length_branches(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    async def fake_getaddrinfo(*_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    monkeypatch.setattr("asyncio.BaseEventLoop.getaddrinfo", fake_getaddrinfo)
    limited = settings.model_copy(update={"max_image_bytes": 3})
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={"content-type": "image/png", "content-length": "4"},
                content=b"",
            )
        )
    ) as client:
        with pytest.raises(ImageInputError, match="exceeds"):
            await ImageLoader(client, limited).load("https://example.com/image.png")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={"content-type": "image/png", "content-length": "not-an-int"},
                content=b"ok",
            )
        )
    ) as client:
        image = await ImageLoader(client, settings).load("https://example.com/image.png")
    assert image.data == b"ok"


@pytest.mark.asyncio
async def test_image_loader_allows_private_urls_when_configured(settings: Settings) -> None:
    allowed = settings.model_copy(update={"allow_private_image_urls": True})
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200, headers={"content-type": "image/png"}, content=b"ok"
            )
        )
    ) as client:
        image = await ImageLoader(client, allowed).load("http://127.0.0.1/image.png")
    assert image.mime_type == "image/png"


def test_prompt_tool_calls_tool_results_and_merging() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "user", "content": "second"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "not-json"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "tool result"},
            ],
        }
    )

    prompt = messages_to_prompt(request.messages)

    assert "first\n\nsecond" in prompt
    assert '<｜DSML｜invoke name="lookup">' in prompt
    assert '<｜DSML｜parameter name="arguments" string="true">not-json</｜DSML｜parameter>' in prompt
    assert '<tool_result>tool result</tool_result>' in prompt


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}], "n": 2},
    ],
)
def test_request_rejects_unsupported_options(payload: dict) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(payload)


@pytest.mark.asyncio
async def test_sse_parser_flushes_eof_data() -> None:
    async def source() -> AsyncIterator[bytes]:
        yield b"data: tail"

    assert [item async for item in iter_sse_data(source())] == ["tail"]


def test_event_state_response_operations_silent_search_and_list_content() -> None:
    state = DeepSeekEventState("deepseek-v4-flash")
    assert state.process({"p": "response", "v": [{"p": "accumulated_token_usage", "v": 7}]}) == []
    assert state.accumulated_token_usage == 7
    state.process({"p": "response", "v": [{"p": "response", "v": {"thinking_enabled": True}}]})
    assert state.current_path == "thinking"
    delta = state.process({"p": "other", "v": [{"v": [{"content": "thought"}]}]})[0]
    assert delta.reasoning_content == "thought"

    silent = DeepSeekEventState("deepseek-v4-flash-search-silent", web_search_enabled=True)
    deltas = silent.process(
        {
            "v": {
                "response": {
                    "fragments": [
                        {
                            "type": "ANSWER",
                            "content": "SEARCH answer[citation:1]",
                            "results": [
                                {
                                    "url": "https://example.com",
                                    "title": "Example",
                                    "citeIndex": 1,
                                }
                            ],
                        }
                    ]
                }
            }
        }
    )
    assert deltas[0].content == "answer"
    assert silent.citations() == ""


@pytest.mark.asyncio
async def test_challenge_retry_and_too_many_images(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, fake_pow: FakePowSolver
) -> None:
    attempts = 0

    async def no_sleep(_delay: float) -> None:
        return None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, json=challenge_payload())

    monkeypatch.setattr("deepseek_python_api.upstream.asyncio.sleep", no_sleep)
    retry_settings = settings.model_copy(update={"challenge_retry_attempts": 2, "max_images": 1})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = DeepSeekClient(settings=retry_settings, http_client=http, pow_solver=fake_pow)
        challenge = await client.get_challenge("access", CHAT_COMPLETION_PATH)
        with pytest.raises(DeepSeekProxyError, match="maximum"):
            await client.upload_images(["one", "two"], "access")

    assert challenge.signature == "signature"
    assert attempts == 2


@pytest.mark.asyncio
async def test_delete_session_without_token_returns_false(
    settings: Settings, fake_pow: FakePowSolver
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(500))
    ) as http:
        client = DeepSeekClient(settings=settings, http_client=http, pow_solver=fake_pow)
        assert await client.delete_session("session") is False


@pytest.mark.asyncio
async def test_wait_for_uploaded_file_failure_modes(
    settings: Settings, fake_pow: FakePowSolver
) -> None:
    async def run_once(response: httpx.Response, expected: str) -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _request: response)
        ) as http:
            client = DeepSeekClient(settings=settings, http_client=http, pow_solver=fake_pow)
            with pytest.raises(UpstreamProtocolError, match=expected):
                await client.wait_for_uploaded_file("file-1", "access")

    await run_once(httpx.Response(500), "HTTP 500")
    await run_once(httpx.Response(200, json={"data": {"biz_data": {"files": []}}}), "not found")
    await run_once(
        httpx.Response(
            200,
            json={"data": {"biz_data": {"files": [{"id": "file-1", "status": "FAILED"}]}}},
        ),
        "processing failed",
    )

    timeout_settings = settings.model_copy(update={"file_poll_attempts": 1})
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={"data": {"biz_data": {"files": [{"id": "file-1", "status": "PENDING"}]}}},
            )
        )
    ) as http:
        client = DeepSeekClient(settings=timeout_settings, http_client=http, pow_solver=fake_pow)
        with pytest.raises(UpstreamProtocolError, match="timed out"):
            await client.wait_for_uploaded_file("file-1", "access")


@pytest.mark.asyncio
async def test_start_completion_error_cleans_session(
    settings: Settings, fake_pow: FakePowSolver
) -> None:
    delete_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal delete_calls
        if request.url.path.endswith("users/current"):
            return httpx.Response(200, json={"data": {"biz_data": {"token": "access"}}})
        if request.url.path.endswith("chat_session/create"):
            return httpx.Response(200, json={"data": {"biz_data": {"id": "session"}}})
        if request.url.path.endswith("create_pow_challenge"):
            return httpx.Response(200, json=challenge_payload())
        if request.url.path.endswith("chat/completion"):
            return httpx.Response(500, content=b"upstream exploded")
        if request.url.path.endswith("chat_session/delete"):
            delete_calls += 1
            return httpx.Response(200, json={"code": 0})
        raise AssertionError(request.url)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = DeepSeekClient(settings=settings, http_client=http, pow_solver=fake_pow)
        with pytest.raises(UpstreamProtocolError, match="HTTP 500"):
            await client.start_completion(
                user_token="user-token",
                prompt="hello",
                model_type="default",
                search_enabled=False,
                thinking_enabled=False,
                image_urls=[],
            )

    assert delete_calls == 1


@pytest.mark.asyncio
async def test_image_loader_additional_error_paths(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    async def public_getaddrinfo(*_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    monkeypatch.setattr("asyncio.BaseEventLoop.getaddrinfo", public_getaddrinfo)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(404))
    ) as client:
        with pytest.raises(ImageInputError, match="HTTP 404"):
            await ImageLoader(client, settings).load("https://example.com/missing.png")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200, headers={"content-type": "text/plain"}, content=b"x"
            )
        )
    ) as client:
        with pytest.raises(ImageInputError, match="only supports image"):
            await ImageLoader(client, settings).load("https://example.com/not-image.txt")

    limited = settings.model_copy(update={"max_image_bytes": 1})
    async with httpx.AsyncClient() as client:
        with pytest.raises(ImageInputError, match="exceeds"):
            await ImageLoader(client, limited).load("data:image/png;base64,eHk=")


@pytest.mark.asyncio
async def test_image_loader_dns_error_paths(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    async def failing_getaddrinfo(*_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        raise socket.gaierror("no such host")

    monkeypatch.setattr("asyncio.BaseEventLoop.getaddrinfo", failing_getaddrinfo)
    async with httpx.AsyncClient() as client:
        with pytest.raises(ImageInputError, match="could not be resolved"):
            await ImageLoader(client, settings).load("https://does-not-resolve.example/image.png")

    async def empty_getaddrinfo(*_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        return []

    monkeypatch.setattr("asyncio.BaseEventLoop.getaddrinfo", empty_getaddrinfo)
    async with httpx.AsyncClient() as client:
        with pytest.raises(ImageInputError, match="did not resolve"):
            await ImageLoader(client, settings).load("https://empty.example/image.png")


@pytest.mark.asyncio
async def test_upload_image_rejects_missing_file_id(
    settings: Settings, fake_pow: FakePowSolver
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("create_pow_challenge"):
            return httpx.Response(200, json=challenge_payload())
        if request.url.path.endswith("upload_file"):
            return httpx.Response(200, json={"data": {"biz_data": {"name": "missing-id"}}})
        raise AssertionError(request.url)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = DeepSeekClient(settings=settings, http_client=http, pow_solver=fake_pow)
        with pytest.raises(UpstreamProtocolError, match=r"file ID|upload failed"):
            await client.upload_image(
                ImageFile("data:", "image.png", "image/png", b"png"), "access"
            )


@pytest.mark.asyncio
async def test_completion_auth_error_closes_stream_context(
    settings: Settings, fake_pow: FakePowSolver
) -> None:
    delete_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal delete_calls
        if request.url.path.endswith("users/current"):
            return httpx.Response(200, json={"data": {"biz_data": {"token": "access"}}})
        if request.url.path.endswith("chat_session/create"):
            return httpx.Response(200, json={"data": {"biz_data": {"id": "session"}}})
        if request.url.path.endswith("create_pow_challenge"):
            return httpx.Response(200, json=challenge_payload())
        if request.url.path.endswith("chat/completion"):
            return httpx.Response(401)
        if request.url.path.endswith("chat_session/delete"):
            delete_calls += 1
            return httpx.Response(200, json={"code": 0})
        raise AssertionError(request.url)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = DeepSeekClient(settings=settings, http_client=http, pow_solver=fake_pow)
        with pytest.raises(DeepSeekProxyError):
            await client.start_completion(
                user_token="user-token",
                prompt="hello",
                model_type="default",
                search_enabled=False,
                thinking_enabled=False,
                image_urls=[],
            )

    assert delete_calls == 1


def test_invalid_settings_api_base() -> None:
    with pytest.raises(ValueError, match="absolute HTTP"):
        Settings(deepseek_api_base="chat.deepseek.test")
