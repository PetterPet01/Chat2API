from __future__ import annotations

import base64
import socket

import httpx
import pytest

from deepseek_python_api.errors import ImageInputError
from deepseek_python_api.images import ImageLoader
from deepseek_python_api.settings import Settings


@pytest.mark.asyncio
async def test_loads_base64_data_url(settings: Settings) -> None:
    loader = ImageLoader(httpx.AsyncClient(), settings)
    try:
        image = await loader.load("data:image/png;base64," + base64.b64encode(b"png").decode())
    finally:
        await loader._client.aclose()
    assert image.mime_type == "image/png"
    assert image.data == b"png"
    assert image.filename.endswith(".png")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [
        "data:image/png,not-base64",
        "data:text/plain;base64,eA==",
        "data:image/png;base64,!!!",
    ],
)
async def test_rejects_invalid_data_urls(settings: Settings, source: str) -> None:
    async with httpx.AsyncClient() as client:
        loader = ImageLoader(client, settings)
        with pytest.raises(ImageInputError):
            await loader.load(source)


@pytest.mark.asyncio
async def test_rejects_private_remote_url(settings: Settings) -> None:
    async with httpx.AsyncClient() as client:
        loader = ImageLoader(client, settings)
        with pytest.raises(ImageInputError, match="private or reserved"):
            await loader.load("http://127.0.0.1/image.png")


@pytest.mark.asyncio
async def test_downloads_public_remote_image(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    async def fake_getaddrinfo(*_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    monkeypatch.setattr("asyncio.BaseEventLoop.getaddrinfo", fake_getaddrinfo)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.com/image.png"
        return httpx.Response(200, headers={"content-type": "image/png"}, content=b"image")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        image = await ImageLoader(client, settings).load("https://example.com/image.png")
    assert image.filename == "image.png"
    assert image.mime_type == "image/png"
    assert image.data == b"image"


@pytest.mark.asyncio
async def test_revalidates_redirect_targets(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    async def fake_getaddrinfo(
        _self: object, hostname: str, *_args: object, **_kwargs: object
    ) -> list[tuple[object, ...]]:
        address = "93.184.216.34" if hostname == "example.com" else "127.0.0.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))]

    monkeypatch.setattr("asyncio.BaseEventLoop.getaddrinfo", fake_getaddrinfo)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://internal.test/image.png"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ImageInputError, match="private or reserved"):
            await ImageLoader(client, settings).load("https://example.com/image.png")


@pytest.mark.asyncio
async def test_enforces_streamed_size_limit(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    limited = settings.model_copy(update={"max_image_bytes": 3})

    async def fake_getaddrinfo(*_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    monkeypatch.setattr("asyncio.BaseEventLoop.getaddrinfo", fake_getaddrinfo)
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, headers={"content-type": "image/png"}, content=b"1234")
    )
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ImageInputError, match="exceeds"):
            await ImageLoader(client, limited).load("https://example.com/image.png")
