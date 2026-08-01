from __future__ import annotations

import asyncio
import base64
import binascii
import ipaddress
import mimetypes
import socket
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import unquote, urljoin, urlsplit
from uuid import uuid4

import httpx

from .errors import ImageInputError
from .settings import Settings


@dataclass(frozen=True, slots=True)
class ImageFile:
    source_url: str
    filename: str
    mime_type: str
    data: bytes


class ImageLoader:
    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    async def load(self, source: str) -> ImageFile:
        if source.startswith("data:"):
            return self._load_data_url(source)
        return await self._load_remote_url(source)

    def _load_data_url(self, source: str) -> ImageFile:
        header, separator, encoded = source.partition(",")
        if not separator or ";base64" not in header.lower():
            raise ImageInputError("DeepSeek vision data URL must be base64 encoded")

        mime_type = header[5:].split(";", 1)[0].strip().lower()
        if not mime_type.startswith("image/"):
            raise ImageInputError(
                f"DeepSeek vision only supports image content, got {mime_type or 'unknown'}"
            )

        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ImageInputError("DeepSeek vision data URL contains invalid base64") from exc

        self._validate_size(data)
        extension = mimetypes.guess_extension(mime_type, strict=False) or ".bin"
        return ImageFile(source, f"{uuid4()}{extension}", mime_type, data)

    async def _load_remote_url(self, source: str) -> ImageFile:
        current_url = source
        for redirect_count in range(self._settings.remote_image_max_redirects + 1):
            parsed = urlsplit(current_url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ImageInputError(
                    "DeepSeek vision remote image_url must be an absolute HTTP(S) URL"
                )

            await self._validate_remote_host(parsed.hostname, parsed.port)
            async with self._client.stream(
                "GET",
                current_url,
                follow_redirects=False,
                timeout=self._settings.request_timeout_seconds,
                headers={"Accept": "image/*"},
            ) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise ImageInputError("Remote image redirect did not include a location")
                    if redirect_count >= self._settings.remote_image_max_redirects:
                        raise ImageInputError("Remote image exceeded the redirect limit")
                    current_url = urljoin(current_url, location)
                    continue

                if response.status_code < 200 or response.status_code >= 300:
                    raise ImageInputError(
                        f"Failed to download DeepSeek vision image: HTTP {response.status_code}"
                    )

                declared_length = response.headers.get("content-length")
                if declared_length:
                    try:
                        if int(declared_length) > self._settings.max_image_bytes:
                            maximum = self._settings.max_image_bytes
                            raise ImageInputError(f"DeepSeek vision image exceeds {maximum} bytes")
                    except ValueError:
                        pass

                chunks: list[bytes] = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > self._settings.max_image_bytes:
                        raise ImageInputError(
                            f"DeepSeek vision image exceeds {self._settings.max_image_bytes} bytes"
                        )
                    chunks.append(chunk)

                data = b"".join(chunks)
                content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                filename = unquote(PurePosixPath(parsed.path).name) or f"{uuid4()}.bin"
                mime_type = (
                    content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
                )
                if not mime_type.startswith("image/"):
                    raise ImageInputError(
                        f"DeepSeek vision only supports image content, got {mime_type}"
                    )
                self._validate_size(data)
                return ImageFile(source, filename, mime_type, data)

        raise ImageInputError("Remote image exceeded the redirect limit")

    async def _validate_remote_host(self, hostname: str, port: int | None) -> None:
        if self._settings.allow_private_image_urls:
            return

        try:
            direct_ip = ipaddress.ip_address(hostname.strip("[]"))
            addresses = {direct_ip}
        except ValueError:
            loop = asyncio.get_running_loop()
            try:
                results = await loop.getaddrinfo(
                    hostname,
                    port or 443,
                    type=socket.SOCK_STREAM,
                )
            except socket.gaierror as exc:
                raise ImageInputError("Remote image hostname could not be resolved") from exc
            addresses = {ipaddress.ip_address(item[4][0]) for item in results}

        if not addresses:
            raise ImageInputError("Remote image hostname did not resolve to an address")
        if any(not address.is_global for address in addresses):
            raise ImageInputError("Remote image URL resolves to a private or reserved address")

    def _validate_size(self, data: bytes) -> None:
        if len(data) > self._settings.max_image_bytes:
            raise ImageInputError(
                f"DeepSeek vision image exceeds {self._settings.max_image_bytes} bytes"
            )
