from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx

from .errors import AuthenticationError, DeepSeekProxyError, UpstreamProtocolError
from .images import ImageFile, ImageLoader
from .pow import Challenge, PowSolverProtocol
from .settings import Settings

LOGGER = logging.getLogger(__name__)
CHAT_COMPLETION_PATH = "/api/v0/chat/completion"
FILE_UPLOAD_PATH = "/api/v0/file/upload_file"
FILE_FETCH_PATH = "/api/v0/file/fetch_files"

FAKE_HEADERS = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
    "Origin": "https://chat.deepseek.com",
    "Referer": "https://chat.deepseek.com/",
    "Sec-Ch-Ua": '"Not/A)Brand";v="99", "Chromium";v="148"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"macOS"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
    ),
    "X-App-Version": "2.0.0",
    "X-Client-Locale": "zh_CN",
    "X-Client-Platform": "web",
    "x-Client-Timezone-Offset": "28800",
    "X-Client-Version": "2.0.0",
}


@dataclass(slots=True)
class TokenCacheEntry:
    access_token: str
    expires_at: float


class DeepSeekStream:
    def __init__(
        self,
        response_context: Any,
        response: httpx.Response,
        client: DeepSeekClient,
        session_id: str,
        user_token: str,
    ) -> None:
        self._response_context = response_context
        self.response = response
        self._client = client
        self.session_id = session_id
        self._user_token = user_token
        self._closed = False

    async def iter_bytes(self) -> AsyncIterator[bytes]:
        async for chunk in self.response.aiter_bytes():
            yield chunk

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._response_context.__aexit__(None, None, None)
        if self._client.settings.delete_sessions:
            await self._client.delete_session(self.session_id, self._user_token)


class DeepSeekClient:
    def __init__(
        self,
        *,
        settings: Settings,
        http_client: httpx.AsyncClient,
        pow_solver: PowSolverProtocol,
    ) -> None:
        self.settings = settings
        self._http = http_client
        self._pow = pow_solver
        self._tokens: dict[str, TokenCacheEntry] = {}
        self._token_locks: dict[str, asyncio.Lock] = {}
        self._image_loader = ImageLoader(http_client, settings)

    async def acquire_token(self, user_token: str, *, force_refresh: bool = False) -> str:
        cached = self._tokens.get(user_token)
        if not force_refresh and cached and cached.expires_at > time.monotonic():
            return cached.access_token

        lock = self._token_locks.setdefault(user_token, asyncio.Lock())
        async with lock:
            cached = self._tokens.get(user_token)
            if not force_refresh and cached and cached.expires_at > time.monotonic():
                return cached.access_token

            response = await self._http.get(
                self._api_url("/api/v0/users/current"),
                headers=self._headers(user_token),
                timeout=self.settings.connect_timeout_seconds,
            )
            if response.status_code in {401, 403}:
                raise AuthenticationError()
            if response.status_code != 200:
                raise UpstreamProtocolError(
                    f"Failed to acquire DeepSeek access token: HTTP {response.status_code}"
                )
            biz_data = _biz_data(response)
            access_token = biz_data.get("token") if isinstance(biz_data, dict) else None
            if not isinstance(access_token, str) or not access_token:
                raise UpstreamProtocolError(
                    "DeepSeek access-token response did not contain a token"
                )
            self._tokens[user_token] = TokenCacheEntry(
                access_token=access_token,
                expires_at=time.monotonic() + self.settings.access_token_ttl_seconds,
            )
            return access_token

    def forget_token(self, user_token: str) -> None:
        self._tokens.pop(user_token, None)
        self._token_locks.pop(user_token, None)

    async def create_session(self, access_token: str) -> str:
        response = await self._http.post(
            self._api_url("/api/v0/chat_session/create"),
            json={},
            headers=self._headers(access_token, cookie=True),
            timeout=self.settings.connect_timeout_seconds,
        )
        biz_data = _biz_data(response)
        session_id = None
        if isinstance(biz_data, dict):
            chat_session = biz_data.get("chat_session")
            if isinstance(chat_session, dict):
                session_id = chat_session.get("id")
            session_id = session_id or biz_data.get("id")
        if response.status_code != 200 or not isinstance(session_id, str):
            raise UpstreamProtocolError(
                f"Failed to create DeepSeek chat session: HTTP {response.status_code}"
            )
        return session_id

    async def delete_session(self, session_id: str, user_token: str | None = None) -> bool:
        try:
            token = (
                await self.acquire_token(user_token)
                if user_token
                else next(
                    (
                        item.access_token
                        for item in self._tokens.values()
                        if item.expires_at > time.monotonic()
                    ),
                    None,
                )
            )
            if not token:
                return False
            response = await self._http.post(
                self._api_url("/api/v0/chat_session/delete"),
                json={"chat_session_id": session_id},
                headers=self._headers(token),
                timeout=self.settings.connect_timeout_seconds,
            )
            return response.status_code == 200 and response.json().get("code") == 0
        except Exception:
            LOGGER.warning("Best-effort DeepSeek session cleanup failed", exc_info=True)
            return False

    async def get_challenge(self, access_token: str, target_path: str) -> Challenge:
        last_error: Exception | None = None
        for attempt in range(self.settings.challenge_retry_attempts):
            try:
                response = await self._http.post(
                    self._api_url("/api/v0/chat/create_pow_challenge"),
                    json={"target_path": target_path},
                    headers=self._headers(access_token),
                    timeout=self.settings.challenge_timeout_seconds,
                )
                biz_data = _biz_data(response)
                challenge = biz_data.get("challenge") if isinstance(biz_data, dict) else None
                if response.status_code != 200 or not isinstance(challenge, dict):
                    raise UpstreamProtocolError(
                        "Failed to get DeepSeek proof-of-work challenge: "
                        f"HTTP {response.status_code}"
                    )
                return Challenge.from_payload(challenge)
            except (httpx.TransportError, httpx.TimeoutException, UpstreamProtocolError) as exc:
                last_error = exc
                if attempt + 1 < self.settings.challenge_retry_attempts:
                    await asyncio.sleep(1)
        if isinstance(last_error, DeepSeekProxyError):
            raise last_error
        raise UpstreamProtocolError(
            "Failed to get DeepSeek proof-of-work challenge"
        ) from last_error

    async def upload_images(self, image_urls: list[str], access_token: str) -> list[str]:
        if len(image_urls) > self.settings.max_images:
            raise DeepSeekProxyError(
                f"A maximum of {self.settings.max_images} images is supported",
                status_code=400,
                code="too_many_images",
                error_type="invalid_request_error",
            )
        file_ids: list[str] = []
        for image_url in image_urls:
            image = await self._image_loader.load(image_url)
            file_ids.append(await self.upload_image(image, access_token))
        return file_ids

    async def upload_image(self, image: ImageFile, access_token: str) -> str:
        challenge = await self.get_challenge(access_token, FILE_UPLOAD_PATH)
        answer = await self._pow.create_answer(challenge, FILE_UPLOAD_PATH)
        response = await self._http.post(
            self._api_url(FILE_UPLOAD_PATH),
            files={"file": (image.filename, image.data, image.mime_type)},
            headers=self._headers(
                access_token,
                cookie=True,
                extra={
                    "X-Ds-Pow-Response": answer,
                    "X-File-Size": str(len(image.data)),
                    "X-Model-Type": "vision",
                },
            ),
            timeout=self.settings.request_timeout_seconds,
        )
        uploaded = _extract_uploaded_file(response)
        if response.status_code != 200 or not uploaded:
            raise UpstreamProtocolError(
                f"DeepSeek vision upload failed: HTTP {response.status_code}"
            )
        file_id = uploaded.get("id")
        if not isinstance(file_id, str):
            raise UpstreamProtocolError("DeepSeek vision upload response did not contain a file ID")
        await self.wait_for_uploaded_file(file_id, access_token)
        return file_id

    async def wait_for_uploaded_file(self, file_id: str, access_token: str) -> dict[str, Any]:
        for attempt in range(self.settings.file_poll_attempts):
            response = await self._http.get(
                self._api_url(FILE_FETCH_PATH),
                params={"file_ids": file_id},
                headers=self._headers(access_token, cookie=True),
                timeout=self.settings.connect_timeout_seconds,
            )
            if response.status_code != 200:
                raise UpstreamProtocolError(
                    f"Failed to fetch DeepSeek vision file status: HTTP {response.status_code}"
                )
            file = next(
                (item for item in _extract_fetched_files(response) if item["id"] == file_id), None
            )
            if not file:
                raise UpstreamProtocolError("DeepSeek vision file was not found after upload")
            if _is_file_ready(file):
                return file
            status = str(file.get("status") or "").upper()
            audit = str(file.get("audit_result") or file.get("auditResult") or "").lower()
            if status in {"FAILED", "FAIL", "ERROR"}:
                raise UpstreamProtocolError(f"DeepSeek vision processing failed: {status}")
            if audit in {"failed", "fail", "reject", "rejected", "block", "blocked", "not_pass"}:
                raise UpstreamProtocolError(f"DeepSeek vision audit failed: {audit}")
            if attempt + 1 < self.settings.file_poll_attempts:
                await asyncio.sleep(self.settings.file_poll_interval_seconds)
        raise UpstreamProtocolError("DeepSeek vision file processing timed out")

    async def start_completion(
        self,
        *,
        user_token: str,
        prompt: str,
        model_type: str,
        search_enabled: bool,
        thinking_enabled: bool,
        image_urls: list[str],
    ) -> DeepSeekStream:
        access_token = await self.acquire_token(user_token)
        session_id = await self.create_session(access_token)
        try:
            file_ids = await self.upload_images(image_urls, access_token) if image_urls else []
            challenge = await self.get_challenge(access_token, CHAT_COMPLETION_PATH)
            answer = await self._pow.create_answer(challenge, CHAT_COMPLETION_PATH)
            request_context = self._http.stream(
                "POST",
                self._api_url(CHAT_COMPLETION_PATH),
                json={
                    "chat_session_id": session_id,
                    "parent_message_id": None,
                    "prompt": prompt,
                    "model_type": "vision" if file_ids else model_type,
                    "ref_file_ids": file_ids,
                    "search_enabled": search_enabled,
                    "thinking_enabled": thinking_enabled,
                    "preempt": False,
                },
                headers=self._headers(
                    access_token,
                    cookie=True,
                    referer=f"https://chat.deepseek.com/a/chat/s/{session_id}",
                    extra={"X-Ds-Pow-Response": answer},
                ),
                timeout=self.settings.request_timeout_seconds,
            )
            response = await request_context.__aenter__()
            if response.status_code in {401, 403}:
                await request_context.__aexit__(None, None, None)
                raise AuthenticationError()
            if response.status_code >= 400:
                body = (await response.aread())[:2048].decode(errors="replace")
                await request_context.__aexit__(None, None, None)
                raise UpstreamProtocolError(
                    f"DeepSeek completion failed: HTTP {response.status_code} {body}".strip(),
                    status_code=502 if response.status_code >= 500 else response.status_code,
                )
            return DeepSeekStream(request_context, response, self, session_id, user_token)
        except Exception:
            if self.settings.delete_sessions:
                await self.delete_session(session_id, user_token)
            raise

    def _api_url(self, target_path: str) -> str:
        if target_path.startswith("/api"):
            target_path = target_path[4:]
        return f"{self.settings.deepseek_api_base}{target_path}"

    @staticmethod
    def _headers(
        token: str,
        *,
        cookie: bool = False,
        referer: str | None = None,
        extra: dict[str, str] | None = None,
    ) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {token}", **FAKE_HEADERS}
        if cookie:
            headers["Cookie"] = _generate_cookie()
        if referer:
            headers["Referer"] = referer
        if extra:
            headers.update(extra)
        return headers


def _generate_cookie() -> str:
    timestamp_ms = int(time.time() * 1000)
    timestamp = timestamp_ms // 1000
    return (
        f"intercom-HWWAFSESTIME={timestamp_ms}; "
        f"HWWAFSESID={secrets.token_hex(9)}; "
        f"Hm_lvt_{uuid4()}={timestamp},{timestamp},{timestamp}; "
        f"Hm_lpvt_{uuid4()}={timestamp}; _frid={uuid4()}; "
        f"_fr_ssid={uuid4()}; _fr_pvid={uuid4()}"
    )


def _json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except json.JSONDecodeError:
        return {}


def _biz_data(response: httpx.Response) -> Any:
    payload = _json(response)
    if not isinstance(payload, dict):
        return {}
    data = payload.get("data")
    if isinstance(data, dict) and "biz_data" in data:
        return data["biz_data"]
    return payload.get("biz_data", data if data is not None else payload)


def _extract_uploaded_file(response: httpx.Response) -> dict[str, Any] | None:
    biz_data = _biz_data(response)
    if not isinstance(biz_data, dict):
        return None
    candidates: list[Any] = [
        biz_data.get("file"),
        biz_data.get("file_info"),
        biz_data.get("fileInfo"),
        biz_data,
    ]
    for key in ("files", "file_infos", "fileInfos"):
        value = biz_data.get(key)
        if isinstance(value, list):
            candidates.extend(value)
    return next(
        (
            candidate
            for candidate in candidates
            if isinstance(candidate, dict) and isinstance(candidate.get("id"), str)
        ),
        None,
    )


def _extract_fetched_files(response: httpx.Response) -> list[dict[str, Any]]:
    biz_data = _biz_data(response)
    if isinstance(biz_data, dict) and isinstance(biz_data.get("id"), str):
        return [biz_data]
    if isinstance(biz_data, dict):
        files: Any = next(
            (biz_data[key] for key in ("files", "file_infos", "fileInfos") if key in biz_data),
            biz_data,
        )
    else:
        files = biz_data
    if isinstance(files, list):
        return [
            item for item in files if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]
    if isinstance(files, dict):
        return [
            item
            for item in files.values()
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]
    return []


def _is_file_ready(file: dict[str, Any]) -> bool:
    status = str(file.get("status") or "").upper()
    audit = str(file.get("audit_result") or file.get("auditResult") or "").lower()
    model_kind = str(file.get("model_kind") or file.get("modelKind") or "").upper()
    return (
        status == "SUCCESS"
        and (not audit or audit == "pass")
        and (not model_kind or model_kind == "VISION")
        and file.get("is_image") is not False
    )
