from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import cast

import pytest

from deepseek_python_api.errors import UpstreamProtocolError
from deepseek_python_api.schemas import ChatCompletionRequest
from deepseek_python_api.service import CompletionContext, CompletionService
from deepseek_python_api.sse import DeepSeekEventState, completion_created_at
from deepseek_python_api.token_manager import TokenLease
from deepseek_python_api.upstream import DeepSeekClient, DeepSeekStream


class FakeStream:
    def __init__(
        self,
        chunks: list[bytes] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.chunks = chunks or []
        self.error = error
        self.closed = False

    async def iter_bytes(self) -> AsyncIterator[bytes]:
        if self.error:
            raise self.error
        for chunk in self.chunks:
            yield chunk

    async def close(self) -> None:
        self.closed = True


class FakeClient:
    def __init__(
        self,
        stream: FakeStream | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.stream = stream or FakeStream()
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def start_completion(
        self,
        *,
        user_token: str,
        prompt: str,
        model_type: str,
        search_enabled: bool,
        thinking_enabled: bool,
        image_urls: list[str],
        proxy_url: str | None = None,
    ) -> DeepSeekStream:
        self.calls.append(
            {
                "user_token": user_token,
                "prompt": prompt,
                "model_type": model_type,
                "search_enabled": search_enabled,
                "thinking_enabled": thinking_enabled,
                "image_urls": image_urls,
                "proxy_url": proxy_url,
            }
        )
        if self.error:
            raise self.error
        return cast("DeepSeekStream", self.stream)


def request(stream: bool = False) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {
            "model": "deepseek-v4-flash",
            "stream": stream,
            "messages": [{"role": "user", "content": "hello"}],
        }
    )


@pytest.mark.asyncio
async def test_start_accepts_legacy_user_token_and_releases_on_failure() -> None:
    stream = FakeStream([b"data: [DONE]\n\n"])
    client = FakeClient(stream=stream)
    service = CompletionService(cast("DeepSeekClient", client))

    context = await service.start(request(), user_token="legacy-token")

    assert client.calls[0]["user_token"] == "legacy-token"
    assert context.lease is not None
    assert context.lease.token == "legacy-token"

    lease = TokenLease("managed", "bad-token", None)
    failing = CompletionService(
        cast("DeepSeekClient", FakeClient(error=UpstreamProtocolError("upstream failed")))
    )
    with pytest.raises(UpstreamProtocolError):
        await failing.start(request(), token_lease=lease)
    assert lease._released is True

    cancel_lease = TokenLease("managed", "cancel-token", None)
    cancelling = CompletionService(
        cast("DeepSeekClient", FakeClient(error=asyncio.CancelledError()))
    )
    with pytest.raises(asyncio.CancelledError):
        await cancelling.start(request(), token_lease=cancel_lease)
    assert cancel_lease._released is True

    with pytest.raises(ValueError, match="user_token or token_lease"):
        await service.start(request())


@pytest.mark.asyncio
async def test_stream_openai_releases_success_and_handles_empty_stream() -> None:
    upstream = FakeStream([b"data: [DONE]\n\n"])
    lease = TokenLease(None, "token", None)
    context = CompletionContext(
        completion_id="chatcmpl-test",
        created=completion_created_at(),
        model="deepseek-v4-flash",
        state=DeepSeekEventState("deepseek-v4-flash"),
        upstream=cast("DeepSeekStream", upstream),
        lease=lease,
    )

    chunks = [
        chunk
        async for chunk in CompletionService(cast("DeepSeekClient", FakeClient())).stream_openai(
            context
        )
    ]

    assert any('"role":"assistant"' in chunk for chunk in chunks)
    assert chunks[-1] == "data: [DONE]\n\n"
    assert lease._released is True
    assert upstream.closed is True


@pytest.mark.asyncio
async def test_stream_openai_releases_on_error_and_cancel() -> None:
    upstream = FakeStream(error=RuntimeError("broken stream"))
    lease = TokenLease(None, "token", None)
    context = CompletionContext(
        completion_id="chatcmpl-test",
        created=completion_created_at(),
        model="deepseek-v4-flash",
        state=DeepSeekEventState("deepseek-v4-flash"),
        upstream=cast("DeepSeekStream", upstream),
        lease=lease,
    )

    with pytest.raises(RuntimeError, match="broken stream"):
        _ = [
            chunk
            async for chunk in CompletionService(
                cast("DeepSeekClient", FakeClient())
            ).stream_openai(context)
        ]

    assert lease._released is True
    assert upstream.closed is True

    cancel_stream = FakeStream(error=asyncio.CancelledError())
    cancel_lease = TokenLease(None, "token", None)
    cancel_context = CompletionContext(
        completion_id="chatcmpl-test",
        created=completion_created_at(),
        model="deepseek-v4-flash",
        state=DeepSeekEventState("deepseek-v4-flash"),
        upstream=cast("DeepSeekStream", cancel_stream),
        lease=cancel_lease,
    )

    with pytest.raises(asyncio.CancelledError):
        _ = [
            chunk
            async for chunk in CompletionService(
                cast("DeepSeekClient", FakeClient())
            ).stream_openai(cancel_context)
        ]
    assert cancel_lease._released is True
    assert cancel_stream.closed is True


@pytest.mark.asyncio
async def test_collect_openai_releases_on_error_and_cancel() -> None:
    error_stream = FakeStream(error=RuntimeError("collect failed"))
    error_lease = TokenLease(None, "token", None)
    error_context = CompletionContext(
        completion_id="chatcmpl-test",
        created=completion_created_at(),
        model="deepseek-v4-flash",
        state=DeepSeekEventState("deepseek-v4-flash"),
        upstream=cast("DeepSeekStream", error_stream),
        lease=error_lease,
    )
    service = CompletionService(cast("DeepSeekClient", FakeClient()))

    with pytest.raises(RuntimeError, match="collect failed"):
        await service.collect_openai(error_context)
    assert error_lease._released is True
    assert error_stream.closed is True

    cancel_stream = FakeStream(error=asyncio.CancelledError())
    cancel_lease = TokenLease(None, "token", None)
    cancel_context = CompletionContext(
        completion_id="chatcmpl-test",
        created=completion_created_at(),
        model="deepseek-v4-flash",
        state=DeepSeekEventState("deepseek-v4-flash"),
        upstream=cast("DeepSeekStream", cancel_stream),
        lease=cancel_lease,
    )

    with pytest.raises(asyncio.CancelledError):
        await service.collect_openai(cancel_context)
    assert cancel_lease._released is True
    assert cancel_stream.closed is True
