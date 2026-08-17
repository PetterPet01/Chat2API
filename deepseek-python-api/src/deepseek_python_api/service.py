from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from .deepseek_v4.dsml import DSMLParseError, parsed_tool_calls_to_openai, parse_completion_text
from .errors import UpstreamProtocolError
from .models import resolve_chat_options
from .prompt import extract_image_urls, messages_to_prompt
from .schemas import ChatCompletionRequest
from .sse import (
    DeepSeekEventState,
    completion_created_at,
    iter_sse_data,
    openai_chunk,
    parse_event,
)
from .token_manager import TokenLease
from .upstream import DeepSeekClient, DeepSeekStream

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class CompletionContext:
    completion_id: str
    created: int
    model: str
    state: DeepSeekEventState
    upstream: DeepSeekStream
    tools: list[dict[str, Any]] | None = None
    lease: TokenLease | None = None


class CompletionService:
    def __init__(self, client: DeepSeekClient) -> None:
        self._client = client

    async def start(
        self,
        request: ChatCompletionRequest,
        user_token: str | None = None,
        token_lease: TokenLease | None = None,
    ) -> CompletionContext:
        options = resolve_chat_options(request.model, request.web_search, request.reasoning_effort)
        image_urls = extract_image_urls(request.messages)
        if token_lease is None:
            if user_token is None:
                raise ValueError("user_token or token_lease is required")
            token_lease = TokenLease(None, user_token, None)
        try:
            upstream = await self._client.start_completion(
                user_token=token_lease.token,
                prompt=messages_to_prompt(request.messages, request.tools),
                model_type=options.model_type,
                search_enabled=options.search_enabled,
                thinking_enabled=options.thinking_enabled,
                image_urls=image_urls,
                proxy_url=token_lease.proxy_url,
            )
        except asyncio.CancelledError as exc:
            await token_lease.release(success=False, error=exc)
            raise
        except Exception as exc:
            await token_lease.release(success=False, error=exc)
            raise
        return CompletionContext(
            completion_id=f"chatcmpl-{uuid4().hex}",
            created=completion_created_at(),
            model=request.model,
            state=DeepSeekEventState(
                semantic_model=request.model,
                web_search_enabled=options.search_enabled,
                reasoning_effort=request.reasoning_effort,
            ),
            upstream=upstream,
            tools=request.tools,
            lease=token_lease,
        )

    async def stream_openai(self, context: CompletionContext) -> AsyncIterator[str]:
        role_sent = False
        try:
            async for data in iter_sse_data(context.upstream.iter_bytes()):
                if data == "[DONE]":
                    break
                event = parse_event(data)
                if not event:
                    continue
                for parsed in context.state.process(event):
                    delta: dict[str, Any] = {}
                    if not role_sent:
                        delta["role"] = "assistant"
                        role_sent = True
                    if parsed.content is not None:
                        delta["content"] = parsed.content
                    if parsed.reasoning_content is not None:
                        delta["reasoning_content"] = parsed.reasoning_content
                    if len(delta) > (1 if "role" in delta else 0):
                        yield openai_chunk(
                            completion_id=context.completion_id,
                            model=context.model,
                            created=context.created,
                            delta=delta,
                        )
            citations = context.state.citations()
            if citations:
                yield openai_chunk(
                    completion_id=context.completion_id,
                    model=context.model,
                    created=context.created,
                    delta={"content": f"\n\n{citations}"},
                )
            if not role_sent:
                yield openai_chunk(
                    completion_id=context.completion_id,
                    model=context.model,
                    created=context.created,
                    delta={"role": "assistant"},
                )
            yield openai_chunk(
                completion_id=context.completion_id,
                model=context.model,
                created=context.created,
                delta={},
                finish_reason="stop",
            )
            yield "data: [DONE]\n\n"
        except asyncio.CancelledError as exc:
            if context.lease:
                await context.lease.release(success=False, error=exc)
            raise
        except Exception as exc:
            if context.lease:
                await context.lease.release(success=False, error=exc)
            raise
        else:
            if context.lease:
                await context.lease.release(success=True)
        finally:
            await context.upstream.close()

    async def collect_openai(self, context: CompletionContext) -> dict[str, Any]:
        content: list[str] = []
        reasoning: list[str] = []
        try:
            async for data in iter_sse_data(context.upstream.iter_bytes()):
                if data == "[DONE]":
                    break
                event = parse_event(data)
                if not event:
                    continue
                for parsed in context.state.process(event):
                    if parsed.content is not None:
                        content.append(parsed.content)
                    if parsed.reasoning_content is not None:
                        reasoning.append(parsed.reasoning_content)
        except asyncio.CancelledError as exc:
            if context.lease:
                await context.lease.release(success=False, error=exc)
            raise
        except Exception as exc:
            if context.lease:
                await context.lease.release(success=False, error=exc)
            raise
        finally:
            await context.upstream.close()

        citations = context.state.citations()
        answer = "".join(content).strip()
        if citations:
            answer = f"{answer}\n\n{citations}" if answer else citations

        reasoning_content = "".join(reasoning).strip()
        try:
            parsed_completion = parse_completion_text(answer, tools=context.tools)
        except DSMLParseError as exc:
            if context.lease:
                await context.lease.release(success=False, error=exc)
            raise UpstreamProtocolError(str(exc), code="malformed_deepseek_dsml") from exc

        message: dict[str, Any] = {"role": "assistant"}
        finish_reason = "stop"
        if parsed_completion.tool_calls:
            message["tool_calls"] = parsed_tool_calls_to_openai(parsed_completion.tool_calls)
            message["content"] = parsed_completion.content or None
            finish_reason = "tool_calls"
        else:
            if not parsed_completion.content and not reasoning_content:
                empty_error = UpstreamProtocolError(
                    "model returned a completed response with no content",
                    code="empty_completion",
                )
                if context.lease:
                    await context.lease.release(success=False, error=empty_error)
                raise empty_error
            message["content"] = parsed_completion.content
        if reasoning_content:
            message["reasoning_content"] = reasoning_content

        result: dict[str, Any] = {
            "id": context.completion_id,
            "model": context.model,
            "object": "chat.completion",
            "created": context.created,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        }
        if context.state.accumulated_token_usage is not None:
            result["usage"] = {"total_tokens": context.state.accumulated_token_usage}
        if context.lease:
            await context.lease.release(success=True)
        return result
