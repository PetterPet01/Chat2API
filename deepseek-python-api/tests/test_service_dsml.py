from __future__ import annotations

import json
from typing import cast

import pytest

from test_service import FakeClient, FakeStream
from deepseek_python_api.errors import UpstreamProtocolError
from deepseek_python_api.service import CompletionContext, CompletionService
from deepseek_python_api.sse import DeepSeekEventState, completion_created_at
from deepseek_python_api.token_manager import TokenLease
from deepseek_python_api.upstream import DeepSeekClient, DeepSeekStream


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
]


def context_for_answer(answer: str, *, reasoning: str = "reason ") -> CompletionContext:
    fragments = []
    if reasoning:
        fragments.append({"type": "THINK", "content": reasoning})
    fragments.append({"type": "ANSWER", "content": answer})
    event = (
        "data: "
        + json.dumps(
            {
                "v": {
                    "response": {
                        "thinking_enabled": True,
                        "fragments": fragments,
                    }
                },
                "response_message_id": "m",
            },
            ensure_ascii=False,
        )
        + "\n\ndata: [DONE]\n\n"
    )
    return CompletionContext(
        completion_id="chatcmpl-dsml",
        created=completion_created_at(),
        model="deepseek-v4-flash-think",
        state=DeepSeekEventState("deepseek-v4-flash-think"),
        upstream=cast("DeepSeekStream", FakeStream([event.encode()])),
        tools=TOOLS,
        lease=TokenLease(None, "token", None),
    )


@pytest.mark.asyncio
async def test_collect_openai_converts_dsml_to_openai_tool_calls() -> None:
    result = await CompletionService(cast("DeepSeekClient", FakeClient())).collect_openai(
        context_for_answer(
            "I'll look."
            "<｜DSML｜tool_calls>"
            '<｜DSML｜invoke name="lookup">'
            '<｜DSML｜parameter name="query" string="true">weather</｜DSML｜parameter>'
            '<｜DSML｜parameter name="count" string="false">2</｜DSML｜parameter>'
            "</｜DSML｜invoke>"
            "</｜DSML｜tool_calls>"
        )
    )

    choice = result["choices"][0]
    message = choice["message"]
    assert choice["finish_reason"] == "tool_calls"
    assert message["content"] == "I'll look."
    assert message["reasoning_content"] == "reason"
    assert "｜DSML｜" not in (message["content"] or "")
    assert message["tool_calls"][0]["type"] == "function"
    assert message["tool_calls"][0]["function"]["name"] == "lookup"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "query": "weather",
        "count": 2,
    }


@pytest.mark.asyncio
async def test_collect_openai_rejects_malformed_dsml_and_releases_failure() -> None:
    context = context_for_answer(
        "<｜DSML｜tool_calls>"
        '<｜DSML｜invoke name="lookup">'
        '<｜DSML｜parameter name="query" string="true">weather'
        "</｜DSML｜invoke>"
    )

    with pytest.raises(UpstreamProtocolError, match="Malformed DeepSeek DSML"):
        await CompletionService(cast("DeepSeekClient", FakeClient())).collect_openai(context)

    assert context.lease is not None
    assert context.lease._released is True


@pytest.mark.asyncio
async def test_collect_openai_rejects_completed_empty_response() -> None:
    context = context_for_answer("", reasoning="")

    with pytest.raises(UpstreamProtocolError, match="no content"):
        await CompletionService(cast("DeepSeekClient", FakeClient())).collect_openai(context)

    assert context.lease is not None
    assert context.lease._released is True
