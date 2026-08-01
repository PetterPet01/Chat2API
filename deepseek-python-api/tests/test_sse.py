from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from deepseek_python_api.sse import DeepSeekEventState, iter_sse_data, openai_chunk, parse_event


async def chunks(*values: bytes) -> AsyncIterator[bytes]:
    for value in values:
        yield value


@pytest.mark.asyncio
async def test_sse_parser_handles_boundaries_crlf_and_multiline() -> None:
    result = [
        item
        async for item in iter_sse_data(
            chunks(b": comment\r\nda", b'ta: {"a":1}\r\ndata: second\r\n\r\ndata: [DONE]\n\n')
        )
    ]
    assert result == ['{"a":1}\nsecond', "[DONE]"]


def test_event_state_splits_thinking_answer_and_citations() -> None:
    state = DeepSeekEventState(
        semantic_model="deepseek-v4-flash-think-search",
        web_search_enabled=True,
        reasoning_effort="medium",
    )
    deltas = state.process(
        {
            "response_message_id": "message-id",
            "v": {
                "response": {
                    "thinking_enabled": True,
                    "fragments": [
                        {"type": "THINK", "content": "SEARCH reason FINISHED"},
                        {
                            "type": "ANSWER",
                            "content": "answer[citation:1]",
                            "results": [
                                {
                                    "url": "https://example.com",
                                    "title": "Example",
                                    "cite_index": 1,
                                }
                            ],
                        },
                    ],
                }
            },
        }
    )
    assert deltas[0].reasoning_content == "reason "
    assert deltas[1].content == "answer[1]"
    assert state.citations() == "[1]: [Example](https://example.com)"
    assert state.message_id == "message-id"


def test_search_result_batch_sets_citation_index() -> None:
    state = DeepSeekEventState("deepseek-v4-flash-search", web_search_enabled=True)
    state.process(
        {
            "p": "response/search_results",
            "v": [{"url": "https://example.com", "title": "Example"}],
        }
    )
    state.process(
        {
            "p": "response/search_results",
            "o": "BATCH",
            "v": [{"p": "0/cite_index", "v": 3}],
        }
    )
    assert state.citations() == "[3]: [Example](https://example.com)"


def test_parse_event_and_openai_chunk() -> None:
    assert parse_event("not-json") == {}
    assert parse_event("[DONE]") is None
    chunk = openai_chunk(
        completion_id="chatcmpl-id",
        model="deepseek-v4-flash",
        created=1,
        delta={"content": "hello"},
    )
    payload = json.loads(chunk.removeprefix("data: ").strip())
    assert payload["choices"][0]["delta"]["content"] == "hello"
