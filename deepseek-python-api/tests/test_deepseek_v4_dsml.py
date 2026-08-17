from __future__ import annotations

import json

import pytest

from deepseek_python_api.deepseek_v4 import (
    DSMLParseError,
    format_tool_calls_as_dsml,
    parse_completion_text,
    render_tools_prompt,
)
from deepseek_python_api.deepseek_v4.dsml import parsed_tool_calls_to_openai
from deepseek_python_api.schemas import ToolCall, ToolCallFunction


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "count": {"type": "integer"},
                    "active": {"type": "boolean"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
]


def test_render_tools_prompt_uses_native_dsml_marker() -> None:
    prompt = render_tools_prompt(TOOLS)

    assert "<｜DSML｜tool_calls>" in prompt
    assert '<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">' in prompt
    assert "lookup" in prompt
    assert "<invoke>" not in prompt


def test_format_historical_tool_calls_as_dsml_with_typed_parameters() -> None:
    rendered = format_tool_calls_as_dsml(
        [
            ToolCall(
                id="call-1",
                type="function",
                function=ToolCallFunction(
                    name="lookup",
                    arguments=json.dumps(
                        {
                            "query": "weather",
                            "count": 3,
                            "active": True,
                            "tags": ["sf", "forecast"],
                        }
                    ),
                ),
            )
        ]
    )

    assert '<｜DSML｜invoke name="lookup">' in rendered
    assert '<｜DSML｜parameter name="query" string="true">weather</｜DSML｜parameter>' in rendered
    assert '<｜DSML｜parameter name="count" string="false">3</｜DSML｜parameter>' in rendered
    assert '<｜DSML｜parameter name="active" string="false">true</｜DSML｜parameter>' in rendered
    assert '<｜DSML｜parameter name="tags" string="false">["sf","forecast"]</｜DSML｜parameter>' in rendered


def test_parse_completion_text_parallel_dsml_tool_calls() -> None:
    parsed = parse_completion_text(
        "Need tools\n"
        "<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="lookup">\n'
        '<｜DSML｜parameter name="query" string="true">weather &amp; alerts</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="count" string="false">3</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        '<｜DSML｜invoke name="lookup">\n'
        '<｜DSML｜parameter name="query" string="true">news</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="active" string="false">true</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>",
        tools=TOOLS,
    )

    assert parsed.content == "Need tools"
    assert [call.name for call in parsed.tool_calls] == ["lookup", "lookup"]
    assert parsed.tool_calls[0].arguments == {"query": "weather & alerts", "count": 3}
    assert parsed.tool_calls[1].arguments == {"query": "news", "active": True}

    openai = parsed_tool_calls_to_openai(parsed.tool_calls)
    assert openai[0]["id"].startswith("call_")
    assert openai[0]["type"] == "function"
    assert json.loads(openai[0]["function"]["arguments"]) == {
        "query": "weather & alerts",
        "count": 3,
    }


def test_parse_completion_text_schema_aware_arguments_wrapper_recovery() -> None:
    parsed = parse_completion_text(
        "<｜DSML｜tool_calls>"
        '<｜DSML｜invoke name="lookup">'
        '<｜DSML｜parameter name="arguments" string="false">{"query":"wrapped"}</｜DSML｜parameter>'
        "</｜DSML｜invoke>"
        "</｜DSML｜tool_calls>",
        tools=TOOLS,
    )

    assert parsed.tool_calls[0].arguments == {"query": "wrapped"}


def test_parse_completion_text_rejects_malformed_native_dsml() -> None:
    with pytest.raises(DSMLParseError):
        parse_completion_text(
            "<｜DSML｜tool_calls>"
            '<｜DSML｜invoke name="lookup">'
            '<｜DSML｜parameter name="query" string="true">missing close'
            "</｜DSML｜invoke>",
            tools=TOOLS,
        )


def test_parse_completion_text_requires_requested_tool_name() -> None:
    with pytest.raises(DSMLParseError):
        parse_completion_text(
            "<｜DSML｜tool_calls>"
            '<｜DSML｜invoke name="unknown">'
            '<｜DSML｜parameter name="query" string="true">x</｜DSML｜parameter>'
            "</｜DSML｜invoke>"
            "</｜DSML｜tool_calls>",
            tools=TOOLS,
        )


def test_parse_completion_text_accepts_missing_outer_wrapper_recovery() -> None:
    parsed = parse_completion_text(
        '<｜DSML｜invoke name="lookup">'
        '<｜DSML｜parameter name="query" string="true">x</｜DSML｜parameter>'
        "</｜DSML｜invoke>",
        tools=TOOLS,
    )

    assert parsed.recovered is True
    assert parsed.tool_calls[0].arguments == {"query": "x"}
