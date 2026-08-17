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


def test_parse_completion_text_accepts_dsh_wrapper_dialect() -> None:
    parsed = parse_completion_text(
        "I will read the plan first.\n"
        "<DSML｜tool_calls>\n"
        '<invoke name="lookup">\n'
        '<parameter name="query" string="true">task_plan.md</parameter>\n'
        "</invoke>\n"
        '<invoke name="lookup">\n'
        '<parameter name="query" string="true">findings.md</parameter>\n'
        "</invoke>\n"
        "</DSML｜tool_calls>",
        tools=TOOLS,
    )

    assert parsed.recovered is True
    assert parsed.content == "I will read the plan first."
    assert [call.arguments for call in parsed.tool_calls] == [
        {"query": "task_plan.md"},
        {"query": "findings.md"},
    ]


def test_parse_completion_text_rejects_malformed_dsh_wrapper_dialect() -> None:
    with pytest.raises(DSMLParseError):
        parse_completion_text(
            "<DSML｜tool_calls>"
            '<invoke name="lookup">'
            '<parameter name="query" string="true">missing close</invoke>'
            "</DSML｜tool_calls>",
            tools=TOOLS,
        )


def test_parse_completion_text_accepts_dsh_dagger_dialect() -> None:
    parsed = parse_completion_text(
        "I'll start by restoring context.\n"
        "<DSML‡tool_calls>\n"
        '<DSML‡invoke name="lookup">\n'
        '<DSML‡parameter name="query" string="true">task_plan.md</DSML‡parameter>\n'
        "</DSML‡invoke>\n"
        '<DSML‡invoke name="lookup">\n'
        '<DSML‡parameter name="query" string="true">findings.md</DSML‡parameter>\n'
        "</DSML‡invoke>\n"
        "</DSML‡tool_calls>",
        tools=TOOLS,
    )

    assert parsed.recovered is True
    assert parsed.content == "I'll start by restoring context."
    assert [call.arguments for call in parsed.tool_calls] == [
        {"query": "task_plan.md"},
        {"query": "findings.md"},
    ]


def test_parse_completion_text_rejects_malformed_dsh_dagger_dialect() -> None:
    with pytest.raises(DSMLParseError):
        parse_completion_text(
            "<DSML‡tool_calls>"
            '<DSML‡invoke name="lookup">'
            '<DSML‡parameter name="query" string="true">missing close</DSML‡invoke>'
            "</DSML‡tool_calls>",
            tools=TOOLS,
        )


def test_parse_completion_text_accepts_structural_dsml_marker_variants() -> None:
    parsed = parse_completion_text(
        "<DSML:tool_calls>"
        '<DSML:invoke name="lookup">'
        '<DSML:parameter name="query" string="true">colon</DSML:parameter>'
        '<DSML:parameter name="count" string="false">2</DSML:parameter>'
        "</DSML:invoke>"
        '<DSML:invoke name="lookup">'
        '<parameter name="query" string="true">plain child</parameter>'
        "</DSML:invoke>"
        "</DSML:tool_calls>",
        tools=TOOLS,
    )

    assert parsed.recovered is True
    assert [call.arguments for call in parsed.tool_calls] == [
        {"query": "colon", "count": 2},
        {"query": "plain child"},
    ]


def test_parse_completion_text_accepts_unseen_unicode_symbol_marker() -> None:
    parsed = parse_completion_text(
        "<DSML§tool_calls>"
        '<DSML§invoke name="lookup">'
        '<DSML§parameter name="query" string="true">section</DSML§parameter>'
        "</DSML§invoke>"
        "</DSML§tool_calls>",
        tools=TOOLS,
    )

    assert parsed.recovered is True
    assert parsed.tool_calls[0].arguments == {"query": "section"}


def test_parse_completion_text_rejects_mismatched_marker_variants() -> None:
    with pytest.raises(DSMLParseError):
        parse_completion_text(
            "<DSML:tool_calls>"
            '<DSML:invoke name="lookup">'
            '<DSML:parameter name="query" string="true">x</DSML:parameter>'
            "</DSML‡invoke>"
            "</DSML:tool_calls>",
            tools=TOOLS,
        )


def test_parse_completion_text_rejects_arbitrary_unmarked_xml() -> None:
    parsed = parse_completion_text(
        "<tool_calls>"
        '<invoke name="lookup">'
        '<parameter name="query" string="true">x</parameter>'
        "</invoke>"
        "</tool_calls>",
        tools=TOOLS,
    )

    assert parsed.tool_calls == []
    assert parsed.content.startswith("<tool_calls>")


def test_parse_completion_text_rejects_dsml_tag_with_extra_letters() -> None:
    with pytest.raises(DSMLParseError):
        parse_completion_text(
            "<xDSMLy_tool_calls>"
            '<xDSMLy_invoke name="lookup">'
            '<xDSMLy_parameter name="query" string="true">x</xDSMLy_parameter>'
            "</xDSMLy_invoke>"
            "</xDSMLy_tool_calls>",
            tools=TOOLS,
        )


def test_parse_completion_text_accepts_dsh_tool_details_markup() -> None:
    parsed = parse_completion_text(
        "<formal notice)>Let me read the planning files.\n"
        "<thought>This should not leak downstream.</thought>\n"
        "Context injectionplanning-with-files\n"
        "I'll resume this work.\n"
        "<details> <summary>🔧 Tool Calls</summary>"
        ' <invoke name="lookup">'
        ' <parameter name="query">task_plan.md</parameter>'
        " </invoke>"
        ' <invoke name="lookup">'
        ' <parameter name="query">findings.md</parameter>'
        " </invoke>"
        " </details>",
        tools=TOOLS,
    )

    assert parsed.recovered is True
    assert parsed.content == "I'll resume this work."
    assert [call.arguments for call in parsed.tool_calls] == [
        {"query": "task_plan.md"},
        {"query": "findings.md"},
    ]


def test_parse_completion_text_accepts_dsh_bash_details_markup() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    parsed = parse_completion_text(
        "<details> <summary>🔧 Tool Calls</summary>"
        ' <invoke name="bash">'
        ' <parameter name="command">'
        "cd /home/pet/Projects/zcode-api && cat task_plan.md 2>/dev/null; "
        'echo "---FINDINGS---"; cat findings.md 2>/dev/null; '
        'echo "---PROGRESS---"; cat progress.md 2>/dev/null'
        "</parameter>"
        ' <parameter name="description">Read planning files</parameter>'
        " </invoke>"
        ' <invoke name="bash">'
        ' <parameter name="command">'
        "cd /home/pet/Projects/zcode-api && git diff --stat 2>/dev/null; "
        'echo "---STATUS---"; git status --short 2>/dev/null; '
        'echo "---WC---"; wc -l src/proxy/captcha.ts'
        "</parameter>"
        ' <parameter name="description">Check git state and captcha file size</parameter>'
        " </invoke>"
        ' <invoke name="bash">'
        ' <parameter name="command">'
        "cd /home/pet/Projects/zcode-api && pwd && "
        "ls -la zcode-proxy 2>/dev/null; "
        'echo "---CONFIG---"; cat config.sandbox.yaml 2>/dev/null'
        "</parameter>"
        ' <parameter name="description">Check working dir, binary, config</parameter>'
        " </invoke>"
        " </details>",
        tools=tools,
    )

    assert parsed.recovered is True
    assert parsed.content == ""
    assert [call.name for call in parsed.tool_calls] == ["bash", "bash", "bash"]
    assert [call.arguments["description"] for call in parsed.tool_calls] == [
        "Read planning files",
        "Check git state and captcha file size",
        "Check working dir, binary, config",
    ]
    assert parsed.tool_calls[0].arguments["command"].startswith(
        "cd /home/pet/Projects/zcode-api && cat task_plan.md"
    )
    assert "wc -l src/proxy/captcha.ts" in parsed.tool_calls[1].arguments["command"]
    assert "cat config.sandbox.yaml" in parsed.tool_calls[2].arguments["command"]


def test_parse_completion_text_rejects_malformed_dsh_tool_details_markup() -> None:
    with pytest.raises(DSMLParseError):
        parse_completion_text(
            "<details><summary>🔧 Tool Calls</summary>"
            '<invoke name="lookup">'
            '<parameter name="query">missing close</invoke>'
            "</details>",
            tools=TOOLS,
        )


def test_parse_completion_text_rejects_truncated_dsh_tool_details_markup() -> None:
    with pytest.raises(DSMLParseError):
        parse_completion_text(
            "<details><summary>🔧 Tool Calls</summary>"
            '<invoke name="lookup">'
            '<parameter name="query">missing close</parameter>'
            "</invoke>",
            tools=TOOLS,
        )


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
