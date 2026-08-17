from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

DSML = "｜DSML｜"
TOOL_START = f"<{DSML}tool_calls>"
TOOL_END = f"</{DSML}tool_calls>"
INVOKE_END = f"</{DSML}invoke>"
PARAM_END = f"</{DSML}parameter>"

TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question.
You can invoke tools by writing a "<{dsml_token}tool_calls>" block like the following:

<{dsml_token}tool_calls>
<{dsml_token}invoke name="$TOOL_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$TOOL_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}tool_calls>

String parameters should be specified as is and set `string="true"`.
For all other types (numbers, booleans, arrays, objects), pass the value
in JSON format and set `string="false"`.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas
to invoke tool calls.
"""

_TOOL_BLOCK_PATTERN = re.compile(
    rf"<{re.escape(DSML)}tool_calls>\s*(?P<body>.*?)\s*</{re.escape(DSML)}tool_calls>",
    re.DOTALL,
)
_INVOKE_PATTERN = re.compile(
    rf"<{re.escape(DSML)}invoke\s+name=(?P<quote>['\"])(?P<name>.*?)(?P=quote)>\s*"
    rf"(?P<body>.*?)\s*</{re.escape(DSML)}invoke>",
    re.DOTALL,
)
_PARAM_PATTERN = re.compile(
    rf"<{re.escape(DSML)}parameter\s+"
    rf"(?P<attrs>[^>]*)>"
    rf"(?P<value>.*?)"
    rf"</{re.escape(DSML)}parameter>",
    re.DOTALL,
)
_ATTR_PATTERN = re.compile(r"([a-zA-Z_][\w:-]*)\s*=\s*(['\"])(.*?)\2", re.DOTALL)
_LEGACY_WRONG_DSML_PATTERN = re.compile(r"<\s*/?｜｜DSML｜｜(?:tool_calls|invoke|parameter)\b")


class DSMLParseError(ValueError):
    """Raised when a native-looking DeepSeek DSML completion is not classifiable."""


@dataclass(frozen=True, slots=True)
class ParsedDSMLToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ParsedDSMLCompletion:
    content: str
    tool_calls: list[ParsedDSMLToolCall] = field(default_factory=list)
    recovered: bool = False


def render_tools_prompt(tools: list[dict[str, Any]]) -> str:
    tool_schemas = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
    return TOOLS_TEMPLATE.format(dsml_token=DSML, tool_schemas=tool_schemas)


def format_tool_calls_as_dsml(tool_calls: Any) -> str:
    calls = list(tool_calls or [])
    if not calls:
        return ""
    blocks = [TOOL_START]
    for call in calls:
        name = _tool_call_name(call)
        args = _tool_call_arguments(call)
        blocks.append(f'<{DSML}invoke name="{_escape_attr(name)}">')
        for key, value in args.items():
            blocks.append(_format_parameter(str(key), value))
        blocks.append(INVOKE_END)
    blocks.append(TOOL_END)
    return "\n".join(blocks)


def parse_completion_text(
    text: str,
    *,
    tools: list[dict[str, Any]] | None = None,
) -> ParsedDSMLCompletion:
    requested = _requested_tool_schemas(tools)
    blocks = list(_TOOL_BLOCK_PATTERN.finditer(text))
    if not blocks:
        if _looks_like_dsml(text):
            recovered = _parse_recoverable_without_wrapper(text, requested)
            if recovered is not None:
                content = _strip_invokes(text, recovered.names).strip()
                return ParsedDSMLCompletion(content=content, tool_calls=recovered.calls, recovered=True)
            raise DSMLParseError("Malformed DeepSeek DSML tool call block")
        return ParsedDSMLCompletion(content=text.strip(), tool_calls=[])

    tool_calls: list[ParsedDSMLToolCall] = []
    for block in blocks:
        parsed = _parse_tool_block(block.group("body"), requested)
        tool_calls.extend(parsed)

    if not tool_calls:
        raise DSMLParseError("DeepSeek DSML tool call block did not contain any invokes")

    content = text
    for block in reversed(blocks):
        content = content[: block.start()] + content[block.end() :]
    return ParsedDSMLCompletion(content=content.strip(), tool_calls=tool_calls)


def parsed_tool_calls_to_openai(calls: list[ParsedDSMLToolCall]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"call_{uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": json.dumps(call.arguments, ensure_ascii=False, separators=(",", ":")),
            },
        }
        for call in calls
    ]


def _parse_tool_block(body: str, requested: dict[str, dict[str, Any]]) -> list[ParsedDSMLToolCall]:
    calls: list[ParsedDSMLToolCall] = []
    consumed: list[tuple[int, int]] = []
    for match in _INVOKE_PATTERN.finditer(body):
        consumed.append(match.span())
        name = html.unescape(match.group("name"))
        calls.append(_parse_invoke(name, match.group("body"), requested))
    remainder = _remove_spans(body, consumed).strip()
    if remainder:
        raise DSMLParseError("Unexpected text inside DeepSeek DSML tool_calls block")
    return calls


def _parse_invoke(
    name: str,
    body: str,
    requested: dict[str, dict[str, Any]],
) -> ParsedDSMLToolCall:
    if requested and name not in requested:
        raise DSMLParseError(f"Unknown DeepSeek DSML tool name: {name}")
    args: dict[str, Any] = {}
    consumed: list[tuple[int, int]] = []
    for match in _PARAM_PATTERN.finditer(body):
        consumed.append(match.span())
        attrs = _parse_attrs(match.group("attrs"))
        param_name = attrs.get("name")
        if not param_name:
            raise DSMLParseError("DeepSeek DSML parameter is missing a name")
        is_string = attrs.get("string")
        if is_string not in {"true", "false"}:
            raise DSMLParseError("DeepSeek DSML parameter is missing string=\"true|false\"")
        args[param_name] = _decode_parameter(match.group("value"), is_string == "true")
    remainder = _remove_spans(body, consumed).strip()
    if remainder:
        raise DSMLParseError("Unexpected text inside DeepSeek DSML invoke block")
    args = _repair_wrapper_argument(name, args, requested)
    _validate_arguments(name, args, requested)
    return ParsedDSMLToolCall(name=name, arguments=args)


def _decode_parameter(value: str, is_string: bool) -> Any:
    decoded = html.unescape(value)
    if is_string:
        return decoded
    try:
        return json.loads(decoded)
    except (json.JSONDecodeError, ValueError):
        return decoded


def _format_parameter(name: str, value: Any) -> str:
    if isinstance(value, str):
        string_flag = "true"
        encoded = value
    else:
        string_flag = "false"
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return (
        f'<{DSML}parameter name="{_escape_attr(name)}" string="{string_flag}">'
        f"{html.escape(encoded, quote=False)}"
        f"{PARAM_END}"
    )


def _tool_call_name(call: Any) -> str:
    function = getattr(call, "function", None)
    if function is not None:
        return str(getattr(function, "name"))
    if isinstance(call, dict):
        fn = call.get("function")
        if isinstance(fn, dict):
            return str(fn.get("name", ""))
        return str(call.get("name", ""))
    raise TypeError("Unsupported tool call object")


def _tool_call_arguments(call: Any) -> dict[str, Any]:
    function = getattr(call, "function", None)
    raw: Any
    if function is not None:
        raw = getattr(function, "arguments", "{}")
    elif isinstance(call, dict):
        fn = call.get("function")
        raw = fn.get("arguments", "{}") if isinstance(fn, dict) else call.get("arguments", {})
    else:
        raw = {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {"arguments": raw}
    return decoded if isinstance(decoded, dict) else {"arguments": decoded}


def _parse_attrs(attrs: str) -> dict[str, str]:
    return {name: html.unescape(value) for name, _quote, value in _ATTR_PATTERN.findall(attrs)}


def _requested_tool_schemas(tools: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    requested: dict[str, dict[str, Any]] = {}
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str):
            continue
        parameters = function.get("parameters")
        requested[name] = parameters if isinstance(parameters, dict) else {}
    return requested


def _repair_wrapper_argument(
    name: str,
    args: dict[str, Any],
    requested: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if len(args) != 1 or name not in requested:
        return args
    wrapper_name = next(iter(args))
    if wrapper_name not in {"arguments", "input"}:
        return args
    schema = requested[name]
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if isinstance(properties, dict) and wrapper_name in properties:
        return args
    wrapped = args[wrapper_name]
    if not isinstance(wrapped, dict):
        return args
    if isinstance(properties, dict) and not set(wrapped).issubset(set(properties)):
        return args
    return wrapped


def _validate_arguments(
    name: str,
    args: dict[str, Any],
    requested: dict[str, dict[str, Any]],
) -> None:
    if not isinstance(args, dict):
        raise DSMLParseError("DeepSeek DSML tool arguments must decode to a JSON object")
    schema = requested.get(name)
    if not isinstance(schema, dict):
        return
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for key in args:
            if key not in properties and schema.get("additionalProperties") is False:
                raise DSMLParseError(f"Unexpected argument for DeepSeek DSML tool {name}: {key}")
    required = schema.get("required")
    if isinstance(required, list):
        missing = [key for key in required if isinstance(key, str) and key not in args]
        if missing:
            raise DSMLParseError(
                f"Missing required argument for DeepSeek DSML tool {name}: {missing[0]}"
            )


def _looks_like_dsml(text: str) -> bool:
    return f"<{DSML}" in text or f"</{DSML}" in text or _LEGACY_WRONG_DSML_PATTERN.search(text) is not None


@dataclass(frozen=True, slots=True)
class _RecoveredInvokes:
    calls: list[ParsedDSMLToolCall]
    names: list[str]


def _parse_recoverable_without_wrapper(
    text: str,
    requested: dict[str, dict[str, Any]],
) -> _RecoveredInvokes | None:
    normalized = text.replace("｜｜DSML｜｜", DSML)
    if TOOL_START in normalized or TOOL_END in normalized:
        return None
    calls: list[ParsedDSMLToolCall] = []
    names: list[str] = []
    for match in _INVOKE_PATTERN.finditer(normalized):
        name = html.unescape(match.group("name"))
        if requested and name not in requested:
            return None
        calls.append(_parse_invoke(name, match.group("body"), requested))
        names.append(name)
    if not calls:
        return None
    return _RecoveredInvokes(calls=calls, names=names)


def _strip_invokes(text: str, _names: list[str]) -> str:
    normalized = text.replace("｜｜DSML｜｜", DSML)
    return _INVOKE_PATTERN.sub("", normalized)


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    pieces: list[str] = []
    cursor = 0
    for start, end in spans:
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _escape_attr(value: str) -> str:
    return html.escape(value, quote=True)
