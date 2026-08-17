from __future__ import annotations

import html
import json
import re
import unicodedata
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

_TAG_NAME_PATTERN = re.compile(r"<\s*(/?)\s*([^\s>/]+)([^>]*)>", re.DOTALL)
_ATTR_PATTERN = re.compile(r"([a-zA-Z_][\w:-]*)\s*=\s*(['\"])(.*?)\2", re.DOTALL)
_DSML_LOOKALIKE_PATTERN = re.compile(r"<\s*/?[^\s>]*DSML[^\s>]*")
_DSH_TOOL_DETAILS_PATTERN = re.compile(
    r"<details\b[^>]*>\s*"
    r"<summary\b[^>]*>\s*[^<]*Tool Calls\s*</summary>"
    r"(?P<body>.*?)</details>",
    re.DOTALL | re.IGNORECASE,
)
_DSH_TOOL_DETAILS_START_PATTERN = re.compile(
    r"<details\b[^>]*>\s*<summary\b[^>]*>\s*[^<]*Tool Calls",
    re.DOTALL | re.IGNORECASE,
)
_DSH_THOUGHT_PATTERN = re.compile(r"<thought>.*?</thought>", re.DOTALL | re.IGNORECASE)
_DSH_FORMAL_NOTICE_PATTERN = re.compile(r"(?m)^\s*<formal notice\)>.*(?:\n|$)")
_DSH_CONTEXT_INJECTION_PATTERN = re.compile(r"(?m)^\s*Context injection[^\n]*(?:\n|$)")


@dataclass(frozen=True, slots=True)
class _Element:
    tag: str
    attrs: str
    body: str
    start: int
    end: int


# DeepSeek Harness has emitted several DSML marker spellings in the wild. Treat
# marker glyphs as syntax around the semantic tag name instead of enumerating
# individual dialects: marker code points may be Unicode punctuation or symbols,
# while letters/numbers beyond the exact "DSML" token are rejected.
def _is_marker_char(char: str) -> bool:
    return unicodedata.category(char)[0] in {"P", "S"}


def _split_dsml_tag(tag: str, semantic: str) -> str | None:
    if not tag.endswith(semantic):
        return None
    marker = tag[: -len(semantic)]
    if "DSML" not in marker:
        return None
    parts = marker.split("DSML")
    if any(any(not _is_marker_char(char) for char in part) for part in parts):
        return None
    return marker


def _is_dsml_tag(tag: str, semantic: str) -> bool:
    return _split_dsml_tag(tag, semantic) is not None


def _uses_compat_tag(tag: str) -> bool:
    return tag != f"{DSML}tool_calls"


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
    blocks = _find_elements(text, "tool_calls", allow_plain=False)
    recovered = any(_uses_compat_tag(block.tag) for block in blocks)
    if not blocks:
        dsh_tool_details = _parse_dsh_tool_details(text, requested)
        if dsh_tool_details is not None:
            content = _clean_dsh_harness_content(_remove_spans(text, dsh_tool_details.spans)).strip()
            return ParsedDSMLCompletion(
                content=content,
                tool_calls=dsh_tool_details.calls,
                recovered=True,
            )
        if _looks_like_dsh_tool_details(text):
            raise DSMLParseError("Malformed DeepSeek DSML tool call block")
        if _looks_like_dsml(text):
            if _has_tag_named(text, "tool_calls"):
                raise DSMLParseError("Malformed DeepSeek DSML tool call block")
            recovered_invokes = _parse_recoverable_without_wrapper(text, requested)
            if recovered_invokes is not None:
                content = _remove_spans(text, recovered_invokes.spans).strip()
                return ParsedDSMLCompletion(
                    content=content,
                    tool_calls=recovered_invokes.calls,
                    recovered=True,
                )
            raise DSMLParseError("Malformed DeepSeek DSML tool call block")
        return ParsedDSMLCompletion(content=text.strip(), tool_calls=[])

    tool_calls: list[ParsedDSMLToolCall] = []
    for block in blocks:
        if block.attrs.strip():
            raise DSMLParseError("Unexpected attributes on DeepSeek DSML tool_calls block")
        parsed = _parse_tool_block(block.body, requested)
        tool_calls.extend(parsed)

    if not tool_calls:
        raise DSMLParseError("DeepSeek DSML tool call block did not contain any invokes")

    content = _remove_spans(text, [(block.start, block.end) for block in blocks]).strip()
    return ParsedDSMLCompletion(content=content, tool_calls=tool_calls, recovered=recovered)


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


def _parse_tool_block(
    body: str,
    requested: dict[str, dict[str, Any]],
    *,
    allow_untyped_parameters: bool = False,
) -> list[ParsedDSMLToolCall]:
    calls: list[ParsedDSMLToolCall] = []
    consumed: list[tuple[int, int]] = []
    for invoke in _find_elements(body, "invoke", allow_plain=True):
        consumed.append((invoke.start, invoke.end))
        attrs = _parse_attrs(invoke.attrs)
        name = attrs.get("name")
        if not name:
            raise DSMLParseError("DeepSeek DSML invoke is missing a name")
        calls.append(
            _parse_invoke(
                name,
                invoke.body,
                requested,
                allow_untyped_parameters=allow_untyped_parameters,
            )
        )
    remainder = _remove_spans(body, consumed).strip()
    if remainder:
        raise DSMLParseError("Unexpected text inside DeepSeek DSML tool_calls block")
    return calls


def _parse_invoke(
    name: str,
    body: str,
    requested: dict[str, dict[str, Any]],
    *,
    allow_untyped_parameters: bool = False,
) -> ParsedDSMLToolCall:
    if requested and name not in requested:
        raise DSMLParseError(f"Unknown DeepSeek DSML tool name: {name}")
    args: dict[str, Any] = {}
    consumed: list[tuple[int, int]] = []
    for param in _find_elements(body, "parameter", allow_plain=True):
        consumed.append((param.start, param.end))
        attrs = _parse_attrs(param.attrs)
        param_name = attrs.get("name")
        if not param_name:
            raise DSMLParseError("DeepSeek DSML parameter is missing a name")
        is_string = attrs.get("string")
        if is_string not in {"true", "false"}:
            if not allow_untyped_parameters:
                raise DSMLParseError("DeepSeek DSML parameter is missing string=\"true|false\"")
            is_string = "true"
        args[param_name] = _decode_parameter(param.body, is_string == "true")
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
    return (
        f"<{DSML}" in text
        or f"</{DSML}" in text
        or "<｜｜DSML｜｜" in text
        or _DSML_LOOKALIKE_PATTERN.search(text) is not None
    )


def _looks_like_dsh_tool_details(text: str) -> bool:
    return _DSH_TOOL_DETAILS_START_PATTERN.search(text) is not None


@dataclass(frozen=True, slots=True)
class _RecoveredInvokes:
    calls: list[ParsedDSMLToolCall]
    spans: list[tuple[int, int]]


def _parse_recoverable_without_wrapper(
    text: str,
    requested: dict[str, dict[str, Any]],
) -> _RecoveredInvokes | None:
    calls: list[ParsedDSMLToolCall] = []
    spans: list[tuple[int, int]] = []
    for invoke in _find_elements(text, "invoke", allow_plain=False):
        attrs = _parse_attrs(invoke.attrs)
        name = attrs.get("name")
        if not name or (requested and name not in requested):
            return None
        calls.append(_parse_invoke(name, invoke.body, requested))
        spans.append((invoke.start, invoke.end))
    if not calls:
        return None
    return _RecoveredInvokes(calls=calls, spans=spans)


def _parse_dsh_tool_details(
    text: str,
    requested: dict[str, dict[str, Any]],
) -> _RecoveredInvokes | None:
    matches = list(_DSH_TOOL_DETAILS_PATTERN.finditer(text))
    if not matches:
        return None
    calls: list[ParsedDSMLToolCall] = []
    spans: list[tuple[int, int]] = []
    for match in matches:
        parsed = _parse_tool_block(
            match.group("body"),
            requested,
            allow_untyped_parameters=True,
        )
        calls.extend(parsed)
        spans.append(match.span())
    if not calls:
        raise DSMLParseError("DeepSeek Harness tool-call details block did not contain any invokes")
    return _RecoveredInvokes(calls=calls, spans=spans)


def _clean_dsh_harness_content(text: str) -> str:
    cleaned = _DSH_THOUGHT_PATTERN.sub("", text)
    cleaned = _DSH_FORMAL_NOTICE_PATTERN.sub("", cleaned)
    cleaned = _DSH_CONTEXT_INJECTION_PATTERN.sub("", cleaned)
    return cleaned


def _find_elements(text: str, semantic: str, *, allow_plain: bool) -> list[_Element]:
    elements: list[_Element] = []
    stack: list[tuple[str, str, str, int, int]] = []
    for match in _TAG_NAME_PATTERN.finditer(text):
        closing, tag, attrs = match.groups()
        semantic_match = tag == semantic if allow_plain else False
        if not semantic_match:
            semantic_match = _is_dsml_tag(tag, semantic)
        if not semantic_match:
            continue
        if closing:
            if not stack or stack[-1][0] != tag:
                raise DSMLParseError("Malformed DeepSeek DSML tool call block")
            _open_tag, open_attrs, _open_raw, start, body_start = stack.pop()
            if not stack:
                elements.append(
                    _Element(
                        tag=tag,
                        attrs=open_attrs,
                        body=text[body_start : match.start()],
                        start=start,
                        end=match.end(),
                    )
                )
            continue
        if attrs.rstrip().endswith("/"):
            raise DSMLParseError("Malformed DeepSeek DSML tool call block")
        stack.append((tag, attrs, match.group(0), match.start(), match.end()))
    if stack:
        raise DSMLParseError("Malformed DeepSeek DSML tool call block")
    return elements


def _has_tag_named(text: str, semantic: str) -> bool:
    for match in _TAG_NAME_PATTERN.finditer(text):
        tag = match.group(2)
        if tag == semantic or _is_dsml_tag(tag, semantic):
            return True
    return False


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
