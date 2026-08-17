"""DeepSeek V4 DSML prompt and tool-call helpers."""

from .dsml import (
    DSMLParseError,
    ParsedDSMLCompletion,
    ParsedDSMLToolCall,
    format_tool_calls_as_dsml,
    parse_completion_text,
    render_tools_prompt,
)

__all__ = [
    "DSMLParseError",
    "ParsedDSMLCompletion",
    "ParsedDSMLToolCall",
    "format_tool_calls_as_dsml",
    "parse_completion_text",
    "render_tools_prompt",
]
