from __future__ import annotations

import re
from typing import Any

from .deepseek_v4 import format_tool_calls_as_dsml, render_tools_prompt
from .schemas import ChatMessage, TextContentPart, ToolCall

_MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[.+\]\(.+\)")


def messages_to_prompt(
    messages: list[ChatMessage],
    tools: list[dict[str, Any]] | None = None,
) -> str:
    processed = _messages_with_tool_results(messages)
    if tools:
        tool_prompt = render_tools_prompt(tools)
        if processed and processed[0][0] == "system":
            role, text = processed[0]
            processed[0] = (role, f"{tool_prompt}\n\n{text}" if text else tool_prompt)
        else:
            processed.insert(0, ("system", tool_prompt))

    merged: list[tuple[str, str]] = []
    for role, text in processed:
        if merged and merged[-1][0] == role:
            previous_role, previous_text = merged[-1]
            merged[-1] = (previous_role, f"{previous_text}\n\n{text}")
        else:
            merged.append((role, text))

    blocks: list[str] = []
    for index, (role, text) in enumerate(merged):
        if role == "assistant":
            blocks.append(f"<｜Assistant｜>{text}<｜end of sentence｜>")
        elif role in {"user", "system"}:
            blocks.append(f"<｜User｜>{text}" if index > 0 else text)
        else:
            blocks.append(text)
    return _MARKDOWN_IMAGE_PATTERN.sub("", "".join(blocks))


def inject_tools_prompt(
    messages: list[ChatMessage],
    tools: list[dict[str, Any]] | None,
) -> list[ChatMessage]:
    """Compatibility shim; native DSML tools are rendered by ``messages_to_prompt``."""

    return messages


def extract_image_urls(messages: list[ChatMessage]) -> list[str]:
    from .schemas import ImageContentPart

    return [
        part.url
        for message in messages
        if isinstance(message.content, list)
        for part in message.content
        if isinstance(part, ImageContentPart)
    ]


def _messages_with_tool_results(messages: list[ChatMessage]) -> list[tuple[str, str]]:
    processed: list[tuple[str, str]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role == "assistant":
            processed.append(("assistant", _assistant_text(message)))
            index += 1
            continue
        if message.role == "tool":
            tool_messages: list[ChatMessage] = []
            while index < len(messages) and messages[index].role == "tool":
                tool_messages.append(messages[index])
                index += 1
            tool_messages = _sort_tool_messages(tool_messages, _last_assistant_tool_calls(messages, index))
            for tool_message in tool_messages:
                processed.append(("user", f"<tool_result>{_message_content_text(tool_message)}</tool_result>"))
            continue
        processed.append((message.role, _message_content_text(message)))
        index += 1
    return processed


def _assistant_text(message: ChatMessage) -> str:
    if message.tool_calls:
        parts = []
        reasoning = message.reasoning_content
        if reasoning:
            parts.append(reasoning)
        content = _message_content_text(message)
        if content:
            parts.append(content)
        parts.append(format_tool_calls_as_dsml(message.tool_calls))
        return "\n".join(part for part in parts if part)
    return _message_content_text(message)


def _message_content_text(message: ChatMessage) -> str:
    if isinstance(message.content, list):
        return "\n".join(
            part.text for part in message.content if isinstance(part, TextContentPart)
        )
    return str(message.content or "")


def _last_assistant_tool_calls(
    messages: list[ChatMessage],
    before_index: int,
) -> list[ToolCall] | None:
    cursor = before_index - 1
    while cursor >= 0:
        candidate = messages[cursor]
        if candidate.role == "assistant":
            return candidate.tool_calls
        if candidate.role not in {"tool"}:
            return None
        cursor -= 1
    return None


def _sort_tool_messages(
    messages: list[ChatMessage],
    tool_calls: list[ToolCall] | None,
) -> list[ChatMessage]:
    if not tool_calls:
        return messages
    order = {call.id: index for index, call in enumerate(tool_calls)}
    return sorted(messages, key=lambda message: order.get(message.tool_call_id or "", len(order)))
