from __future__ import annotations

import re

from .schemas import ChatMessage, TextContentPart, tool_calls_to_prompt

_MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[.+\]\(.+\)")


def messages_to_prompt(messages: list[ChatMessage]) -> str:
    processed: list[tuple[str, str]] = []
    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            text = tool_calls_to_prompt(message.tool_calls)
        elif message.role == "tool" and message.tool_call_id:
            text = (
                f'<tool_result tool_call_id="{message.tool_call_id}">'
                f"{message.content or ''}</tool_result>"
            )
        elif isinstance(message.content, list):
            text = "\n".join(
                part.text for part in message.content if isinstance(part, TextContentPart)
            )
        else:
            text = str(message.content or "")
        processed.append((message.role, text))

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
        elif role == "tool":
            blocks.append(f"<｜User｜>{text}")
        else:
            blocks.append(text)
    return _MARKDOWN_IMAGE_PATTERN.sub("", "".join(blocks))


def extract_image_urls(messages: list[ChatMessage]) -> list[str]:
    from .schemas import ImageContentPart

    return [
        part.url
        for message in messages
        if isinstance(message.content, list)
        for part in message.content
        if isinstance(part, ImageContentPart)
    ]
