from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

_SEARCH_CONTROL_MARKER = re.compile(r"^(SEARCH|WEB_SEARCH|SEARCHING)(?:\s+|$)", re.IGNORECASE)
_SEARCH_RESULTS_PATH = re.compile(r"^response/fragments/-?\d+/results$")


@dataclass(slots=True)
class ParsedDelta:
    content: str | None = None
    reasoning_content: str | None = None


@dataclass(slots=True)
class DeepSeekEventState:
    semantic_model: str
    web_search_enabled: bool = False
    reasoning_effort: str | None = None
    message_id: str = ""
    current_path: str = ""
    accumulated_token_usage: int | None = None
    search_results: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_thinking_model(self) -> bool:
        model = self.semantic_model.lower()
        return any(alias in model for alias in ("think", "r1", "reasoner")) or bool(
            self.reasoning_effort
        )

    @property
    def is_search_silent(self) -> bool:
        return "search-silent" in self.semantic_model.lower()

    @property
    def should_strip_search_marker(self) -> bool:
        return self.web_search_enabled or "search" in self.semantic_model.lower()

    def process(self, chunk: dict[str, Any]) -> list[ParsedDelta]:
        if chunk.get("response_message_id") and not self.message_id:
            self.message_id = str(chunk["response_message_id"])

        deltas: list[ParsedDelta] = []
        value = chunk.get("v")
        if isinstance(value, dict) and isinstance(value.get("response"), dict):
            response = value["response"]
            if "thinking_enabled" in response:
                self.current_path = "thinking" if response["thinking_enabled"] else "content"
            fragments = response.get("fragments")
            if isinstance(fragments, list):
                deltas.extend(self._process_fragments(fragments))
        elif chunk.get("p") == "response/fragments" and isinstance(value, list):
            deltas.extend(self._process_fragments(value))
        elif chunk.get("p") == "response" and isinstance(value, list):
            for operation in value:
                if not isinstance(operation, dict):
                    continue
                if operation.get("p") == "accumulated_token_usage" and isinstance(
                    operation.get("v"), int
                ):
                    self.accumulated_token_usage = operation["v"]
                nested = operation.get("v")
                if operation.get("p") == "response" and isinstance(nested, dict):
                    if nested.get("thinking_enabled") is True:
                        self.current_path = "thinking"

        path = str(chunk.get("p") or "")
        if (path == "response/search_results" or _SEARCH_RESULTS_PATH.match(path)) and isinstance(
            value, list
        ):
            if chunk.get("o") == "BATCH":
                self._apply_search_batch(value)
            else:
                self._merge_search_results(value)
            return deltas

        if isinstance(value, str):
            delta = self._content_delta(value, self.current_path)
            if delta:
                deltas.append(delta)
        elif isinstance(value, list) and path not in {"response", "response/fragments"}:
            content = "".join(
                str(item.get("content", ""))
                for operation in value
                if isinstance(operation, dict) and isinstance(operation.get("v"), list)
                for item in operation["v"]
                if isinstance(item, dict)
            )
            delta = self._content_delta(content, self.current_path)
            if delta:
                deltas.append(delta)
        return deltas

    def citations(self) -> str:
        if self.is_search_silent:
            return ""
        unique: dict[str, dict[str, Any]] = {}
        for result in self.search_results:
            url = result.get("url")
            title = result.get("title")
            cite_index = result.get("cite_index")
            if isinstance(url, str) and isinstance(title, str) and isinstance(cite_index, int):
                unique.setdefault(url, result)
        return "\n".join(
            f"[{item['cite_index']}]: [{item['title']}]({item['url']})"
            for item in sorted(unique.values(), key=lambda item: item["cite_index"])
        )

    def _process_fragments(self, fragments: list[Any]) -> list[ParsedDelta]:
        deltas: list[ParsedDelta] = []
        for fragment in fragments:
            if not isinstance(fragment, dict):
                continue
            results = fragment.get("results")
            if isinstance(results, list):
                self._merge_search_results(results)
            content = fragment.get("content")
            fragment_type = fragment.get("type")
            if not isinstance(content, str):
                continue
            if fragment_type == "THINK":
                self.current_path = "thinking"
                delta = self._content_delta(content, "thinking")
            elif fragment_type in {"ANSWER", "RESPONSE"}:
                self.current_path = "content"
                delta = self._content_delta(content, "content")
            else:
                delta = None
            if delta:
                deltas.append(delta)
        return deltas

    def _content_delta(self, content: str, path: str) -> ParsedDelta | None:
        if not content:
            return None
        cleaned = content.replace("FINISHED", "")
        if self.should_strip_search_marker:
            cleaned = _SEARCH_CONTROL_MARKER.sub("", cleaned)
        cleaned = (
            re.sub(r"\[citation:(\d+)\]", "", cleaned)
            if self.is_search_silent
            else re.sub(r"\[citation:(\d+)\]", r"[\1]", cleaned)
        )
        if not cleaned:
            return None
        effective_path = path or ("thinking" if self.is_thinking_model else "content")
        if effective_path == "thinking":
            return ParsedDelta(reasoning_content=cleaned)
        return ParsedDelta(content=cleaned)

    def _merge_search_results(self, results: list[Any]) -> None:
        for result in results:
            if not isinstance(result, dict):
                continue
            url = result.get("url")
            title = result.get("title")
            if not isinstance(url, str) or not isinstance(title, str):
                continue
            normalized = dict(result)
            if "citeIndex" in normalized and "cite_index" not in normalized:
                normalized["cite_index"] = normalized.pop("citeIndex")
            index = next(
                (i for i, existing in enumerate(self.search_results) if existing.get("url") == url),
                -1,
            )
            if index >= 0:
                self.search_results[index].update(normalized)
            else:
                self.search_results.append(normalized)

    def _apply_search_batch(self, operations: list[Any]) -> None:
        for operation in operations:
            if not isinstance(operation, dict):
                continue
            match = re.match(r"^(\d+)/cite_index$", str(operation.get("p") or ""))
            index = int(match.group(1)) if match else -1
            value = operation.get("v")
            if 0 <= index < len(self.search_results) and isinstance(value, int):
                self.search_results[index]["cite_index"] = value


async def iter_sse_data(chunks: AsyncIterator[bytes]) -> AsyncIterator[str]:
    buffer = ""
    data_lines: list[str] = []
    async for chunk in chunks:
        buffer += chunk.decode("utf-8", errors="replace")
        while True:
            newline = buffer.find("\n")
            if newline < 0:
                break
            line = buffer[:newline].rstrip("\r")
            buffer = buffer[newline + 1 :]
            if line == "":
                if data_lines:
                    yield "\n".join(data_lines)
                    data_lines.clear()
                continue
            if line.startswith(":"):
                continue
            field, separator, value = line.partition(":")
            if field == "data":
                data_lines.append(value[1:] if separator and value.startswith(" ") else value)
    if buffer:
        line = buffer.rstrip("\r")
        if line.startswith("data:"):
            value = line[5:]
            data_lines.append(value[1:] if value.startswith(" ") else value)
    if data_lines:
        yield "\n".join(data_lines)


def parse_event(data: str) -> dict[str, Any] | None:
    if data == "[DONE]":
        return None
    try:
        value = json.loads(data)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def openai_chunk(
    *,
    completion_id: str,
    model: str,
    created: int,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> str:
    payload = {
        "id": completion_id,
        "model": model,
        "object": "chat.completion.chunk",
        "created": created,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def completion_created_at() -> int:
    return int(time.time())
