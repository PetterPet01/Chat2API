from __future__ import annotations

from dataclasses import dataclass

MODELS = (
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "deepseek-v4-flash-think",
    "deepseek-v4-flash-search",
    "deepseek-v4-flash-think-search",
    "deepseek-v4-pro-think",
    "deepseek-v4-pro-search",
    "deepseek-v4-pro-think-search",
    "deepseek-chat",
    "deepseek-reasoner",
    "DeepSeek-V3.2",
    "DeepSeek-Search",
    "DeepSeek-R1",
    "DeepSeek-R1-Search",
)


@dataclass(frozen=True, slots=True)
class ChatOptions:
    model_type: str
    search_enabled: bool
    thinking_enabled: bool


def resolve_chat_options(model: str, web_search: bool, reasoning_effort: str | None) -> ChatOptions:
    model_lower = model.lower()
    is_pro = "deepseek-v4-pro" in model_lower or "expert" in model_lower
    is_search = "search" in model_lower
    is_thinking = any(alias in model_lower for alias in ("think", "r1", "reasoner"))
    return ChatOptions(
        model_type="expert" if is_pro else "default",
        search_enabled=web_search or is_search,
        thinking_enabled=reasoning_effort is not None or is_thinking,
    )
