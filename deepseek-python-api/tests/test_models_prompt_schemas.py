from __future__ import annotations

import pytest
from pydantic import ValidationError

from deepseek_python_api.models import resolve_chat_options
from deepseek_python_api.prompt import extract_image_urls, messages_to_prompt
from deepseek_python_api.schemas import ChatCompletionRequest


def test_resolve_chat_options_from_model_aliases() -> None:
    options = resolve_chat_options("deepseek-v4-pro-think-search", False, None)
    assert options.model_type == "expert"
    assert options.search_enabled is True
    assert options.thinking_enabled is True


def test_resolve_chat_options_from_request_flags() -> None:
    options = resolve_chat_options("deepseek-v4-flash", True, "medium")
    assert options.model_type == "default"
    assert options.search_enabled is True
    assert options.thinking_enabled is True


def test_prompt_conversion_preserves_history_and_ignores_images() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "deepseek-v4-flash",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Answer"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe it"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/a.jpg"},
                        },
                    ],
                },
            ],
        }
    )
    assert messages_to_prompt(request.messages) == (
        "Be concise.<｜User｜>First<｜Assistant｜>Answer<｜end of sentence｜><｜User｜>Describe it"
    )
    assert extract_image_urls(request.messages) == ["https://example.com/a.jpg"]


def test_string_image_url_compatibility() -> None:
    request = ChatCompletionRequest.model_validate(
        {
            "model": "deepseek-v4-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": "data:image/png;base64,eA=="}],
                }
            ],
        }
    )
    assert extract_image_urls(request.messages) == ["data:image/png;base64,eA=="]


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "deepseek-v4-flash", "messages": []},
        {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "x"}],
            "n": 2,
        },
        {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": None}],
        },
        {
            "model": "deepseek-v4-flash",
            "messages": [{"role": "tool", "content": "result"}],
        },
    ],
)
def test_rejects_unsupported_request_shapes(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(payload)
