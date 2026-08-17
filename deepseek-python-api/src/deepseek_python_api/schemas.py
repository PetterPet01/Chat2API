from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TextContentPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"]
    text: str


class ImageUrlValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    detail: Literal["auto", "low", "high"] | None = None


class ImageContentPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["image_url"]
    image_url: ImageUrlValue | str

    @property
    def url(self) -> str:
        return self.image_url if isinstance(self.image_url, str) else self.image_url.url


ContentPart = Annotated[TextContentPart | ImageContentPart, Field(discriminator="type")]


class ToolCallFunction(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    arguments: str = "{}"


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentPart] | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    reasoning_content: str | None = None

    @model_validator(mode="after")
    def validate_message_shape(self) -> ChatMessage:
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        if self.content is None and not (self.role == "assistant" and self.tool_calls):
            raise ValueError("message content may be null only for assistant tool calls")
        return self


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = None
    web_search: bool = False
    reasoning_effort: Literal["low", "medium", "high", "max"] | None = None
    n: int = Field(default=1, ge=1)
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None

    @field_validator("reasoning_effort", mode="before")
    @classmethod
    def normalize_reasoning_effort(cls, value: Any) -> Any:
        return "high" if value == "medium" else value

    @model_validator(mode="after")
    def validate_supported_options(self) -> ChatCompletionRequest:
        if self.n != 1:
            raise ValueError("only n=1 is supported")
        if self.tool_choice not in (None, "auto"):
            raise ValueError("only tool_choice='auto' is supported")
        return self


class ErrorDetail(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "deepseek-web"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject]


def tool_calls_to_prompt(tool_calls: list[ToolCall]) -> str:
    from .deepseek_v4 import format_tool_calls_as_dsml

    return format_tool_calls_as_dsml(tool_calls)
