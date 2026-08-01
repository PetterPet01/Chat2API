from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
from pydantic import SecretStr

from deepseek_python_api.settings import Settings


class FakePowSolver:
    def __init__(self) -> None:
        self.calls: list[tuple[object, str]] = []

    async def create_answer(self, challenge: object, target_path: str) -> str:
        self.calls.append((challenge, target_path))
        return f"pow:{target_path}"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        deepseek_auth_token=SecretStr("user-token"),
        deepseek_api_base="https://chat.deepseek.test/api",
        challenge_retry_attempts=1,
        file_poll_attempts=2,
        file_poll_interval_seconds=0,
        max_image_bytes=1024 * 1024,
    )


@pytest.fixture
def fake_pow() -> FakePowSolver:
    return FakePowSolver()


@pytest.fixture
async def plain_http_client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as client:
        yield client
