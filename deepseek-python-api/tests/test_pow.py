from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from deepseek_python_api.errors import UpstreamProtocolError
from deepseek_python_api.pow import Challenge, DeepSeekHashSolver


def test_challenge_validates_payload() -> None:
    with pytest.raises(UpstreamProtocolError):
        Challenge.from_payload({"algorithm": "DeepSeekHashV1"})


@pytest.mark.asyncio
async def test_rejects_unknown_pow_algorithm() -> None:
    solver = DeepSeekHashSolver(timeout_seconds=1, workers=1)
    try:
        challenge = Challenge("other", "c", "s", 1, 1, "sig")
        with pytest.raises(UpstreamProtocolError, match="Unsupported"):
            await solver.create_answer(challenge, "/target")
    finally:
        solver.close()


@pytest.mark.asyncio
async def test_pow_solver_matches_recorded_challenge_shape() -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "pow_challenge.json"
    if not fixture_path.exists():
        pytest.skip("Sanitized PoW fixture has not been generated")
    fixture = json.loads(fixture_path.read_text())
    solver = DeepSeekHashSolver(timeout_seconds=30, workers=1)
    try:
        encoded = await solver.create_answer(
            Challenge.from_payload(fixture["challenge"]), fixture["target_path"]
        )
    finally:
        solver.close()
    answer = json.loads(base64.b64decode(encoded))
    assert answer["target_path"] == fixture["target_path"]
    assert isinstance(answer["answer"], int | float)
    assert answer["answer"] >= 0
    assert answer["signature"] == fixture["challenge"]["signature"]
