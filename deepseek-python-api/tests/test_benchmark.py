from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx
import pytest

from deepseek_python_api import benchmark


def args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "base_url": "https://proxy.test",
        "api_key": "secret-client-key",
        "model": "deepseek-v4-flash",
        "prompt": "hello",
        "requests": 2,
        "concurrency": 1,
        "warmup": 0,
        "timeout": 10.0,
        "stream": False,
        "web_search": False,
        "reasoning_effort": None,
        "image_url": None,
        "image_file": None,
        "json_path": None,
        "min_success_rate": 1.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_percentile_and_summary() -> None:
    assert benchmark.percentile([], 0.5) is None
    assert benchmark.percentile([1, 3], 0.5) == 2
    report = benchmark.summarize(
        [
            benchmark.Sample(True, 200, 10, 4),
            benchmark.Sample(False, 500, 30, error="upstream failed"),
        ],
        2,
    )
    assert report["success_rate"] == 0.5
    assert report["requests_per_second"] == 1
    assert report["latency_ms"]["p50"] == 20
    assert report["ttft_ms"]["p50"] == 4
    assert report["errors"] == {"upstream failed": 1}


@pytest.mark.asyncio
async def test_non_streaming_and_streaming_requests() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.headers["authorization"] == "Bearer secret-client-key"
        if payload["stream"]:
            return httpx.Response(
                200,
                text='data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async with httpx.AsyncClient(
        base_url="https://proxy.test", transport=httpx.MockTransport(handler)
    ) as client:
        regular = await benchmark.one_request(client, args(), stream=False)
        streaming = await benchmark.one_request(client, args(), stream=True)
    assert regular.ok
    assert streaming.ok
    assert streaming.ttft_ms is not None


@pytest.mark.asyncio
async def test_http_failure_is_sanitized() -> None:
    async with httpx.AsyncClient(
        base_url="https://proxy.test",
        transport=httpx.MockTransport(lambda _: httpx.Response(503, text="bad\n response")),
    ) as client:
        sample = await benchmark.one_request(client, args(), stream=False)
    assert not sample.ok
    assert sample.status_code == 503
    assert sample.error == "bad response"


def test_payload_variants_and_validation(tmp_path: Path) -> None:
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"jpeg")
    payload = benchmark.make_payload(
        args(image_file=str(image), web_search=True, reasoning_effort="medium"), stream=True
    )
    content = payload["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert payload["web_search"] is True
    assert payload["reasoning_effort"] == "medium"
    with pytest.raises(ValueError, match="choose only one"):
        benchmark.validate(args(image_url="https://example.test/a.jpg", image_file=str(image)))
    with pytest.raises(ValueError, match="between 0 and 1"):
        benchmark.validate(args(min_success_rate=2))


@pytest.mark.asyncio
async def test_async_main_writes_secret_free_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def fake_preflight(_: argparse.Namespace) -> None:
        return None

    async def fake_run(_: argparse.Namespace, *, stream: bool) -> list[benchmark.Sample]:
        assert not stream
        return [benchmark.Sample(True, 200, 12)]

    monkeypatch.setattr(benchmark, "preflight", fake_preflight)
    monkeypatch.setattr(benchmark, "run_scenario", fake_run)
    output = tmp_path / "report.json"
    status = await benchmark.async_main(args(json_path=output, api_key="do-not-write-this"))
    data = output.read_text(encoding="utf-8")
    assert status == 0
    assert "do-not-write-this" not in data
    assert json.loads(data)["summary"]["succeeded"] == 1


@pytest.mark.asyncio
async def test_success_rate_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_preflight(_: argparse.Namespace) -> None:
        return None

    async def fake_run(_: argparse.Namespace, *, stream: bool) -> list[benchmark.Sample]:
        return [benchmark.Sample(False, 500, 2, error="failed")]

    monkeypatch.setattr(benchmark, "preflight", fake_preflight)
    monkeypatch.setattr(benchmark, "run_scenario", fake_run)
    assert await benchmark.async_main(args(min_success_rate=0.5)) == 1
