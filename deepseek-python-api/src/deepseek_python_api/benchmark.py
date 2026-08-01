from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(slots=True)
class Sample:
    ok: bool
    status_code: int | None
    latency_ms: float
    ttft_ms: float | None = None
    error: str | None = None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def safe_error(value: str) -> str:
    return " ".join(value.replace("\n", " ").split())[:240]


def auth_headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def make_payload(args: argparse.Namespace, *, stream: bool) -> dict[str, Any]:
    content: str | list[dict[str, Any]] = args.prompt
    if args.image_url:
        content = [
            {"type": "text", "text": args.prompt},
            {"type": "image_url", "image_url": {"url": args.image_url}},
        ]
    elif args.image_file:
        encoded = __import__("base64").b64encode(Path(args.image_file).read_bytes()).decode()
        content = [
            {"type": "text", "text": args.prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
        ]
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
    }
    if args.web_search:
        payload["web_search"] = True
    if args.reasoning_effort:
        payload["reasoning_effort"] = args.reasoning_effort
    return payload


async def one_request(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    *,
    stream: bool,
) -> Sample:
    started = time.perf_counter()
    ttft: float | None = None
    try:
        if stream:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                headers=auth_headers(args.api_key),
                json=make_payload(args, stream=True),
            ) as response:
                if response.status_code >= 400:
                    return Sample(
                        False,
                        response.status_code,
                        elapsed_ms(started),
                        error=safe_error((await response.aread()).decode(errors="replace")),
                    )
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line[6:] != "[DONE]" and ttft is None:
                        ttft = elapsed_ms(started)
            return Sample(True, response.status_code, elapsed_ms(started), ttft)
        response = await client.post(
            "/v1/chat/completions",
            headers=auth_headers(args.api_key),
            json=make_payload(args, stream=False),
        )
        if response.status_code >= 400:
            return Sample(
                False, response.status_code, elapsed_ms(started), error=safe_error(response.text)
            )
        return Sample(True, response.status_code, elapsed_ms(started))
    except Exception as exc:  # benchmark must record one failed request and continue
        return Sample(
            False, None, elapsed_ms(started), error=safe_error(f"{type(exc).__name__}: {exc}")
        )


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


async def run_scenario(args: argparse.Namespace, *, stream: bool) -> list[Sample]:
    limits = httpx.Limits(
        max_connections=args.concurrency, max_keepalive_connections=args.concurrency
    )
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"), timeout=timeout, limits=limits
    ) as client:
        for _ in range(args.warmup):
            await one_request(client, args, stream=stream)
        semaphore = asyncio.Semaphore(args.concurrency)

        async def worker() -> Sample:
            async with semaphore:
                return await one_request(client, args, stream=stream)

        return await asyncio.gather(*(worker() for _ in range(args.requests)))


def summarize(samples: list[Sample], elapsed: float) -> dict[str, Any]:
    latencies = [sample.latency_ms for sample in samples]
    ttfts = [sample.ttft_ms for sample in samples if sample.ttft_ms is not None]
    successes = sum(sample.ok for sample in samples)
    failures = len(samples) - successes
    errors = Counter(
        sample.error or f"HTTP {sample.status_code}" for sample in samples if not sample.ok
    )

    def stats(values: list[float]) -> dict[str, float | None]:
        return {
            "min": min(values) if values else None,
            "mean": statistics.fmean(values) if values else None,
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": max(values) if values else None,
        }

    return {
        "attempted": len(samples),
        "succeeded": successes,
        "failed": failures,
        "success_rate": successes / len(samples) if samples else 0.0,
        "elapsed_seconds": elapsed,
        "requests_per_second": len(samples) / elapsed if elapsed else 0.0,
        "latency_ms": stats(latencies),
        "ttft_ms": stats(ttfts),
        "errors": dict(errors),
    }


async def preflight(args: argparse.Namespace) -> None:
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"), timeout=args.timeout
    ) as client:
        health = await client.get("/health")
        if health.status_code >= 400:
            raise RuntimeError(f"health check failed: HTTP {health.status_code}")
        models = await client.get("/v1/models", headers=auth_headers(args.api_key))
        if models.status_code >= 400:
            raise RuntimeError(f"model check failed: HTTP {models.status_code}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Conservative benchmark for the DeepSeek Python API"
    )
    result.add_argument(
        "--base-url", default=os.getenv("DEEPSEEK_API_URL", "http://127.0.0.1:8000")
    )
    result.add_argument("--api-key", default=os.getenv("API_KEY"))
    result.add_argument("--model", default="deepseek-v4-flash")
    result.add_argument(
        "--prompt", default="Reply with one short sentence confirming the API is working."
    )
    result.add_argument("--requests", type=int, default=3)
    result.add_argument("--concurrency", type=int, default=1)
    result.add_argument("--warmup", type=int, default=1)
    result.add_argument("--timeout", type=float, default=180.0)
    result.add_argument("--stream", action="store_true")
    result.add_argument("--web-search", action="store_true")
    result.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    result.add_argument("--image-url")
    result.add_argument("--image-file")
    result.add_argument("--json", dest="json_path", type=Path)
    result.add_argument("--min-success-rate", type=float, default=1.0)
    return result


def validate(args: argparse.Namespace) -> None:
    if args.requests < 1 or args.concurrency < 1 or args.warmup < 0 or args.timeout <= 0:
        raise ValueError(
            "requests and concurrency must be positive; warmup non-negative; timeout positive"
        )
    if not 0 <= args.min_success_rate <= 1:
        raise ValueError("min-success-rate must be between 0 and 1")
    if args.image_url and args.image_file:
        raise ValueError("choose only one of --image-url and --image-file")
    if args.image_file and not Path(args.image_file).is_file():
        raise ValueError(f"image file does not exist: {args.image_file}")


async def async_main(args: argparse.Namespace) -> int:
    validate(args)
    await preflight(args)
    started = time.perf_counter()
    samples = await run_scenario(args, stream=args.stream)
    report = {
        "configuration": {
            "base_url": args.base_url,
            "model": args.model,
            "requests": args.requests,
            "concurrency": args.concurrency,
            "warmup": args.warmup,
            "stream": args.stream,
            "web_search": args.web_search,
            "reasoning_effort": args.reasoning_effort,
            "image": bool(args.image_url or args.image_file),
        },
        "summary": summarize(samples, time.perf_counter() - started),
    }
    print(json.dumps(report["summary"], indent=2))
    if args.json_path:
        args.json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["summary"]["success_rate"] >= args.min_success_rate else 1


def main() -> None:
    try:
        raise SystemExit(asyncio.run(async_main(parser().parse_args())))
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
