"""Tests for the mixed static + rotating proxy pool."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from deepseek_python_api.proxies import (
    ProxyPool,
    RotatingProxyEntry,
    StaticProxyEntry,
    _parse_colon_quad,
    _parse_proxy_line,
    _parse_rotating_line,
    assign_proxy,
    load_proxy_entries,
    normalize_proxy_url,
)


# ---------------------------------------------------------------------------
# normalize_proxy_url
# ---------------------------------------------------------------------------


class TestNormalizeProxyUrl:
    def test_plain_host_port(self) -> None:
        assert normalize_proxy_url("1.2.3.4:8080") == "http://1.2.3.4:8080"

    def test_user_pass_at_host_port(self) -> None:
        assert normalize_proxy_url("user:pass@1.2.3.4:8080") == "http://user:pass@1.2.3.4:8080"

    def test_already_http_scheme(self) -> None:
        assert normalize_proxy_url("http://u:p@host.com:3128") == "http://u:p@host.com:3128"

    def test_https_scheme(self) -> None:
        assert normalize_proxy_url("https://u:p@host.com:3128") == "https://u:p@host.com:3128"

    def test_colon_quad_format(self) -> None:
        result = normalize_proxy_url("180.93.2.169:3131:emmettdani22268:proxyhub8686")
        assert result == "http://emmettdani22268:proxyhub8686@180.93.2.169:3131"

    def test_blank_line(self) -> None:
        assert normalize_proxy_url("") is None
        assert normalize_proxy_url("   ") is None

    def test_comment_line(self) -> None:
        assert normalize_proxy_url("# a comment") is None

    def test_invalid_port(self) -> None:
        assert normalize_proxy_url("host.com:notaport") is None

    def test_missing_port(self) -> None:
        # host without port should fail (port is None)
        assert normalize_proxy_url("host.com") is None


# ---------------------------------------------------------------------------
# _parse_colon_quad
# ---------------------------------------------------------------------------


class TestParseColonQuad:
    def test_valid(self) -> None:
        assert _parse_colon_quad("1.2.3.4:8080:user:pass") == "http://user:pass@1.2.3.4:8080"

    def test_wrong_segments(self) -> None:
        assert _parse_colon_quad("1.2.3.4:8080") is None

    def test_bad_port(self) -> None:
        assert _parse_colon_quad("1.2.3.4:abc:user:pass") is None


# ---------------------------------------------------------------------------
# _parse_proxy_line
# ---------------------------------------------------------------------------


class TestParseProxyLine:
    def test_static_plain(self) -> None:
        entry = _parse_proxy_line("1.2.3.4:8080")
        assert isinstance(entry, StaticProxyEntry)
        assert entry.url == "http://1.2.3.4:8080"

    def test_static_colon_quad(self) -> None:
        entry = _parse_proxy_line("1.2.3.4:3131:emmettdani22268:proxyhub8686")
        assert isinstance(entry, StaticProxyEntry)
        assert "emmettdani22268" in entry.url

    def test_rotating_basic(self) -> None:
        line = "rotating:http://u:p@1.2.3.4:3131|rotate_url=http://rotate.example.com/api"
        entry = _parse_proxy_line(line)
        assert isinstance(entry, RotatingProxyEntry)
        assert entry.rotate_url == "http://rotate.example.com/api"
        assert entry.min_interval == 60.0

    def test_rotating_custom_interval(self) -> None:
        line = "rotating:http://u:p@1.2.3.4:3131|rotate_url=http://r.example.com|min_interval=90"
        entry = _parse_proxy_line(line)
        assert isinstance(entry, RotatingProxyEntry)
        assert entry.min_interval == 90.0

    def test_blank_returns_none(self) -> None:
        assert _parse_proxy_line("") is None
        assert _parse_proxy_line("  # comment  ") is None

    def test_rotating_missing_rotate_url_falls_back_to_none(self) -> None:
        # No rotate_url → warning logged, returns None
        line = "rotating:http://u:p@1.2.3.4:3131"
        entry = _parse_proxy_line(line)
        assert entry is None


# ---------------------------------------------------------------------------
# load_proxy_entries
# ---------------------------------------------------------------------------


class TestLoadProxyEntries:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        result = load_proxy_entries(tmp_path / "nonexistent.txt")
        assert result == []

    def test_loads_mixed_entries(self, tmp_path: Path) -> None:
        content = (
            "# comment\n"
            "1.2.3.4:8080\n"
            "user:pass@5.6.7.8:3128\n"
            "rotating:http://u:p@9.9.9.9:3131|rotate_url=http://rotate.test/api\n"
            "\n"
        )
        f = tmp_path / "proxies.txt"
        f.write_text(content)
        entries = load_proxy_entries(f)
        assert len(entries) == 3
        assert isinstance(entries[0], StaticProxyEntry)
        assert isinstance(entries[1], StaticProxyEntry)
        assert isinstance(entries[2], RotatingProxyEntry)

    def test_deduplicates_static(self, tmp_path: Path) -> None:
        content = "1.2.3.4:8080\n1.2.3.4:8080\nhttp://1.2.3.4:8080\n"
        f = tmp_path / "proxies.txt"
        f.write_text(content)
        entries = load_proxy_entries(f)
        # All three normalise to the same URL
        assert len(entries) == 1

    def test_colon_quad_format(self, tmp_path: Path) -> None:
        content = "180.93.2.169:3131:emmettdani22268:proxyhub8686\n"
        f = tmp_path / "proxies.txt"
        f.write_text(content)
        entries = load_proxy_entries(f)
        assert len(entries) == 1
        assert isinstance(entries[0], StaticProxyEntry)
        assert "emmettdani22268" in entries[0].url


# ---------------------------------------------------------------------------
# StaticProxyEntry.redacted
# ---------------------------------------------------------------------------


class TestStaticProxyRedacted:
    def test_with_credentials(self) -> None:
        e = StaticProxyEntry("http://user:secret@1.2.3.4:8080")
        assert "***" in e.redacted
        assert "secret" not in e.redacted

    def test_without_credentials(self) -> None:
        e = StaticProxyEntry("http://1.2.3.4:8080")
        assert e.redacted == "http://1.2.3.4:8080"


# ---------------------------------------------------------------------------
# RotatingProxyEntry
# ---------------------------------------------------------------------------


class TestRotatingProxyEntry:
    def _make_entry(self, min_interval: float = 60.0) -> RotatingProxyEntry:
        return RotatingProxyEntry(
            url="http://u:p@1.2.3.4:3131",
            rotate_url="http://rotate.test/api",
            min_interval=min_interval,
        )

    def test_initially_ready(self) -> None:
        entry = self._make_entry()
        assert entry.ready_to_rotate is True
        assert entry.seconds_until_next_rotation == 0.0

    def test_not_ready_immediately_after_rotation(self) -> None:
        entry = self._make_entry()
        entry.last_rotated_at = time.monotonic()
        assert entry.ready_to_rotate is False
        assert entry.seconds_until_next_rotation > 0

    @pytest.mark.asyncio
    async def test_rotate_skips_when_not_ready(self) -> None:
        entry = self._make_entry()
        entry.last_rotated_at = time.monotonic()  # just rotated

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        result = await entry.rotate(client=mock_client)
        assert result is False
        mock_client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_rotate_success(self) -> None:
        entry = self._make_entry()  # last_rotated_at=0 → ready

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await entry.rotate(client=mock_client)
        assert result is True
        assert entry.last_rotated_at > 0
        mock_client.get.assert_awaited_once_with(
            "http://rotate.test/api",
            timeout=15.0,
            follow_redirects=True,
        )

    @pytest.mark.asyncio
    async def test_rotate_handles_http_error(self) -> None:
        entry = self._make_entry()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

        result = await entry.rotate(client=mock_client)
        assert result is False

    @pytest.mark.asyncio
    async def test_concurrent_rotation_only_fires_once(self) -> None:
        """Two concurrent callers — only the first should actually hit the API."""
        entry = self._make_entry()

        call_count = 0

        async def fake_get(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            # After first call mark as rotated
            entry.last_rotated_at = time.monotonic()
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.status_code = 200
            return resp

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = fake_get

        # Release the lock inside the entry to simulate concurrent access.
        # We test that the _lock serialises the calls so the second sees
        # last_rotated_at set and skips.
        results = await asyncio.gather(
            entry.rotate(client=mock_client),
            entry.rotate(client=mock_client),
        )
        # Exactly one should succeed (the second will see ready_to_rotate=False
        # inside the lock because last_rotated_at was set by the first).
        assert results.count(True) == 1
        assert results.count(False) == 1


# ---------------------------------------------------------------------------
# ProxyPool
# ---------------------------------------------------------------------------


class TestProxyPool:
    def _pool(self) -> tuple[ProxyPool, StaticProxyEntry, StaticProxyEntry, RotatingProxyEntry]:
        s1 = StaticProxyEntry("http://1.1.1.1:8080")
        s2 = StaticProxyEntry("http://2.2.2.2:8080")
        r1 = RotatingProxyEntry(
            url="http://u:p@9.9.9.9:3131",
            rotate_url="http://rotate.test/api",
        )
        pool = ProxyPool([s1, s2, r1])
        return pool, s1, s2, r1

    def test_pick_round_robin(self) -> None:
        pool, s1, s2, r1 = self._pool()
        assert pool.pick(0) is s1
        assert pool.pick(1) is s2
        assert pool.pick(2) is r1
        assert pool.pick(3) is s1  # wraps

    def test_pick_empty_pool(self) -> None:
        pool = ProxyPool([])
        assert pool.pick(0) is None

    def test_static_and_rotating_counts(self) -> None:
        pool, *_ = self._pool()
        assert pool.static_count == 2
        assert pool.rotating_count == 1
        assert pool.count == 3

    def test_find_by_url(self) -> None:
        pool, s1, s2, r1 = self._pool()
        assert pool.find_by_url("http://1.1.1.1:8080") is s1
        assert pool.find_by_url("http://u:p@9.9.9.9:3131") is r1
        assert pool.find_by_url("http://unknown:9999") is None

    def test_rotate_ready(self) -> None:
        pool, _, _, r1 = self._pool()
        ready = pool.rotate_ready()
        assert r1 in ready

    def test_rotate_ready_excludes_not_ready(self) -> None:
        pool, _, _, r1 = self._pool()
        r1.last_rotated_at = time.monotonic()  # just rotated
        assert pool.rotate_ready() == []

    @pytest.mark.asyncio
    async def test_rotate_all(self) -> None:
        pool, _, _, r1 = self._pool()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)

        results = await pool.rotate_all(client=mock_client)
        assert r1.redacted in results
        assert results[r1.redacted] is True

    @pytest.mark.asyncio
    async def test_rotate_for_proxy_url_static_returns_false(self) -> None:
        pool, s1, _, _ = self._pool()
        result = await pool.rotate_for_proxy_url(s1.url)
        assert result is False

    @pytest.mark.asyncio
    async def test_rotate_for_proxy_url_unknown_returns_false(self) -> None:
        pool, _, _, _ = self._pool()
        result = await pool.rotate_for_proxy_url("http://not-in-pool:9999")
        assert result is False

    @pytest.mark.asyncio
    async def test_rotate_for_proxy_url_rotating(self) -> None:
        pool, _, _, r1 = self._pool()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await pool.rotate_for_proxy_url(r1.url, client=mock_client)
        assert result is True

    def test_summary_contains_expected_keys(self) -> None:
        pool, *_ = self._pool()
        s = pool.summary()
        assert s["total"] == 3
        assert s["static"] == 2
        assert s["rotating"] == 1
        assert isinstance(s["rotating_entries"], list)
        assert len(s["rotating_entries"]) == 1


# ---------------------------------------------------------------------------
# assign_proxy (backward-compat helper)
# ---------------------------------------------------------------------------


class TestAssignProxy:
    def test_empty(self) -> None:
        assert assign_proxy([], 0) is None

    def test_wraps(self) -> None:
        entries = [StaticProxyEntry("http://a:1"), StaticProxyEntry("http://b:2")]
        assert assign_proxy(entries, 0) is entries[0]
        assert assign_proxy(entries, 1) is entries[1]
        assert assign_proxy(entries, 2) is entries[0]
        assert assign_proxy(entries, 5) is entries[1]
