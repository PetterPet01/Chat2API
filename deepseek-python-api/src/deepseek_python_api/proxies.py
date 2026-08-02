"""Proxy pool management with support for static and rotating (IP-changer) proxies.

Static proxy format (all equivalent):
    host:port
    user:pass@host:port
    http://user:pass@host:port
    https://user:pass@host:port
    host:port:user:pass          ← colon-separated alternate (e.g. HomeProxy style)

Rotating proxy format (line starts with "rotating:"):
    rotating:<proxy_url>|rotate_url=<rotation_api_url>
    rotating:<proxy_url>|rotate_url=<rotation_api_url>|min_interval=60

  Example line in proxies.txt:
    rotating:http://emmettdani22268:proxyhub8686@180.93.2.169:3131|rotate_url=https://buyproxy.studio/api/v3/users/rotatev2?token=...

If no prefix is given the line is treated as a static proxy (backward compat).

Selection algorithm:
  - Both types implement a common ProxyEntry interface (.url, .kind, .proxy_id).
  - assign_proxy() and ProxyPool pick entries by index (round-robin) across the
    combined pool, skipping disabled entries.
  - For rotating proxies, rotate() is async and enforces the 60-second minimum
    interval between IP changes.  A failed rotation (too early) is surfaced as a
    warning but the proxy entry remains usable with its current IP.
  - ProxyPool.rotate_all() rotates every *enabled* rotating proxy past its interval.

Enable/disable state:
  - Individual proxies (static or rotating) can be disabled at runtime via the
    management API or dashboard.
  - Disabled entries are never picked for new token assignments or rotations.
  - State is persisted to a JSON sidecar file (proxy_state_file in settings) so
    it survives restarts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

import httpx

from .errors import DeepSeekProxyError

LOGGER = logging.getLogger(__name__)

DEFAULT_PROXIES_FILE = "proxies.txt"
_DEFAULT_MIN_ROTATION_INTERVAL = 60.0  # seconds — hard floor imposed by the provider


# ---------------------------------------------------------------------------
# Stable proxy ID helper
# ---------------------------------------------------------------------------

def _make_proxy_id(url: str) -> str:
    """Return a short stable hex ID derived from the proxy URL."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Static proxy
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class StaticProxyEntry:
    """An ordinary fixed HTTP(S) proxy."""

    url: str
    kind: Literal["static"] = field(default="static", init=False)

    @property
    def proxy_id(self) -> str:
        return _make_proxy_id(self.url)

    @property
    def redacted(self) -> str:
        parts = urlsplit(self.url)
        host = parts.hostname or ""
        port = f":{parts.port}" if parts.port else ""
        username = parts.username
        netloc = f"***@{host}{port}" if username else f"{host}{port}"
        return urlunsplit((parts.scheme, netloc, "", "", ""))


# ---------------------------------------------------------------------------
# Rotating proxy
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RotatingProxyEntry:
    """A proxy whose upstream IP can be rotated via an HTTP API call.

    The provider enforces a minimum of ``min_interval`` seconds between
    successive rotations.  We track ``last_rotated_at`` and skip the API call
    when it would be refused anyway.
    """

    url: str
    rotate_url: str
    min_interval: float = _DEFAULT_MIN_ROTATION_INTERVAL
    last_rotated_at: float = field(default=0.0)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    kind: Literal["rotating"] = field(default="rotating", init=False)

    @property
    def proxy_id(self) -> str:
        return _make_proxy_id(self.url)

    @property
    def redacted(self) -> str:
        parts = urlsplit(self.url)
        host = parts.hostname or ""
        port = f":{parts.port}" if parts.port else ""
        username = parts.username
        netloc = f"***@{host}{port}" if username else f"{host}{port}"
        return urlunsplit((parts.scheme, netloc, "", "", ""))

    @property
    def seconds_until_next_rotation(self) -> float:
        """How many seconds must pass before another rotation is allowed."""
        return max(0.0, self.min_interval - (time.monotonic() - self.last_rotated_at))

    @property
    def ready_to_rotate(self) -> bool:
        return self.seconds_until_next_rotation == 0.0

    async def rotate(self, *, client: httpx.AsyncClient | None = None) -> bool:
        """Request an IP rotation.

        Returns True if the rotation succeeded (or was accepted by the provider).
        Returns False if it was too early or the API call failed.
        Does *not* raise; failures are logged.
        """
        async with self._lock:
            wait = self.seconds_until_next_rotation
            if wait > 0:
                LOGGER.info(
                    "Rotating proxy %s: too early by %.1fs — skipping",
                    self.redacted,
                    wait,
                )
                return False

            own_client = client is None
            if own_client:
                client = httpx.AsyncClient()
            try:
                resp = await client.get(
                    self.rotate_url,
                    timeout=15.0,
                    follow_redirects=True,
                )
                resp.raise_for_status()
                self.last_rotated_at = time.monotonic()
                LOGGER.info(
                    "Rotating proxy %s: rotation OK (HTTP %s)",
                    self.redacted,
                    resp.status_code,
                )
                return True
            except Exception as exc:
                LOGGER.warning(
                    "Rotating proxy %s: rotation request failed — %s",
                    self.redacted,
                    exc,
                )
                return False
            finally:
                if own_client:
                    await client.aclose()


# Unified type alias
ProxyEntry = StaticProxyEntry | RotatingProxyEntry


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_colon_quad(value: str) -> str | None:
    """Parse ``host:port:user:pass`` into a normalised ``http://user:pass@host:port`` URL."""
    parts = value.split(":")
    if len(parts) != 4:
        return None
    host, port_str, user, password = parts
    try:
        port = int(port_str)
    except ValueError:
        return None
    if not host or not (1 <= port <= 65535):
        return None
    return f"http://{user}:{password}@{host}:{port}"


def normalize_proxy_url(value: str) -> str | None:
    """Normalise an arbitrary static-proxy string to ``http(s)://…`` form.

    Accepts:
      - ``host:port``
      - ``user:pass@host:port``
      - ``http://…`` / ``https://…``
      - ``host:port:user:pass``  (colon-quad, HomeProxy-style)
    Returns None for blank lines, comments, and unparseable values.
    """
    trimmed = value.strip()
    if not trimmed or trimmed.startswith("#"):
        return None

    # Try colon-quad first (4 colon-separated segments, last two not looking like a URL scheme)
    if "://" not in trimmed and trimmed.count(":") == 3:
        candidate = _parse_colon_quad(trimmed)
        if candidate:
            return candidate

    with_scheme = trimmed if "://" in trimmed else f"http://{trimmed}"
    parts = urlsplit(with_scheme)
    try:
        port = parts.port
    except ValueError:
        return None
    if parts.scheme not in {"http", "https"} or not parts.hostname or port is None:
        return None
    return with_scheme


def _parse_rotating_line(line: str) -> RotatingProxyEntry | None:
    """Parse a ``rotating:<proxy_url>|rotate_url=<url>[|min_interval=N]`` line."""
    # Strip the "rotating:" prefix (case-insensitive)
    body = line[len("rotating:"):].strip()
    segments = [s.strip() for s in body.split("|")]
    if not segments:
        return None

    proxy_url_raw = segments[0]
    proxy_url = normalize_proxy_url(proxy_url_raw)
    if not proxy_url:
        LOGGER.warning("Rotating proxy line has invalid proxy URL: %r", proxy_url_raw)
        return None

    params: dict[str, str] = {}
    for seg in segments[1:]:
        if "=" in seg:
            k, _, v = seg.partition("=")
            params[k.strip().lower()] = v.strip()

    rotate_url = params.get("rotate_url")
    if not rotate_url:
        LOGGER.warning(
            "Rotating proxy line missing rotate_url — treating as static: %r", line
        )
        return None

    min_interval = _DEFAULT_MIN_ROTATION_INTERVAL
    if "min_interval" in params:
        try:
            min_interval = max(0.0, float(params["min_interval"]))
        except ValueError:
            pass

    return RotatingProxyEntry(
        url=proxy_url,
        rotate_url=rotate_url,
        min_interval=min_interval,
    )


def _parse_proxy_line(line: str) -> ProxyEntry | None:
    """Parse one line from a proxy file into a ProxyEntry (static or rotating)."""
    trimmed = line.strip()
    if not trimmed or trimmed.startswith("#"):
        return None
    lower = trimmed.lower()
    if lower.startswith("rotating:"):
        return _parse_rotating_line(trimmed)
    url = normalize_proxy_url(trimmed)
    if url is None:
        return None
    return StaticProxyEntry(url)


# ---------------------------------------------------------------------------
# File loader
# ---------------------------------------------------------------------------

def load_proxy_entries(path: str | Path) -> list[ProxyEntry]:
    """Load all proxy entries from *path*, skipping blank lines and comments.

    Deduplication is done by URL so that duplicate static proxies are dropped;
    rotating entries with the same proxy URL but different rotate_url values are
    kept (they have distinct rotate_url keys).
    """
    file_path = Path(path).expanduser()
    if not file_path.exists():
        return []
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DeepSeekProxyError(
            f"Failed to read proxies file: {file_path}",
            status_code=500,
            code="proxy_file_error",
            error_type="configuration_error",
        ) from exc

    entries: list[ProxyEntry] = []
    seen_urls: set[str] = set()
    for line in lines:
        entry = _parse_proxy_line(line)
        if entry is None:
            continue
        key = (entry.url, getattr(entry, "rotate_url", ""))
        if key not in seen_urls:
            seen_urls.add(key)
            entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Assignment helpers (used by TokenManager)
# ---------------------------------------------------------------------------

def assign_proxy(entries: list[ProxyEntry], index: int) -> ProxyEntry | None:
    """Round-robin assignment: return entries[index % len(entries)] or None."""
    if not entries:
        return None
    return entries[index % len(entries)]


# ---------------------------------------------------------------------------
# ProxyPool — high-level pool with enable/disable and rotation support
# ---------------------------------------------------------------------------

class ProxyPool:
    """Manages a mixed pool of static and rotating proxies.

    Each proxy can be individually enabled or disabled at runtime.  Disabled
    entries are excluded from ``pick()``, ``rotate_ready()``, and ``rotate_all()``.
    The enabled/disabled state is persisted to a JSON sidecar file so it
    survives restarts.

    Provides:
      - ``pick(index)`` — select an *enabled* proxy by index (round-robin).
      - ``enable_by_id(proxy_id)`` / ``disable_by_id(proxy_id)`` — toggle a proxy.
      - ``rotate_ready()`` — list of *enabled* rotating entries ready to rotate.
      - ``rotate_all(client)`` — trigger rotation for every ready enabled rotating entry.
      - ``rotate_for_proxy_url(url, client)`` — rotate one specific rotating proxy.
      - ``summary()`` — full status dict including all entries with id/enabled/kind.
    """

    def __init__(self, entries: list[ProxyEntry]) -> None:
        self._entries = entries
        self._disabled: set[str] = set()  # set of disabled proxy URLs

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def entries(self) -> list[ProxyEntry]:
        return self._entries

    @property
    def count(self) -> int:
        return len(self._entries)

    @property
    def enabled_entries(self) -> list[ProxyEntry]:
        return [e for e in self._entries if e.url not in self._disabled]

    @property
    def static_count(self) -> int:
        return sum(1 for e in self._entries if isinstance(e, StaticProxyEntry))

    @property
    def rotating_count(self) -> int:
        return sum(1 for e in self._entries if isinstance(e, RotatingProxyEntry))

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def find_by_url(self, url: str) -> ProxyEntry | None:
        return next((e for e in self._entries if e.url == url), None)

    def find_by_id(self, proxy_id: str) -> ProxyEntry | None:
        return next((e for e in self._entries if e.proxy_id == proxy_id), None)

    def is_enabled(self, url: str) -> bool:
        return url not in self._disabled

    # ------------------------------------------------------------------
    # Enable / disable
    # ------------------------------------------------------------------

    def enable_by_id(self, proxy_id: str) -> ProxyEntry:
        """Enable the proxy with *proxy_id*.  Returns the entry.  Raises if not found."""
        entry = self.find_by_id(proxy_id)
        if entry is None:
            raise DeepSeekProxyError(
                f"Proxy not found: {proxy_id}",
                status_code=404,
                code="proxy_not_found",
                error_type="invalid_request_error",
            )
        self._disabled.discard(entry.url)
        LOGGER.info("Proxy enabled: %s (%s)", entry.redacted, proxy_id)
        return entry

    def disable_by_id(self, proxy_id: str) -> ProxyEntry:
        """Disable the proxy with *proxy_id*.  Returns the entry.  Raises if not found."""
        entry = self.find_by_id(proxy_id)
        if entry is None:
            raise DeepSeekProxyError(
                f"Proxy not found: {proxy_id}",
                status_code=404,
                code="proxy_not_found",
                error_type="invalid_request_error",
            )
        self._disabled.add(entry.url)
        LOGGER.info("Proxy disabled: %s (%s)", entry.redacted, proxy_id)
        return entry

    # ------------------------------------------------------------------
    # Picking
    # ------------------------------------------------------------------

    def pick(self, index: int) -> ProxyEntry | None:
        """Return the entry at *index* mod len(enabled_entries), or None."""
        enabled = self.enabled_entries
        if not enabled:
            return None
        return enabled[index % len(enabled)]

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def rotate_ready(self) -> list[RotatingProxyEntry]:
        """Return *enabled* rotating entries whose cooldown has expired."""
        return [
            e for e in self._entries
            if isinstance(e, RotatingProxyEntry)
            and self.is_enabled(e.url)
            and e.ready_to_rotate
        ]

    async def rotate_all(self, client: httpx.AsyncClient | None = None) -> dict[str, bool]:
        """Rotate every *enabled* rotating proxy that is ready.  Returns {redacted_url: success}."""
        ready = self.rotate_ready()
        if not ready:
            return {}
        results = await asyncio.gather(
            *(e.rotate(client=client) for e in ready), return_exceptions=True
        )
        return {
            e.redacted: bool(r) if not isinstance(r, BaseException) else False
            for e, r in zip(ready, results)
        }

    async def rotate_for_proxy_url(
        self, proxy_url: str, client: httpx.AsyncClient | None = None
    ) -> bool:
        """Rotate the proxy entry that owns *proxy_url*, if it is rotating and ready."""
        entry = self.find_by_url(proxy_url)
        if not isinstance(entry, RotatingProxyEntry):
            return False
        return await entry.rotate(client=client)

    # ------------------------------------------------------------------
    # Persistence (enabled/disabled state)
    # ------------------------------------------------------------------

    def load_state(self, path: str | Path) -> None:
        """Load enabled/disabled state from *path* (JSON sidecar).  Silent if missing."""
        file_path = Path(path).expanduser()
        if not file_path.exists():
            return
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            disabled = data.get("disabled", [])
            if isinstance(disabled, list):
                self._disabled = {url for url in disabled if isinstance(url, str)}
        except Exception as exc:
            LOGGER.warning("Failed to load proxy state from %s: %s", file_path, exc)

    def save_state(self, path: str | Path) -> None:
        """Persist enabled/disabled state to *path* atomically."""
        file_path = Path(path).expanduser()
        payload: dict[str, Any] = {"disabled": sorted(self._disabled)}
        file_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{file_path.name}.", dir=file_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
            os.replace(tmp, file_path)
        except Exception as exc:
            LOGGER.warning("Failed to save proxy state to %s: %s", file_path, exc)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, object]:
        all_entries = []
        for e in self._entries:
            entry_dict: dict[str, object] = {
                "id": e.proxy_id,
                "kind": e.kind,
                "proxy": e.redacted,
                "enabled": self.is_enabled(e.url),
            }
            if isinstance(e, RotatingProxyEntry):
                entry_dict["min_interval_seconds"] = e.min_interval
                entry_dict["seconds_until_next_rotation"] = round(
                    e.seconds_until_next_rotation, 1
                )
                entry_dict["ready_to_rotate"] = e.ready_to_rotate
            all_entries.append(entry_dict)

        enabled_count = len(self.enabled_entries)
        return {
            "total": self.count,
            "enabled": enabled_count,
            "disabled": self.count - enabled_count,
            "static": self.static_count,
            "rotating": self.rotating_count,
            "entries": all_entries,
            # Keep legacy key for backward compat with older dashboard/tests
            "rotating_entries": [
                e for e in all_entries if e["kind"] == "rotating"
            ],
        }
