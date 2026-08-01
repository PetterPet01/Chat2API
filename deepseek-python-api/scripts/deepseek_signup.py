#!/usr/bin/env python3
"""
Automated DeepSeek Platform (platform.deepseek.com) Account Creator — SeleniumBase UC Edition
=======================================================================================
Modeled on the QwenAA qwen_signup.py approach.

Flow:
  1. Create a disposable email address (iCloud HME, Graph API Hotmail, Guerrilla Mail, Mail.tm, or Mail.gw)
  2. Open platform.deepseek.com/sign_up in an undetected Chrome session
  3. Fill in the registration form (email + password + optional captcha)
  4. Wait for email verification OTP / link, submit it in the browser
  5. Extract the auth token from localStorage ('userToken')
  6. POST the token to deepseek-python-api's management endpoint
     POST /v0/management/tokens  with  Authorization: Bearer <MANAGEMENT_API_KEY>
  7. Save the account to a local JSON file

Auth endpoint (reverse-engineered from platform.deepseek.com):
  GET  /api/v0/users/current    → {biz_data: {token: "<access-token>"}}
  Token stored in localStorage['userToken']

Mail service options:
  icloud    — Apple iCloud Hide My Email (@icloud.com alias). RECOMMENDED: passes
              DeepSeek's domain blocklist. Requires iCloud cookies + IMAP credentials.
  graphapi  — Microsoft Graph API (Hotmail / Outlook / Live). Credentials are
              pipe-separated: email|pass|refresh_token[|client_id].
              Refresh token is exchanged for a live access token via Microsoft's
              own OAuth endpoint, then inbox polled via the official Graph API.
              @hotmail.com / @outlook.com pass DeepSeek's domain check.
  guerrilla — Guerrilla Mail. Blocked by DeepSeek domain check.
  mailtm    — Mail.tm (hydra API). May be geo-blocked. Falls back to guerrilla.
  mailgw    — Mail.gw (hydra API). Falls back to guerrilla.
  auto      — Tries icloud (if configured) → graphapi (if configured)
              → guerrilla → mailtm → mailgw.

Usage:
    # RECOMMENDED — iCloud Hide My Email (passes DeepSeek domain check)
    python scripts/deepseek_signup.py --manual-captcha \\
        --mail-service icloud \\
        --icloud-cookies icloud_cookies.txt \\
        --icloud-imap-user you@icloud.com \\
        --icloud-imap-pass xxxx-xxxx-xxxx-xxxx

    # Microsoft Hotmail / Outlook via Graph API refresh token
    python scripts/deepseek_signup.py --manual-captcha \\
        --mail-service graphapi \\
        --graphapi-creds 'you@hotmail.com|YourPass|M.C556_BAY.0...refresh_token...|client-id-uuid'

    python scripts/deepseek_signup.py --count 3
    python scripts/deepseek_signup.py --headless
    python scripts/deepseek_signup.py --debug
    python scripts/deepseek_signup.py --manual-captcha
    python scripts/deepseek_signup.py --api-url http://localhost:8000 --api-key nowsthetime
    python scripts/deepseek_signup.py --no-upload

Setting up iCloud HME:
  1. Go to icloud.com/settings in your browser (log in with iCloud+)
  2. Install Cookie-Editor extension, click Export → Header String
  3. Paste into icloud_cookies.txt
  4. Generate an App-Specific Password at appleid.apple.com
  5. Pass it as --icloud-imap-pass  (IMAP host: imap.mail.me.com, port 993)

Setting up Microsoft Graph API mail:
  1. Get a refresh_token for your Hotmail / Outlook account (e.g. from dongvanfb.net
     or any tool that exports Microsoft OAuth tokens).
  2. The credential string is pipe-separated:
       email|password|refresh_token|client_id
     client_id defaults to the Outlook mobile app ID if omitted.
  3. Pass via --graphapi-creds 'email|pass|token[|client_id]'
"""

from __future__ import annotations

import argparse
import asyncio
import imaplib
import json
import logging
import random
import re
import string
import time
from abc import ABC, abstractmethod
from email import policy
from email.parser import BytesParser
from pathlib import Path

import requests

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("deepseek_signup")

# ── Constants ─────────────────────────────────────────────────────────────────
MAIL_TM_BASE = "https://api.mail.tm"
MAIL_GW_BASE = "https://api.mail.gw"
GUERRILLA_BASE = "https://api.guerrillamail.com/ajax.php"
HME_API_HOST = "p68-maildomainws.icloud.com"  # iCloud HME API
HME_IMAP_HOST = "imap.mail.me.com"  # iCloud IMAP
HME_IMAP_PORT = 993
# Microsoft Graph API (Hotmail / Outlook / Live)
MS_TOKEN_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
MS_GRAPH_BASE = "https://graph.microsoft.com/v1.0"
# Default client_id: Microsoft's own Outlook mobile app (public client, no secret needed)
MS_DEFAULT_CLIENT_ID = "27922004-5251-4030-b22d-91ecd9a37ea4"
DEEPSEEK_SIGNUP_URL = "https://platform.deepseek.com/sign_up"
DEEPSEEK_LOGIN_URL = "https://chat.deepseek.com/sign_in"  # Chat subdomain for real token
DEEPSEEK_CHAT_URL = "https://chat.deepseek.com"
POLL_TIMEOUT = 120
POLL_INTERVAL = 4

# ── API server defaults ───────────────────────────────────────────────────────
DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_API_KEY = "nowsthetime"


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — Temporary Email abstraction
# ══════════════════════════════════════════════════════════════════════════════


class TempMailBackend(ABC):
    """Abstract base for a disposable mail provider."""

    address: str = ""
    password: str = ""

    @abstractmethod
    def create_account(self) -> tuple[str, str]:
        """Create inbox. Returns (email_address, password)."""

    @abstractmethod
    def poll_for_otp(self, timeout: int = POLL_TIMEOUT, interval: int = POLL_INTERVAL) -> str:
        """Block until a 6-digit OTP is found. Raises TimeoutError."""

    @abstractmethod
    def poll_for_verification_link(
        self, timeout: int = 90, interval: int = POLL_INTERVAL
    ) -> str | None:
        """Block until a verification link is found. Returns None on timeout."""

    def seed_seen_ids(self) -> None:
        """Snapshot inbox to ignore pre-existing emails. Call before Send Code."""
        pass


# ── Guerrilla Mail backend ────────────────────────────────────────────────────


class GuerrillaMailAdapter(TempMailBackend):
    """
    Guerrilla Mail via https://api.guerrillamail.com/ajax.php
    No account creation required — just GET an address and poll.
    Proven to work from this environment.
    """

    def __init__(self) -> None:
        self.s = requests.Session()
        self.s.headers["User-Agent"] = (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/124.0.0.0"
        )
        self.address = ""
        self.password = ""
        self._sid = ""
        self._seen: set[str] = set()

    def create_account(self) -> tuple[str, str]:
        r = self.s.get(GUERRILLA_BASE + "?f=get_email_address", timeout=15)
        r.raise_for_status()
        data = r.json()
        self.address = data["email_addr"]
        self._sid = data["sid_token"]
        self.password = ""
        log.info(f"  ✓ Guerrilla Mail inbox: {self.address}")
        return self.address, self.password

    def _fetch_messages(self) -> list[dict]:
        r = self.s.get(
            GUERRILLA_BASE + f"?f=get_email_list&offset=0&sid_token={self._sid}",
            timeout=15,
        )
        r.raise_for_status()
        return r.json().get("list", [])

    def _fetch_body(self, mail_id: str) -> str:
        r = self.s.get(
            GUERRILLA_BASE + f"?f=fetch_email&email_id={mail_id}&sid_token={self._sid}",
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("mail_body", "") or data.get("mail_excerpt", "")

    def poll_for_otp(self, timeout: int = POLL_TIMEOUT, interval: int = POLL_INTERVAL) -> str:
        log.info(f"  Polling Guerrilla inbox for OTP (up to {timeout}s)...")
        deadline = time.time() + timeout
        while time.time() < deadline:
            for msg in self._fetch_messages():
                mid = str(msg.get("mail_id", ""))
                subj = msg.get("mail_subject", "")
                if mid in self._seen or mid == "0":
                    continue
                self._seen.add(mid)
                log.info(f"  📧 New email: subject='{subj}'")
                excerpt = msg.get("mail_excerpt", "")
                otp = _extract_otp(excerpt)
                if otp:
                    log.info(f"  ✓ OTP from excerpt: {otp}")
                    return otp
                body = self._fetch_body(mid)
                otp = _extract_otp(body)
                if otp:
                    log.info(f"  ✓ OTP from body: {otp}")
                    return otp
            time.sleep(interval)
        raise TimeoutError(f"No OTP received in {timeout}s.")

    def seed_seen_ids(self) -> None:
        """Pre-populate _seen with current message IDs."""
        log.info("  Seeding Guerrilla Mail seen IDs...")
        try:
            for msg in self._fetch_messages():
                self._seen.add(str(msg.get("mail_id", "")))
            log.info(f"  ✓ Seeded {len(self._seen)} existing message ID(s).")
        except Exception as e:
            log.warning(f"  Could not seed seen IDs: {e}")

    def poll_for_verification_link(
        self, timeout: int = 90, interval: int = POLL_INTERVAL
    ) -> str | None:
        log.info(f"  Polling Guerrilla inbox for verification link (up to {timeout}s)...")
        deadline = time.time() + timeout
        while time.time() < deadline:
            for msg in self._fetch_messages():
                mid = str(msg.get("mail_id", ""))
                subj = msg.get("mail_subject", "")
                if mid in self._seen or mid == "0":
                    continue
                self._seen.add(mid)
                log.info(f"  📧 New email: subject='{subj}'")
                excerpt = msg.get("mail_excerpt", "")
                link = _extract_verification_link(excerpt)
                if link:
                    return link
                body = self._fetch_body(mid)
                link = _extract_verification_link(body)
                if link:
                    return link
            time.sleep(interval)
        log.info(f"  No verification link found in {timeout}s.")
        return None


# ── iCloud Hide My Email backend ─────────────────────────────────────────────


class ICloudHMEAdapter(TempMailBackend):
    """
    iCloud Hide My Email — generates a real @icloud.com alias via Apple's
    private maildomain API, then reads the forwarded OTP from your real
    ICloud inbox via IMAP.
    """

    def __init__(
        self,
        cookies: str | Path,
        imap_user: str,
        imap_pass: str,
        label: str = "deepseek-auto",
        imap_host: str = HME_IMAP_HOST,
        imap_port: int = HME_IMAP_PORT,
        hme_api_host: str = HME_API_HOST,
    ) -> None:
        if isinstance(cookies, Path) or (isinstance(cookies, str) and Path(cookies).exists()):
            cookie_path = Path(cookies)
            lines = cookie_path.read_text(encoding="utf-8").splitlines()
            non_comment = [l for l in lines if l.strip() and not l.strip().startswith("//")]
            self._cookies = non_comment[0].strip() if non_comment else ""
        else:
            self._cookies = str(cookies).strip()

        self._imap_user = imap_user
        self._imap_pass = imap_pass
        self._label = label
        self._imap_host = imap_host
        self._imap_port = imap_port
        self._hme_host = hme_api_host
        self.address = ""
        self.password = ""
        self._anonymous_id = ""
        self._created_after: float = 0.0
        self._seen_uids: set[bytes] = set()

    def _hme_request(self, method: str, path: str, **kwargs) -> dict:
        import ssl
        import certifi
        import aiohttp

        url = f"https://{self._hme_host}{path}"
        params = {
            "clientBuildNumber": "2626Build17",
            "clientMasteringNumber": "2626Build17",
            "clientId": "",
            "dsid": "",
        }
        headers = {
            "Connection": "keep-alive",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/150.0.0.0"
            ),
            "Content-Type": "text/plain",
            "Accept": "*/*",
            "Origin": "https://www.icloud.com",
            "Referer": "https://www.icloud.com/",
            "Sec-Fetch-Site": "same-site",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Dest": "empty",
            "Accept-Language": "en-US,en;q=0.7",
            "Cookie": self._cookies,
        }

        async def _run() -> dict:
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_ctx)
            async with aiohttp.ClientSession(
                headers=headers,
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as session:
                async with session.request(method, url, params=params, **kwargs) as resp:
                    return await resp.json()

        try:
            return asyncio.run(_run())
        except Exception as e:
            return {"error": 1, "reason": str(e)}

    def _generate_email(self) -> str:
        result = self._hme_request("POST", "/v1/hme/generate", json={"langCode": "en-us"})
        if result.get("error"):
            raise RuntimeError(f"HME generate failed: {result.get('reason', result)}")
        hme = result.get("hme") or (result.get("result", {}) or {}).get("hme", "")
        if not hme:
            hme = _search_dict_for_str(result, "hme") or ""
        if not hme:
            raise RuntimeError(f"HME generate: no 'hme' in response: {result}")
        return hme

    def _reserve_email(self, hme: str) -> str:
        result = self._hme_request(
            "POST",
            "/v1/hme/reserve",
            json={"hme": hme, "label": self._label, "note": "auto deepseek signup"},
        )
        if result.get("error"):
            raise RuntimeError(f"HME reserve failed: {result.get('reason', result)}")
        anon_id = result.get("anonymousId") or (result.get("result", {}) or {}).get(
            "anonymousId", ""
        )
        return anon_id or ""

    def create_account(self) -> tuple[str, str]:
        log.info("  Generating iCloud Hide My Email alias...")
        hme = self._generate_email()
        log.info(f"  Generated: {hme}")
        anon_id = self._reserve_email(hme)
        self.address = hme
        self.password = ""
        self._anonymous_id = anon_id
        self._created_after = time.time()
        log.info(
            f"  ✓ Reserved HME alias: {hme}"
        )
        return hme, ""

    def _imap_connect(self) -> imaplib.IMAP4_SSL:
        mb = imaplib.IMAP4_SSL(self._imap_host, self._imap_port)
        mb.login(self._imap_user, self._imap_pass)
        mb.select("INBOX")
        return mb

    def _imap_search_new(self, mb: imaplib.IMAP4_SSL) -> list[bytes]:
        import email.utils
        from datetime import datetime, timezone

        dt = datetime.fromtimestamp(self._created_after, tz=timezone.utc)
        since_str = dt.strftime("%d-%b-%Y")
        _, data = mb.uid("search", None, f'SINCE "{since_str}"')
        uids = data[0].split() if data and data[0] else []
        return uids

    def _imap_fetch_body(self, mb: imaplib.IMAP4_SSL, uid: bytes) -> str:
        _, msg_data = mb.uid("fetch", uid, "(RFC822)")
        raw = b""
        for part in msg_data or []:
            if isinstance(part, tuple):
                raw += part[1]
        if not raw:
            return ""
        msg = BytesParser(policy=policy.default).parsebytes(raw)
        subject = str(msg.get("Subject", ""))
        body = _imap_get_text(msg)
        return f"{subject}\n{body}"

    def seed_seen_ids(self) -> None:
        """Snapshot current UIDs."""
        log.info("  Seeding iCloud IMAP inbox...")
        try:
            mb = self._imap_connect()
            uids = self._imap_search_new(mb)
            self._seen_uids.update(uids)
            mb.logout()
            log.info(f"  ✓ Seeded {len(self._seen_uids)} existing email(s).")
        except Exception as e:
            log.warning(f"  Could not seed iCloud inbox: {e}")

    def poll_for_otp(self, timeout: int = POLL_TIMEOUT, interval: int = POLL_INTERVAL) -> str:
        log.info(f"  Polling iCloud IMAP inbox for OTP (up to {timeout}s)...")
        deadline = time.time() + timeout
        mb: imaplib.IMAP4_SSL | None = None
        try:
            mb = self._imap_connect()
        except Exception as e:
            raise RuntimeError(f"IMAP connection failed: {e}") from e

        try:
            while time.time() < deadline:
                try:
                    mb.noop()
                    uids = self._imap_search_new(mb)
                except Exception:
                    try: mb.logout()
                    except: pass
                    mb = self._imap_connect()
                    uids = self._imap_search_new(mb)

                for uid in uids:
                    if uid in self._seen_uids:
                        continue
                    self._seen_uids.add(uid)
                    body = self._imap_fetch_body(mb, uid)
                    otp = _extract_otp(body)
                    if otp:
                        log.info(f"  ✓ OTP found in iCloud inbox: {otp}")
                        return otp
                time.sleep(interval)
        finally:
            try: mb.logout()
            except: pass
        raise TimeoutError(f"No OTP found in iCloud inbox in {timeout}s.")

    def poll_for_verification_link(
        self, timeout: int = 90, interval: int = POLL_INTERVAL
    ) -> str | None:
        log.info(f"  Polling iCloud IMAP for verification link...")
        deadline = time.time() + timeout
        mb: imaplib.IMAP4_SSL | None = None
        try:
            mb = self._imap_connect()
        except Exception as e:
            return None
        try:
            while time.time() < deadline:
                try:
                    mb.noop()
                    uids = self._imap_search_new(mb)
                except Exception:
                    try: mb.logout()
                    except: pass
                    mb = self._imap_connect()
                    uids = self._imap_search_new(mb)

                for uid in uids:
                    if uid in self._seen_uids:
                        continue
                    self._seen_uids.add(uid)
                    body = self._imap_fetch_body(mb, uid)
                    link = _extract_verification_link(body)
                    if link:
                        return link
                time.sleep(interval)
        finally:
            try: mb.logout()
            except: pass
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Microsoft Graph API mail backend
# ══════════════════════════════════════════════════════════════════════════════


def parse_graphapi_credential(cred_str: str) -> dict:
    parts = cred_str.strip().split("|")
    if len(parts) < 3:
        raise ValueError("Graph API credential must have at least 3 segments: 'email|pass|refresh_token[|client_id]'.")
    email = parts[0].strip()
    if not email:
        raise ValueError("Email cannot be empty.")
    password = parts[1].strip()
    refresh_token = parts[2].strip()
    if not refresh_token:
        raise ValueError("refresh_token cannot be empty.")
    client_id = parts[3].strip() if len(parts) >= 4 and parts[3].strip() else MS_DEFAULT_CLIENT_ID
    return {"email": email, "password": password, "refresh_token": refresh_token, "client_id": client_id}


class GraphAPIMailAdapter(TempMailBackend):
    def __init__(
        self,
        email: str,
        password: str,
        refresh_token: str,
        client_id: str = MS_DEFAULT_CLIENT_ID,
    ) -> None:
        import time
        # Use nanoseconds so parallel workers spawned in the same second get unique tags
        self.address = email.replace("@", f"+ds{time.time_ns() // 1_000_000}@")
        self.password = password
        self._refresh_tok = refresh_token
        self._client_id = client_id
        self._access_tok = ""
        self._seen_ids: set[str] = set()
        self.s = requests.Session()
        self.s.headers["User-Agent"] = (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/124.0.0.0"
        )

    def _refresh_access_token(self) -> str:
        r = self.s.post(
            MS_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "client_id": self._client_id,
                "refresh_token": self._refresh_tok,
                "scope": "https://graph.microsoft.com/Mail.Read offline_access",
            },
            timeout=20,
        )
        if r.status_code != 200:
            raise RuntimeError(f"MS token refresh failed ({r.status_code}): {r.text[:300]}")
        data = r.json()
        self._access_tok = data["access_token"]
        if "refresh_token" in data:
            self._refresh_tok = data["refresh_token"]
        return self._access_tok

    def _graph_get(self, path: str, **kwargs) -> dict:
        if not self._access_tok:
            self._refresh_access_token()
        headers = {"Authorization": f"Bearer {self._access_tok}"}
        r = self.s.get(f"{MS_GRAPH_BASE}{path}", headers=headers, timeout=20, **kwargs)
        if r.status_code == 401:
            self._refresh_access_token()
            headers = {"Authorization": f"Bearer {self._access_tok}"}
            r = self.s.get(f"{MS_GRAPH_BASE}{path}", headers=headers, timeout=20, **kwargs)
        r.raise_for_status()
        return r.json()

    def create_account(self) -> tuple[str, str]:
        self._refresh_access_token()
        self._graph_get("/me/messages", params={"$top": 1, "$select": "id"})
        return self.address, self.password

    def seed_seen_ids(self) -> None:
        log.info("  Seeding Graph API inbox...")
        try:
            data = self._graph_get("/me/messages", params={"$top": 20, "$select": "id"})
            for msg in data.get("value", []):
                self._seen_ids.add(msg["id"])
            log.info(f"  ✓ Seeded {len(self._seen_ids)} message ID(s).")
        except Exception as e:
            log.warning(f"  Seed failed: {e}")

    def _fetch_messages(self, top: int = 10) -> list[dict]:
        """Fetch the latest *top* messages from the Graph API inbox."""
        data = self._graph_get(
            "/me/messages",
            params={
                "$top": top,
                # Include toRecipients so we can route OTPs to the correct plus-address
                "$select": "id,subject,body,bodyPreview,toRecipients",
            },
        )
        return data.get("value", [])

    def _body_of(self, msg: dict) -> str:
        import html as html_mod

        subject = msg.get("subject", "")
        body = msg.get("body", {})
        content = body.get("content", "") or msg.get("bodyPreview", "")
        ctype = body.get("contentType", "text")
        if ctype == "html":
            import re as _re

            content = _re.sub(r"<[^>]+>", " ", html_mod.unescape(content))
        return f"{subject}\n{content}"

    @staticmethod
    def _to_addresses(msg: dict) -> list[str]:
        """Return lowercase list of 'To:' recipient addresses from a message."""
        recipients = []
        for r in msg.get("toRecipients", []):
            addr = r.get("emailAddress", {}).get("address", "")
            if addr:
                recipients.append(addr.lower())
        return recipients

    def poll_for_otp(
        self,
        timeout: int = POLL_TIMEOUT,
        interval: int = POLL_INTERVAL,
        to_filter: str | None = None,
    ) -> str:
        """
        Block until a 6-digit OTP arrives.

        Args:
            to_filter: If set (e.g. "user+ds123@hotmail.com"), only accept OTP
                       emails addressed to this exact plus-address.  This
                       enables safe parallel account creation: each worker
                       listens only for its own OTP.
        """
        filter_lower = to_filter.lower() if to_filter else None
        log.info(
            f"  Polling Graph API inbox for OTP (up to {timeout}s)"
            + (f" [to={filter_lower}]" if filter_lower else "") + "..."
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                messages = self._fetch_messages(top=20)
            except Exception as e:
                log.warning(f"  Graph API fetch error: {e} — retrying...")
                time.sleep(interval)
                continue
            for msg in messages:
                mid = msg.get("id", "")
                if mid in self._seen_ids:
                    continue
                # Plus-address routing: skip messages not addressed to our tag
                if filter_lower:
                    to_addrs = self._to_addresses(msg)
                    if to_addrs and filter_lower not in to_addrs:
                        log.debug(f"  ⏭  Skipping email for {to_addrs} (want {filter_lower})")
                        continue  # Not our OTP — leave unseen so sibling worker can pick it up
                self._seen_ids.add(mid)
                subj = msg.get("subject", "")
                log.info(f"  📧 New email: subject='{subj}'")
                body = self._body_of(msg)
                otp = _extract_otp(body)
                if otp:
                    log.info(f"  ✓ OTP found: {otp}")
                    return otp
            time.sleep(interval)
        raise TimeoutError(f"No OTP found in Graph API inbox in {timeout}s.")

    def poll_for_verification_link(
        self, timeout: int = 90, interval: int = POLL_INTERVAL
    ) -> str | None:
        log.info(f"  Polling Graph API inbox for verification link (up to {timeout}s)...")
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                messages = self._fetch_messages(top=10)
            except Exception as e:
                log.warning(f"  Graph API fetch error: {e} — retrying...")
                time.sleep(interval)
                continue
            for msg in messages:
                mid = msg.get("id", "")
                if mid in self._seen_ids:
                    continue
                self._seen_ids.add(mid)
                subj = msg.get("subject", "")
                log.info(f"  📧 New email: subject='{subj}'")
                body = self._body_of(msg)
                link = _extract_verification_link(body)
                if link:
                    log.info(f"  ✓ Verification link found")
                    return link
            time.sleep(interval)
        log.info(f"  No verification link found in {timeout}s.")
        return None


def _imap_get_text(msg) -> str:
    """Extract plaintext (or stripped HTML) from an email.message.Message."""
    import html as html_mod
    import re as _re

    plain_parts: list[str] = []
    html_parts: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_disposition() == "attachment":
                continue
            ctype = part.get_content_type()
            try:
                content = part.get_content()
            except Exception:
                continue
            if not isinstance(content, str):
                continue
            if ctype == "text/plain":
                plain_parts.append(content)
            elif ctype == "text/html":
                stripped = _re.sub(r"<[^>]+>", " ", html_mod.unescape(content))
                html_parts.append(stripped)
    else:
        try:
            content = msg.get_content()
        except Exception:
            content = ""
        if isinstance(content, str):
            if msg.get_content_type() == "text/html":
                stripped = _re.sub(r"<[^>]+>", " ", html_mod.unescape(content))
                html_parts.append(stripped)
            else:
                plain_parts.append(content)
    return "\n".join(plain_parts) or "\n".join(html_parts)


def _search_dict_for_str(data, key: str) -> str | None:
    """Recursively find `key` in a nested dict/list and return its string value."""
    if isinstance(data, dict):
        if key in data and isinstance(data[key], str):
            return data[key]
        for v in data.values():
            r = _search_dict_for_str(v, key)
            if r:
                return r
    if isinstance(data, list):
        for item in data:
            r = _search_dict_for_str(item, key)
            if r:
                return r
    return None


# ── Hydra-style backend (mail.tm / mail.gw) ───────────────────────────────────


class HydraMailAdapter(TempMailBackend):
    """
    Mail.tm / Mail.gw — both use the same Hydra/JSON-LD REST API.
    """

    def __init__(self, base_url: str = MAIL_TM_BASE) -> None:
        self.base = base_url.rstrip("/")
        self.s = requests.Session()
        self.s.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )
        self.address = ""
        self.password = ""
        self._token = ""

    @staticmethod
    def _rstr(n: int = 10) -> str:
        return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))

    def _get(self, path: str) -> dict:
        r = self.s.get(f"{self.base}{path}", timeout=15)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, **kw) -> requests.Response:
        return self.s.post(f"{self.base}{path}", timeout=15, **kw)

    def _get_domain(self) -> str:
        data = self._get("/domains")
        members = data if isinstance(data, list) else data.get("hydra:member", [])
        if not members:
            raise RuntimeError(f"{self.base}: no active domains available.")
        return members[0]["domain"]

    def create_account(self) -> tuple[str, str]:
        domain = self._get_domain()
        uname = self._rstr(10)
        password = self._rstr(12) + "Aa1!"
        address = f"{uname}@{domain}"
        log.info(f"  Creating temp email via {self.base}: {address}")
        r = self._post("/accounts", json={"address": address, "password": password})
        if r.status_code not in (200, 201):
            raise RuntimeError(f"Mail account creation failed ({r.status_code}): {r.text[:200]}")
        self.address = address
        self.password = password
        # Authenticate immediately
        r2 = self._post("/token", json={"address": address, "password": password})
        r2.raise_for_status()
        self._token = r2.json()["token"]
        self.s.headers["Authorization"] = f"Bearer {self._token}"
        log.info(f"  ✓ Temp email created and authenticated.")
        return address, password

    def _messages(self) -> list[dict]:
        data = self._get("/messages")
        return data if isinstance(data, list) else data.get("hydra:member", [])

    def _message_body(self, mid: str) -> str:
        full = self._get(f"/messages/{mid}")
        texts = full.get("text") or []
        htmls = full.get("html") or []
        body = " ".join(texts if isinstance(texts, list) else [texts])
        body += " ".join(htmls if isinstance(htmls, list) else [htmls])
        return body

    def poll_for_otp(self, timeout: int = POLL_TIMEOUT, interval: int = POLL_INTERVAL) -> str:
        log.info(f"  Polling inbox for OTP (up to {timeout}s)...")
        deadline = time.time() + timeout
        seen: set[str] = set()
        while time.time() < deadline:
            for msg in self._messages():
                mid = msg["id"]
                if mid in seen:
                    continue
                seen.add(mid)
                log.info(f"  📧 New email: subject='{msg.get('subject', '')}'")
                otp = _extract_otp(msg.get("intro", ""))
                if otp:
                    return otp
                otp = _extract_otp(self._message_body(mid))
                if otp:
                    return otp
            time.sleep(interval)
        raise TimeoutError(f"No OTP received in {timeout}s.")

    def poll_for_verification_link(
        self, timeout: int = 90, interval: int = POLL_INTERVAL
    ) -> str | None:
        log.info(f"  Polling inbox for verification link (up to {timeout}s)...")
        deadline = time.time() + timeout
        seen: set[str] = set()
        while time.time() < deadline:
            for msg in self._messages():
                mid = msg["id"]
                if mid in seen:
                    continue
                seen.add(mid)
                log.info(f"  📧 New email: subject='{msg.get('subject', '')}'")
                link = _extract_verification_link(msg.get("intro", ""))
                if link:
                    return link
                link = _extract_verification_link(self._message_body(mid))
                if link:
                    return link
            time.sleep(interval)
        log.info(f"  No verification link found in {timeout}s.")
        return None


# ── Factory with auto-failover ─────────────────────────────────────────────────

# Module-level storage for iCloud / Graph API credentials (populated by main)
_ICLOUD_KWARGS: dict = {}
_GRAPHAPI_POOL: list[dict] = []  # list of parsed credential dicts
_GRAPHAPI_IDX: int = 0  # round-robin index into _GRAPHAPI_POOL


def _create_mail_backend(service: str) -> TempMailBackend:
    """
    Create and test a TempMailBackend for `service`.
    On network failure, try the next provider in the failover chain.

    Priority chain:
      'icloud'     → ICloudHMEAdapter     (requires _ICLOUD_KWARGS)
      'graphapi'   → GraphAPIMailAdapter  (round-robins through _GRAPHAPI_POOL)
      'guerrilla'  → GuerrillaMailAdapter
      'mailtm'     → HydraMailAdapter(mail.tm) → fallback to guerrilla
      'mailgw'     → HydraMailAdapter(mail.gw) → fallback to guerrilla
      'auto'       → icloud → graphapi → guerrilla → mailtm → mailgw
    """
    global _GRAPHAPI_IDX
    candidates: list[tuple[str, object]] = []

    def _next_graphapi_factory():
        """Return a factory that picks the next account from the pool round-robin."""
        global _GRAPHAPI_IDX
        if not _GRAPHAPI_POOL:
            raise RuntimeError(
                "graphapi service selected but no credentials provided. "
                "Pass --graphapi-creds 'email|pass|refresh_token[|client_id]'."
            )
        # Snapshot index so the lambda captures the right value
        idx = _GRAPHAPI_IDX % len(_GRAPHAPI_POOL)
        _GRAPHAPI_IDX = idx + 1
        kwargs = _GRAPHAPI_POOL[idx]
        log.info(
            f"  Graph API pool: using account {idx + 1}/{len(_GRAPHAPI_POOL)} ({kwargs['email']})"
        )
        return lambda: GraphAPIMailAdapter(**kwargs)

    if service == "icloud":
        if not _ICLOUD_KWARGS:
            raise RuntimeError(
                "iCloud service selected but no credentials provided. "
                "Pass --icloud-cookies, --icloud-imap-user, --icloud-imap-pass."
            )
        candidates = [("icloud", lambda: ICloudHMEAdapter(**_ICLOUD_KWARGS))]
    elif service == "graphapi":
        candidates = [("graphapi", _next_graphapi_factory())]
    elif service == "guerrilla":
        candidates = [("guerrilla", GuerrillaMailAdapter)]
    elif service == "mailtm":
        candidates = [
            ("mailtm", lambda: HydraMailAdapter(MAIL_TM_BASE)),
            ("guerrilla", GuerrillaMailAdapter),
        ]
    elif service == "mailgw":
        candidates = [
            ("mailgw", lambda: HydraMailAdapter(MAIL_GW_BASE)),
            ("guerrilla", GuerrillaMailAdapter),
        ]
    else:  # 'auto'
        icloud_factory = (
            [("icloud", lambda: ICloudHMEAdapter(**_ICLOUD_KWARGS))] if _ICLOUD_KWARGS else []
        )
        graphapi_factory = [("graphapi", _next_graphapi_factory())] if _GRAPHAPI_POOL else []
        candidates = [
            *icloud_factory,
            *graphapi_factory,
            ("guerrilla", GuerrillaMailAdapter),
            ("mailtm", lambda: HydraMailAdapter(MAIL_TM_BASE)),
            ("mailgw", lambda: HydraMailAdapter(MAIL_GW_BASE)),
        ]

    last_error: Exception | None = None
    for name, factory in candidates:
        try:
            backend = factory()  # type: ignore[operator]
            backend.create_account()
            log.info(f"  ✓ Mail backend '{name}' connected.")
            return backend
        except Exception as e:
            log.warning(f"  ⚠️  Mail backend '{name}' failed: {e} — trying next...")
            last_error = e

    raise RuntimeError(f"All mail backends failed. Last error: {last_error}")


# Legacy alias used by tests (kept for backwards compatibility)
class TempMail(HydraMailAdapter):
    """Alias for HydraMailAdapter — used by legacy callers and tests."""

    def __init__(self, base_url: str = MAIL_TM_BASE) -> None:
        super().__init__(base_url)

    def authenticate(self) -> str:
        """Legacy compat: authenticate separately (create_account now authenticates inline)."""
        if self._token:
            return self._token
        r = self._post("/token", json={"address": self.address, "password": self.password})
        r.raise_for_status()
        self._token = r.json()["token"]
        self.s.headers["Authorization"] = f"Bearer {self._token}"
        return self._token

    # Expose .token attribute expected by some test code
    @property
    def token(self) -> str:
        return self._token

    @token.setter
    def token(self, v: str) -> None:
        self._token = v


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers: OTP + link extraction
# ══════════════════════════════════════════════════════════════════════════════


def _extract_otp(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"(?<!\d)(\d{6})(?!\d)", text)
    return m.group(1) if m else None


def _extract_verification_link(text: str) -> str | None:
    """
    Find a DeepSeek email-verification link.
    DeepSeek sends links containing 'verify', 'confirm', 'activate', or 'deepseek'.
    """
    if not text:
        return None
    keywords = ["verify", "verif", "confirm", "activat", "deepseek", "account"]
    urls = re.findall(r'https://[^\s"\'><\]]+', text)
    for url in urls:
        lower = url.lower()
        if any(kw in lower for kw in keywords):
            url = url.rstrip(".,;)")
            log.info(f"  🔗 Verification link found: {url[:80]}{'...' if len(url) > 80 else ''}")
            return url
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Token Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _scan_storage_for_token(driver) -> str | None:
    """Scan localStorage + sessionStorage + cookies for a DeepSeek auth token."""
    try:
        items = driver.execute_script("""
            return (function() {
                const out = [];
                // Primary key
                const ut = localStorage.getItem('userToken');
                if (ut && ut.length > 20) out.push({k: 'userToken', v: ut});

                // Scan all localStorage
                for (let i = 0; i < localStorage.length; i++) {
                    const k = localStorage.key(i);
                    if (k === 'userToken') continue;
                    const v = localStorage.getItem(k);
                    if (v && v.length > 30) {
                        out.push({k: k, v: v});
                    }
                }
                // sessionStorage
                for (let i = 0; i < sessionStorage.length; i++) {
                    const k = sessionStorage.key(i);
                    const v = sessionStorage.getItem(k);
                    if (v && v.length > 30) {
                        out.push({k: 'ss:' + k, v: v});
                    }
                }
                // Cookies
                document.cookie.split(';').forEach(c => {
                    const parts = c.trim().split('=');
                    if (parts.length >= 2 && parts[0].toLowerCase().includes('token')) {
                        out.push({k: 'cookie:' + parts[0], v: parts.slice(1).join('=')});
                    }
                });
                return out;
            })()
        """)
        for item in items or []:
            key = item.get("k", "")
            val = item.get("v", "")
            if key == "userToken" and val and len(val) > 20:
                if val.startswith("{"):
                    try:
                        parsed = json.loads(val)
                        if "value" in parsed:
                            if isinstance(parsed["value"], str) and len(parsed["value"]) > 20:
                                t = parsed["value"]
                                log.info(f"  🔑 Token found in localStorage['userToken'] (JSON value): {t[:12]}...")
                                return t
                            else:
                                continue # Invalid/null value, keep polling
                    except Exception as e:
                        log.warning(f"  Failed to parse userToken JSON: {e}")
                else:
                    log.info(f"  🔑 Token found in localStorage['userToken']: {val[:12]}...")
                    return val


        found_keys = [item.get("k") for item in (items or [])]
        if found_keys:
            log.info(f"  [Scan] No token found yet. Storage keys present: {', '.join(k for k in found_keys if k)}")
        else:
            log.info("  [Scan] Storage is completely empty!")
    except Exception as e:
        log.error(f"  Storage scan error: {e}")
    return None


def _search_dict_for_token(data) -> str | None:
    """Recursively search a dict/list for a DeepSeek token."""
    if isinstance(data, str):
        if len(data) >= 40:
            return data
        return None
    if isinstance(data, dict):
        for key in (
            "token",
            "userToken",
            "user_token",
            "accessToken",
            "access_token",
            "auth_token",
            "value",
        ):
            if key in data and isinstance(data[key], str) and len(data[key]) >= 20:
                return data[key]
        for v in data.values():
            r = _search_dict_for_token(v)
            if r:
                return r
    if isinstance(data, list):
        for item in data:
            r = _search_dict_for_token(item)
            if r:
                return r
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — Token Submission to deepseek-python-api
# ══════════════════════════════════════════════════════════════════════════════


def submit_token_to_api(
    token: str,
    name: str,
    api_url: str = DEFAULT_API_URL,
    api_key: str = DEFAULT_API_KEY,
) -> bool:
    """
    POST the DeepSeek auth token to deepseek-python-api's management endpoint.

    POST /v0/management/tokens
    Authorization: Bearer <api_key>
    {"token": "<token>", "name": "<name>", "enabled": true, "check": false}

    Returns True on success, False on any error.
    """
    url = f"{api_url.rstrip('/')}/v0/management/tokens"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "token": token,
        "name": name,
        "enabled": True,
        "check": False,
    }
    log.info(f"  ↑ Submitting token to deepseek-python-api: {url}")
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=20)
        data = r.json() if r.content else {}
        if r.status_code in (200, 201):
            tok_id = data.get("id", "?")
            status = data.get("status", "?")
            log.info(f"  ✅ Token submitted — id={tok_id}, status={status}")
            return True
        elif r.status_code == 409:
            log.warning(f"  ⚠️  Token already exists in API (duplicate).")
            return True
        else:
            log.warning(f"  ⚠️  Token submission failed ({r.status_code}): {data}")
            return False
    except Exception as e:
        log.error(f"  ❌ Token submission error: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — SeleniumBase UC Browser Automation
# ══════════════════════════════════════════════════════════════════════════════



class DeepSeekAPI:
    """Pure API-based DeepSeek account creation (Bypasses Selenium)."""
    def __init__(self, debug: bool = False):
        self.debug = debug
        import requests
        self.client = requests.Session()
        self.client.headers.update({
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
            "content-type": "application/json",
            "origin": "https://platform.deepseek.com",
            "referer": "https://platform.deepseek.com/sign_up",
        })

    def _generate_device_id(self) -> str:
        import secrets
        import base64
        return base64.b64encode(secrets.token_bytes(64)).decode('utf-8')

    def signup(self, email: str, password: str, otp_callback, seed_callback=None) -> str | None:
        """
        Register a new DeepSeek account via platform API.
        Returns None — caller must use signin() afterwards to get the chat token
        (the platform register token != chat token; WAF on chat.deepseek.com
        requires a browser for the first login).
        """
        import sys, os, json, base64, asyncio, time
        try:
            from deepseek_python_api.pow import DeepSeekHashSolver, Challenge
        except ImportError:
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
            from deepseek_python_api.pow import DeepSeekHashSolver, Challenge

        if seed_callback: seed_callback()
        device_id = self._generate_device_id()
        import logging
        log = logging.getLogger("deepseek_signup")
        log.info(f"Starting API signup → {email}")

        # Step 1: Request OTP
        res = self.client.post(
            "https://platform.deepseek.com/auth-api/v0/users/create_email_verification_code",
            json={"email": email, "turnstile_token": "", "locale": "en_US",
                  "device_id": device_id, "scenario": "register"},
        )
        if res.status_code != 200 or res.json().get("code") != 0:
            log.error(f"Failed to send OTP via API: {res.text}")
            return None

        log.info("OTP sent via API. Waiting for email...")
        try:
            otp = otp_callback()
        except TimeoutError:
            return None

        # Step 2: Solve PoW challenge
        res = self.client.post(
            "https://platform.deepseek.com/auth-api/v0/users/create_guest_challenge",
            json={"target_path": "/auth-api/v0/users/register"},
        )
        challenge = Challenge.from_payload(res.json()["data"]["biz_data"]["guest_challenge"])
        solver = DeepSeekHashSolver()
        b64_full = asyncio.run(solver.create_answer(challenge, "/auth-api/v0/users/register"))
        full_pow = json.loads(base64.b64decode(b64_full).decode())
        pow_header = base64.b64encode(
            json.dumps({"salt": full_pow["salt"], "answer": full_pow["answer"]},
                       separators=(',', ':')).encode()
        ).decode()
        solver.close()

        # Step 3: Register
        self.client.headers["x-ds-guest-pow-response"] = pow_header
        res = self.client.post(
            "https://platform.deepseek.com/auth-api/v0/users/register",
            json={
                "locale": "en_US", "region": "VN",
                "payload": {"email": email, "email_verification_code": otp,
                            "password": password},
                "device_id": device_id, "os": "web",
            },
        )
        if res.status_code != 200 or res.json().get("code") != 0:
            log.error(f"Registration failed: {res.text}")
            return None
        log.info("✓ Account registered via API!")
        # Return None — signin() is needed to get the chat.deepseek.com token.
        return None

    def signin(self, email: str, password: str, otp_callback=None, seed_callback=None) -> str | None:
        """
        Sign in to chat.deepseek.com using a headless Selenium browser.
        Pure HTTP login is blocked by AWS WAF (202 JS challenge), so we use a
        lightweight SeleniumBase session just for the sign-in step.
        Returns the chat token string or None.
        """
        import logging
        log = logging.getLogger("deepseek_signup")
        log.info(f"Signing in (browser-assisted, headless): {email}")
        sb_browser = DeepSeekBrowserSB(headless=True, debug=self.debug, manual_captcha=False)
        return sb_browser.signin(
            email=email,
            password=password,
            otp_callback=otp_callback,
            seed_callback=seed_callback,
        )

class DeepSeekBrowserSB:

    """
    SeleniumBase UC (undetected-chromedriver) signup automation for platform.deepseek.com.

    DeepSeek's registration flow:
      1. Visit https://platform.deepseek.com/sign_up
      2. Enter email + password
      3. Click "Send Code" → enter 6-digit OTP from email
      4. Check ToS checkbox → Submit
      5. Token stored in localStorage['userToken']
    """

    def __init__(
        self, headless: bool = False, debug: bool = False, manual_captcha: bool = False
    ) -> None:
        self.headless = headless
        self.debug = debug
        self.manual_captcha = manual_captcha
        self._shot_idx = 0

    def _shot(self, sb, label: str) -> None:
        if not self.debug:
            return
        fname = f"ds_debug_{self._shot_idx:02d}_{label}.png"
        self._shot_idx += 1
        try:
            sb.save_screenshot(fname)
            log.debug(f"  Screenshot: {fname}")
        except Exception:
            pass

    # ── CAPTCHA Detection & Handling ──────────────────────────────────────────

    def _detect_captcha(self, sb) -> str | None:
        """Detect common CAPTCHAs. Returns a string type if found, else None."""
        try:
            result = sb.execute_script("""
                (function() {
                    if (document.querySelector('iframe[src*="challenges.cloudflare.com"]'))
                        return 'cloudflare_turnstile';
                    if (document.querySelector('.g-recaptcha, iframe[src*="recaptcha"]'))
                        return 'recaptcha';
                    const body = document.body.innerText || '';
                    if (body.includes('verify you are human') || body.includes('I am not a robot'))
                        return 'human_check';
                    const els = document.querySelectorAll('[class*="captcha"], [id*="captcha"]');
                    if (els.length > 0) return 'captcha_element';
                    return null;
                })()
            """)
            return result
        except Exception:
            return None

    def _handle_captcha(self, sb) -> bool:
        """Handle detected CAPTCHA. Returns True if resolved."""
        captcha_type = self._detect_captcha(sb)
        if not captcha_type:
            log.info("  ✓ No CAPTCHA detected.")
            return True

        log.info(f"  ⚠️  CAPTCHA detected: {captcha_type}")
        self._shot(sb, "captcha_detected")

        if self.manual_captcha:
            return self._wait_for_human_captcha(sb)

        log.info("  [1/2] Trying uc_gui_click_captcha()...")
        try:
            sb.uc_gui_click_captcha()
            sb.sleep(3)
            if not self._detect_captcha(sb):
                log.info("  ✓ uc_gui_click_captcha() solved it.")
                return True
        except Exception as e:
            log.debug(f"  uc_gui_click_captcha failed: {e}")

        log.info("  [2/2] Auto-solve failed — falling back to manual mode.")
        return self._wait_for_human_captcha(sb)

    def _wait_for_human_captcha(self, sb, timeout: int = 300) -> bool:
        import sys

        banner = """
╔══════════════════════════════════════════════════════════════╗
║        🧩  CAPTCHA — HUMAN ACTION REQUIRED  🧩              ║
║                                                              ║
║  A CAPTCHA has appeared in the browser window.               ║
║                                                              ║
║  👉  Solve it in the browser, then press ENTER here.         ║
║                                                              ║
║  (You have up to 5 minutes before the session times out.)    ║
╚══════════════════════════════════════════════════════════════╝"""
        print(banner, flush=True)
        log.info("  ⏸  Waiting for human to solve CAPTCHA...")
        try:
            import select

            print("  ▶  Press ENTER when done: ", end="", flush=True)
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if ready:
                sys.stdin.readline()
            else:
                log.warning(f"  CAPTCHA wait timed out after {timeout}s — continuing.")
        except Exception:
            try:
                input("  ▶  Press ENTER when done: ")
            except Exception:
                pass
        log.info("  ▶  Resuming after human CAPTCHA...")
        self._shot(sb, "captcha_human_done")
        return True

    # ── Form Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _js_set_value(sb, selector: str, value: str) -> bool:
        """Set an input's value via React-friendly event dispatching."""
        import json as _json

        val_js = _json.dumps(value)
        sel_js = _json.dumps(selector)
        result = sb.execute_script(f"""
            (function() {{
                const el = document.querySelector({sel_js});
                if (!el) return false;
                const setter = Object.getOwnPropertyDescriptor(
                    window.HTMLInputElement.prototype, 'value').set;
                setter.call(el, {val_js});
                el.dispatchEvent(new Event('input',  {{bubbles: true}}));
                el.dispatchEvent(new Event('change', {{bubbles: true}}));
                el.dispatchEvent(new Event('blur',   {{bubbles: true}}));
                return true;
            }})()
        """)
        return bool(result)

    @staticmethod
    def _js_click(sb, selector: str) -> bool:
        """Click an element via JavaScript."""
        import json as _json

        sel_js = _json.dumps(selector)
        result = sb.execute_script(f"""
            (function() {{
                const el = document.querySelector({sel_js});
                if (!el) return false;
                el.click();
                return true;
            }})()
        """)
        return bool(result)

    # ── Main signup flow ──────────────────────────────────────────────────────

    def signup(self, email: str, password: str, otp_callback, seed_callback=None) -> str | None:
        """
        Complete DeepSeek signup flow.

        Args:
            email:         Temporary email address
            password:      Account password to register with
            otp_callback:  Callable() → str that returns the OTP when called.
            seed_callback: Optional callable() that snapshots the inbox *before*
                           clicking Send Code, so pre-existing emails are ignored.

        Returns token string or None.
        """
        from seleniumbase import SB

        token = None
        log.info(f"Starting SeleniumBase UC signup → {email}")
        if self.manual_captcha:
            log.info("  ℹ️  Manual-CAPTCHA mode: browser will be VISIBLE.")

        try:
            with SB(
                test=False,
                uc=True,
                headless=False,
                headless2=(False if self.manual_captcha else self.headless),
                chromium_arg=(
                    "--no-sandbox,"
                    "--disable-setuid-sandbox,"
                    "--disable-dev-shm-usage,"
                    "--disable-blink-features=AutomationControlled"
                ),
            ) as sb:
                # ── 1. Load signup page ───────────────────────────────────────
                log.info(f"  Loading {DEEPSEEK_SIGNUP_URL}...")
                sb.uc_open_with_reconnect(DEEPSEEK_SIGNUP_URL, reconnect_time=4)
                sb.sleep(3)
                self._shot(sb, "01_page_loaded")

                self._handle_captcha(sb)
                sb.sleep(1)

                # ── 2. Wait for the signup form ───────────────────────────────
                log.info("  Waiting for signup form...")
                form_selectors = [
                    "input[placeholder*='Email' i]",
                    "input[type='email']",
                    "input[name='email']",
                    "input[placeholder*='mail' i]",
                ]
                form_found = False
                for sel in form_selectors:
                    try:
                        sb.wait_for_element_visible(sel, timeout=20)
                        log.info(f"  ✓ Email input found: {sel}")
                        form_found = True
                        break
                    except Exception:
                        continue

                if not form_found:
                    log.error("  Signup form not found.")
                    self._shot(sb, "error_no_form")
                    return None

                self._shot(sb, "02_signup_form")

                # ── 3. Fill email ─────────────────────────────────────────────
                # Use sb.type() FIRST — it fires real keyboard events that React
                # needs to validate the email and enable the Send Code button.
                log.info(f"  Filling email: {email}")
                email_filled = False
                for sel in form_selectors:
                    try:
                        sb.type(sel, email)
                        log.info(f"  ✓ Email filled via sb.type ({sel})")
                        email_filled = True
                        break
                    except Exception:
                        continue

                if not email_filled:
                    for sel in form_selectors:
                        if self._js_set_value(sb, sel, email):
                            log.info(f"  ✓ Email filled via JS fallback ({sel})")
                            email_filled = True
                            break

                if not email_filled:
                    log.error("  Could not fill email field.")
                    self._shot(sb, "error_no_email")
                    return None

                sb.sleep(0.8)

                # ── 4. Fill password ──────────────────────────────────────────
                log.info("  Filling password...")
                import json as _json

                pwd_js = _json.dumps(password)
                pw_result = sb.execute_script(f"""
                    (function() {{
                        function setVal(el, v) {{
                            const setter = Object.getOwnPropertyDescriptor(
                                window.HTMLInputElement.prototype, 'value').set;
                            setter.call(el, v);
                            el.dispatchEvent(new Event('input',  {{bubbles: true}}));
                            el.dispatchEvent(new Event('change', {{bubbles: true}}));
                            el.dispatchEvent(new Event('blur',   {{bubbles: true}}));
                        }}
                        const pws = document.querySelectorAll("input[type='password']");
                        for (const pw of pws) {{ setVal(pw, {pwd_js}); }}
                        return pws.length;
                    }})()
                """)
                log.info(f"  ✓ Password filled ({pw_result} field(s))")
                sb.sleep(0.5)
                self._shot(sb, "03_form_filled")

                # ── 5. Seed inbox before clicking Send Code ───────────────────
                # Snapshot current IDs so pre-existing emails are never returned
                # as the new OTP.
                if seed_callback:
                    try:
                        seed_callback()
                    except Exception as _se:
                        log.warning(f"  Inbox seed failed (non-fatal): {_se}")

                # ── 6. Click "Send Code" button ───────────────────────────────
                log.info("  Looking for 'Send Code' / OTP button...")
                send_code_clicked = False

                # Wait a moment for React to render the button after filling email
                sb.sleep(1)

                clicked_js = sb.execute_script("""
                    (function() {
                        const keywords = [
                            'send code', 'get code', 'send otp', 'verify email',
                            'send verification', 'get verification',
                            '发送验证码', '获取验证码', '发送', '获取'
                        ];
                        // Search buttons AND spans/divs that look clickable
                        const candidates = document.querySelectorAll(
                            'button, [role="button"], div.ds-button, [class*="send"], [class*="Send"]'
                        );
                        for (const el of candidates) {
                            const txt = (el.textContent || '').trim().toLowerCase();
                            for (const kw of keywords) {
                                if (txt.includes(kw)) {
                                    el.click();
                                    return txt;
                                }
                            }
                        }
                        return null;
                    })()
                """)
                if clicked_js:
                    log.info(f"  ✓ OTP send button clicked (text: '{clicked_js}')")
                    send_code_clicked = True

                if not send_code_clicked:
                    for sel in [
                        "button:contains('Send Code')",
                        "button:contains('Get Code')",
                        "button:contains('Send OTP')",
                        "button:contains('Verify')",
                        "button:contains('Send')",
                    ]:
                        try:
                            if sb.is_element_visible(sel):
                                sb.click(sel)
                                log.info(f"  ✓ OTP send button clicked: {sel}")
                                send_code_clicked = True
                                break
                        except Exception:
                            continue

                if not send_code_clicked:
                    log.warning(
                        "  'Send Code' button not confirmed clicked — "
                        "will still poll for OTP in case it was triggered."
                    )

                sb.sleep(2)
                self._shot(sb, "04_after_send_code")

                # ── 7. Get OTP and enter it ───────────────────────────────────
                # Always poll for OTP — even if we couldn't confirm the button
                # click, DeepSeek may have sent the code anyway.
                log.info("  Retrieving OTP from inbox...")
                otp = None
                try:
                    otp = otp_callback()
                    log.info(f"  ✓ OTP received: {otp}")
                except TimeoutError as e:
                    log.warning(f"  OTP polling timed out: {e}")
                    self._shot(sb, "error_otp_timeout")

                if otp:
                        otp_filled = False
                        for sel in [
                            "input[type='tel']",
                            "input[placeholder='Code']",
                            "input[placeholder='code']",
                            "input[placeholder*='code' i]",
                            "input[placeholder*='Code' i]",
                            "input[placeholder*='OTP' i]",
                            "input[maxlength='6']",
                        ]:
                            if self._js_set_value(sb, sel, otp):
                                log.info(f"  ✓ OTP entered via JS ({sel})")
                                otp_filled = True
                                break

                        if not otp_filled:
                            # Try split digit inputs
                            digit_count = sb.execute_script("""
                                return document.querySelectorAll(
                                    'input[maxlength="1"][type="text"], input[maxlength="1"]'
                                ).length;
                            """)
                            if digit_count == 6:
                                otp_js = _json.dumps(list(otp))
                                sb.execute_script(f"""
                                    (function() {{
                                        const digits = {otp_js};
                                        const inputs = document.querySelectorAll(
                                            'input[maxlength="1"]'
                                        );
                                        function setVal(el, v) {{
                                            const setter = Object.getOwnPropertyDescriptor(
                                                window.HTMLInputElement.prototype, 'value').set;
                                            setter.call(el, v);
                                            el.dispatchEvent(new Event('input',  {{bubbles:true}}));
                                            el.dispatchEvent(new Event('change', {{bubbles:true}}));
                                        }}
                                        inputs.forEach((el, i) => {{
                                            if (i < digits.length) setVal(el, digits[i]);
                                        }});
                                    }})()
                                """)
                                log.info("  ✓ OTP entered digit-by-digit")
                                otp_filled = True

                        if not otp_filled:
                            log.warning("  Could not find OTP input field.")
                        sb.sleep(1)
                        self._shot(sb, "05_otp_entered")

                # ── 8. Check ToS checkbox ─────────────────────────────────────
                log.info("  Checking ToS checkbox (if any)...")
                tos_result = sb.execute_script("""
                    (function() {
                        const cbs = document.querySelectorAll("input[type='checkbox']");
                        let clicked = 0;
                        for (const cb of cbs) {
                            if (!cb.checked) { cb.click(); clicked++; }
                        }
                        return clicked;
                    })()
                """)
                if tos_result:
                    log.info(f"  ✓ Checked {tos_result} checkbox(es)")
                sb.sleep(0.3)

                # ── 8. Submit registration form ───────────────────────────────
                log.info("  Submitting registration form...")
                submit_result = sb.execute_script("""
                    (function() {
                        const keywords = ['sign up', 'register', 'create', '注册', 'continue', 'next', 'submit', 'start'];
                        const btns = document.querySelectorAll("button[type='submit'], button, div[role='button'], div.ds-button");
                        for (const btn of btns) {
                            const txt = btn.textContent.trim().toLowerCase();
                            const disabled = btn.disabled || btn.getAttribute('aria-disabled') === 'true' || btn.classList.contains('ds-button--disabled');
                            if (keywords.some(kw => txt.includes(kw)) && !disabled) {
                                btn.click();
                                return 'clicked:' + btn.textContent.trim();
                            }
                        }
                        const sub = document.querySelector("button[type='submit']:not([disabled])");
                        if (sub) { sub.click(); return 'submit:' + sub.textContent.trim(); }
                        return 'not_found';
                    })()
                """)
                log.info(f"  Submit result: {submit_result}")
                sb.sleep(3)
                self._shot(sb, "06_after_submit")

                # ── 9. Handle post-submit CAPTCHA ─────────────────────────────
                self._handle_captcha(sb)
                sb.sleep(2)
                self._shot(sb, "07_post_captcha")

                # ── 10. Scan for token ────────────────────────────────────────
                log.info("  Scanning for DeepSeek auth token (up to 45s)...")
                redirected_to_chat = False
                for attempt in range(15):
                    current_url = sb.get_current_url()
                    log.info(f"  [{attempt + 1}/15] URL: {current_url[:80]}")

                    try:
                        page_text = sb.execute_script("return document.body.innerText || '';")
                        if (
                            "already exists" in page_text.lower()
                            or "already registered" in page_text.lower()
                        ):
                            log.warning("  ⚠️  Email already registered — trying signin")
                            return None
                    except Exception:
                        pass

                    token = _scan_storage_for_token(sb.driver)
                    if token:
                        log.info("  ✅ Token found!")
                        break

                    if not redirected_to_chat and "platform.deepseek.com" in current_url and "/sign" not in current_url:
                        log.info(f"  ✓ Signup succeeded! Navigating to chat.deepseek.com to get token...")
                        sb.uc_open_with_reconnect("https://chat.deepseek.com", reconnect_time=4)
                        redirected_to_chat = True
                        continue

                    captcha = self._detect_captcha(sb)
                    if captcha:
                        log.info(f"  CAPTCHA reappeared: {captcha} — handling...")
                        self._handle_captcha(sb)
                        sb.sleep(2)

                    sb.sleep(3)

                self._shot(sb, "08_final_state")

        except Exception as e:
            log.error(f"SeleniumBase error during signup: {e}", exc_info=True)

        return token

    def signin(self, email: str, password: str, otp_callback=None, seed_callback=None) -> str | None:
        """
        Sign in to an existing DeepSeek account via chat.deepseek.com.
        DeepSeek's email login sends an OTP — otp_callback is required for this.

        Args:
            email:         Registered DeepSeek email address
            password:      Account password (used if password login is available)
            otp_callback:  Callable() -> str that returns the OTP from email.
            seed_callback: Optional callable() to snapshot inbox before triggering OTP.

        Returns token string or None.
        """
        from seleniumbase import SB
        import json as _json

        token = None
        log.info(f"Signing in via chat.deepseek.com: {email}")

        try:
            with SB(
                test=False,
                uc=True,
                headless=False,
                headless2=(False if self.manual_captcha else self.headless),
                chromium_arg=(
                    "--no-sandbox,"
                    "--disable-setuid-sandbox,"
                    "--disable-dev-shm-usage,"
                    "--disable-blink-features=AutomationControlled"
                ),
            ) as sb:
                # ── 1. Go directly to platform.deepseek.com/sign_in ───────────────
                log.info("  Loading https://platform.deepseek.com/sign_in...")
                sb.uc_open_with_reconnect("https://platform.deepseek.com/sign_in", reconnect_time=4)
                sb.sleep(3)
                sb.execute_script("try { localStorage.removeItem('userToken'); } catch(e) {}")
                
                self._handle_captcha(sb)
                sb.sleep(1)
                self._shot(sb, "signin_01_page")

                # ── 5. Wait for email input ───────────────────────────────────
                email_sel = None
                for sel in [
                    "input[placeholder*='Email' i]",
                    "input[type='email']",
                    "input[name='email']",
                    "input[placeholder*='phone' i]",
                ]:
                    try:
                        sb.wait_for_element_visible(sel, timeout=15)
                        log.info(f"  ✓ Email input found: {sel}")
                        email_sel = sel
                        break
                    except Exception:
                        continue

                if not email_sel:
                    log.error("  ❌ Could not find email input on signin page.")
                    self._shot(sb, "signin_error_no_form")
                    return None

                # ── 6. Fill email ─────────────────────────────────────────────
                # Use sb.type() first (fires real keyboard events that React needs),
                # then JS setVal as backup to ensure value is set.
                try:
                    sb.type(email_sel, email)
                    log.info(f"  ✓ Email filled via sb.type")
                except Exception:
                    self._js_set_value(sb, email_sel, email)
                    log.info(f"  ✓ Email filled via JS fallback")
                sb.sleep(0.8)

                # ── 7. Fill password ──────────────────────────────────────────────────────
                pass_sel = "input[type='password'], input[placeholder*='Password' i]"
                try:
                    sb.type(pass_sel, password, timeout=3)
                    log.info("  ✓ Password filled via sb.type")
                except Exception as _e:
                    log.info(f"  Password field not found: {_e} — assuming OTP-only flow")
                
                try:
                    self._js_set_value(sb, pass_sel, password)
                    log.info("  ✓ Password filled via JS fallback")
                except Exception:
                    pass

                sb.sleep(0.5)
                self._shot(sb, "signin_02_filled")

                try:
                    sb.press_keys(pass_sel, "\\n")
                    log.info("  ✓ Pressed ENTER on password field")
                except Exception as _e:
                    log.info(f"  Could not press ENTER: {_e}")

                sb.sleep(1)

                try:
                    with open("page_source.html", "w", encoding="utf-8") as f:
                        f.write(sb.get_page_source())
                except: pass
                # ── 8. Seed inbox before triggering OTP ───────────────────────
                if seed_callback:
                    try:
                        seed_callback()
                    except Exception as _se:
                        log.warning(f"  Inbox seed failed (non-fatal): {_se}")

                # ── 9. Submit the email form ─────────────────────────────────
                # Strategy: use sb.type() with the Return key appended so it
                # fires real keyboard events that React responds to.
                sb.sleep(1)
                log.info("  Submitting email form...")

                # Debug: dump outerHTML of first button (to identify it)
                # and verify the email field actually contains the email
                btn_html = sb.execute_script("""
                    const btn = document.querySelector('button,[role="button"]');
                    return btn ? btn.outerHTML.slice(0, 200) : 'none';
                """)
                log.info(f"  First button HTML: {btn_html}")
                email_val = sb.execute_script(f"""
                    const el = document.querySelector("input[placeholder*='Email' i], input[type='email']");
                    return el ? el.value : 'NOT FOUND';
                """)
                log.info(f"  Email field value: {email_val!r}")

                self._shot(sb, "signin_02b_pre_click")

                # Click the Log in button using React internal props or SeleniumBase
                clicked = None
                for text in ["Log in", "Continue", "Sign in", "登录", "Đăng nhập"]:
                    try:
                        # React internal hack
                        sb.execute_script(f"""
                            const btns = document.querySelectorAll("div.ds-button, button");
                            for (let btn of btns) {{
                                if ((btn.textContent || '').trim().toLowerCase().includes("{text}".toLowerCase())) {{
                                    const key = Object.keys(btn).find(k => k.startsWith("__reactProps$"));
                                    if (key && btn[key] && btn[key].onClick) {{
                                        btn[key].onClick({{ preventDefault: () => {{}}, stopPropagation: () => {{}} }});
                                        return true;
                                    }}
                                    btn.click();
                                    return true;
                                }}
                            }}
                            return false;
                        """)
                        # Fallback to real CDP click
                        sb.click(f'div.ds-button:contains("{text}"), button:contains("{text}")', timeout=2)
                        log.info(f"  ✓ Clicked button with text: {text}")
                        clicked = text
                        break
                    except Exception:
                        continue

                log.info(f"  Final click result: {clicked}")

                sb.sleep(2)
                self._handle_captcha(sb)
                sb.sleep(2)
                self._shot(sb, "signin_03_post_click")

                # ── 10. Handle OTP if DeepSeek sends one ─────────────────────
                # DeepSeek email login may show an OTP step
                otp_input_sel = None
                for sel in [
                    "input[type='tel']",
                    "input[placeholder*='code' i]",
                    "input[placeholder*='Code' i]",
                    "input[placeholder*='OTP' i]",
                    "input[maxlength='6']",
                ]:
                    try:
                        sb.wait_for_element_visible(sel, timeout=5)
                        otp_input_sel = sel
                        log.info(f"  OTP input detected: {sel}")
                        break
                    except Exception:
                        continue

                # Also check for 6 split digit inputs
                if not otp_input_sel:
                    digit_count = sb.execute_script("""
                        return document.querySelectorAll(
                            'input[maxlength="1"][type="text"], input[maxlength="1"]'
                        ).length;
                    """)
                    if digit_count == 6:
                        otp_input_sel = "split_digits"
                        log.info("  OTP input detected: 6 split digit inputs")

                if otp_input_sel and otp_callback:
                    log.info("  OTP step detected — polling inbox...")
                    try:
                        otp = otp_callback()
                        log.info(f"  ✓ OTP received: {otp}")
                        if otp_input_sel == "split_digits":
                            otp_js = _json.dumps(list(otp))
                            sb.execute_script(f"""
                                (function() {{
                                    const digits = {otp_js};
                                    const inputs = document.querySelectorAll('input[maxlength="1"]');
                                    function setVal(el, v) {{
                                        const setter = Object.getOwnPropertyDescriptor(
                                            window.HTMLInputElement.prototype, 'value').set;
                                        setter.call(el, v);
                                        el.dispatchEvent(new Event('input',  {{bubbles:true}}));
                                        el.dispatchEvent(new Event('change', {{bubbles:true}}));
                                    }}
                                    inputs.forEach((el, i) => {{
                                        if (i < digits.length) setVal(el, digits[i]);
                                    }});
                                }})()
                            """)
                            log.info("  ✓ OTP entered digit-by-digit")
                        else:
                            self._js_set_value(sb, otp_input_sel, otp)
                            log.info(f"  ✓ OTP entered into {otp_input_sel}")
                        sb.sleep(1)
                        # Submit after OTP
                        sb.execute_script("""
                            (function() {
                                const keywords = ['confirm', 'verify', 'submit', 'continue', '确认', '验证'];
                                const btns = document.querySelectorAll(
                                    "button[type='submit'], button, div.ds-button, div[role='button']"
                                );
                                for (const btn of btns) {
                                    const txt = (btn.textContent || '').trim().toLowerCase();
                                    const disabled = btn.disabled
                                        || btn.getAttribute('aria-disabled') === 'true'
                                        || btn.classList.contains('ds-button--disabled');
                                    if (!disabled && (keywords.some(kw => txt.includes(kw)) || txt.length < 20)) {
                                        btn.click();
                                        return txt;
                                    }
                                }
                                const sub = document.querySelector("button[type='submit']:not([disabled])");
                                if (sub) sub.click();
                            })()
                        """)
                        sb.sleep(3)
                        self._handle_captcha(sb)
                        sb.sleep(2)
                        self._shot(sb, "signin_04_after_otp")
                    except TimeoutError as e:
                        log.warning(f"  OTP polling timed out during signin: {e}")
                elif otp_input_sel and not otp_callback:
                    log.warning("  OTP input found but no otp_callback provided — cannot complete signin.")

                # ── 11. Wait for redirect and scan for token ──────────────────
                log.info("  Waiting for token in chat.deepseek.com storage...")
                redirected_to_chat = False
                for attempt in range(20):
                    if attempt == 5:
                        sb.save_screenshot("debug_login.png")
                        log.info("  Saved screenshot to debug_login.png")
                        try:
                            with open("debug_login.html", "w", encoding="utf-8") as f:
                                f.write(sb.get_page_source())
                        except: pass
                    
                    token = _scan_storage_for_token(sb.driver)
                    if token:
                        log.info("  ✅ Token found!")
                        break
                    
                    current_url = sb.get_current_url()
                    log.debug(f"  [{attempt+1}/20] URL: {current_url[:80]}")
                    
                    # If we are on platform.deepseek.com and no longer on the signin page
                    if not redirected_to_chat and "platform.deepseek.com" in current_url and "/sign" not in current_url:
                        log.info(f"  ✓ Signin succeeded! Navigating to chat.deepseek.com to get token...")
                        sb.uc_open_with_reconnect(DEEPSEEK_CHAT_URL, reconnect_time=4)
                        redirected_to_chat = True
                        sb.sleep(3)
                        continue

                    # If we somehow landed somewhere else
                    if "chat.deepseek.com" not in current_url and "platform.deepseek.com" not in current_url:
                        log.info("  Navigating back to chat.deepseek.com...")
                        sb.uc_open_with_reconnect(DEEPSEEK_CHAT_URL, reconnect_time=4)
                        sb.sleep(3)
                        continue

                    sb.sleep(2)

                self._shot(sb, "signin_05_final")

        except Exception as e:
            log.error(f"Signin error: {e}", exc_info=True)

        return token


# ══════════════════════════════════════════════════════════════════════════════
#  Main Orchestrator
# ══════════════════════════════════════════════════════════════════════════════


def create_one_account(
    mail_service: str = "icloud",
    headless: bool = False,
    debug: bool = False,
    manual_captcha: bool = False,
    api_url: str | None = None,
    api_key: str | None = None,
    use_api_mode: bool = False,
) -> dict | None:
    """
    Full account creation flow:
      1. Create temp email (with auto-failover between providers)
      2. Browser signup with OTP callback
      3. If no token: follow verification link
      4. If still no token: signin fallback
      5. Submit token to deepseek-python-api
    """
    # Step 1: Temp email with failover
    try:
        mail = _create_mail_backend(mail_service)
    except Exception as e:
        log.error(f"Mail setup failed: {e}")
        return None

    email = mail.address
    password = mail.password

    # Generate a strong password
    ds_pw = (
        "".join(random.choices(string.ascii_uppercase, k=2))
        + "".join(random.choices(string.digits, k=4))
        + "".join(random.choices(string.ascii_lowercase, k=6))
        + "!@"
    )
    log.info(f"  DeepSeek password will be: {ds_pw}")

    if use_api_mode:
        browser = DeepSeekAPI(debug=debug)
    else:
        browser = DeepSeekBrowserSB(headless=headless, debug=debug, manual_captcha=manual_captcha)

    # For GraphAPIMailAdapter, pass plus-address filter so parallel workers
    # don't accidentally steal each other's OTPs.
    _plus_filter: str | None = None
    if hasattr(mail, "poll_for_otp") and hasattr(mail, "_to_addresses"):
        _plus_filter = email  # email is already the plus-addressed version

    def get_otp() -> str:
        kwargs = {"timeout": 120, "interval": 4}
        if _plus_filter is not None:
            kwargs["to_filter"] = _plus_filter  # type: ignore[assignment]
        return mail.poll_for_otp(**kwargs)  # type: ignore[call-arg]

    # Step 2: Signup (API or browser)
    token = browser.signup(
        email=email,
        password=ds_pw,
        otp_callback=get_otp,
        seed_callback=mail.seed_seen_ids,
    )

    def _finish(acc: dict) -> dict:
        log.info(f"\n✅  SUCCESS!")
        log.info(f"  Email   : {acc['email']}")
        log.info(f"  Password: {acc['password']}")
        log.info(f"  Token   : {acc['token'][:20]}...")
        if api_url:
            submit_token_to_api(
                acc["token"],
                name=f"auto-{acc['email']}",
                api_url=api_url,
                api_key=api_key or DEFAULT_API_KEY,
            )
        return acc

    if token:
        return _finish({"email": email, "password": ds_pw, "token": token})

    # Note: DeepSeek uses OTP, so we skip the verification link step.

    # Step 4: Signin fallback (pass the mail backend so OTP can be handled)
    log.info("  Trying signin as fallback via chat.deepseek.com...")
    # Reseed the inbox so we only see new OTP emails during signin
    try:
        mail.seed_seen_ids()
    except Exception:
        pass
    token = browser.signin(
        email=email,
        password=ds_pw,
        seed_callback=mail.seed_seen_ids,
    )
    if token:
        return _finish({"email": email, "password": ds_pw, "token": token})

    log.error("❌ Failed to obtain token.")
    return None


def create_accounts_parallel(
    count: int,
    *,
    mail_service: str = "graphapi",
    headless: bool = False,
    debug: bool = False,
    manual_captcha: bool = False,
    api_url: str | None = None,
    api_key: str | None = None,
    use_api_mode: bool = False,
    max_workers: int = 10,
) -> list[dict]:
    """
    Create *count* accounts in parallel, up to *max_workers* at a time.

    With plus-addressing (GraphAPI backend), each worker uses a unique
    plus-tag so OTP emails are routed to the correct worker with no
    cross-contamination.  The number of truly parallel workers is bounded
    by *max_workers* (default 10) — one per real email account.
    """
    import concurrent.futures
    import threading

    results: list[dict] = []
    lock = threading.Lock()

    def worker(i: int) -> dict | None:
        log.info(f"\n{'=' * 60}")
        log.info(f"  Parallel worker {i + 1}/{count}")
        log.info(f"{'=' * 60}")
        acc = create_one_account(
            mail_service=mail_service,
            headless=headless,
            debug=debug,
            manual_captcha=manual_captcha,
            api_url=api_url,
            api_key=api_key,
            use_api_mode=use_api_mode,
        )
        if acc:
            with lock:
                results.append(acc)
                log.info(f"  Worker {i + 1}: ✅ {acc['email']}")
        else:
            log.warning(f"  Worker {i + 1}: ❌ failed")
        return acc

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(count, max_workers)) as pool:
        futures = [pool.submit(worker, i) for i in range(count)]
        concurrent.futures.wait(futures)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated platform.deepseek.com account creator (SeleniumBase UC mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recommended: iCloud HME (passes DeepSeek domain check)
  python scripts/deepseek_signup.py --manual-captcha \\
      --mail-service icloud \\
      --icloud-cookies icloud_cookies.txt \\
      --icloud-imap-user you@icloud.com \\
      --icloud-imap-pass xxxx-xxxx-xxxx-xxxx

  python scripts/deepseek_signup.py --count 3 --mail-service icloud [...]
  python scripts/deepseek_signup.py --no-upload
  python scripts/deepseek_signup.py --api-url http://localhost:8000 --api-key mykey
  python scripts/deepseek_signup.py --headless --debug
        """,
    )
    parser.add_argument(
        "--api-mode",
        action="store_true",
        help=(
            "Hybrid API mode: registration via platform API (fast, no browser + PoW), "
            "sign-in via headless browser (handles AWS WAF on chat.deepseek.com). "
            "Removes need for full Selenium signup flow."
        ),
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Create accounts in parallel using N concurrent workers (default: 1 = sequential). "
            "Requires --mail-service graphapi (plus-addressing). Each worker uses a unique "
            "plus-tag (e.g. user+ds<epoch>N@hotmail.com) so OTPs are routed correctly. "
            "Recommended max: 10 (one per real inbox)."
        ),
    )
    parser.add_argument("--count", type=int, default=1, help="Number of accounts to create")
    parser.add_argument("--headless", action="store_true", help="Run browser headlessly")
    parser.add_argument(
        "--manual-captcha",
        action="store_true",
        help="Show browser and let you solve CAPTCHAs (most reliable)",
    )
    parser.add_argument("--debug", action="store_true", help="Save debug screenshots")
    parser.add_argument(
        "--mail-service",
        choices=["icloud", "graphapi", "guerrilla", "mailtm", "mailgw", "auto"],
        default="icloud",
        help="Mail provider: icloud (default), graphapi (Hotmail/Outlook), "
        "guerrilla, mailtm, mailgw, auto",
    )
    parser.add_argument(
        "--output", default="deepseek_accounts.json", help="Output file for saved accounts"
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"deepseek-python-api base URL (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help=f"Management API key (default: {DEFAULT_API_KEY})",
    )
    parser.add_argument(
        "--no-upload", action="store_true", help="Skip submitting tokens to deepseek-python-api"
    )
    # iCloud HME credentials
    parser.add_argument(
        "--icloud-cookies",
        default="icloud_cookies.txt",
        help="Path to iCloud cookies file (Cookie-Editor → Header String)",
    )
    parser.add_argument(
        "--icloud-imap-user",
        default="",
        help="iCloud Apple ID email for IMAP (e.g. you@icloud.com)",
    )
    parser.add_argument(
        "--icloud-imap-pass", default="", help="iCloud App-Specific Password for IMAP"
    )
    # Microsoft Graph API credentials (Hotmail / Outlook) — multiple accounts supported
    parser.add_argument(
        "--graphapi-creds",
        action="append",
        default=[],
        metavar="EMAIL|PASS|TOKEN[|CLIENT_ID]",
        help="Graph API credential string 'email|pass|refresh_token[|client_id]'. "
        "Repeat the flag for multiple accounts (round-robin). "
        "Also accepts a file path (one credential per line). "
        "Also read from GRAPHAPI_CREDS env var (newline-separated).",
    )
    parser.add_argument(
        "--graphapi-creds-file",
        default="",
        metavar="FILE",
        help="Path to a file containing Graph API credentials, one per line "
        "(email|pass|refresh_token[|client_id]). Comments (#) are ignored.",
    )
    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)

    if args.manual_captcha and args.headless:
        parser.error("--manual-captcha and --headless are mutually exclusive.")

    global _ICLOUD_KWARGS, _GRAPHAPI_POOL, _GRAPHAPI_IDX

    # ── iCloud credentials ────────────────────────────────────────────────────
    cookie_path = Path(args.icloud_cookies)
    if args.mail_service in ("icloud", "auto"):
        if not cookie_path.exists():
            if args.mail_service == "icloud":
                parser.error(
                    f"--mail-service icloud requires a cookies file at '{cookie_path}'.\n"
                    "  1. Go to icloud.com/settings in your browser\n"
                    "  2. Install Cookie-Editor extension\n"
                    "  3. Click Export → Header String → paste into icloud_cookies.txt"
                )
            else:
                log.warning(
                    f"iCloud cookies file '{cookie_path}' not found — skipping icloud backend."
                )
        elif not args.icloud_imap_user or not args.icloud_imap_pass:
            if args.mail_service == "icloud":
                parser.error(
                    "--mail-service icloud requires --icloud-imap-user and --icloud-imap-pass.\n"
                    "  Generate an App-Specific Password at appleid.apple.com → Security."
                )
            else:
                log.warning("iCloud IMAP credentials missing — skipping icloud backend.")
        else:
            _ICLOUD_KWARGS = {
                "cookies": cookie_path,
                "imap_user": args.icloud_imap_user,
                "imap_pass": args.icloud_imap_pass,
            }
            log.info(f"  iCloud HME enabled: {args.icloud_imap_user}")

    # ── Microsoft Graph API credentials (multi-account pool) ─────────────────
    import os as _os

    def _load_cred_lines(raw_lines: list[str]) -> list[str]:
        """Expand any file paths within raw_lines and return individual credential strings."""
        out: list[str] = []
        for entry in raw_lines:
            entry = entry.strip()
            if not entry:
                continue
            # Credential strings always contain '|' (email|pass|token...).
            # Only attempt a filesystem check when there is NO pipe — this avoids
            # OSError ENAMETOOLONG when the refresh token is hundreds of chars long.
            if "|" not in entry:
                p = Path(entry)
                try:
                    is_file = p.exists() and p.is_file()
                except OSError:
                    is_file = False
                if is_file:
                    file_count = 0
                    for line in p.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line and not line.startswith("#"):
                            out.append(line)
                            file_count += 1
                    log.info(
                        f"  Loaded Graph API credentials from file: {p} ({file_count} account(s))"
                    )
                    continue
            out.append(entry)
        return out

    # Gather raw credential strings from all sources
    raw_cred_entries: list[str] = list(args.graphapi_creds)  # from repeated --graphapi-creds flags

    # --graphapi-creds-file
    if args.graphapi_creds_file:
        fp = Path(args.graphapi_creds_file)
        if not fp.exists():
            parser.error(f"--graphapi-creds-file not found: {fp}")
        raw_cred_entries.append(str(fp))  # will be expanded by _load_cred_lines

    # GRAPHAPI_CREDS env var (newline or comma separated)
    env_creds = _os.environ.get("GRAPHAPI_CREDS", "")
    if env_creds:
        raw_cred_entries.extend(env_creds.replace(",", "\n").splitlines())

    cred_lines = _load_cred_lines(raw_cred_entries)

    for raw in cred_lines:
        try:
            parsed = parse_graphapi_credential(raw)
            _GRAPHAPI_POOL.append(parsed)
        except ValueError as e:
            if args.mail_service == "graphapi":
                parser.error(f"Invalid Graph API credential: {e}")
            else:
                log.warning(f"Skipping invalid Graph API credential: {e}")

    if _GRAPHAPI_POOL:
        _GRAPHAPI_IDX = 0
        emails = ", ".join(c["email"] for c in _GRAPHAPI_POOL)
        log.info(f"  Graph API pool: {len(_GRAPHAPI_POOL)} account(s) — {emails}")
    elif args.mail_service == "graphapi":
        parser.error(
            "--mail-service graphapi requires at least one credential.\n"
            "  Use --graphapi-creds 'email|pass|token[|client_id]' (repeat for multiple)\n"
            "  or --graphapi-creds-file accounts.txt (one per line)\n"
            "  or set GRAPHAPI_CREDS env var."
        )

    api_url = None if args.no_upload else args.api_url
    api_key = args.api_key

    if api_url:
        log.info(f"Token upload enabled → {api_url}")
    else:
        log.info("Token upload disabled (--no-upload)")

    out_path = Path(args.output)
    results: list[dict] = []
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text())
            log.info(f"Loaded {len(results)} existing account(s) from {out_path}")
        except Exception:
            pass

    common_kwargs = dict(
        mail_service=args.mail_service,
        headless=args.headless,
        debug=args.debug,
        manual_captcha=args.manual_captcha,
        api_url=api_url,
        api_key=api_key,
        use_api_mode=args.api_mode,
    )

    if args.parallel > 1:
        log.info(f"\n🚀 Parallel mode: {args.count} account(s) in batches of {args.parallel} workers")
        if args.mail_service != "graphapi":
            log.warning(
                "⚠  --parallel works best with --mail-service graphapi (plus-addressing). "
                "With other backends OTP routing is sequential and may be unreliable."
            )
        new_results = create_accounts_parallel(
            args.count,
            **common_kwargs,
            max_workers=args.parallel,
        )
        results.extend(new_results)
        out_path.write_text(json.dumps(results, indent=2))
    else:
        for i in range(args.count):
            log.info(f"\n{'=' * 60}")
            log.info(f"  Account {i + 1}/{args.count}")
            log.info(f"{'=' * 60}")

            acc = create_one_account(**common_kwargs)

            if acc:
                results.append(acc)
                out_path.write_text(json.dumps(results, indent=2))
                log.info(f"  Saved to {out_path}")
                print(f"\n  ✅ Token: {acc['token'][:40]}...")
            else:
                log.warning(f"  Account {i + 1}/{args.count} failed.")

            if i < args.count - 1:
                delay = random.randint(8, 15)
                log.info(f"  Waiting {delay}s before next account...")
                time.sleep(delay)

    log.info(f"\nDone. {len(results)} account(s) saved to {out_path}")
    if results:
        print("\n" + "=" * 60)
        print(f"{'EMAIL':<35} {'TOKEN (first 40 chars)'}")
        print("-" * 60)
        for acc in results:
            print(f"{acc['email']:<35} {acc['token'][:40]}...")
        print("=" * 60)


if __name__ == "__main__":
    main()
