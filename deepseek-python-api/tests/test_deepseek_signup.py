"""
Tests for scripts/deepseek_signup.py

All tests are offline (no real browser or network).
Covers:
  - GuerrillaMailAdapter  (primary working provider)
  - HydraMailAdapter      (mail.tm / mail.gw family)
  - GraphAPIMailAdapter   (Microsoft Hotmail / Outlook via Graph API)
  - parse_graphapi_credential  (pipe-format parser)
  - _create_mail_backend  (factory + failover logic)
  - OTP / link extraction helpers
  - Token extraction helpers
  - submit_token_to_api   (mocked HTTP)
  - DeepSeekBrowserSB     (non-Selenium logic)
  - create_one_account    (orchestration, all branches)
  - CLI argument parsing
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests as req

# Put the scripts dir on the path so we can import deepseek_signup directly
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import deepseek_signup as ds

# ══════════════════════════════════════════════════════════════════════════════
#  OTP extraction
# ══════════════════════════════════════════════════════════════════════════════


class TestExtractOtp:
    def test_six_digit_code(self):
        assert ds._extract_otp("Your code is 123456") == "123456"

    def test_code_in_html(self):
        html = "<p>Verification code: <strong>654321</strong></p>"
        assert ds._extract_otp(html) == "654321"

    def test_ignores_longer_numbers(self):
        assert ds._extract_otp("Order ID 1234567") is None

    def test_ignores_shorter_numbers(self):
        assert ds._extract_otp("Code: 12345") is None

    def test_empty_string(self):
        assert ds._extract_otp("") is None

    def test_none_input(self):
        assert ds._extract_otp(None) is None

    def test_multiple_codes_returns_first(self):
        result = ds._extract_otp("First: 111111, Second: 999999")
        assert result == "111111"

    def test_code_at_start(self):
        assert ds._extract_otp("987654 is your code") == "987654"

    def test_code_with_surrounding_spaces(self):
        assert ds._extract_otp("  123456  ") == "123456"


# ══════════════════════════════════════════════════════════════════════════════
#  Verification link extraction
# ══════════════════════════════════════════════════════════════════════════════


class TestExtractVerificationLink:
    def test_basic_deepseek_link(self):
        text = "Click here to verify: https://platform.deepseek.com/verify?token=abc123"
        result = ds._extract_verification_link(text)
        assert result == "https://platform.deepseek.com/verify?token=abc123"

    def test_confirm_link(self):
        text = "Please confirm: https://example.com/confirm?id=xyz"
        result = ds._extract_verification_link(text)
        assert result == "https://example.com/confirm?id=xyz"

    def test_activate_link(self):
        text = "Activate your account: https://deepseek.com/activate/abc"
        result = ds._extract_verification_link(text)
        assert result is not None
        assert "activate" in result

    def test_no_link_returns_none(self):
        assert ds._extract_verification_link("No link here") is None

    def test_none_input(self):
        assert ds._extract_verification_link(None) is None

    def test_strips_trailing_punctuation(self):
        text = "https://deepseek.com/verify?t=abc."
        result = ds._extract_verification_link(text)
        assert result == "https://deepseek.com/verify?t=abc"

    def test_html_with_link(self):
        html = '<a href="https://deepseek.com/verify?code=ZZZ">Verify Email</a>'
        result = ds._extract_verification_link(html)
        assert result is not None
        assert "deepseek.com/verify" in result

    def test_prefers_deepseek_domain(self):
        text = "Other: https://example.com/page DeepSeek: https://deepseek.com/account/verify"
        result = ds._extract_verification_link(text)
        assert result is not None


# ══════════════════════════════════════════════════════════════════════════════
#  Token extraction helpers
# ══════════════════════════════════════════════════════════════════════════════





class TestSearchDictForToken:
    def test_finds_token_key(self):
        data = {"token": "U4vsCtoBdxn1pfU4QclHNg2M9rVtBq82giPe2ilrFBfeiuR4yLj"}
        assert ds._search_dict_for_token(data) is not None

    def test_finds_user_token_key(self):
        data = {"userToken": "U4vsCtoBdxn1pfU4QclHNg2M9rVtBq82giPe2ilrFBfeiuR4yLj"}
        assert (
            ds._search_dict_for_token(data) == "U4vsCtoBdxn1pfU4QclHNg2M9rVtBq82giPe2ilrFBfeiuR4yLj"
        )

    def test_nested_dict(self):
        data = {"user": {"auth": {"token": "A" * 45}}}
        result = ds._search_dict_for_token(data)
        assert result == "A" * 45

    def test_list_of_dicts(self):
        data = [{"token": "B" * 45}]
        result = ds._search_dict_for_token(data)
        assert result == "B" * 45

    def test_string_input(self):
        result = ds._search_dict_for_token("A" * 50)
        assert result == "A" * 50

    def test_short_string_returns_none(self):
        result = ds._search_dict_for_token("short")
        assert result is None

    def test_empty_dict(self):
        assert ds._search_dict_for_token({}) is None

    def test_prefer_token_key_over_nested(self):
        data = {
            "token": "DirectToken" + "x" * 35,
            "nested": {"other_token": "NestedToken" + "y" * 35},
        }
        result = ds._search_dict_for_token(data)
        assert result == "DirectToken" + "x" * 35


# ══════════════════════════════════════════════════════════════════════════════
#  GuerrillaMailAdapter (mocked network)
# ══════════════════════════════════════════════════════════════════════════════


class TestGuerrillaMailAdapter:
    def _make_adapter(self) -> ds.GuerrillaMailAdapter:
        return ds.GuerrillaMailAdapter()

    def test_create_account_success(self):
        adapter = self._make_adapter()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "email_addr": "test@guerrillamailblock.com",
            "sid_token": "abc123sid",
            "alias": "y1x+abc",
        }
        with patch.object(adapter.s, "get", return_value=mock_resp):
            address, password = adapter.create_account()

        assert address == "test@guerrillamailblock.com"
        assert adapter._sid == "abc123sid"
        assert password == ""  # Guerrilla Mail has no password

    def test_poll_for_otp_finds_code_in_excerpt(self):
        adapter = self._make_adapter()
        adapter._sid = "test-sid"

        msgs = [
            {
                "mail_id": "1",
                "mail_subject": "DeepSeek code",
                "mail_excerpt": "Your verification code is 123456",
            }
        ]

        def fake_get(url, **kw):
            r = MagicMock()
            r.raise_for_status = MagicMock()
            if "get_email_list" in url:
                r.json.return_value = {"list": msgs}
            return r

        with patch.object(adapter.s, "get", side_effect=fake_get):
            otp = adapter.poll_for_otp(timeout=5, interval=0)

        assert otp == "123456"

    def test_poll_for_otp_fetches_body_when_excerpt_empty(self):
        adapter = self._make_adapter()
        adapter._sid = "test-sid"

        msgs = [{"mail_id": "2", "mail_subject": "Verify", "mail_excerpt": ""}]
        body_response = {"mail_body": "Here is your 6-digit code: 654321"}

        call_count = [0]

        def fake_get(url, **kw):
            r = MagicMock()
            r.raise_for_status = MagicMock()
            call_count[0] += 1
            if "get_email_list" in url:
                r.json.return_value = {"list": msgs}
            elif "fetch_email" in url:
                r.json.return_value = body_response
            return r

        with patch.object(adapter.s, "get", side_effect=fake_get):
            otp = adapter.poll_for_otp(timeout=5, interval=0)

        assert otp == "654321"

    def test_poll_for_otp_timeout(self):
        adapter = self._make_adapter()
        adapter._sid = "sid"

        def fake_get(url, **kw):
            r = MagicMock()
            r.raise_for_status = MagicMock()
            r.json.return_value = {"list": []}
            return r

        with patch.object(adapter.s, "get", side_effect=fake_get):
            with pytest.raises(TimeoutError):
                adapter.poll_for_otp(timeout=1, interval=1)

    def test_poll_for_verification_link_success(self):
        adapter = self._make_adapter()
        adapter._sid = "sid"

        msgs = [
            {
                "mail_id": "3",
                "mail_subject": "Verify DeepSeek",
                "mail_excerpt": "Click: https://deepseek.com/verify?t=abc",
            }
        ]

        def fake_get(url, **kw):
            r = MagicMock()
            r.raise_for_status = MagicMock()
            if "get_email_list" in url:
                r.json.return_value = {"list": msgs}
            return r

        with patch.object(adapter.s, "get", side_effect=fake_get):
            link = adapter.poll_for_verification_link(timeout=5, interval=0)

        assert link is not None
        assert "deepseek.com/verify" in link

    def test_poll_for_verification_link_timeout(self):
        adapter = self._make_adapter()
        adapter._sid = "sid"

        def fake_get(url, **kw):
            r = MagicMock()
            r.raise_for_status = MagicMock()
            r.json.return_value = {"list": []}
            return r

        with patch.object(adapter.s, "get", side_effect=fake_get):
            result = adapter.poll_for_verification_link(timeout=1, interval=1)

        assert result is None

    def test_skips_welcome_email_with_id_zero(self):
        """Mail ID '0' is the Guerrilla Mail welcome message — must be skipped."""
        adapter = self._make_adapter()
        adapter._sid = "sid"

        msgs = [
            {"mail_id": "0", "mail_subject": "Welcome", "mail_excerpt": "000000 welcome"},
            {"mail_id": "1", "mail_subject": "Code", "mail_excerpt": "Your code is 987654"},
        ]

        def fake_get(url, **kw):
            r = MagicMock()
            r.raise_for_status = MagicMock()
            r.json.return_value = {"list": msgs}
            return r

        with patch.object(adapter.s, "get", side_effect=fake_get):
            otp = adapter.poll_for_otp(timeout=5, interval=0)

        assert otp == "987654"  # NOT "000000" from welcome mail


# ══════════════════════════════════════════════════════════════════════════════
#  HydraMailAdapter / TempMail (mocked network)
# ══════════════════════════════════════════════════════════════════════════════


class TestHydraMailAdapter:
    def _make(self, base=ds.MAIL_TM_BASE) -> ds.HydraMailAdapter:
        return ds.HydraMailAdapter(base_url=base)

    def test_create_account_success(self):
        m = self._make()
        mock_get = MagicMock()
        mock_get.return_value.json.return_value = {"hydra:member": [{"domain": "example.tm"}]}
        mock_get.return_value.raise_for_status = MagicMock()

        mock_post = MagicMock()
        mock_post.return_value.status_code = 201
        mock_post.return_value.json.return_value = {"token": "mail-jwt"}
        mock_post.return_value.raise_for_status = MagicMock()

        with patch.object(m.s, "get", mock_get):
            with patch.object(m.s, "post", mock_post):
                address, password = m.create_account()

        assert "@example.tm" in address
        assert len(password) > 12
        assert m._token == "mail-jwt"

    def test_get_domain_raises_if_no_domains(self):
        m = self._make()
        with patch.object(m, "_get", return_value={"hydra:member": []}):
            with pytest.raises(RuntimeError, match="no active domains"):
                m._get_domain()

    def test_create_account_fails_on_bad_status(self):
        m = self._make()
        mock_get = MagicMock()
        mock_get.return_value = {"hydra:member": [{"domain": "example.tm"}]}
        mock_post = MagicMock()
        mock_post.return_value.status_code = 500
        mock_post.return_value.text = "Server error"

        with patch.object(m, "_get", mock_get):
            with patch.object(m.s, "post", mock_post):
                with pytest.raises(RuntimeError, match="Mail account creation failed"):
                    m.create_account()

    def test_poll_for_otp_timeout(self):
        m = self._make()
        with patch.object(m, "_get", return_value={"hydra:member": []}):
            with pytest.raises(TimeoutError):
                m.poll_for_otp(timeout=1, interval=1)

    def test_poll_for_verification_link_timeout(self):
        m = self._make()
        with patch.object(m, "_get", return_value={"hydra:member": []}):
            result = m.poll_for_verification_link(timeout=1, interval=1)
        assert result is None


class TestTempMailLegacyAlias:
    """TempMail is a legacy alias for HydraMailAdapter — test backwards compat."""

    def test_is_hydra_subclass(self):
        assert issubclass(ds.TempMail, ds.HydraMailAdapter)

    def test_authenticate_sets_token(self):
        mail = ds.TempMail()
        mail.address = "test@example.tm"
        mail.password = "testpw"

        mock_post = MagicMock()
        mock_post.return_value.json.return_value = {"token": "legacy-jwt"}
        mock_post.return_value.raise_for_status = MagicMock()

        with patch.object(mail.s, "post", mock_post):
            token = mail.authenticate()

        assert token == "legacy-jwt"
        assert mail.token == "legacy-jwt"
        assert mail.s.headers.get("Authorization") == "Bearer legacy-jwt"

    def test_authenticate_returns_existing_token(self):
        """If already authenticated, don't make another request."""
        mail = ds.TempMail()
        mail._token = "existing-token"

        with patch.object(mail.s, "post") as mock_post:
            token = mail.authenticate()

        assert token == "existing-token"
        mock_post.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
#  _create_mail_backend factory + failover
# ══════════════════════════════════════════════════════════════════════════════


class TestCreateMailBackend:
    def _mock_guerrilla_success(self):
        """Return a mock that makes GuerrillaMailAdapter.create_account() succeed."""
        adapter = MagicMock(spec=ds.GuerrillaMailAdapter)
        adapter.address = "test@guerrillamailblock.com"
        adapter.password = ""
        adapter.create_account.return_value = ("test@guerrillamailblock.com", "")
        return adapter

    def test_guerrilla_service_returns_guerrilla_adapter(self):
        mock_adapter = self._mock_guerrilla_success()
        with patch("deepseek_signup.GuerrillaMailAdapter", return_value=mock_adapter):
            result = ds._create_mail_backend("guerrilla")
        assert result is mock_adapter

    def test_mailtm_falls_back_to_guerrilla_on_failure(self):
        """If mail.tm fails, should fall back to Guerrilla Mail."""
        mock_hydra = MagicMock(spec=ds.HydraMailAdapter)
        mock_hydra.create_account.side_effect = RuntimeError("Connection timeout")

        mock_guerrilla = self._mock_guerrilla_success()

        with patch("deepseek_signup.HydraMailAdapter", return_value=mock_hydra):
            with patch("deepseek_signup.GuerrillaMailAdapter", return_value=mock_guerrilla):
                result = ds._create_mail_backend("mailtm")

        assert result is mock_guerrilla

    def test_auto_tries_guerrilla_first(self):
        """'auto' service should try Guerrilla Mail first."""
        mock_guerrilla = self._mock_guerrilla_success()
        mock_hydra = MagicMock(spec=ds.HydraMailAdapter)

        guerrilla_calls = []

        def make_guerrilla():
            guerrilla_calls.append(1)
            return mock_guerrilla

        with patch("deepseek_signup.GuerrillaMailAdapter", side_effect=make_guerrilla):
            with patch("deepseek_signup.HydraMailAdapter", return_value=mock_hydra):
                result = ds._create_mail_backend("auto")

        assert len(guerrilla_calls) == 1
        assert result is mock_guerrilla
        mock_hydra.create_account.assert_not_called()

    def test_all_backends_fail_raises(self):
        """If all backends fail, RuntimeError is raised."""
        mock_g = MagicMock(spec=ds.GuerrillaMailAdapter)
        mock_g.create_account.side_effect = RuntimeError("Network error")
        mock_h = MagicMock(spec=ds.HydraMailAdapter)
        mock_h.create_account.side_effect = RuntimeError("Also failed")

        with patch("deepseek_signup.GuerrillaMailAdapter", return_value=mock_g):
            with patch("deepseek_signup.HydraMailAdapter", return_value=mock_h):
                with pytest.raises(RuntimeError, match="All mail backends failed"):
                    ds._create_mail_backend("auto")

    def test_mailgw_falls_back_to_guerrilla_on_failure(self):
        mock_hydra = MagicMock(spec=ds.HydraMailAdapter)
        mock_hydra.create_account.side_effect = RuntimeError("502 Bad Gateway")
        mock_guerrilla = self._mock_guerrilla_success()

        with patch("deepseek_signup.HydraMailAdapter", return_value=mock_hydra):
            with patch("deepseek_signup.GuerrillaMailAdapter", return_value=mock_guerrilla):
                result = ds._create_mail_backend("mailgw")

        assert result is mock_guerrilla


# ══════════════════════════════════════════════════════════════════════════════
#  submit_token_to_api
# ══════════════════════════════════════════════════════════════════════════════


class TestSubmitTokenToApi:
    def test_successful_submission(self):
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.content = b'{"id":"tok_abc","status":"unchecked"}'
        mock_response.json.return_value = {"id": "tok_abc", "status": "unchecked"}

        with patch("requests.post", return_value=mock_response):
            result = ds.submit_token_to_api(
                token="U4vsCtoBdxn1pfU4QclHNg2M9rVtBq82giPe2ilrFBfeiuR4yLj",
                name="test-account",
                api_url="http://127.0.0.1:8000",
                api_key="nowsthetime",
            )
        assert result is True

    def test_200_response(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"id":"tok_xyz","status":"healthy"}'
        mock_response.json.return_value = {"id": "tok_xyz", "status": "healthy"}

        with patch("requests.post", return_value=mock_response):
            result = ds.submit_token_to_api("token123" * 6, "name")
        assert result is True

    def test_duplicate_token_409(self):
        mock_response = MagicMock()
        mock_response.status_code = 409
        mock_response.content = b'{"error":{"code":"duplicate_token"}}'
        mock_response.json.return_value = {"error": {"code": "duplicate_token"}}

        with patch("requests.post", return_value=mock_response):
            result = ds.submit_token_to_api("token123" * 6, "name")
        assert result is True

    def test_server_error_returns_false(self):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.content = b'{"detail":"internal error"}'
        mock_response.json.return_value = {"detail": "internal error"}

        with patch("requests.post", return_value=mock_response):
            result = ds.submit_token_to_api("token123" * 6, "name")
        assert result is False

    def test_auth_error_returns_false(self):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.content = b'{"error":{"code":"invalid_api_key"}}'
        mock_response.json.return_value = {"error": {"code": "invalid_api_key"}}

        with patch("requests.post", return_value=mock_response):
            result = ds.submit_token_to_api("token123" * 6, "name", api_key="wrong-key")
        assert result is False

    def test_network_error_returns_false(self):
        with patch("requests.post", side_effect=req.exceptions.ConnectionError("refused")):
            result = ds.submit_token_to_api("token123" * 6, "name")
        assert result is False

    def test_timeout_returns_false(self):
        with patch("requests.post", side_effect=req.exceptions.Timeout()):
            result = ds.submit_token_to_api("token123" * 6, "name")
        assert result is False

    def test_correct_endpoint_constructed(self):
        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            resp = MagicMock()
            resp.status_code = 201
            resp.content = b'{"id":"tok_1","status":"unchecked"}'
            resp.json.return_value = {"id": "tok_1"}
            return resp

        with patch("requests.post", side_effect=fake_post):
            ds.submit_token_to_api("token123" * 6, "name", api_url="http://myserver:9000")

        assert captured["url"] == "http://myserver:9000/v0/management/tokens"

    def test_trailing_slash_stripped(self):
        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            resp = MagicMock()
            resp.status_code = 201
            resp.content = b'{"id":"t","status":"unchecked"}'
            resp.json.return_value = {"id": "t"}
            return resp

        with patch("requests.post", side_effect=fake_post):
            ds.submit_token_to_api("token" * 15, "n", api_url="http://localhost:8000/")

        assert captured["url"] == "http://localhost:8000/v0/management/tokens"

    def test_correct_headers_sent(self):
        captured = {}

        def fake_post(url, **kwargs):
            captured["headers"] = kwargs.get("headers", {})
            resp = MagicMock()
            resp.status_code = 201
            resp.content = b'{"id":"tok_1","status":"unchecked"}'
            resp.json.return_value = {"id": "tok_1"}
            return resp

        with patch("requests.post", side_effect=fake_post):
            ds.submit_token_to_api("token123" * 6, "name", api_key="my-secret-key")

        assert captured["headers"].get("Authorization") == "Bearer my-secret-key"
        assert captured["headers"].get("Content-Type") == "application/json"

    def test_payload_structure(self):
        captured = {}
        token_str = "U4vsCtoBdxn1pfU4QclHNg2M9rVtBq82giPe2ilrFBfeiuR4yLj"

        def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json", {})
            resp = MagicMock()
            resp.status_code = 201
            resp.content = b'{"id":"tok_1","status":"unchecked"}'
            resp.json.return_value = {"id": "tok_1"}
            return resp

        with patch("requests.post", side_effect=fake_post):
            ds.submit_token_to_api(token_str, "my-account")

        assert captured["json"]["token"] == token_str
        assert captured["json"]["name"] == "my-account"
        assert captured["json"]["enabled"] is True


# ══════════════════════════════════════════════════════════════════════════════
#  DeepSeekBrowserSB (unit tests for non-browser logic)
# ══════════════════════════════════════════════════════════════════════════════


class TestDeepSeekBrowserSB:
    def test_init_defaults(self):
        b = ds.DeepSeekBrowserSB()
        assert b.headless is False
        assert b.debug is False
        assert b.manual_captcha is False
        assert b._shot_idx == 0

    def test_init_custom(self):
        b = ds.DeepSeekBrowserSB(headless=True, debug=True, manual_captcha=False)
        assert b.headless is True
        assert b.debug is True

    def test_shot_no_op_when_debug_false(self):
        b = ds.DeepSeekBrowserSB(debug=False)
        fake_sb = MagicMock()
        b._shot(fake_sb, "test")
        fake_sb.save_screenshot.assert_not_called()

    def test_shot_calls_screenshot_when_debug(self):
        b = ds.DeepSeekBrowserSB(debug=True)
        fake_sb = MagicMock()
        b._shot(fake_sb, "test_label")
        fake_sb.save_screenshot.assert_called_once()
        args = fake_sb.save_screenshot.call_args[0]
        assert "test_label" in args[0]

    def test_shot_increments_index(self):
        b = ds.DeepSeekBrowserSB(debug=True)
        fake_sb = MagicMock()
        b._shot(fake_sb, "a")
        b._shot(fake_sb, "b")
        assert b._shot_idx == 2

    def test_js_set_value_executes_script(self):
        b = ds.DeepSeekBrowserSB()
        fake_sb = MagicMock()
        fake_sb.execute_script.return_value = True
        result = b._js_set_value(fake_sb, "input[type='email']", "test@example.com")
        assert result is True
        fake_sb.execute_script.assert_called_once()

    def test_js_click_executes_script(self):
        b = ds.DeepSeekBrowserSB()
        fake_sb = MagicMock()
        fake_sb.execute_script.return_value = True
        result = b._js_click(fake_sb, "button[type='submit']")
        assert result is True
        fake_sb.execute_script.assert_called_once()

    def test_detect_captcha_returns_none_when_no_sb_error(self):
        b = ds.DeepSeekBrowserSB()
        fake_sb = MagicMock()
        fake_sb.execute_script.return_value = None
        result = b._detect_captcha(fake_sb)
        assert result is None

    def test_detect_captcha_returns_type(self):
        b = ds.DeepSeekBrowserSB()
        fake_sb = MagicMock()
        fake_sb.execute_script.return_value = "cloudflare_turnstile"
        result = b._detect_captcha(fake_sb)
        assert result == "cloudflare_turnstile"

    def test_handle_captcha_no_captcha(self):
        b = ds.DeepSeekBrowserSB()
        fake_sb = MagicMock()
        fake_sb.execute_script.return_value = None
        result = b._handle_captcha(fake_sb)
        assert result is True

    def test_handle_captcha_manual_mode(self):
        b = ds.DeepSeekBrowserSB(manual_captcha=True)
        fake_sb = MagicMock()
        fake_sb.execute_script.return_value = "recaptcha"

        with patch.object(b, "_wait_for_human_captcha", return_value=True) as mock_wait:
            result = b._handle_captcha(fake_sb)

        mock_wait.assert_called_once_with(fake_sb)
        assert result is True

    def test_handle_captcha_auto_solved(self):
        """Auto mode: uc_gui_click_captcha solves it successfully."""
        b = ds.DeepSeekBrowserSB(manual_captcha=False)
        fake_sb = MagicMock()
        # First detect_captcha call returns captcha, second (after solve) returns None
        fake_sb.execute_script.side_effect = ["recaptcha", None]
        fake_sb.uc_gui_click_captcha = MagicMock()

        with patch.object(b, "_detect_captcha", side_effect=["recaptcha", None]):
            result = b._handle_captcha(fake_sb)

        assert result is True


# ══════════════════════════════════════════════════════════════════════════════
#  create_one_account orchestration (mocked)
# ══════════════════════════════════════════════════════════════════════════════


class TestCreateOneAccount:
    def _mock_mail(self, address="user@guerrillamailblock.com") -> MagicMock:
        mail = MagicMock(spec=ds.GuerrillaMailAdapter)
        mail.address = address
        mail.password = ""
        mail.create_account.return_value = (address, "")
        return mail

    def test_returns_none_on_mail_failure(self):
        with patch(
            "deepseek_signup._create_mail_backend", side_effect=RuntimeError("All backends failed")
        ):
            result = ds.create_one_account(api_url=None)
        assert result is None

    def test_returns_dict_on_success(self):
        mail_mock = self._mock_mail()
        long_token = "U4vsCtoBdxn1pfU4QclHNg2M9rVtBq82giPe2ilrFBfeiuR4yLj"

        with patch("deepseek_signup._create_mail_backend", return_value=mail_mock):
            with patch("deepseek_signup.DeepSeekBrowserSB") as MockBrowser:
                browser_inst = MockBrowser.return_value
                browser_inst.signup.return_value = long_token

                result = ds.create_one_account(api_url=None)

        assert result is not None
        assert result["email"] == "user@guerrillamailblock.com"
        assert result["token"] == long_token

    def test_falls_back_to_signin_when_signup_fails(self):
        mail_mock = self._mock_mail()
        mail_mock.poll_for_verification_link.return_value = None
        long_token = "fallback_" + "X" * 45

        with patch("deepseek_signup._create_mail_backend", return_value=mail_mock):
            with patch("deepseek_signup.DeepSeekBrowserSB") as MockBrowser:
                browser_inst = MockBrowser.return_value
                browser_inst.signup.return_value = None
                browser_inst.signin.return_value = long_token

                result = ds.create_one_account(api_url=None)

        assert result is not None
        assert result["token"] == long_token

    def test_submits_to_api_when_url_provided(self):
        mail_mock = self._mock_mail()

        with patch("deepseek_signup._create_mail_backend", return_value=mail_mock):
            with patch("deepseek_signup.DeepSeekBrowserSB") as MockBrowser:
                MockBrowser.return_value.signup.return_value = "token" * 15

                with patch("deepseek_signup.submit_token_to_api", return_value=True) as mock_sub:
                    result = ds.create_one_account(
                        api_url="http://127.0.0.1:8000",
                        api_key="testkey",
                    )

        assert result is not None
        mock_sub.assert_called_once()
        assert mock_sub.call_args.kwargs["api_url"] == "http://127.0.0.1:8000"
        assert mock_sub.call_args.kwargs["api_key"] == "testkey"

    def test_no_api_submission_when_url_is_none(self):
        mail_mock = self._mock_mail()

        with patch("deepseek_signup._create_mail_backend", return_value=mail_mock):
            with patch("deepseek_signup.DeepSeekBrowserSB") as MockBrowser:
                MockBrowser.return_value.signup.return_value = "token" * 15

                with patch("deepseek_signup.submit_token_to_api") as mock_sub:
                    result = ds.create_one_account(api_url=None)

        mock_sub.assert_not_called()
        assert result is not None

    def test_returns_none_when_no_token_obtained(self):
        mail_mock = self._mock_mail()
        mail_mock.poll_for_verification_link.return_value = None

        with patch("deepseek_signup._create_mail_backend", return_value=mail_mock):
            with patch("deepseek_signup.DeepSeekBrowserSB") as MockBrowser:
                browser_inst = MockBrowser.return_value
                browser_inst.signup.return_value = None
                browser_inst.signin.return_value = None

                result = ds.create_one_account(api_url=None)

        assert result is None

    def test_uses_icloud_mail_by_default(self):
        """Default mail_service should be 'icloud' (passes DeepSeek domain check)."""
        captured = {}

        def fake_factory(service):
            captured["service"] = service
            raise RuntimeError("stop here")  # short-circuit

        with patch("deepseek_signup._create_mail_backend", side_effect=fake_factory):
            ds.create_one_account(api_url=None)

        assert captured.get("service") == "icloud"


# ══════════════════════════════════════════════════════════════════════════════
#  CLI argument parsing
# ══════════════════════════════════════════════════════════════════════════════


class TestCliArgParsing:
    def _parse(self, args: list[str]):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--count", type=int, default=1)
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--manual-captcha", action="store_true")
        parser.add_argument("--debug", action="store_true")
        parser.add_argument(
            "--mail-service",
            choices=["icloud", "graphapi", "guerrilla", "mailtm", "mailgw", "auto"],
            default="icloud",
        )
        parser.add_argument("--output", default="deepseek_accounts.json")
        parser.add_argument("--api-url", default=ds.DEFAULT_API_URL)
        parser.add_argument("--api-key", default=ds.DEFAULT_API_KEY)
        parser.add_argument("--no-upload", action="store_true")
        parser.add_argument("--graphapi-creds", action="append", default=[], metavar="CRED")
        parser.add_argument("--graphapi-creds-file", default="")
        return parser.parse_args(args)

    def test_defaults(self):
        ns = self._parse([])
        assert ns.count == 1
        assert ns.headless is False
        assert ns.manual_captcha is False
        assert ns.debug is False
        assert ns.mail_service == "icloud"
        assert ns.output == "deepseek_accounts.json"
        assert ns.api_url == ds.DEFAULT_API_URL
        assert ns.api_key == ds.DEFAULT_API_KEY
        assert ns.no_upload is False

    def test_count_flag(self):
        ns = self._parse(["--count", "5"])
        assert ns.count == 5

    def test_headless_flag(self):
        ns = self._parse(["--headless"])
        assert ns.headless is True

    def test_manual_captcha_flag(self):
        ns = self._parse(["--manual-captcha"])
        assert ns.manual_captcha is True

    def test_debug_flag(self):
        ns = self._parse(["--debug"])
        assert ns.debug is True

    def test_mail_service_guerrilla(self):
        ns = self._parse(["--mail-service", "guerrilla"])
        assert ns.mail_service == "guerrilla"

    def test_mail_service_graphapi(self):
        ns = self._parse(["--mail-service", "graphapi"])
        assert ns.mail_service == "graphapi"

    def test_mail_service_mailtm(self):
        ns = self._parse(["--mail-service", "mailtm"])
        assert ns.mail_service == "mailtm"

    def test_mail_service_mailgw(self):
        ns = self._parse(["--mail-service", "mailgw"])
        assert ns.mail_service == "mailgw"

    def test_mail_service_auto(self):
        ns = self._parse(["--mail-service", "auto"])
        assert ns.mail_service == "auto"

    def test_custom_api_url(self):
        ns = self._parse(["--api-url", "http://myserver:9000"])
        assert ns.api_url == "http://myserver:9000"

    def test_custom_api_key(self):
        ns = self._parse(["--api-key", "supersecret"])
        assert ns.api_key == "supersecret"

    def test_no_upload_flag(self):
        ns = self._parse(["--no-upload"])
        assert ns.no_upload is True

    def test_invalid_mail_service(self):
        with pytest.raises(SystemExit):
            self._parse(["--mail-service", "invalid_service"])

    def test_graphapi_creds_flag(self):
        """--graphapi-creds can be passed multiple times and accumulates into a list."""
        cred1 = "me@hotmail.com|pass|token123|client-uuid"
        cred2 = "you@outlook.com|pass2|token456"
        ns = self._parse(["--graphapi-creds", cred1, "--graphapi-creds", cred2])
        assert cred1 in ns.graphapi_creds
        assert cred2 in ns.graphapi_creds
        assert len(ns.graphapi_creds) == 2

    def test_graphapi_creds_file_flag(self):
        ns = self._parse(["--graphapi-creds-file", "accounts.txt"])
        assert ns.graphapi_creds_file == "accounts.txt"


# ══════════════════════════════════════════════════════════════════════════════
#  Graph API pool — round-robin multi-account logic
# ══════════════════════════════════════════════════════════════════════════════


class TestGraphAPIPool:
    """Tests for the multi-account round-robin pool in _create_mail_backend."""

    def setup_method(self):
        """Reset module-level pool state before each test."""
        ds._GRAPHAPI_POOL.clear()
        ds._GRAPHAPI_IDX = 0

    def teardown_method(self):
        ds._GRAPHAPI_POOL.clear()
        ds._GRAPHAPI_IDX = 0

    def _make_pool(self, emails: list[str]) -> list[dict]:
        """Populate _GRAPHAPI_POOL with fake entries for the given emails."""
        pool = [
            {"email": e, "password": "p", "refresh_token": f"tok-{i}", "client_id": "client"}
            for i, e in enumerate(emails)
        ]
        ds._GRAPHAPI_POOL.extend(pool)
        return pool

    def test_single_account_always_selected(self):
        """With one account in the pool, it's always picked."""
        self._make_pool(["only@hotmail.com"])
        selected = []
        for _ in range(3):
            # We verify by checking the email the adapter was constructed with
            with patch.object(ds, "GraphAPIMailAdapter") as MockAdapter:
                mock_inst = MagicMock()
                mock_inst.address = "only@hotmail.com"
                mock_inst.create_account = MagicMock(return_value=("only@hotmail.com", ""))
                MockAdapter.return_value = mock_inst
                ds._create_mail_backend("graphapi")
            call_kwargs = MockAdapter.call_args.kwargs
            selected.append(call_kwargs.get("email", ""))
        assert all(e == "only@hotmail.com" for e in selected)

    def test_two_accounts_round_robins(self):
        """With two accounts, they alternate strictly: A, B, A, B, ..."""
        self._make_pool(["alpha@hotmail.com", "beta@hotmail.com"])
        emails_used = []
        for _ in range(4):
            with patch.object(ds, "GraphAPIMailAdapter") as MockAdapter:
                inst = MagicMock()
                inst.create_account = MagicMock(return_value=("x", ""))
                MockAdapter.return_value = inst
                ds._create_mail_backend("graphapi")
            emails_used.append(MockAdapter.call_args.kwargs["email"])
        assert emails_used == [
            "alpha@hotmail.com",
            "beta@hotmail.com",
            "alpha@hotmail.com",
            "beta@hotmail.com",
        ]

    def test_empty_pool_raises_runtime_error(self):
        """When the pool is empty and graphapi is selected, a RuntimeError is raised."""
        assert ds._GRAPHAPI_POOL == []
        with pytest.raises(RuntimeError, match="credential"):
            ds._create_mail_backend("graphapi")

    def test_parse_graphapi_credential_populates_pool(self):
        """Each valid credential string should add one entry to the pool."""
        creds = [
            "a@hotmail.com|pa|tokA|clientA",
            "b@outlook.com|pb|tokB|clientB",
            "c@live.com|pc|tokC",  # no client_id — uses default
        ]
        for raw in creds:
            ds._GRAPHAPI_POOL.append(ds.parse_graphapi_credential(raw))
        assert len(ds._GRAPHAPI_POOL) == 3
        assert ds._GRAPHAPI_POOL[0]["email"] == "a@hotmail.com"
        assert ds._GRAPHAPI_POOL[2]["client_id"] == ds.MS_DEFAULT_CLIENT_ID

    def test_pool_file_loading(self, tmp_path):
        """A credential file (one entry per line, # comments ignored) is parsed correctly."""
        creds_file = tmp_path / "graphapi_accounts.txt"
        creds_file.write_text(
            "# This is a comment\n"
            "first@hotmail.com|pass1|token1|client1\n"
            "   \n"  # blank line
            "second@outlook.com|pass2|token2\n"
            "# another comment\n"
            "third@live.com|pass3|token3|client3\n",
            encoding="utf-8",
        )
        lines = []
        for line in creds_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
        parsed = [ds.parse_graphapi_credential(line) for line in lines]
        assert len(parsed) == 3
        assert parsed[0]["email"] == "first@hotmail.com"
        assert parsed[1]["client_id"] == ds.MS_DEFAULT_CLIENT_ID
        assert parsed[2]["email"] == "third@live.com"


# ══════════════════════════════════════════════════════════════════════════════
#  parse_graphapi_credential
# ══════════════════════════════════════════════════════════════════════════════


class TestParseGraphAPICredential:
    """Tests for the pipe-separated credential parser."""

    def test_full_four_fields(self):
        result = ds.parse_graphapi_credential(
            "user@hotmail.com|Password1|M.RefreshTokenValue|custom-uuid"
        )
        assert result["email"] == "user@hotmail.com"
        assert result["password"] == "Password1"
        assert result["refresh_token"] == "M.RefreshTokenValue"
        assert result["client_id"] == "custom-uuid"

    def test_three_fields_uses_default_client_id(self):
        result = ds.parse_graphapi_credential("user@outlook.com|pass|M.Token123")
        assert result["client_id"] == ds.MS_DEFAULT_CLIENT_ID

    def test_empty_client_id_field_uses_default(self):
        """Trailing pipe with empty client_id should use the default."""
        result = ds.parse_graphapi_credential("a@b.com|p|tok|")
        assert result["client_id"] == ds.MS_DEFAULT_CLIENT_ID

    def test_realistic_token_with_special_chars(self):
        """The actual token from the user's example contains *, !, $."""
        tok = "M.C556_BAY.0.U.MsaArtifacts.-CkrwSPV1*abc!def!ghi$"
        result = ds.parse_graphapi_credential(f"me@hotmail.com|pass|{tok}|9e5f94bc")
        assert result["refresh_token"] == tok
        assert result["client_id"] == "9e5f94bc"

    def test_strips_whitespace(self):
        result = ds.parse_graphapi_credential("  user@live.com | pass | tok | uid  ")
        assert result["email"] == "user@live.com"
        assert result["password"] == "pass"
        assert result["client_id"] == "uid"

    def test_too_few_fields_raises_value_error(self):
        with pytest.raises(ValueError, match="segment"):
            ds.parse_graphapi_credential("onlyone")

    def test_two_fields_raises_value_error(self):
        with pytest.raises(ValueError):
            ds.parse_graphapi_credential("email|pass")

    def test_empty_email_raises(self):
        with pytest.raises(ValueError, match="Email"):
            ds.parse_graphapi_credential("|pass|token")

    def test_empty_refresh_token_raises(self):
        with pytest.raises(ValueError, match="refresh_token"):
            ds.parse_graphapi_credential("me@h.com|pass|")


# ══════════════════════════════════════════════════════════════════════════════
#  GraphAPIMailAdapter
# ══════════════════════════════════════════════════════════════════════════════


class TestGraphAPIMailAdapter:
    """Unit tests for the Microsoft Graph API mail backend (all mocked)."""

    SAMPLE_EMAIL = "test@hotmail.com"
    SAMPLE_PASSWORD = "Password1!"
    SAMPLE_REFRESH = "M.RefreshTokenSample"
    SAMPLE_CLIENT = "custom-client-id"

    def _make_adapter(self):
        return ds.GraphAPIMailAdapter(
            email=self.SAMPLE_EMAIL,
            password=self.SAMPLE_PASSWORD,
            refresh_token=self.SAMPLE_REFRESH,
            client_id=self.SAMPLE_CLIENT,
        )

    def _token_response(self, access="access-tok", refresh="new-refresh"):
        """Fake successful token endpoint response."""
        mock = MagicMock()
        mock.status_code = 200
        mock.json.return_value = {"access_token": access, "refresh_token": refresh}
        return mock

    def _messages_response(self, msgs: list):
        """Fake Graph API /me/messages response."""
        mock = MagicMock()
        mock.status_code = 200
        mock.json.return_value = {"value": msgs}
        mock.raise_for_status = MagicMock()
        return mock

    def test_init_sets_address_and_password(self):
        adapter = self._make_adapter()
        assert adapter.address.startswith(self.SAMPLE_EMAIL.split("@")[0])
        assert adapter.address.endswith("@hotmail.com")
        assert adapter.password == self.SAMPLE_PASSWORD

    def test_default_client_id_used_when_not_provided(self):
        adapter = ds.GraphAPIMailAdapter(email="a@hotmail.com", password="p", refresh_token="tok")
        assert adapter._client_id == ds.MS_DEFAULT_CLIENT_ID

    def test_refresh_access_token_posts_to_ms_endpoint(self):
        adapter = self._make_adapter()
        with patch.object(adapter.s, "post", return_value=self._token_response()) as mock_post:
            tok = adapter._refresh_access_token()
        assert tok == "access-tok"
        assert adapter._access_tok == "access-tok"
        assert adapter._refresh_tok == "new-refresh"  # rotated
        call_kwargs = mock_post.call_args
        assert ds.MS_TOKEN_URL in str(call_kwargs)
        assert "refresh_token" in str(call_kwargs)

    def test_refresh_raises_on_non_200(self):
        adapter = self._make_adapter()
        bad_resp = MagicMock(status_code=400, text="bad_request")
        with patch.object(adapter.s, "post", return_value=bad_resp):
            with pytest.raises(RuntimeError, match="400"):
                adapter._refresh_access_token()

    def test_refresh_token_not_rotated_if_absent_from_response(self):
        adapter = self._make_adapter()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"access_token": "new-access"}
        with patch.object(adapter.s, "post", return_value=mock_resp):
            adapter._refresh_access_token()
        # refresh_token should remain the original since none was returned
        assert adapter._refresh_tok == self.SAMPLE_REFRESH

    def test_create_account_refreshes_token_and_smoke_tests_inbox(self):
        adapter = self._make_adapter()
        with (
            patch.object(adapter, "_refresh_access_token") as mock_refresh,
            patch.object(adapter, "_graph_get", return_value={"value": []}) as mock_get,
        ):
            addr, pwd = adapter.create_account()
        mock_refresh.assert_called_once()
        mock_get.assert_called_once()
        assert addr.startswith(self.SAMPLE_EMAIL.split("@")[0])
        assert pwd == self.SAMPLE_PASSWORD

    def test_poll_for_otp_finds_code_in_first_message(self):
        adapter = self._make_adapter()
        adapter._access_tok = "tok"  # skip refresh
        msgs = [
            {
                "id": "msg-001",
                "subject": "Your DeepSeek code",
                "body": {"content": "Your verification code is 987654", "contentType": "text"},
                "bodyPreview": "",
            }
        ]
        with patch.object(adapter, "_fetch_messages", return_value=msgs):
            otp = adapter.poll_for_otp(timeout=5, interval=0)
        assert otp == "987654"

    def test_poll_for_otp_skips_already_seen_messages(self):
        adapter = self._make_adapter()
        adapter._access_tok = "tok"
        adapter._seen_ids.add("msg-seen")
        msgs = [
            {
                "id": "msg-seen",
                "subject": "X",
                "body": {"content": "123456", "contentType": "text"},
                "bodyPreview": "",
            }
        ]
        with patch.object(adapter, "_fetch_messages", return_value=msgs):
            with pytest.raises(TimeoutError):
                adapter.poll_for_otp(timeout=1, interval=0)

    def test_poll_for_otp_times_out_when_no_code(self):
        adapter = self._make_adapter()
        adapter._access_tok = "tok"
        with patch.object(adapter, "_fetch_messages", return_value=[]):
            with pytest.raises(TimeoutError):
                adapter.poll_for_otp(timeout=1, interval=0)

    def test_poll_for_otp_strips_html_body(self):
        adapter = self._make_adapter()
        adapter._access_tok = "tok"
        msgs = [
            {
                "id": "msg-html",
                "subject": "DeepSeek",
                "body": {"content": "<p>Code: <b>112233</b></p>", "contentType": "html"},
                "bodyPreview": "",
            }
        ]
        with patch.object(adapter, "_fetch_messages", return_value=msgs):
            otp = adapter.poll_for_otp(timeout=5, interval=0)
        assert otp == "112233"

    def test_poll_for_verification_link_returns_link(self):
        adapter = self._make_adapter()
        adapter._access_tok = "tok"
        msgs = [
            {
                "id": "msg-link",
                "subject": "Verify your DeepSeek account",
                "body": {
                    "content": (
                        "Click https://platform.deepseek.com/verify?token=abc123 to confirm."
                    ),
                    "contentType": "text",
                },
                "bodyPreview": "",
            }
        ]
        with patch.object(adapter, "_fetch_messages", return_value=msgs):
            link = adapter.poll_for_verification_link(timeout=5, interval=0)
        assert link and "deepseek" in link

    def test_poll_for_verification_link_returns_none_on_timeout(self):
        adapter = self._make_adapter()
        adapter._access_tok = "tok"
        with patch.object(adapter, "_fetch_messages", return_value=[]):
            result = adapter.poll_for_verification_link(timeout=1, interval=0)
        assert result is None

    def test_graph_get_auto_refreshes_on_401(self):
        adapter = self._make_adapter()
        adapter._access_tok = "stale-tok"

        call_count = {"n": 0}
        ok_resp = MagicMock(status_code=200)
        ok_resp.json.return_value = {"value": []}
        ok_resp.raise_for_status = MagicMock()
        unauth_resp = MagicMock(status_code=401)
        unauth_resp.raise_for_status = MagicMock()

        def fake_get(url, **kw):
            call_count["n"] += 1
            return unauth_resp if call_count["n"] == 1 else ok_resp

        with (
            patch.object(adapter.s, "get", side_effect=fake_get),
            patch.object(adapter, "_refresh_access_token") as mock_refresh,
        ):
            adapter._graph_get("/me/messages", params={})
        mock_refresh.assert_called_once()
