from seleniumbase import SB
import json
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("deepseek")

def _search_dict_for_token(data) -> str | None:
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
    return None

def _find_deepseek_token_in_str(s: str) -> str | None:
    return None

def _scan_storage_for_token(driver) -> str | None:
    try:
        items = driver.execute_script("""
            (function() {
                const out = [];
                const ut = localStorage.getItem('userToken');
                if (ut && ut.length > 20) out.push({k: 'userToken', v: ut});

                for (let i = 0; i < localStorage.length; i++) {
                    const k = localStorage.key(i);
                    if (k === 'userToken') continue;
                    const v = localStorage.getItem(k);
                    if (v && v.length > 30) {
                        out.push({k: k, v: v});
                    }
                }
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
                        if "value" in parsed and isinstance(parsed["value"], str) and len(parsed["value"]) > 20:
                            t = parsed["value"]
                            log.info(f"  🔑 Token found in localStorage['userToken'] (JSON value): {t[:12]}...")
                            return t
                    except Exception as e:
                        log.warning(f"  Failed to parse userToken JSON: {e}")
                
                log.info(f"  🔑 Token found in localStorage['userToken']: {val[:12]}...")
                return val
            if val.startswith("{") or val.startswith("["):
                try:
                    parsed = json.loads(val)
                    t = _search_dict_for_token(parsed)
                    if t:
                        log.info(f"  🔑 Token found in storage['{key}'] (JSON): {t[:12]}...")
                        return t
                except Exception:
                    pass
            t = _find_deepseek_token_in_str(val)
            if t and len(t) >= 40:
                log.info(f"  🔑 Token found in storage['{key}']: {t[:12]}...")
                return t
        
        found_keys = [item.get("k") for item in (items or [])]
        if found_keys:
            log.info(f"  [Scan] No token found yet. Storage keys present: {', '.join(k for k in found_keys if k)}")
        else:
            log.info("  [Scan] Storage is completely empty!")

    except Exception as e:
        log.error(f"  Storage scan error: {e}")
    return None

with SB(uc=True, headless=False) as sb:
    sb.uc_open_with_reconnect('https://chat.deepseek.com/sign_in', reconnect_time=4)
    sb.sleep(5)
    sb.type('input[placeholder*=\'Email\' i]', 'VasseurMaxim1982+ds1785606164@hotmail.com')
    sb.type('input[type=\'password\'], input[placeholder*=\'Password\' i]', 'CR1419njouhd!@')
    from selenium.webdriver.common.keys import Keys
    sb.press_keys('input[type=\'password\'], input[placeholder*=\'Password\' i]', Keys.RETURN)
    sb.sleep(10)
    
    token = _scan_storage_for_token(sb.driver)
    print("EXTRACTED TOKEN:", token)
