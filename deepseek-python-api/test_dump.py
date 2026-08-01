from seleniumbase import SB
import json
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("deepseek")

def _scan_storage_for_token(driver) -> str | None:
    try:
        items = driver.execute_script("""
            (function() {
                const out = [];
                const ut = localStorage.getItem('userToken');
                if (ut && ut.length > 20) out.push({k: 'userToken', v: ut});
                return out;
            })()
        """)
        return items
    except Exception as e:
        return None

with SB(uc=True, headless=False) as sb:
    sb.uc_open_with_reconnect('https://chat.deepseek.com/sign_in', reconnect_time=4)
    sb.sleep(5)
    sb.type('input[placeholder*=\'Email\' i]', 'VasseurMaxim1982+ds1785606164@hotmail.com')
    sb.type('input[type=\'password\'], input[placeholder*=\'Password\' i]', 'CR1419njouhd!@')
    from selenium.webdriver.common.keys import Keys
    sb.press_keys('input[type=\'password\'], input[placeholder*=\'Password\' i]', Keys.RETURN)
    sb.sleep(10)
    
    print("URL:", sb.get_current_url())
    
    ls = sb.execute_script('''
        var out = {};
        for (var i = 0; i < localStorage.length; i++) {
            var k = localStorage.key(i);
            out[k] = localStorage.getItem(k);
        }
        return out;
    ''')
    print("DICT DUMP:", ls.keys())
    
    arr_dump = _scan_storage_for_token(sb.driver)
    print("ARR DUMP:", arr_dump)
