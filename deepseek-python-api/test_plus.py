from seleniumbase import SB
import time

email = "VasseurMaxim1982+test@hotmail.com"

with SB(uc=True, headless=True) as sb:
    sb.open("https://platform.deepseek.com/sign_up")
    sb.sleep(3)
    sb.type("input[type='email'], input[placeholder*='Email' i]", email)
    sb.type("input[type='password']", "TestPwd123!@#")
    sb.click("div.ds-checkbox")
    sb.click("button:contains('Sign up')")
    sb.sleep(3)
    # Check if there is an error message
    error = sb.execute_script("""
        const el = document.querySelector('.ds-form-item-message-error');
        return el ? el.innerText : 'NO_ERROR';
    """)
    print("Error:", error)
