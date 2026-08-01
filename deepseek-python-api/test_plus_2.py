from seleniumbase import SB
import time

email = "VasseurMaxim1982+test1@hotmail.com"

with SB(uc=True, headless=True) as sb:
    sb.open("https://platform.deepseek.com/sign_up")
    sb.sleep(3)
    sb.type("input[type='email'], input[placeholder*='Email' i]", email)
    sb.type("input[type='password']", "TestPwd123!@#")
    
    # We just need to check if the 'Please enter a valid email address' appears.
    # Usually it appears immediately after typing or blur.
    sb.execute_script("document.querySelector('input[type=\"email\"]').blur();")
    sb.sleep(1)
    
    error = sb.execute_script("""
        const el = document.querySelector('.ds-form-item-message-error');
        return el ? el.innerText : 'NO_ERROR';
    """)
    print("Error:", error)
