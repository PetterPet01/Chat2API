from seleniumbase import SB
import time

email = "VasseurMaxim1982@hotmail.com"

with SB(uc=True, headless=True) as sb:
    sb.open("https://chat.deepseek.com")
    sb.sleep(3)
    sb.type("input[type='email'], input[placeholder*='Email' i]", email)
    sb.sleep(2)
    # The first screen just asks for email? Or does it show password immediately?
    # Let's take a shot
    sb.save_screenshot("ds_test_1.png")
    
    # If it's a two-step, we might need to click continue
    try:
        sb.click('div[role="button"]:contains("Continue"), button:contains("Continue")')
        sb.sleep(2)
    except:
        pass

    sb.save_screenshot("ds_test_2.png")
    
    # Click forgot password
    try:
        sb.click('a:contains("Forgot password"), div:contains("Forgot password"), span:contains("Forgot password")')
        sb.sleep(2)
        sb.save_screenshot("ds_test_3_forgot.png")
    except Exception as e:
        print("No forgot password:", e)

