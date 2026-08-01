from seleniumbase import SB
import time

with SB(test=True, headless=True) as sb:
    sb.open("https://platform.deepseek.com/sign_up")
    time.sleep(5)
    inputs = sb.find_elements("input")
    for i, inp in enumerate(inputs):
        print(f"Input {i}: placeholder='{inp.get_attribute('placeholder')}' type='{inp.get_attribute('type')}' class='{inp.get_attribute('class')}'")
    buttons = sb.find_elements("div[role='button'], button")
    for i, btn in enumerate(buttons):
        print(f"Button {i}: text='{btn.text}' type='{btn.get_attribute('type')}'")
