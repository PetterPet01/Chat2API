import json

with open('2026-08-02-011305.har', 'r', encoding='utf-8-sig') as f:
    har = json.load(f)

for entry in har['log']['entries']:
    req = entry['request']
    url = req['url']
    if 'deepseek' in url and req['method'] in ('POST', 'PUT', 'PATCH'):
        print(f"URL: {url}")
        
        if 'postData' in req:
            text = req['postData'].get('text', '')
            print(f"Payload: {text}")
        print("-" * 40)
