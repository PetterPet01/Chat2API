import json

with open('2026-08-02-011305.har', 'r', encoding='utf-8-sig') as f:
    har = json.load(f)

for entry in har['log']['entries']:
    req = entry['request']
    url = req['url']
    if 'register' in url:
        print(f"URL: {url}")
        headers = {h['name'].lower(): h['value'] for h in req['headers']}
        print(f"Headers: {headers.keys()}")
        for k, v in headers.items():
            if 'pow' in k or 'challenge' in k:
                print(f"  {k}: {v}")
