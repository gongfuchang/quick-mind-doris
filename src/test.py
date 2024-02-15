import requests
import json
url = "http://localhost:8000/"

with requests.get(url, stream=True) as r:
    for chunk in r.iter_content(1024):  # or, for line in r.iter_lines():
        print(chunk)

# with requests.post(url, stream=True, json=json.loads('{"query": "如何进入强制模式"}')) as r:
#     for chunk in r.iter_content(1024):  # or, for line in r.iter_lines():
#         print(chunk)