import requests
import json

url = 'http://localhost:19327/v1/completions'
headers = {
    'Content-Type': 'application/json'
}

data = {
    'prompt': "美国总统是谁"
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.json())