import requests
import json

url = 'http://localhost:19327/v1/chat/completions'
headers = {
    'Content-Type': 'application/json'
}

data = {
    'messages': [
        {'role': 'user', 'message': '给我介绍一下北京吧'}
    ],
    'repetition_penalty': 1.0
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.json())