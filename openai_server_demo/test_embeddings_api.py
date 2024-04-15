import requests
import json

url = 'http://localhost:19327/v1/embeddings'
headers = {
    'Content-Type': 'application/json'
}

data = {
    'input': "今天天气很好"
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.json())