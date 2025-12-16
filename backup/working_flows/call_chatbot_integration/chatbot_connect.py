import requests
import json

url = "http://15.165.60.45:5000/chat"

payload = json.dumps({
  "session_id": "38bb86ff-9a62-4e04-af1e-2dacdbda8f99",
  "question": "안녕하세요"
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)