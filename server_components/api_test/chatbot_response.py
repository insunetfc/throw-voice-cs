import requests
import json
import time

# -------- CONFIG --------
# URL = "https://99c0ad659065.ngrok-free.app/respond"
URL = "https://honest-trivially-buffalo.ngrok-free.app/chatbot/respond"

# -------- TEST INPUT --------
payload = {
    "text": "수수료 몇 %에요?"   # any test sentence
}

headers = {
    "Content-Type": "application/json"
}

# -------- REQUEST --------
start_time = time.time()
response = requests.post(URL, headers=headers, data=json.dumps(payload))
elapsed = time.time() - start_time

# -------- OUTPUT --------
print("Response JSON:")
try:
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))
except Exception:
    print(response.text)
print(f"\n⏱️ Elapsed time: {elapsed:.3f} sec")
