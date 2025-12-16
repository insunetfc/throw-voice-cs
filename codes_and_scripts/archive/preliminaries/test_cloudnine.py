import os
import json
import sys
import urllib.request

# Use env var first; fallback to your public IP; for local dev you can set this to http://127.0.0.1:8000/process
API_URL = os.getenv("API_URL", "http://3.38.171.85:8000/process")

def lambda_handler(event, context):
    text = (event or {}).get("text", "Hello from Lambda")

    payload = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(API_URL, data=payload, headers={"Content-Type": "application/json"})

    # add a timeout so Lambda doesn’t hang forever
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp_data = resp.read().decode("utf-8")
    return json.loads(resp_data)

# ----- Local runner -----
if __name__ == "__main__":
    # Usage:
    #   API_URL=http://127.0.0.1:8000/process python bridge_client.py "안녕하세요"
    text = "Hello from local"
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    result = lambda_handler({"text": text}, None)
    print(result)
