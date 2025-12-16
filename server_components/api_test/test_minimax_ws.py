# minimax_tts_local.py
import requests, base64, sys, json, time

GROUP_ID = "1987687468056449719"
API_KEY  = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiLquYDsiJjtmLgiLCJVc2VyTmFtZSI6Iuq5gOyImO2YuCIsIkFjY291bnQiOiIiLCJTdWJqZWN0SUQiOiIxOTg3Njg3NDY4MDYwNjQ4MTE5IiwiUGhvbmUiOiIiLCJHcm91cElEIjoiMTk4NzY4NzQ2ODA1NjQ0OTcxOSIsIlBhZ2VOYW1lIjoiIiwiTWFpbCI6InBtQGluc3VuZXRmYy5jb20iLCJDcmVhdGVUaW1lIjoiMjAyNS0xMS0xMCAxMDoyMjoxNCIsIlRva2VuVHlwZSI6MSwiaXNzIjoibWluaW1heCJ9.eDIqZYbtrToy5op-Fl879l6AwhJkHL80xygDS3HTCgzjWfAnXtjvpavhYjGDWGifbgn9tlUw4py5nG4IkCErmkKUQuNc9SWaeweAP_vkoF_GOGbNwiIKaktYY05I7R_0ACuByv9u4dpL9sbJ1peDPFynJadfHnI0Kgz0NLDif_J5ArkQE1V2zH1ptAKmANVr0xUqq3WqvU37jPsIQxh3Jcj8ceoNVuO1Tbjmc2G3NzGh6XVG7pR035R-mTmqevaxMbddKRIO07zDK5OCfZtkszgtyiYVIBPHrSx6OOASPOtpsOkE5UKbW2Ll3RgPDAKIkw5nsNIQCdQ8MOY_MuUy-Q"

TTS_URL = "https://api.minimax.io/v1/t2a_async_v2"
QUERY_URL = "https://api.minimax.io/v1/query/t2a_async_query_v2"
DOWNLOAD_URL = "https://api.minimax.io/v1/files/retrieve_content"

TEXT = "안녕하세요 고객님, 미니맥스 비동기 TTS 테스트입니다."
MODEL = "speech-2.6-hd"   # if this errors, try "speech-01"
VOICE_ID = "English_expressive_narrator"  # or whatever your acct supports

# 1) create TTS task
create_payload = {
    "model": MODEL,
    "text": TEXT,                     # 👈 THIS was missing before
    "language_boost": "auto",
    "voice_setting": {
        "voice_id": VOICE_ID,
        "speed": 1,
        "vol": 10,
        "pitch": 1
    },
    "audio_setting": {
        "audio_sample_rate": 32000,
        "bitrate": 128000,
        "format": "mp3",
        "channel": 2
    }
}
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

print("📤 creating MiniMax TTS task...")
resp = requests.post(TTS_URL, headers=headers, json=create_payload)
resp.raise_for_status()
data = resp.json()
# expect: {"base_resp": {"status_code": 0,...}, "task_id": "..."}
if data.get("base_resp", {}).get("status_code") != 0:
    print("create error:", data)
    raise SystemExit

task_id = data["task_id"]
print("✅ task created:", task_id)

# 2) poll for result
while True:
    time.sleep(1.5)
    q = requests.get(
        f"{QUERY_URL}?task_id={task_id}",
        headers=headers,
    )
    q.raise_for_status()
    qd = q.json()
    # expect: status: PROCESSING / SUCCESS
    status_code = qd.get("base_resp", {}).get("status_code", -1)
    task_status = qd.get("task_status")
    print("⏳ status:", task_status, status_code)
    if task_status == "SUCCESS":
        file_id = qd["file_id"]
        break
    if task_status in ("FAILED", "ERROR"):
        print("task failed:", qd)
        raise SystemExit

print("✅ task success, file_id:", file_id)

# 3) download file
d = requests.get(
    f"{DOWNLOAD_URL}?file_id={file_id}",
    headers=headers,
)
d.raise_for_status()
out_path = "minimax_async.mp3"
with open(out_path, "wb") as f:
    f.write(d.content)
print("✅ saved", out_path)