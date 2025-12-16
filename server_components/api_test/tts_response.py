import requests
import os
import logging
logging.basicConfig(level=logging.INFO)

BASE = "http://honest-trivially-buffalo.ngrok-free.app/tts"
AUTH = {"Authorization": "Bearer YOUR_TOKEN"}  # same auth as before

# Optional: path to your reference audio file (leave None to skip)
REF_PATH = "/home/tiongsik/Python/outbound_calls/files/audio/KoreanPrompt.wav"  # or None

def upload_reference():
    if not REF_PATH or not os.path.exists(REF_PATH):
        print("⚠️ No reference WAV provided, skipping upload.")
        return None
    print("📼 Uploading reference...")
    files = {"ref_wav": open(REF_PATH, "rb")}
    r = requests.post(f"{BASE}/reference", files=files, headers=AUTH)
    r.raise_for_status()
    data = r.json()
    print("✅ Uploaded reference:", data)
    return data["ref_id"]

def synthesize(text, ref_id=None):
    payload = {"text": text}
    if ref_id:
        payload["ref_id"] = ref_id
    print("🎙️ Sending synthesis request:", payload)
    r = requests.post(f"{BASE}/synthesize2", json=payload, headers=AUTH)
    r.raise_for_status()
    data = r.json()
    print("✅ Synthesis done:", data)
    if data is None:
        raise RuntimeError("Server returned null JSON")
    if "error" in data:
        raise RuntimeError(f"Server error: {data['error']}")
    return data

def download_playable(s3_url):
    print("⬇️ Downloading playable version from server...")
    r = requests.get(
        f"{BASE}/download_unified",
        params={"s3_url": s3_url, "convert": True},  # Added convert=True
        headers=AUTH,
        timeout=300,
    )
    r.raise_for_status()
    out_name = "playable.wav"
    with open(out_name, "wb") as f:
        f.write(r.content)
    print(f"🎧 Saved playable file: {out_name} ({len(r.content)} bytes)")

if __name__ == "__main__":
    print("=" * 50)
    print("TEST 1: Upload reference")
    ref_id = upload_reference()
    print(f"Got ref_id: {ref_id}")
    
    print("\n" + "=" * 50)
    print("TEST 2: Synthesize with reference")
    data = synthesize("안녕하세요, 테스트입니다.", ref_id=ref_id)
    
    print("\n" + "=" * 50)
    print("TEST 3: Download audio")
    s3_url = data.get("url") or data.get("s3_url")
    if s3_url:
        download_playable(s3_url)
        print("✅ Check playable.wav - it should sound like your reference!")