# bridge_api/adapters/elevenlabs.py
import requests, os, json, io
from typing import Optional

API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
BASE_URL = "https://api.elevenlabs.io/v1/text-to-speech"

VOICE_ID = os.getenv("ELEVENLABS_VOICE", "")
if not VOICE_ID or len(VOICE_ID) < 10:
    VOICE_ID = "AW5wrnG1jVizOYY7R1Oo"  # your JiYoung ID

MODEL_ID = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")


def fetch_elevenlabs_audio(text: str, voice_id: str | None = None) -> tuple[bytes, str]:
    use_voice = voice_id or VOICE_ID  # VOICE_ID is your default JiYoung
    headers = {
        "xi-api-key": API_KEY,
        "Accept": "audio/wav",
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": MODEL_ID,
        "voice_settings": {
            "stability": 0.6,
            "similarity_boost": 0.8,
        },
    }
    url = f"{BASE_URL}/{use_voice}"
    resp = requests.post(url, headers=headers, data=json.dumps(payload))
    resp.raise_for_status()
    ctype = resp.headers.get("Content-Type", "").lower()
    return resp.content, ctype
