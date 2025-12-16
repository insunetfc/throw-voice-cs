import os
import base64
import json
import requests

# Env vars you will fill in tomorrow
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_GROUP_ID = os.getenv("MINIMAX_GROUP_ID", "")
MINIMAX_HOST = os.getenv("MINIMAX_API_HOST", "https://api.minimax.io")

# Manager mentioned "Speech-02 Turbo"
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "speech-02-turbo")

# A safe Korean-ish female system voice as placeholder;
# after you log in you can list voices and replace this.
DEFAULT_VOICE_ID = os.getenv("MINIMAX_VOICE_ID", "female-yue-ningmeng")  # placeholder


class MiniMaxError(RuntimeError):
    pass


def _build_url() -> str:
    if not MINIMAX_GROUP_ID:
        # you said you'll add it tomorrow, so we just fail loudly for now
        raise MiniMaxError("MINIMAX_GROUP_ID is not set")
    return f"{MINIMAX_HOST}/v1/t2a_v2?GroupId={MINIMAX_GROUP_ID}"


def generate_minimax_audio(text: str) -> bytes:
    """
    Call MiniMax T2A v2 API and return raw audio bytes.
    MiniMax returns base64-encoded audio in 'audio_file'.
    Docs: https://api.minimax.io/v1/t2a_v2  (see public doc) :contentReference[oaicite:2]{index=2}
    """
    if not MINIMAX_API_KEY:
        raise MiniMaxError("MINIMAX_API_KEY is not set")

    url = _build_url()
    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MINIMAX_MODEL,
        "text": text,
        # basic voice settings; you can swap to your cloned voice later
        "voice_id": DEFAULT_VOICE_ID,
        "speed": 1.0,
        "vol": 1.0,
        "pitch": 0,
        # we ask MiniMax for something higher (32k) and downsample to 8k mulaw later
        "audio_sample_rate": 32000,
        "format": "wav",
    }

    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
    if resp.status_code != 200:
        raise MiniMaxError(f"HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    base_resp = data.get("base_resp") or {}
    if base_resp.get("status_code") != 0:
        raise MiniMaxError(f"MiniMax error: {base_resp.get('status_msg')}")

    b64_audio = data.get("audio_file")
    if not b64_audio:
        raise MiniMaxError("MiniMax response missing 'audio_file'")

    return base64.b64decode(b64_audio)
