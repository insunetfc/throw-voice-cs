# app.py (top-level on NIPA)
from fastapi import FastAPI
import importlib

app = FastAPI(title="Combined FishSpeech + Chatbot + PhoneCall")

# simple in-memory mapping: key -> elevenlabs voice_id
VOICE_REGISTRY: dict[str, str] = {}

# 1) TTS (your existing fishspeech app)
tts_module = importlib.import_module("fishspeech.fish-speech.app")
tts_app = getattr(tts_module, "app")
app.mount("/tts", tts_app)

# 2) Chatbot (your existing one)
from chatbot.app import app as chatbot_app
app.mount("/chatbot", chatbot_app)

# 3) Phone call (the one we just made in phone_call/app.py)
from phone_call.app import app as phone_app
app.mount("/phone", phone_app)

# 4) Eleven Labs and GPT Voice mounted on the bridge
from bridge_api.app import app as bridge_app
app.mount("/voice", bridge_app)