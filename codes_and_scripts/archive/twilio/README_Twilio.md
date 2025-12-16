# Twilio Front Door for Korean Voice Bot (Pilot)

This starter shows how to use **Twilio** as a front door for barge‑in and then either:
- **A) Gather-based speech capture** (quickest path) and/or
- **B) Bidirectional Media Streams** to stream TTS audio back to the caller while you still control capture.

You can then **hand off to Amazon Connect** by dialing your Connect DID. Your existing Connect contact flow remains unchanged.

> **Note:** Twilio trial accounts: only verified caller IDs can call your number and a trial announcement will play. Upgrade to remove these limits.

---

## Topology

```
Caller → Twilio Number
  ├─ A) /voice_gather  → <Gather input="speech"> (barge-in) → /handle_speech → (optional) <Connect><Stream> to WS
  └─ B) /voice_stream  → <Connect><Stream> to your WebSocket server (own audio/STT pipeline)

[Optional handoff] → <Dial> +82... (your Amazon Connect DID)
```

---

## Quick Start

### 1) Run the Flask app (TwiML webhooks)

```bash
pip install flask python-dotenv
python twilio_flask_app.py
# Expose to the internet, e.g. using ngrok:
# ngrok http 5000
```

Set your **Twilio number → Voice webhook** (HTTP POST) to:
- `https://<your-ngrok>/voice_gather`  (to try A)
- or `https://<your-ngrok>/voice_stream` (to try B directly)

### 2) (For B) Run the WebSocket media server

```bash
pip install websockets boto3 requests
python twilio_ws_server.py
# Expose with a TLS terminator / a reverse proxy that provides wss://
# For quick tests, some tunnels can terminate TLS and proxy to ws://localhost:8765
```

Configure your Twilio `<Connect><Stream url="wss://...">` to point at your public WSS endpoint.

---

## Environment

Create a `.env` in the same folder (used by both scripts if you like):

```
# For handoff to Amazon Connect (optional)
CONNECT_DID=+82XXXXXXXXXX

# For your existing TTS service (optional, used by the WS server)
TTS_URL=http://localhost:8000/synthesize_stream_start
TTS_BUCKET=your-tts-bucket
AWS_REGION=ap-northeast-2
KEY_PREFIX=sessions
```

- `CONNECT_DID`: your Connect entry number to transfer to agents (optional).
- `TTS_URL`: your FishSpeech streaming endpoint that returns a `job_id` and uploads parts to S3.
- `TTS_BUCKET`, `AWS_REGION`, `KEY_PREFIX`: where the μ‑law 8kHz `.wav` parts are uploaded.

---

## A) Gather-based (fastest pilot)

- Barge‑in behavior: The caller can talk over the prompt inside `<Gather>`; Twilio stops playback and posts the transcript to your `/handle_speech` webhook.
- In this starter, `/handle_speech` simply **echoes** what was said and optionally **dials your Connect DID** to keep your existing flow.

Endpoints:
- `POST /voice_gather` → Returns TwiML with `<Gather input="speech">`
- `POST /handle_speech` → Receives `SpeechResult`, responds with `<Say>` and optionally `<Dial>` to Connect

---

## B) Media Streams (bidirectional)

- Use `<Connect><Stream url="wss://...">` to open a **bidirectional** media stream.
- The included `twilio_ws_server.py` parses Twilio's `start/media/stop` events, then:
  1. Starts your **TTS** (`TTS_URL`) using the text passed via `<Parameter name="text">` (or you can wire your own STT).
  2. Polls S3 for `partN.wav` and sends **base64 μ‑law frames** back to Twilio as they appear.
- This gives you **“barge‑in + process the same breath”** control since you own the audio.

---

## Handoff to Amazon Connect

At any point, respond with TwiML:

```xml
<Response>
  <Dial>${CONNECT_DID}</Dial>
</Response>
```

The call lands in your existing Connect entry point; your contact flow runs unchanged.

---

## Notes

- Twilio trial: only verified numbers can call in; upgrade for public testing.
- For WSS in dev, terminate TLS at a proxy (Caddy, Nginx, cloud tunnel) and forward to `ws://localhost:8765`.
- The media server assumes your TTS parts are **μ‑law mono 8kHz WAV** (RIFF). It strips WAV headers and sends μ‑law frames to Twilio.

---

## Files

- `twilio_flask_app.py` – TwiML endpoints for Gather and Stream.
- `twilio_ws_server.py` – WebSocket server that streams S3 parts back to Twilio.
- `README_Twilio.md` – This file.
