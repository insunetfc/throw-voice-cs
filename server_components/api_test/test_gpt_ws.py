import asyncio, json, base64, websockets, wave

WS_URL = "wss://honest-trivially-buffalo.ngrok-free.app/voice/gpt"

async def main():
    audio_bytes = bytearray()

    async with websockets.connect(WS_URL) as ws:
        # 1) minimal session setup
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": "You are a polite Korean call agent."
            }
        }))

        # 2) ask for a short spoken response
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {
                "instructions": "안녕하세요 고객님, 이번 달 자동차 보험 갱신 할인 안내드립니다. 짧게 말씀해 주세요."
            }
        }))

        # 3) collect base64 chunks safely
        while True:
            raw = await ws.recv()
            evt = json.loads(raw)
            t = evt.get("type")

            if t == "response.output_audio.delta":
                # OpenAI realtime now uses "delta" for audio data
                chunk = evt.get("delta")
                if chunk:
                    audio_bytes.extend(base64.b64decode(chunk))
            elif t in ("response.output_audio.done", "response.done"):
                break

    # 4) save as WAV (16 kHz mono, 16-bit)
    with wave.open("gpt_voice.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(audio_bytes)

    print(f"✅ saved gpt_voice.wav ({len(audio_bytes)} bytes)")

asyncio.run(main())
