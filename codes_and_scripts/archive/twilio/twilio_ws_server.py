import asyncio, base64, json, os, wave, io, time
import websockets
import boto3, botocore
import requests

AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
TTS_BUCKET = os.getenv("TTS_BUCKET")
KEY_PREFIX = os.getenv("KEY_PREFIX", "sessions")
TTS_URL = os.getenv("TTS_URL", "http://localhost:8000/synthesize_stream_start")

s3 = boto3.client("s3", region_name=AWS_REGION)

async def send_ulaw_wav(ws, stream_sid: str, wav_bytes: bytes, frame_ms: int = 20):
    """
    Send mu-law mono 8kHz WAV audio to Twilio as media frames.
    Splits into ~20ms frames (~160 samples) for smoother playback.
    """
    # Strip WAV header and read frames
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        nch, sampwidth, fr, nframes, _, _ = w.getparams()
        assert fr == 8000, f"Expected 8kHz, got {fr}"
        assert nch == 1, f"Expected mono, got {nch}"
        raw = w.readframes(nframes)

    samples_per_frame = int(8000 * frame_ms / 1000)
    for i in range(0, len(raw), samples_per_frame):
        chunk = raw[i:i+samples_per_frame]
        payload = base64.b64encode(chunk).decode("ascii")
        msg = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload}
        }
        await ws.send(json.dumps(msg))
        await asyncio.sleep(frame_ms/1000.0)

async def stream_tts_parts(ws, stream_sid: str, text: str):
    """
    Start TTS job, then poll S3 for partN.wav and stream each to Twilio.
    """
    # 1) Kick off TTS
    try:
        r = requests.post(TTS_URL, json={"text": text, "sample_rate": 8000}, timeout=5)
        r.raise_for_status()
        job_id = r.json().get("job_id")
        if not job_id:
            raise RuntimeError("No job_id from TTS")
    except Exception as e:
        return

    index = 0
    final_seen = False
    prefix = f"{KEY_PREFIX}/{job_id}/"
    while True:
        # Check for final.wav
        try:
            s3.head_object(Bucket=TTS_BUCKET, Key=prefix + "final.wav")
            final_seen = True
        except botocore.exceptions.ClientError:
            pass

        # Try next part
        key = f"{prefix}part{index}.wav"
        try:
            obj = s3.get_object(Bucket=TTS_BUCKET, Key=key)
            data = obj["Body"].read()
            await send_ulaw_wav(ws, stream_sid, data)
            index += 1
        except botocore.exceptions.ClientError:
            await asyncio.sleep(0.2)

        if final_seen and index > 0:
            await asyncio.sleep(0.2)
            break

async def handler(ws):
    stream_sid = None
    seed_text = "안녕하세요. 테스트 오디오를 재생합니다."
    try:
        async for message in ws:
            evt = json.loads(message)
            et = evt.get("event")
            if et == "start":
                stream_sid = evt.get("start", {}).get("streamSid")
                params = evt.get("start", {}).get("customParameters") or {}
                seed_text = params.get("text", seed_text)
                asyncio.create_task(stream_tts_parts(ws, stream_sid, seed_text))
            elif et == "media":
                # Inbound caller audio frames arrive here (evt['media']['payload'])
                pass
            elif et == "stop":
                break
    except websockets.exceptions.ConnectionClosed:
        pass

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765, max_size=2**22, ping_interval=None):
        print("Twilio WS server listening on ws://0.0.0.0:8765/twilio")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
