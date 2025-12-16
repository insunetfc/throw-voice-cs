import os, json, time, asyncio, logging
import numpy as np
import boto3
from amazon_transcribe.client import TranscribeStreamingClient

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("asr-bridge")

# --- ENV / CONFIG ---
REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
CONNECT_INSTANCE_ID = os.environ.get("CONNECT_INSTANCE_ID", "5b83741e-7823-4d70-952a-519d1ac05e63")
TRANSCRIBE_LANG = os.environ.get("TRANSCRIBE_LANGUAGE_CODE", "ko-KR")
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "8000"))
LEX_BOT_ID = os.environ.get("LEX_BOT_ID", "ENOV0CLDHS")
LEX_ALIAS_ID = os.environ.get("LEX_ALIAS_ID", "LPAEZPXVLI")
LEX_LOCALE = os.environ.get("LEX_LOCALE", "ko_KR")
SILENCE_MS = int(os.environ.get("SILENCE_MS", "450"))
FIFO_PATH = os.environ.get("PCM_FIFO", "/tmp/customer.pcm")

FRAME_MS = 20                         # 20 ms frames
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2       # 16-bit mono
SILENCE_FRAMES = max(1, SILENCE_MS // FRAME_MS)
RMS_FLOOR = float(os.environ.get("RMS_FLOOR", "250.0"))

session = boto3.Session(region_name=REGION)
lex = session.client("lexv2-runtime")
connect = session.client("connect")

def is_silent(pcm_i16: np.ndarray, rms_floor=RMS_FLOOR):
    rms = np.sqrt(np.mean(pcm_i16.astype(np.float64) ** 2))
    return rms < rms_floor

async def fifo_pcm_frames(path=FIFO_PATH):
    """Yield 20ms raw PCM16 mono frames from a FIFO (no WAV header)."""
    loop = asyncio.get_running_loop()
    # Open blocks until writer connects
    with open(path, "rb", buffering=0) as f:
        buf = b""
        while True:
            chunk = await loop.run_in_executor(None, f.read, 4096)
            if not chunk:
                await asyncio.sleep(0.01)
                continue
            buf += chunk
            while len(buf) >= FRAME_BYTES:
                out = buf[:FRAME_BYTES]
                buf = buf[FRAME_BYTES:]
                yield out

async def stream_to_transcribe(pcm_iter, contact_id, session_id):
    client = TranscribeStreamingClient(region=REGION)
    stream = await client.start_stream_transcription(
        language_code=TRANSCRIBE_LANG,
        media_sample_rate_hz=SAMPLE_RATE,
        media_encoding="pcm",
        enable_partial_results_stabilization=True,
        partial_results_stability="medium",
    )

    final_text = ""
    silent_streak = 0

    async def write_audio():
        nonlocal silent_streak, final_text
        async for frame in pcm_iter:
            await stream.input_stream.send_audio_event(audio_chunk=frame)
            pcm = np.frombuffer(frame, dtype=np.int16)
            silent_streak = silent_streak + 1 if is_silent(pcm) else 0
            if silent_streak >= SILENCE_FRAMES:
                await stream.input_stream.end_stream()
                return
        await stream.input_stream.end_stream()

    async def read_transcripts():
        nonlocal final_text
        async for event in stream.output_stream:
            for res in event.transcript.results:
                if res.is_partial:
                    continue
                for alt in res.alternatives:
                    text = (alt.transcript or "").strip()
                    if text:
                        final_text = text

    await asyncio.gather(write_audio(), read_transcripts())
    log.info(f"[{contact_id}] Final transcript: {final_text}")
    return final_text

def update_connect_attrs(instance_id, contact_id, attrs: dict):
    connect.update_contact_attributes(
        InstanceId=instance_id,
        InitialContactId=contact_id,
        Attributes=attrs
    )

def recognize_text_lex(session_id: str, text: str):
    resp = lex.recognize_text(
        botId=LEX_BOT_ID,
        botAliasId=LEX_ALIAS_ID,
        localeId=LEX_LOCALE,
        sessionId=session_id,
        text=text
    )
    st = resp.get("sessionState", {})
    intent = st.get("intent", {}).get("name")
    slots = st.get("intent", {}).get("slots")
    return intent, slots, resp

async def main():
    contact_id = os.environ.get("CONTACT_ID", "b7c0c1be-a84a-489e-a428-e13505c85aca")
    if not contact_id:
        raise RuntimeError("Set CONTACT_ID (e.g., export CONTACT_ID=<InitialContactId>).")
    session_id = contact_id

    # 1) FIFO (raw PCM) -> Transcribe
    text = await stream_to_transcribe(fifo_pcm_frames(), contact_id, session_id)

    # 2) Lex NLU
    intent, slots, _ = recognize_text_lex(session_id, text or "")

    # 3) Update Connect attributes
    attrs = {
        "transcript": text or "",
        "nlu_intent": intent or "",
        "nlu_slots": json.dumps(slots or {}, ensure_ascii=False),
        "asr_engine": "transcribe",
        "eos_ms": str(SILENCE_MS),
        "sample_rate": str(SAMPLE_RATE),
    }
    update_connect_attrs(CONNECT_INSTANCE_ID, contact_id, attrs)

if __name__ == "__main__":
    asyncio.run(main())
