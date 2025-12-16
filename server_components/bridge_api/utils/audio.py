import io, soundfile as sf, numpy as np
from scipy import signal
import audioop, wave

def ensure_mulaw_8k(raw_wav_bytes: bytes) -> bytes:
    """Convert arbitrary audio to 8 kHz μ-law mono WAV (Connect compatible)."""
    try:
        data, sr = sf.read(io.BytesIO(raw_wav_bytes), dtype="float32")
    except Exception as e:
        print(f"[Audio] Read error, assuming 16k PCM: {e}")
        data, sr = sf.read(io.BytesIO(raw_wav_bytes), dtype="float32", samplerate=16000)
    if data.ndim > 1:
        data = data.mean(axis=1)
    resampled = signal.resample_poly(data, 8000, sr)
    pcm16 = np.int16(np.clip(resampled, -1, 1) * 32767)
    mulaw = audioop.lin2ulaw(pcm16.tobytes(), 2)
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(8000)
        wf.writeframes(mulaw)
    return out.getvalue()
