from fastapi import FastAPI
from pydantic import BaseModel
import os

# Reuse functions from infer_tts.py when running as a package/module
# Or duplicate minimal wrappers if running this file directly
from infer_tts import synthesize_to_wav, s3_upload_and_presign, _read_yaml

class SynthesizeReq(BaseModel):
    text: str
    speaker_wav_local: str | None = None  # path on the server; or extend to accept base64
    key_prefix: str | None = None
    sample_rate: int | None = None
    temperature: float | None = None
    top_p: float | None = None

app = FastAPI()

CFG_PATH = os.environ.get("FISHSPEECH_CFG", "/home/work/VALL-E/fish-speech/fishspeech_infer/config.yaml")
DEFAULT_BUCKET = os.environ.get("TTS_BUCKET")  # required

@app.post("/synthesize")
def synthesize(req: SynthesizeReq):
    if not DEFAULT_BUCKET:
        return {"error": "Missing env TTS_BUCKET"}

    outdir = os.environ.get("TTS_OUTDIR", "outputs")
    local_wav = synthesize_to_wav(
        cfg_path=CFG_PATH,
        text=req.text,
        outdir=outdir,
        speaker_wav=req.speaker_wav_local,
        override_sr=req.sample_rate,
        temperature=req.temperature,
        top_p=req.top_p,
    )

    cfg = _read_yaml(CFG_PATH)
    expires = cfg.get("presign_expires", 3600)

    result = s3_upload_and_presign(
        local_path=local_wav,
        bucket=DEFAULT_BUCKET,
        key_prefix=req.key_prefix or cfg.get("s3", {}).get("key_prefix", "tts-out"),
        expires=expires,
    )
    return result