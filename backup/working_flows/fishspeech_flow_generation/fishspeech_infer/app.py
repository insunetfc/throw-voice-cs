import os, io, uuid, base64, re, time, tempfile
from typing import Optional, Tuple, List

import numpy as np
import soundfile as sf
import yaml
import boto3
from botocore.client import Config as BotoConfig
from functools import lru_cache

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# ---- FishSpeech imports ----
import torch
from loguru import logger
from fish_speech.inference_engine.utils import InferenceResult
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest

# ===================== env/config =====================
CFG_PATH        = os.getenv("FISH_CONFIG", "/home/work/VALL-E/fish-speech/fishspeech_infer/config.yaml")
AWS_REGION      = os.getenv("AWS_REGION", "ap-northeast-2")
TTS_BUCKET      = os.getenv("TTS_BUCKET", "seoul-bucket-65432")
KEY_PREFIX_DEF  = os.getenv("KEY_PREFIX", "sessions/demo")
PRESIGN_EXPIRES = int(os.getenv("PRESIGN_EXPIRES", "600"))
API_TOKEN       = os.getenv("API_TOKEN", "")  # optional bearer

s3 = boto3.client("s3", region_name=AWS_REGION, config=BotoConfig(signature_version="s3v4"))
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high") 

def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _ensure_mono(audio: np.ndarray) -> np.ndarray:
    """Make mono. Accepts [N] or [N, C]; returns [N]."""
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 2:
        if a.shape[1] == 1:
            a = a[:, 0]
        else:
            a = np.mean(a, axis=1)
    return a

def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Lightweight linear resampler (no scipy/librosa)."""
    if src_sr == dst_sr or audio.size == 0:
        return audio
    src_t = np.linspace(0.0, 1.0, num=audio.size, endpoint=False, dtype=np.float64)
    dst_len = int(round(audio.size * (dst_sr / float(src_sr))))
    dst_t = np.linspace(0.0, 1.0, num=max(dst_len, 1), endpoint=False, dtype=np.float64)
    return np.interp(dst_t, src_t, audio).astype(np.float32)

    
# ===================== audio helpers =====================
def _ensure_int16(audio: np.ndarray) -> np.ndarray:
    if audio.dtype not in (np.float32, np.float64): audio = audio.astype(np.float32)
    max_val = float(np.max(np.abs(audio))) if audio.size else 1.0
    max_val = max(max_val, 1e-6)
    audio = audio / max_val
    return np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)

def _extract_audio_sr(obj, fallback_sr: int) -> Tuple[np.ndarray, int]:
    import soundfile as _sf
    # tuple
    if isinstance(obj, tuple) and len(obj) == 2:
        a, b = obj
        if isinstance(a, (int, np.integer)) and isinstance(b, np.ndarray): return b.astype(np.float32), int(a)
        if isinstance(a, np.ndarray) and isinstance(b, (int, np.integer)): return a.astype(np.float32), int(b)
        if isinstance(a, (bytes, bytearray)):
            audio, sr2 = _sf.read(io.BytesIO(a), dtype="float32"); return audio.astype(np.float32), int(sr2 or fallback_sr)
        if isinstance(b, (bytes, bytearray)):
            audio, sr2 = _sf.read(io.BytesIO(b), dtype="float32"); return audio.astype(np.float32), int(sr2 or fallback_sr)
        for p in (a, b):
            try:
                audio, sr = _sf.read(str(p), dtype="float32"); return audio.astype(np.float32), int(sr or fallback_sr)
            except Exception: pass
    # dict
    if isinstance(obj, dict):
        if "audio" in obj:
            a = obj["audio"]; sr = int(obj.get("sample_rate") or obj.get("sr") or fallback_sr)
            if isinstance(a, (bytes, bytearray)):
                audio, sr2 = _sf.read(io.BytesIO(a), dtype="float32"); return audio.astype(np.float32), int(sr or sr2 or fallback_sr)
            return np.asarray(a, dtype=np.float32), sr
        if "wav_path" in obj:
            audio, sr = _sf.read(obj["wav_path"], dtype="float32"); return audio.astype(np.float32), int(sr)
    # InferenceResult
    if isinstance(obj, InferenceResult):
        a = getattr(obj, "audio", None)
        if a is not None:
            if isinstance(a, (list, tuple)) and len(a) == 2:
                first, second = a
                if isinstance(first, (int, np.integer)) and isinstance(second, np.ndarray): return second.astype(np.float32), int(first)
                if isinstance(first, np.ndarray) and isinstance(second, (int, np.integer)): return first.astype(np.float32), int(second)
                if isinstance(first, (bytes, bytearray)):
                    audio, sr2 = _sf.read(io.BytesIO(first), dtype="float32"); return audio.astype(np.float32), int(sr2 or fallback_sr)
                if isinstance(second, (bytes, bytearray)):
                    audio, sr2 = _sf.read(io.BytesIO(second), dtype="float32"); return audio.astype(np.float32), int(sr2 or fallback_sr)
            if isinstance(a, (bytes, bytearray)):
                audio, sr = _sf.read(io.BytesIO(a), dtype="float32"); return audio.astype(np.float32), int(sr or fallback_sr)
            if isinstance(a, np.ndarray): return a.astype(np.float32), int(fallback_sr)
            if isinstance(a, list): return np.asarray(a, dtype=np.float32), int(fallback_sr)
    raise RuntimeError(f"Unsupported inference output type: {type(obj)}")

# ===================== engine =====================
class FishEngineAdapter:
    def __init__(self, cfg: dict):
        device = cfg.get("device", "cuda")
        half = bool(cfg.get("half", False))
        precision = torch.half if half else torch.bfloat16
        compile_flag = bool(cfg.get("compile", False))
        load_embeddings = bool(cfg.get("load_embeddings", True))

        llama_ckpt   = cfg.get("llama_checkpoint_path", "/home/work/VALL-E/fish-speech/checkpoints/openaudio-s1-mini")
        decoder_cfg  = cfg.get("decoder_config_name", "modded_dac_vq")
        decoder_ckpt = cfg.get("decoder_checkpoint_path", "/home/work/VALL-E/fish-speech/checkpoints/openaudio-s1-mini/codec.pth")
        self.sample_rate = int(cfg.get("audio", {}).get("sample_rate", 22050))

        logger.info("Launching LLM queue…")
        self.llama_queue = launch_thread_safe_queue(
            checkpoint_path=llama_ckpt, device=device, precision=precision, compile=compile_flag,
        )
        logger.info("Loading decoder model (%s)…", decoder_cfg)
        self.decoder_model = load_decoder_model(config_name=decoder_cfg, checkpoint_path=decoder_ckpt, device=device)

        logger.info("Constructing TTSInferenceEngine…")
        self.engine = TTSInferenceEngine(
            llama_queue=self.llama_queue,
            decoder_model=self.decoder_model,
            compile=compile_flag,
            precision=precision,
            load_embeddings=load_embeddings,
            embedding_path=cfg.get("embedding_path", "/home/work/VALL-E/fish-speech/cached_ref.pt"),
        )
        self.default_gen = cfg.get("sampling", {})

    def synthesize(
        self,
        text: str,
        speaker_wav: Optional[str] = None,
        sr: Optional[int] = None,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        chunk_length: Optional[int] = None,
        use_memory_cache: Optional[bool] = None,
    ) -> Tuple[np.ndarray, int]:
        params = {**self.default_gen}
        if temperature is not None: params["temperature"] = temperature
        if top_p is not None: params["top_p"] = top_p
        if seed is not None: params["seed"] = seed
        if repetition_penalty is not None: params["repetition_penalty"] = repetition_penalty
        if max_new_tokens is not None: params["max_new_tokens"] = max_new_tokens
        if chunk_length is not None: params["chunk_length"] = chunk_length
        if use_memory_cache is not None: params["use_memory_cache"] = use_memory_cache

        umc_val = params.get("use_memory_cache", False)
        umc_str = "on" if (str(umc_val).lower() == "on" or bool(umc_val)) else "off"

        req = ServeTTSRequest(
            text=text,
            references=[],
            reference_id=None,
            reference_audio=speaker_wav,
            reference_text=None,
            max_new_tokens=int(params.get("max_new_tokens", 0)),
            chunk_length=int(params.get("chunk_length", 300)),
            top_p=float(params.get("top_p", 0.8)),
            repetition_penalty=float(params.get("repetition_penalty", 1.1)),
            temperature=float(params.get("temperature", 0.8)),
            seed=int(params.get("seed", 42)),
            use_memory_cache=umc_str,  # 'on' | 'off'
            format="wav",
            stream=False,
        )

        stream = self.engine.inference(req)
        items = list(stream)
        if not items:
            raise RuntimeError("Engine returned no output")
        out = items[-1]
        audio_np, sample_rate = _extract_audio_sr(out, self.sample_rate)

        if sr is not None and sr != sample_rate:
            logger.warning("Requested SR %s != model SR %s; returning model SR.", sr, sample_rate)
        return np.asarray(audio_np, dtype=np.float32), int(sample_rate)

@lru_cache(maxsize=1)
def get_engine() -> FishEngineAdapter:
    cfg = _read_yaml(CFG_PATH)
    return FishEngineAdapter(cfg)

# ===================== FastAPI =====================
app = FastAPI()

class SynthesizeIn(BaseModel):
    text: str
    key_prefix: Optional[str] = None
    sample_rate: Optional[int] = None
    speaker_wav: Optional[str] = None  # path to ref wav (optional)
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    seed: Optional[int] = 42
    repetition_penalty: Optional[float] = 1.1
    max_new_tokens: Optional[int] = 0
    chunk_length: Optional[int] = 300
    use_memory_cache: Optional[bool] = False

def _check_auth(authorization: Optional[str]):
    if API_TOKEN and authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="unauthorized")

@app.on_event("startup")
def _warm_engine():
    logger.info("Loading FishSpeech engine at startup…")
    _ = get_engine()
    logger.info("Engine loaded.")

@app.get("/health")
def health():
    return {"ok": True, "bucket": TTS_BUCKET, "region": AWS_REGION}

@app.post("/synthesize")
def synthesize(req: SynthesizeIn, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    t0 = time.time()

    # Target sample rate: default 8000 for Amazon Connect; honor request if given
    target_sr = int(req.sample_rate or 8000)

    eng = get_engine()
    audio_f32, model_sr = eng.synthesize(
        text=req.text,
        speaker_wav=req.speaker_wav,
        sr=req.sample_rate,  # still pass through; engine may ignore
        temperature=req.temperature,
        top_p=req.top_p,
        seed=req.seed,
        repetition_penalty=req.repetition_penalty,
        max_new_tokens=req.max_new_tokens,
        chunk_length=req.chunk_length,
        use_memory_cache=req.use_memory_cache,
    )

    # ---- Ensure mono + target SR, then 16-bit PCM
    audio_mono = _ensure_mono(audio_f32)
    audio_res  = _resample_linear(audio_mono, model_sr, target_sr)
    pcm16      = _ensure_int16(audio_res)

    # Write WAV @ target_sr, PCM_16
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, pcm16, target_sr, subtype="PCM_16")
        local_wav = tmp.name

    # Upload & presign
    key_prefix = (req.key_prefix or KEY_PREFIX_DEF).rstrip("/")
    key = f"{key_prefix}/{uuid.uuid4().hex}.wav"

    # Optional: tag sample rate in metadata for debugging
    s3.upload_file(
        local_wav,
        TTS_BUCKET,
        key,
        ExtraArgs={
            "ContentType": "audio/wav",
            "Metadata": {"sample_rate": str(target_sr), "channels": "1"}
        }
    )

    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": TTS_BUCKET, "Key": key},
        ExpiresIn=PRESIGN_EXPIRES
    )

    logger.info(f"[TTS OUT] text_len={len(req.text)} seed={req.seed} model_sr={model_sr} out_sr={target_sr} key={key}")

    latency = int((time.time() - t0) * 1000)
    return {
        "bucket": TTS_BUCKET,
        "key": key,
        "url": url,
        "s3_url": url,     # compatibility
        "latency_ms": latency,
        "sample_rate": target_sr,
        "text": req.text,
    }