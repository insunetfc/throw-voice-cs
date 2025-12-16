import os, io, uuid, base64, re, time, tempfile
from typing import Optional, Tuple, List
import subprocess, tempfile, boto3, os

import numpy as np
import soundfile as sf
import yaml
import boto3
from botocore.client import Config as BotoConfig
from functools import lru_cache
import logging, asyncio, warnings

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# ---- FishSpeech imports ----
import torch
import torchaudio
from loguru import logger
from fish_speech.inference_engine.utils import InferenceResult
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest
from fastapi import BackgroundTasks
import threading, time

def _run_stream_job(job_id: str, text: str, target_sr: int):
    eng = get_engine()
    req_obj = ServeTTSRequest(text=text, format="wav", stream=True, max_new_tokens=512, chunk_length=128)
    upload_every = target_sr
    total_target_samples = 0
    cur_part = 0
    pcm_chunks = []
    last_sr_src = eng.sample_rate

    for out in eng.engine.inference(req_obj):
        audio_np, sr_src = _extract_audio_sr(out, eng.sample_rate)
        last_sr_src = sr_src
        pcm_chunks.append(audio_np)

        src_cat = np.concatenate(pcm_chunks) if pcm_chunks else np.zeros(0, dtype=np.float32)
        float_mono, _ = _to_float_mono(src_cat, sr_src, target_sr)

        while len(float_mono) - total_target_samples >= upload_every:
            start = total_target_samples
            end   = start + upload_every
            part  = float_mono[start:end]
            total_target_samples = end

            buf = io.BytesIO()
            sf.write(buf, part, target_sr, subtype="ULAW", format="WAV")  # 8k μ-law
            buf.seek(0)
            key = f"{job_id}/part{cur_part}.wav"
            _put_ulaw_wav(buf, key)
            cur_part += 1

        # small sleep to reduce CPU thrash; optional
        time.sleep(0.005)

    # flush remainder
    if pcm_chunks:
        src_cat = np.concatenate(pcm_chunks)
        float_mono, _ = _to_float_mono(src_cat, last_sr_src, target_sr)
        if len(float_mono) > total_target_samples:
            part = float_mono[total_target_samples:]
            buf = io.BytesIO()
            sf.write(buf, part, target_sr, subtype="ULAW", format="WAV")
            buf.seek(0)
            _put_ulaw_wav(buf, f"{job_id}/final.wav")

torch.backends.cudnn.benchmark = True

# ===================== env/config =====================
CFG_PATH        = os.getenv("FISH_CONFIG", "/home/work/VALL-E/fish-speech/fishspeech_infer/config.yaml")
AWS_REGION      = os.getenv("AWS_REGION", "ap-northeast-2")
TTS_BUCKET      = os.getenv("TTS_BUCKET", "seoul-bucket-65432")
KEY_PREFIX_DEF  = os.getenv("KEY_PREFIX", "sessions/demo")
PRESIGN_EXPIRES = int(os.getenv("PRESIGN_EXPIRES", "600"))
KMS_KEY_ARN = os.getenv("KMS_KEY_ARN") 
API_TOKEN       = os.getenv("API_TOKEN", "")  # optional bearer

warnings.filterwarnings(
    "ignore",
    message="torchaudio._backend.list_audio_backends has been deprecated"
)

s3 = boto3.client("s3", region_name=AWS_REGION, config=BotoConfig(signature_version="s3v4"))
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high") 

PAD_MS = 80  # tail padding

def encode_ulaw_wav(part_f32_mono: np.ndarray, sr: int) -> bytes:
    """
    Input: float32 mono in [-1, 1], at desired sample rate (8000).
    Output: bytes of WAV (pcm_mulaw), verified.
    """
    # tiny tail pad to avoid clip truncation on some carriers
    pad = np.zeros(int(sr * 0.08), dtype=np.float32)  # 80 ms
    data = np.concatenate([part_f32_mono.astype(np.float32), pad])

    # write ULAW
    buf = io.BytesIO()
    sf.write(buf, data, sr, subtype="ULAW", format="WAV")
    buf.seek(0)

    # verify
    info = sf.info(buf)
    if info.subtype.upper() != "ULAW" or info.samplerate != sr or info.channels != 1:
        raise RuntimeError(f"ULAW encode failed: subtype={info.subtype}, sr={info.samplerate}, ch={info.channels}")

    return buf.getvalue()

def write_ulaw_wav(buf, samples_f32, sr):
    import soundfile as sf, numpy as np
    pad = np.zeros(int(sr * (PAD_MS/1000.0)), dtype=np.float32)
    sf.write(buf, np.concatenate([samples_f32, pad]), sr, subtype="ULAW", format="WAV")

def _to_float_mono(audio: np.ndarray | torch.Tensor, sr_src: int, sr_dst: int) -> tuple[np.ndarray, int]:
    # torch -> numpy float32
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().float().numpy()
    audio = np.asarray(audio, dtype=np.float32)

    # to mono
    if audio.ndim == 2:
        if 2 in audio.shape:
            audio = audio.mean(axis=0 if audio.shape[0] == 2 else 1)
        else:
            audio = audio.squeeze()

    # resample
    if sr_src != sr_dst:
        t = torch.from_numpy(audio).float()
        t = torchaudio.functional.resample(t, sr_src, sr_dst)
        audio = t.cpu().numpy()

    # normalize to [-1, 1] safely
    maxv = float(np.max(np.abs(audio))) if audio.size else 1.0
    if maxv < 1e-6: maxv = 1.0
    audio = np.clip(audio / maxv, -1.0, 1.0).astype(np.float32)
    return audio, sr_dst

def _put_ulaw_wav(buf: io.BytesIO, key: str):
    extra = {"ContentType": "audio/wav", "CacheControl": "no-cache"}
    if KMS_KEY_ARN:
        extra.update({"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": KMS_KEY_ARN})
    s3.upload_fileobj(buf, TTS_BUCKET, key, ExtraArgs=extra)
    
def _to_pcm16_mono(audio: np.ndarray | torch.Tensor, sr_src: int, sr_dst: int) -> tuple[np.ndarray, int]:
    # torch tensor -> numpy float32 mono
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().float().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    # ensure 1D mono
    if audio.ndim == 2:
        if audio.shape[0] == 2 or audio.shape[1] == 2:
            # average stereo to mono
            if audio.shape[0] == 2: audio = audio.mean(axis=0)
            else:                   audio = audio.mean(axis=1)
        else:
            audio = audio.squeeze()
    # resample to 8kHz
    if sr_src != sr_dst:
        # use torchaudio resample for quality
        t = torch.from_numpy(audio).float()
        t = torchaudio.functional.resample(t, sr_src, sr_dst)
        audio = t.cpu().numpy()
    # normalize and convert to int16 PCM
    # (use your existing helper if you prefer)
    maxv = np.max(np.abs(audio)) if audio.size else 1.0
    maxv = max(maxv, 1e-6)
    audio = np.clip(audio / maxv, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    return pcm16, sr_dst

def _presign(key: str, expires: int = 3600) -> str:
    """
    Generate a presigned URL for an object in the TTS bucket.
    """
    try:
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": TTS_BUCKET, "Key": key},
            ExpiresIn=expires,
        )
    except ClientError as e:
        raise RuntimeError(f"Failed to presign S3 URL: {e}")

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

def _warm_once():
    """
    Try warming using whatever API the engine exposes:
    - .infer(text, ...)                (some adapters)
    - .inference(ServeTTSRequest)      (FishSpeech reference engine => generator)
    - .synthesize(text, ...)           (some wrappers)
    """
    eng = get_engine()

    # 1) Adapter with .infer(...)
    if hasattr(eng, "infer") and callable(getattr(eng, "infer")):
        logger.info("Warming via eng.infer(...)")
        try:
            eng.infer("짧은 테스트입니다.", speed=1.1)
            return
        except Exception as e:
            logger.warning(f"warm infer() failed: {e}")

    # 2) Reference engine with .inference(...) generator
    if hasattr(eng, "inference") and callable(getattr(eng, "inference")) and ServeTTSRequest:
        logger.info("Warming via eng.inference(ServeTTSRequest)")
        try:
            req = ServeTTSRequest(
                text="안녕하세요.",
                references=[],
                reference_id=None,
                max_new_tokens=256,
                chunk_length=128,
                top_p=0.7,
                repetition_penalty=1.2,
                temperature=0.7,
                format="wav",
            )
            it = eng.inference(req)
            # consume just the first chunk to trigger graph/kernels
            try:
                next(iter(it))
            except StopIteration:
                pass
            return
        except Exception as e:
            logger.warning(f"warm inference() failed: {e}")

    # 3) Some wrappers expose .synthesize(...)
    if hasattr(eng, "synthesize") and callable(getattr(eng, "synthesize")):
        logger.info("Warming via eng.synthesize(...)")
        try:
            eng.synthesize("짧은 테스트입니다.")
            return
        except Exception as e:
            logger.warning(f"warm synthesize() failed: {e}")

    logger.warning("Warmup skipped: no known engine API found.")
    
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
    use_memory_cache: Optional[bool] = True

def _check_auth(authorization: Optional[str]):
    if API_TOKEN and authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="unauthorized")

@app.post("/synthesize/warmup")
async def warmup():
    logger.info("Loading FishSpeech engine at startup…")
    # Do the warmup in a background task so startup doesn't fail if warmup errors
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _warm_once)
    logger.info("Engine warmup done.")
    return {"ok": True}
        
@app.on_event("startup")
async def _warm_engine():
    logger.info("Loading FishSpeech engine at startup…")
    # Do the warmup in a background task so startup doesn't fail if warmup errors
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _warm_once)
    logger.info("Engine warmup done.")
    
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


@app.post("/synthesize_stream")
def synthesize_stream(req: SynthesizeIn, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    eng = get_engine()
    target_sr = int(req.sample_rate or 8000)

    job_id = uuid.uuid4().hex
    keys: list[str] = []

    req_obj = ServeTTSRequest(
        text=req.text,
        references=[],
        reference_id=None,
        max_new_tokens=req.max_new_tokens or 512,
        chunk_length=req.chunk_length or 128,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        temperature=req.temperature,
        format="wav",
        stream=True,
    )

    pcm_chunks: list[np.ndarray] = []
    stream = eng.engine.inference(req_obj)

    upload_every = target_sr  # ~1s at target SR
    total_target_samples = 0
    cur_part = 0
    last_sr_src = eng.sample_rate

    for out in stream:
        audio_np, sr_src = _extract_audio_sr(out, eng.sample_rate)
        last_sr_src = sr_src
        pcm_chunks.append(audio_np)

        # concatenate source-rate floats we have so far
        src_cat = np.concatenate(pcm_chunks) if pcm_chunks else np.zeros(0, dtype=np.float32)

        # convert to float mono @ target_sr
        float_mono, _ = _to_float_mono(src_cat, sr_src, target_sr)

        if len(float_mono) - total_target_samples >= upload_every:
            start = total_target_samples
            end   = start + upload_every
            part  = float_mono[start:end]
            total_target_samples = end

            # write μ-law WAV
            buf = io.BytesIO()
            write_ulaw_wav(buf, part, target_sr)
            buf.seek(0)

            key = f"{job_id}/part{cur_part}.wav"
            _put_ulaw_wav(buf, key)
            keys.append(key)
            cur_part += 1

            # reset accumulation to keep latency low
            pcm_chunks = []

    # flush remainder
    if pcm_chunks:
        src_cat = np.concatenate(pcm_chunks)
        float_mono, _ = _to_float_mono(src_cat, last_sr_src, target_sr)
        if len(float_mono) > total_target_samples:
            part = float_mono[total_target_samples:]

            buf = io.BytesIO()
            write_ulaw_wav(buf, part, target_sr)
            buf.seek(0)

            key = f"{job_id}/final.wav"
            _put_ulaw_wav(buf, key)
            keys.append(key)

    # Return KEYS (no presign); your Lambda will construct clean regional URLs
    return {"keys": keys, "bucket": TTS_BUCKET, "region": "ap-northeast-2"}

@app.post("/synthesize_stream_start")
def synthesize_stream_start(req: SynthesizeIn, background: BackgroundTasks, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    job_id = uuid.uuid4().hex
    target_sr = int(req.sample_rate or 8000)
    background.add_task(_run_stream_job, job_id, req.text, target_sr)
    return {"job_id": job_id, "bucket": TTS_BUCKET, "region": "ap-northeast-2"}

@app.get("/synthesize_stream_status/{job_id}")
def synthesize_stream_status(job_id: str, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    # Option 1 (simple): list S3 to see ready parts
    part_indices, has_final = _list_ready_parts_from_s3(job_id)
    return {"ready_parts": part_indices, "has_final": has_final}