import torch, os
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")  # PyTorch 2.x
torch.backends.cudnn.benchmark = True  

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
from fastapi import Request

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import audioop, math
from twilio_host import ws_for_call, stream_sid_for_call, send_clear, send_ulaw_media, send_mark
from pydantic import BaseModel
from threading import Event
import threading, queue, time

MEL_QUEUE_MAX = 8   # backpressure; tune if GPU is large
PCM_QUEUE_MAX = 8

# ---- FishSpeech imports ----
import torch
import torchaudio
from loguru import logger
from fish_speech.inference_engine.utils import InferenceResult
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest
from fastapi import BackgroundTasks, Header
import threading, time
from fastapi.responses import Response

USE_COMPILE = os.getenv("TORCH_COMPILE", "0") == "1"
def maybe_compile(m):
    return torch.compile(m, mode="reduce-overhead", fullgraph=False) if USE_COMPILE else m

HTTP_BASE = "https://honest-trivially-buffalo.ngrok-free.app"  # your HTTP ngrok (app.py)
WS_BASE   = "a30904a4985f.ngrok-free.app"    
CANCEL_EVENTS = {}
chunk_length = 64 # originally 128

# ---- chunker (tune for ko-KR / general) ----
import re, io, numpy as np
from typing import List

def _split_into_chunks(text: str, max_chars: int = 220):
    """Split text on sentence boundaries, then pack into ~max_chars chunks."""
    sents = re.split(r'(?<=[.!?。！？])\s+', text.strip())
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) + (1 if cur else 0) <= max_chars:
            cur = (cur + " " + s).strip() if cur else s
        else:
            if cur:
                chunks.append(cur)
            if len(s) <= max_chars:
                cur = s
            else:
                # hard-wrap very long sentence
                for i in range(0, len(s), max_chars):
                    chunks.append(s[i:i+max_chars])
                cur = ""
    if cur:
        chunks.append(cur)
    return chunks

def _is_cancelled(job_id: str) -> bool:
    ev = CANCEL_EVENTS.get(job_id)
    return ev.is_set() if ev else False

def _mark_cancelled(job_id: str):
    ev = CANCEL_EVENTS.get(job_id)
    if not ev:
        ev = Event()
        CANCEL_EVENTS[job_id] = ev
    ev.set()

class CancelIn(BaseModel):
    job_id: str

def to_8k_pcm16(pcm16_bytes, sr_in):
    if sr_in == 8000: 
        return pcm16_bytes
    # naive but works: audioop.ratecv (mono, width=2)
    out, _ = audioop.ratecv(pcm16_bytes, 2, 1, sr_in, 8000, None)
    return out

def pcm16_to_ulaw_frames(pcm16_8k):
    frames = []
    for i in range(0, len(pcm16_8k), SAMPLES_PER_FRAME*2):
        chunk = pcm16_8k[i:i+SAMPLES_PER_FRAME*2]
        if len(chunk) < SAMPLES_PER_FRAME*2: break
        frames.append(audioop.lin2ulaw(chunk, 2))  # -> 160 bytes
    return frames

# --- 1) Chunk text ---
def _chunk_text(text: str, max_len: int = 140) -> list[str]:
    import re
    parts = re.split(r'([\.!\?。！？…]\s*|\n+)', text)
    sents = []
    for i in range(0, len(parts), 2):
        seg = (parts[i] or "").strip()
        delim = (parts[i+1] if i+1 < len(parts) else "")
        if seg: sents.append(seg + delim)

    chunks, buf = [], ""
    for s in sents:
        if not buf:
            buf = s
        elif len(buf) + len(s) <= max_len:
            buf += s
        else:
            chunks.append(buf.strip())
            buf = s
    if buf: chunks.append(buf.strip())
    return chunks or [text.strip()]

# --- 2) Synthesize & upload one chunk ---
def _synthesize_chunk_to_key(chunk: str, target_sr: int, job_id: str, idx: int) -> str:
    eng = get_engine()
    with torch.inference_mode(), torch.amp.autocast(dtype=torch.float16, device_type='cuda'):
        audio_f32, model_sr = eng.synthesize(  # <- consumes generator internally
            text=chunk,
            sr=target_sr,
            temperature=TTS_TEMP,
            top_p=TTS_TOP_P,
            repetition_penalty=TTS_REP,
            max_new_tokens=512,
            chunk_length=chunk_length,
            use_memory_cache=True,
        )
    float_mono, _ = _to_float_mono(audio_f32, model_sr, target_sr)

    import io
    buf = io.BytesIO()
    write_ulaw_wav(buf, float_mono, target_sr)
    buf.seek(0)

    key = f"{job_id}/part{idx}.wav"
    _put_ulaw_wav(buf, key)
    logger.info(f"[CHUNK] uploaded {key} dur≈{len(float_mono)/target_sr:.2f}s len={len(chunk)}")
    return key

def _split_into_sentences(text: str):
    """
    Return a list of sentences, preserving terminal punctuation.
    Supports English/Korean/JP/CN punctuation.
    """
    if not text:
        return []
    # Keep punctuation as part of the sentence
    pieces = re.findall(r'.+?(?:[.!?。！？…]+|\n+|$)', text.strip(), flags=re.S)
    # Clean and drop empties
    return [p.strip() for p in pieces if p.strip()]

# --- CHUNKED MODE: one sentence -> one partN.wav, with cancel between sentences ---
def _run_stream_job_chunked(job_id: str, text: str, target_sr: int, start_idx: int = 0):
    sentences = _split_into_sentences(text)
    for idx, sent in enumerate(sentences[start_idx:], start=start_idx):
        if _is_cancelled(job_id):
            tail = np.zeros(int(target_sr * 0.2), dtype=np.float32)
            buf = io.BytesIO(); write_ulaw_wav(buf, tail, target_sr); buf.seek(0)
            _put_ulaw_wav(buf, f"{job_id}/final.wav")
            logger.info(f"[CHUNK] job {job_id} cancelled; uploaded final.wav")
            return
        try:
            _synthesize_chunk_to_key(sent, target_sr, job_id, idx)
            logger.info(f"[CHUNK] uploaded {job_id}/part{idx}.wav")
        except Exception:
            logger.exception(f"[CHUNK] sentence {idx} failed; skipping")

    # final sentinel
    tail = np.zeros(int(target_sr * 0.2), dtype=np.float32)
    buf = io.BytesIO(); write_ulaw_wav(buf, tail, target_sr); buf.seek(0)
    _put_ulaw_wav(buf, f"{job_id}/final.wav")
    logger.info(f"[CHUNK] uploaded {job_id}/final.wav")

def _upload_part(job_id, idx, pcm_list):
    import numpy as np
    pcm = torch.cat(pcm_list, dim=-1).squeeze().numpy()
    # If you do 8k telephony µ-law:
    wav_bytes = encode_pcm_mu_law_8k(pcm)  # or 16-bit PCM if you prefer
    key = f"{job_id}/part{idx}.wav"
    s3_put_bytes(key, wav_bytes)
    
def run_stream_job(text, sr=8000, stream_upload_seconds=0.25):
    mel_q = queue.Queue(maxsize=MEL_QUEUE_MAX)
    pcm_q = queue.Queue(maxsize=PCM_QUEUE_MAX)
    stop_flag = {"stop": False}

    def acoustic_worker():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            # your generator should yield mel chunks (e.g., 10–20 frames)
            for mel_chunk in acoustic_generate_mel_streaming(text, chunk_frames=20):
                if stop_flag["stop"]: break
                mel_q.put(mel_chunk)   # [B, n_mels, Tchunk] on cuda
        mel_q.put(None)  # sentinel

    def vocoder_worker():
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            while True:
                item = mel_q.get()
                if item is None: break
                # Vocoder forward on the chunk
                pcm = vocoder(item.contiguous())    # [B, 1, T_audio_chunk] cuda
                pcm_q.put(pcm.detach().cpu())       # stage back to CPU for mux/encode
        pcm_q.put(None)

    def uploader_worker(job_id):
        # aggregate pcm until ~stream_upload_seconds then write partN.wav
        buf, t0, part_idx = [], time.perf_counter(), 0
        while True:
            pcm = pcm_q.get()
            if pcm is None:
                if buf:
                    _upload_part(job_id, part_idx, buf)
                break
            buf.append(pcm)
            if time.perf_counter() - t0 >= stream_upload_seconds:
                _upload_part(job_id, part_idx, buf)
                buf, t0, part_idx = [], time.perf_counter(), part_idx + 1

    job_id = new_job_id()
    t1 = threading.Thread(target=acoustic_worker, daemon=True)
    t2 = threading.Thread(target=vocoder_worker, daemon=True)
    t3 = threading.Thread(target=uploader_worker, args=(job_id,), daemon=True)
    t1.start(); t2.start(); t3.start()
    t1.join(); t2.join(); t3.join()
    _finalize_job(job_id)
    return job_id

def _run_stream_job_streaming(job_id: str, text: str, target_sr: int):
    eng = get_engine()
    req = ServeTTSRequest(
        text=text,
        references=[],
        reference_id=None,
        max_new_tokens=512,
        chunk_length=chunk_length,
        top_p=TTS_TOP_P,
        repetition_penalty=TTS_REP,
        temperature=TTS_TEMP,
        format="wav",
        stream=True,
    )

    upload_every = int(target_sr * float(os.getenv("STREAM_UPLOAD_SECONDS", "0.25")))
    buf = np.zeros(0, dtype=np.float32)   # rolling mono @ target_sr
    flushed = 0                           # number of samples already uploaded from buf
    cur_part = 0

    for item in eng.engine.inference(req):
        if _is_cancelled(job_id):
            tail = np.zeros(int(target_sr * 0.2), dtype=np.float32)
            b = io.BytesIO(); write_ulaw_wav(b, tail, target_sr); b.seek(0)
            _put_ulaw_wav(b, f"{job_id}/final.wav")
            logger.info(f"[STREAM] job {job_id} cancelled; uploaded final.wav")
            return

        # extract audio and convert to target_sr now
        chunk_np, sr_src = _extract_audio_sr(item, eng.sample_rate)
        chunk_f32, _ = _to_float_mono(chunk_np, sr_src, target_sr)
        if chunk_f32.size == 0:
            continue

        # append to rolling buffer
        buf = np.concatenate([buf, chunk_f32])

        # while we have at least one “upload_every” window ready, emit a part
        while buf.size - flushed >= upload_every:
            part = buf[flushed:flushed + upload_every]
            b = io.BytesIO()
            write_ulaw_wav(b, part, target_sr); b.seek(0)
            key = f"{job_id}/part{cur_part}.wav"
            _put_ulaw_wav(b, key)
            logger.info(f"[STREAM] uploaded {key}")
            cur_part += 1
            flushed += upload_every

        # keep memory bounded: drop consumed prefix
        if flushed >= 4 * upload_every:
            buf = buf[flushed:]
            flushed = 0

    # flush tail (anything not yet uploaded)
    if buf.size - flushed > 0:
        tail = buf[flushed:]
        b = io.BytesIO(); write_ulaw_wav(b, tail, target_sr); b.seek(0)
        _put_ulaw_wav(b, f"{job_id}/part{cur_part}.wav")
        logger.info(f"[STREAM] uploaded {job_id}/part{cur_part}.wav")

    # sentinel
    endpad = np.zeros(int(target_sr * 0.2), dtype=np.float32)
    b = io.BytesIO(); write_ulaw_wav(b, endpad, target_sr); b.seek(0)
    _put_ulaw_wav(b, f"{job_id}/final.wav")
    logger.info(f"[STREAM] uploaded {job_id}/final.wav")


# Add this helper to continue from an index
def _run_stream_job_from_idx(job_id: str, text: str, target_sr: int, start_idx: int):
    chunks = _chunk_text(text, max_len=160)
    for idx, chunk in enumerate(chunks[start_idx:], start=start_idx):
        try:
            _synthesize_chunk_to_key(chunk, target_sr, job_id, idx)
            logger.info(f"[STREAM] uploaded {job_id}/part{idx}.wav")
        except Exception:
            logger.exception(f"chunk {idx} failed; skipping")

    # sentinel to mark completion
    import numpy as np, io
    tail = np.zeros(int(target_sr * 0.2), dtype=np.float32)
    buf = io.BytesIO()
    write_ulaw_wav(buf, tail, target_sr)
    buf.seek(0)
    _put_ulaw_wav(buf, f"{job_id}/final.wav")
    logger.info(f"[STREAM] uploaded {job_id}/final.wav")

torch.backends.cudnn.benchmark = True

# ===================== env/config =====================
CFG_PATH        = os.getenv("FISH_CONFIG", "/home/work/VALL-E/fish-speech/fishspeech_infer/config.yaml")
AWS_REGION      = os.getenv("AWS_REGION", "ap-northeast-2")
TTS_BUCKET      = os.getenv("TTS_BUCKET", "seoul-bucket-65432")
KEY_PREFIX_DEF  = os.getenv("KEY_PREFIX", "sessions/demo")
PRESIGN_EXPIRES = int(os.getenv("PRESIGN_EXPIRES", "600"))
KMS_KEY_ARN = os.getenv("KMS_KEY_ARN") 
API_TOKEN       = os.getenv("API_TOKEN", "")  # optional bearer
S3_REGION = AWS_REGION
TTS_TOP_P = float(os.getenv("TTS_TOP_P", "0.9"))
TTS_TEMP  = float(os.getenv("TTS_TEMP",  "0.7"))
TTS_REP   = float(os.getenv("TTS_REP",   "1.0")) 
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "100"))

warnings.filterwarnings(
    "ignore",
    message="torchaudio._backend.list_audio_backends has been deprecated"
)

from botocore.config import Config as BotoConfig  # <-- use botocore.config
s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    config=BotoConfig(signature_version="s3v4", max_pool_connections=50),
)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high") 

PAD_MS = 30  # tail padding
SAMPLE_RATE = 8000     # add this near your config section
FRAME_S = 0.02
SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_S)

def _list_ready_parts_from_s3(job_id: str):
    """
    Returns (sorted_part_indices: List[int], has_final: bool)
    Looks for keys: <job_id>/part0.wav, part1.wav, ..., final.wav
    """
    prefix = f"{job_id}/"
    client = globals().get("s3") or boto3.client("s3", region_name=S3_REGION)

    part_indices = []
    has_final = False
    token = None

    while True:
        kwargs = {"Bucket": TTS_BUCKET, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = client.list_objects_v2(**kwargs)

        for obj in resp.get("Contents", []):
            k = obj["Key"]
            m = _PART_RE.search(k)
            if m:
                part_indices.append(int(m.group(1)))
            elif k.endswith("final.wav"):
                has_final = True

        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break

    part_indices.sort()
    return part_indices, has_final

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
                chunk_length=chunk_length,
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
            with torch.inference_mode(), torch.amp.autocast(dtype=torch.float16, device_type='cuda'):
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
            chunk_length=64, #int(params.get("chunk_length", 300)),
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
    eng = FishEngineAdapter(cfg)
    try:
        import torch
        eng.engine = torch.compile(eng.engine, mode="reduce-overhead", fullgraph=False)
        logger.info("torch.compile enabled")
    except Exception:
        logger.info("torch.compile not available; continuing")
    return eng

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
    with torch.inference_mode(), torch.amp.autocast(dtype=torch.float16, device_type='cuda'):
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
    
    # ---- Ensure mono + target SR as float32 in [-1, 1]
    float_mono, _ = _to_float_mono(audio_f32, model_sr, target_sr)

    # Encode to WAV/ULAW and upload once
    buf = io.BytesIO()
    write_ulaw_wav(buf, float_mono, target_sr)
    buf.seek(0)

    key_prefix = (req.key_prefix or KEY_PREFIX_DEF).rstrip("/")
    key = f"{key_prefix}/{uuid.uuid4().hex}.wav"

    # SINGLE upload: uses your helper that sets ContentType and KMS if configured
    _put_ulaw_wav(buf, key)

    # presign as before
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": TTS_BUCKET, "Key": key},
        ExpiresIn=PRESIGN_EXPIRES
    )

    logger.info(f"[TTS OUT] text_len={len(req.text)} seed={req.seed} model_sr={model_sr} out_sr={target_sr} key={key}")
    return {
        "bucket": TTS_BUCKET,
        "key": key,
        "url": url,
        "s3_url": url,
        "latency_ms": int((time.time() - t0) * 1000),
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
        chunk_length=chunk_length, #req.chunk_length or 128,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        temperature=req.temperature,
        format="wav",
        stream=True,
    )

    pcm_chunks: list[np.ndarray] = []
    stream = eng.engine.inference(req_obj)
    
    upload_every = int(target_sr * float(os.getenv("STREAM_UPLOAD_SECONDS", "0.25")))
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
    return {"keys": keys, "bucket": TTS_BUCKET, "region": S3_REGION}


@app.post("/transcript")
async def on_transcript(request: Request):
    # Twilio sends application/x-www-form-urlencoded
    ctype = request.headers.get("content-type", "")
    if "application/json" in ctype:
        payload = await request.json()
    else:
        form = await request.form()
        payload = dict(form)

    # Event gate
    event = payload.get("TranscriptionEvent") or payload.get("event")
    if event != "transcription-content":
        return {"ok": True}

    # Final flag
    final_val = payload.get("Final")
    is_final = final_val if isinstance(final_val, bool) else (str(final_val).lower() == "true")

    # Transcript text (TranscriptionData is a JSON string)
    text = ""
    tdata = payload.get("TranscriptionData")
    if isinstance(tdata, str):
        try:
            td = json.loads(tdata)
            text = td.get("transcript") or td.get("Transcript") or ""
        except Exception:
            text = ""
    if not text:
        return {"ok": True}

    # Map CallSid -> (ws, streamSid)
    call_sid = payload.get("CallSid") or payload.get("callSid")
    ws = ws_for_call(call_sid)
    sid = stream_sid_for_call(call_sid)
    if not ws or not sid:
        return {"ok": True}

    # BARGE-IN: partials => clear immediately
    if not is_final:
        send_clear(ws, sid)
        return {"ok": True}

    # On final: echo reply
    eng = get_engine()
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
        audio_f32, model_sr = eng.synthesize(text=text, sr=None)
    pcm16_np, _ = _to_pcm16_mono(audio_f32, model_sr, 8000)
    for ulaw in pcm16_to_ulaw_frames(pcm16_np.tobytes()):
        send_ulaw_media(ws, sid, ulaw)
    send_mark(ws, sid, "bot-end")
    return {"ok": True}


@app.api_route("/answer", methods=["GET","POST"])
def answer():
    twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Start>
    <Transcription statusCallbackUrl="{HTTP_BASE}/transcript"
                   transcriptionEngine="google"
                   partialResults="true"
                   languageCode="ko-KR"/>
  </Start>
  <Connect>
    <Stream url="wss://{WS_BASE}/twilio" />
  </Connect>
</Response>'''
    return Response(twiml, media_type="text/xml")

@app.post("/synthesize_stream_start")
def synthesize_stream_start(req: SynthesizeIn, background: BackgroundTasks,
                            authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    import uuid, time

    job_id = uuid.uuid4().hex
    target_sr = int(req.sample_rate or 8000)

    # Kick off chunked uploader (generate->upload per chunk)
    background.add_task(_run_stream_job_chunked, job_id, req.text, target_sr)
#     background.add_task(_run_stream_job_streaming, job_id, req.text, target_sr)

    # --- NEW: opportunistically return first_url/first_key if part0.wav appears fast ---
    # Tunables (keep small so we never block long)
    wait_ms   = int(os.getenv("FIRST_URL_WAIT_MS", "800"))        # total budget to wait
    poll_ms   = int(os.getenv("FIRST_URL_POLL_MS", "50"))         # polling interval
    pres_ttl  = int(os.getenv("PRESIGN_TTL_SEC", "120"))          # presigned URL TTL

    first_key = None
    first_url = None

    # Only bother if we have a positive wait budget
    if wait_ms > 0:
        deadline = time.time() + (wait_ms / 1000.0)
        # Poll for part0.wav
        while time.time() < deadline:
            # light-weight check: see if part 0 is among ready keys
            part_indices, _ = _list_ready_parts_from_s3(job_id)
            if 0 in part_indices:
                first_key = f"{job_id}/part0.wav"
                # Use your existing presign helper (region/bucket are already known)
                first_url = _presign(first_key, expires=pres_ttl)
                break
            time.sleep(poll_ms / 1000.0)

    resp = {"job_id": job_id, "bucket": TTS_BUCKET, "region": S3_REGION}
    if first_key and first_url:
        resp["first_key"] = first_key
        resp["first_url"] = first_url
    return resp


@app.get("/synthesize_stream_status/{job_id}")
def status(job_id: str, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    parts, has_final = _list_ready_parts_from_s3(job_id)
    return {"ready_parts": parts, "has_final": has_final}


@app.get("/synthesize_stream_batch")
def synthesize_stream_batch(
    job_id: str,
    start_idx: int = 0,
    limit: int = 4,
    expires: int = PRESIGN_EXPIRES,
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)

    part_indices, has_final = _list_ready_parts_from_s3(job_id)
    remaining = [i for i in part_indices if i >= int(start_idx)]
    take = remaining[:max(1, int(limit))]

    urls = [_presign(f"{job_id}/part{i}.wav", expires=int(expires)) for i in take]
    next_idx_out = (take[-1] + 1) if take else int(start_idx)
    has_more = len(remaining) > len(take)

    # attribute names align with your Connect/Lambda flow
    return {
        "JobId": job_id,
        "AudioS3Urls": urls,
        "AudioS3UrlCount": len(urls),
        "BatchCount": len(urls),
        "HasMore": "true" if has_more else "false",
        "HasFinal": "true" if has_final else "false",
        "NextIndexOut": next_idx_out,
        "Bucket": TTS_BUCKET,
        "Region": S3_REGION,
    }

@app.post("/synthesize_stream_cancel")
def synthesize_stream_cancel(req: CancelIn, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    _mark_cancelled(req.job_id)
    return {"ok": True, "job_id": req.job_id}