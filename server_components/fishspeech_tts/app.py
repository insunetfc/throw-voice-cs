import torch, os
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")  # PyTorch 2.x
torch.backends.cudnn.benchmark = True  

import os, io, uuid, base64, re, time, tempfile
from typing import Optional, Tuple, List
import subprocess, tempfile, boto3, os
from g2pk import G2p
from fastapi import UploadFile, File, Form
from pathlib import Path
import numpy as np
import soundfile as sf
import yaml
import boto3
from botocore.client import Config as BotoConfig
from functools import lru_cache
import logging, asyncio, warnings
from fastapi import Request
import sys

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import audioop, math
from pydantic import BaseModel
from threading import Event
from fastapi.responses import FileResponse
import threading, queue, time
from fastapi.responses import StreamingResponse

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
import io
import wave
import audioop
import subprocess
import tempfile
import numpy as np
import torch

# adjust these if your repo layout is different
FISH_ROOT = Path("/home/work/VALL-E/fishspeech/fish-speech").resolve()
DAC_SCRIPT = FISH_ROOT / "fish_speech" / "models" / "dac" / "inference.py"
DAC_CKPT = FISH_ROOT / "checkpoints" / "openaudio-s1-mini" / "codec.pth"

USE_COMPILE = os.getenv("TORCH_COMPILE", "0") == "1"
def maybe_compile(m):
    return torch.compile(m, mode="reduce-overhead", fullgraph=False) if USE_COMPILE else m

HTTP_BASE = "https://honest-trivially-buffalo.ngrok-free.app"  # your HTTP ngrok (app.py)
WS_BASE   = "302cac3fa5c9.ngrok-free.app"    
CANCEL_EVENTS = {}
chunk_length = 128 # originally 128
_PART_RE = re.compile(r"(?:^|.*/)?part(\d+)\.wav$")
REF_DIR = Path("/home/work/VALL-E/audio_samples/refs")
REF_DIR.mkdir(parents=True, exist_ok=True)
REF_AUDIO: dict[str, str] = {}

from fish_speech.inference_engine import TTSInferenceEngine
_original_inference = TTSInferenceEngine.inference

@torch.inference_mode()
def patched_inference(self, req: ServeTTSRequest):
    """
    Patched inference that checks REF_AUDIO for uploaded references
    """
    ref_id: str | None = req.reference_id
    prompt_tokens, prompt_texts = [], []

    req_ref_audio = getattr(req, "reference_audio", None)
    req_refs = getattr(req, "references", None)

    # Check if request actually brought a reference
    use_req_ref = (
        (req_ref_audio is not None)
        or (req_refs and len(req_refs) > 0)
        or (ref_id is not None)
    )

    # Try to load request-specific reference first
    if use_req_ref:
        if ref_id is not None:
            # Check our uploaded references
            if ref_id in REF_AUDIO:
                wav_path = REF_AUDIO[ref_id]
                pt_path = REF_DIR / f"{ref_id}.pt"
                
                if pt_path.exists():
                    logger.info(f"✅ [PATCHED] Loading uploaded ref from .pt: {pt_path}")
                    data = torch.load(str(pt_path), map_location="cpu")
                    prompt_tokens = data["prompt_tokens"]
                    prompt_texts = data["prompt_texts"]
                else:
                    logger.info(f"✅ [PATCHED] Loading uploaded ref from wav: {wav_path}")
                    try:
                        prompt_tokens, prompt_texts = self.load_from_path(wav_path, req.use_memory_cache)
                    except Exception as e:
                        logger.error(f"❌ Failed to load from path: {e}")
            else:
                # Check disk
                found = False
                for ext in (".wav", ".m4a", ".mp3"):
                    candidate = REF_DIR / f"{ref_id}{ext}"
                    if candidate.exists():
                        logger.info(f"✅ [PATCHED] Found ref on disk: {candidate}")
                        REF_AUDIO[ref_id] = str(candidate)
                        pt_path = REF_DIR / f"{ref_id}.pt"
                        if pt_path.exists():
                            data = torch.load(str(pt_path), map_location="cpu")
                            prompt_tokens = data["prompt_tokens"]
                            prompt_texts = data["prompt_texts"]
                        else:
                            prompt_tokens, prompt_texts = self.load_from_path(str(candidate), req.use_memory_cache)
                        found = True
                        break
                
                if not found:
                    logger.warning(f"⚠️ [PATCHED] ref_id {ref_id} not found, trying original loader")
                    # Try original method if it exists
                    if hasattr(self, 'load_by_id'):
                        try:
                            prompt_tokens, prompt_texts = self.load_by_id(ref_id, req.use_memory_cache)
                        except Exception as e:
                            logger.warning(f"Original load_by_id failed: {e}")
                            
        elif req_refs:
            prompt_tokens, prompt_texts = self.load_by_hash(req_refs, req.use_memory_cache)
            logger.info(f"✅ [PATCHED] Loaded reference by hash")
        elif req_ref_audio is not None:
            prompt_tokens, prompt_texts = self.load_from_path(req_ref_audio, req.use_memory_cache)
            logger.info(f"✅ [PATCHED] Loaded reference from path: {req_ref_audio}")

    # Fall back to default embedding only if no request reference was loaded
    if (not prompt_tokens or not prompt_texts) and self.load_embeddings:
        logger.info(f"📦 [PATCHED] Loading default embedding from {self.embedding_path}")
        data = torch.load(self.embedding_path, map_location="cpu")
        prompt_tokens = data["prompt_tokens"]
        prompt_texts = data["prompt_texts"]

    # Final check
    if not prompt_tokens or not prompt_texts:
        logger.warning("⚠️ [PATCHED] No prompt tokens/texts available")

    logger.info(f'[PATCHED] Done loading embeddings. Using {len(prompt_tokens)} prompt token(s).')

    # Now call the rest of the original inference logic
    # Set the random seed if provided
    if req.seed is not None:
        from fish_speech.utils import set_seed
        set_seed(req.seed)
        logger.warning(f"set seed: {req.seed}")

    # Get the symbolic tokens from the LLAMA model
    response_queue = self.send_Llama_request(req, prompt_tokens, prompt_texts)

    # Get the sample rate from the decoder model
    if hasattr(self.decoder_model, "spec_transform"):
        sample_rate = self.decoder_model.spec_transform.sample_rate
    else:
        sample_rate = self.decoder_model.sample_rate

    # If streaming, send the header
    if req.streaming:
        from fish_speech.inference_engine.utils import InferenceResult, wav_chunk_header
        yield InferenceResult(
            code="header",
            audio=(
                sample_rate,
                np.array(wav_chunk_header(sample_rate=sample_rate)),
            ),
            error=None,
        )

    segments = []

    while True:
        from fish_speech.models.text2semantic.inference import GenerateResponse, WrappedGenerateResponse
        from fish_speech.inference_engine.utils import InferenceResult
        
        # Get the response from the LLAMA model
        wrapped_result: WrappedGenerateResponse = response_queue.get()
        if wrapped_result.status == "error":
            yield InferenceResult(
                code="error",
                audio=None,
                error=(
                    wrapped_result.response
                    if isinstance(wrapped_result.response, Exception)
                    else Exception("Unknown error")
                ),
            )
            break

        # Check the response type
        if not isinstance(wrapped_result.response, GenerateResponse):
            raise TypeError(
                f"Expected GenerateResponse, got {type(wrapped_result.response).__name__}"
            )

        result: GenerateResponse = wrapped_result.response
        if result.action != "next":
            segment = self.get_audio_segment(result)

            if req.streaming:
                yield InferenceResult(
                    code="segment",
                    audio=(sample_rate, segment),
                    error=None,
                )
            segments.append(segment)
        else:
            break

    # Clean up the memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    # Edge case: no audio generated
    if len(segments) == 0:
        yield InferenceResult(
            code="error",
            audio=None,
            error=RuntimeError("No audio generated, please check the input text."),
        )
    else:
        # Streaming or not, return the final audio
        audio = np.concatenate(segments, axis=0)
        yield InferenceResult(
            code="final",
            audio=(sample_rate, audio),
            error=None,
        )

    return None

# Apply the patch
TTSInferenceEngine.inference = patched_inference
logger.info("✅ Patched TTSInferenceEngine.inference to use REF_AUDIO")

# ---- chunker (tune for ko-KR / general) ----
import re, io, numpy as np
from typing import List
ENGINE = None
WARMED = False
TGT_SE = None
g2p = G2p()
# sensible defaults
MAX_NEW_DEFAULT   = int(os.getenv("MAX_NEW_TOKENS", "120"))  # <- cap
CHUNK_LEN_DEFAULT = int(os.getenv("CHUNK_LENGTH", "192"))
reference_audio = "/home/work/VALL-E/audio_samples/latest_reference.m4a"


import time
if not hasattr(torchaudio, "list_audio_backends"):
    def _list_audio_backends():
        # With dispatcher, we don't need to choose; return a safe placeholder.
        return ["soundfile"]
    torchaudio.list_audio_backends = _list_audio_backends

# Optional: make get_audio_backend non-None for any code that prints it
if not hasattr(torchaudio, "get_audio_backend"):
    torchaudio.get_audio_backend = lambda: "soundfile"

async def _save_upload_to_disk(upload: UploadFile, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    # keep original suffix if present
    suffix = Path(upload.filename).suffix or ".wav"
    out_path = dest_dir / f"{uuid.uuid4().hex}{suffix}"
    data = await upload.read()
    with out_path.open("wb") as f:
        f.write(data)
    return out_path


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

def mulaw_wav_to_pcm16_wav_bytes(mulaw_bytes: bytes) -> bytes:
    """
    Read a μ-law WAV (what we upload to S3) and re-encode it as
    normal PCM 16-bit WAV, using soundfile which understands ULAW.
    """
    import soundfile as sf

    src_buf = io.BytesIO(mulaw_bytes)

    # soundfile can decode ULAW-in-WAV directly
    data, sr = sf.read(src_buf, dtype="int16")  # data is now PCM int16 array

    out_buf = io.BytesIO()
    # write as normal PCM WAV → playable on PC
    sf.write(out_buf, data, sr, format="WAV", subtype="PCM_16")
    out_buf.seek(0)
    return out_buf.getvalue()

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

import re, unicodedata

def normalize_for_key(s: str) -> str:
    import unicodedata, re
    s = unicodedata.normalize("NFKC", s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    # keep %, numbers, basic punctuation; avoid stripping sentence finals
    return s

# For TTS (preserve prosody)
def prepare_for_tts(s: str) -> str:
    s = s.strip()
    # ensure a sentence-final mark to keep a natural tail
    if not s.endswith((".", "!", "?", "…", "~")):
        s = s + "~"  # a soft tail; pick what your model likes
    return s

def _voiced_endpoint(samples: np.ndarray, sr: int, frame_ms: int = 20, rms_thr: float = 0.02, hang_ms: int = 120):
    """Return sample index of last voiced frame using simple RMS VAD with a short hangover."""
    if samples.ndim > 1:
        samples = samples.mean(axis=0)
    frame = max(1, int(sr * (frame_ms / 1000.0)))
    hang = max(1, int(sr * (hang_ms / 1000.0)))
    last_voiced = 0
    for i in range(0, len(samples) - frame, frame):
        rms = float(np.sqrt(np.mean(np.square(samples[i:i+frame])) + 1e-9))
        if rms > rms_thr:
            last_voiced = i + frame
    return min(len(samples), last_voiced + hang)

_ZWJ = "\u2060"  # WORD JOINER (prevents '겠' dropout)

def tts_prepare(text: str) -> str:
    """
    TTS-only preprocessing (do NOT use for hashing):
      - NFKC normalize & trim
      - Strip simple markdown markers
      - Make URLs speakable in Korean
      - Expand % -> '퍼센트'
      - Prevent '겠' skip with WORD JOINER
      - Gentle sentence-final tail
    """
    s = unicodedata.normalize("NFKC", text).strip()

    # 1) Strip simple markdown
    s = s.replace("**", "").replace("__", "").replace("*", "").replace("_", "").replace("`", "")

    # 2) URLs -> speakable
    def _speak_url(m):
        full = m.group(0)
        core = re.sub(r"^https?://", "", full, flags=re.I)
        core = re.sub(r"^www\.", "더블유 더블유 더블유 점 ", core, flags=re.I)
        core = core.replace(".com", " 닷컴")
        core = core.replace(".kr", " 점 케이알")
        core = core.replace(".", " 점 ")
        core = core.replace("/", " 슬래시 ")
        return core.strip()
    s = re.sub(r"https?://\S+|www\.\S+", _speak_url, s, flags=re.I)

    # 3) Safe glyph & numeric expansions
    s = s.replace("켇", "켜")                 # rare corruption guard
    s = re.sub(r"(\d+)\s*%", r"\1 퍼센트", s)  # 15% -> 15 퍼센트

    s = re.sub(
        r"[\U0001F600-\U0001F64F"
        r"\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF"
        r"\U0001F1E0-\U0001F1FF"
        r"\u2600-\u26FF\u2700-\u27BF]+",
        "",
        s
    )
    s = re.sub(r"[:;][\-\^]?[)D(]+", "", s)

    # 4) Collapse whitespace
    s = re.sub(r"\s+", " ", s)

    # 5) Prevent '겠' dropout before common endings (TTS-only)
    s = re.sub(r"겠(?=(다|습니다|어요|에요|지요|네요|군요|고요|죠))", "겠" + _ZWJ, s)

    # 6) Ensure a gentle sentence ending (so engines don't truncate)
    if not re.search(r"[.!?…~다요]$", s):
        s += "…"
        
    def _pct(m):
        n = int(m.group(1))
        return f"{_num_to_korean_under_10000(n)} 퍼센트"
    s = re.sub(r"\b(\d{1,4})\s*%", _pct, s)

    def _num(m):
        n = int(m.group(0))
        return _num_to_korean_under_10000(n)
    s = re.sub(r"\b\d{1,4}\b", _num, s)
    
    return s

_KD = {0:"영",1:"일",2:"이",3:"삼",4:"사",5:"오",6:"육",7:"칠",8:"팔",9:"구"}
_KU = ["","십","백","천"]

def _num_to_korean_under_10000(n: int) -> str:
    if n == 0: return _KD[0]
    parts = []
    i = 0
    while n > 0:
        d = n % 10
        if d:
            # omit "일" before 십/백/천 (10,100,1000)
            if d == 1 and i > 0:
                parts.append(_KU[i])
            else:
                parts.append(_KD[d] + _KU[i])
        n //= 10; i += 1
    return "".join(reversed(parts))


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
#     chunk = g2p(chunk)
    with torch.inference_mode(), torch.amp.autocast(dtype=torch.float16, device_type='cuda'):
#         audio_f32, model_sr = eng.synthesize(  # <- consumes generator internally
#             text=chunk,
#             sr=target_sr,
#             temperature=TTS_TEMP,
#             top_p=TTS_TOP_P,
#             repetition_penalty=TTS_REP,
#             max_new_tokens=512,
#             chunk_length=chunk_length,
#             use_memory_cache=True,
#         )
        audio_f32, model_sr = synthesize_with_guard(
            eng,# <- consumes generator internally
            text=tts_prepare(chunk),
            speaker_wav=reference_audio,
            sr=target_sr,
            temperature=TTS_TEMP,
            top_p=TTS_TOP_P,
            repetition_penalty=TTS_REP,
            max_new_tokens=512,
            chunk_length=chunk_length,
            use_memory_cache=False,
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
    text = g2p(text)
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

def wav_to_prompt_pt_via_dac_cli(wav_path: str, pt_path: str) -> str:
    """
    1. run: python fish_speech/models/dac/inference.py -i <wav> --checkpoint-path <ckpt>
       (exactly like run_fish_pipeline.py)
    2. that script will drop fake.npy in the current working directory
    3. load fake.npy -> wrap -> save pt_path
    """
    # 1) run the same command as your script
    cmd = [
        sys.executable,
        str(DAC_SCRIPT),
        "-i",
        str(wav_path),
        "--checkpoint-path",
        str(DAC_CKPT),
    ]
    # IMPORTANT: run it in repo root so it writes fake.npy where we expect
    subprocess.run(cmd, cwd=str(FISH_ROOT), check=True)

    fake_npy = FISH_ROOT / "fake.npy"
    if not fake_npy.exists():
        raise FileNotFoundError(f"Expected {fake_npy} to be created by DAC script")

    # 2) load codes
    codes = np.load(str(fake_npy))        # this is how your convert_tokens.py does it

    # 3) wrap like cached_ref.pt
    codes_tensor = torch.tensor(codes, dtype=torch.long)
    prompt_tokens = [codes_tensor]
    prompt_texts = [""]

    torch.save(
        {
            "prompt_tokens": prompt_tokens,
            "prompt_texts": prompt_texts,
        },
        pt_path,
    )

    # optional: clean up fake.npy so concurrent calls don’t collide
    try:
        fake_npy.unlink()
    except OSError:
        pass

    return pt_path

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
CFG_PATH        = os.getenv("FISH_CONFIG", "/home/work/VALL-E/fishspeech/fish-speech/config.yaml")
AWS_REGION      = os.getenv("AWS_REGION", "ap-northeast-2")
TTS_BUCKET      = os.getenv("TTS_BUCKET", "tts-bucket-250810")
KEY_PREFIX_DEF  = os.getenv("KEY_PREFIX", "sessions/demo")
PRESIGN_EXPIRES = int(os.getenv("PRESIGN_EXPIRES", "600"))
KMS_KEY_ARN = os.getenv("KMS_KEY_ARN") 
API_TOKEN       = os.getenv("API_TOKEN", "")  # optional bearer
S3_REGION = AWS_REGION
TTS_TOP_P = float(os.getenv("TTS_TOP_P", "0.9"))
TTS_TEMP  = float(os.getenv("TTS_TEMP",  "0.9"))
TTS_REP   = float(os.getenv("TTS_REP",   "1.25")) 
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

PAD_HEAD_MS = 40      # avoids first-phoneme chop
PAD_TAIL_MS = 80     # avoids last-syllable chop on telephony
MIN_DURATION_SEC = 0.40
SAMPLE_RATE = 8000     # add this near your config section
FRAME_S = 0.02
SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_S)
REF_DIR = Path("/home/work/VALL-E/audio_samples/refs")
REF_DIR.mkdir(parents=True, exist_ok=True)

# point to your existing codec checkpoint – this matches your server config
DAC_CKPT_PATH = Path("/home/work/VALL-E/fishspeech/fish-speech/checkpoints/openaudio-s1-mini/codec.pth")

_dac_model = None

import numpy as np
import soundfile as sf

def _pad_and_write_wav(audio: np.ndarray, sr: int) -> bytes:
    # Head/tail pad
    head = np.zeros(int(PAD_HEAD_MS/1000.0 * sr), dtype=audio.dtype)
    tail = np.zeros(int(PAD_TAIL_MS/1000.0 * sr), dtype=audio.dtype)
    out = np.concatenate([head, audio, tail])

    # Enforce minimum duration
    need = int(max(0.0, MIN_DURATION_SEC - (out.shape[0] / sr)) * sr)
    if need > 0:
        out = np.concatenate([out, np.zeros(need, dtype=audio.dtype)])

    bio = io.BytesIO()
    sf.write(bio, out, sr, format="WAV", subtype="PCM_16")
    return bio.getvalue()


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
    assert _PART_RE.search("abc/part0.wav").group(1) == "0"
    assert _PART_RE.search("part12.wav").group(1) == "12"
    assert _PART_RE.search("xyz/final.wav") is None

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
    import numpy as np, soundfile as sf
    head = np.zeros(int(sr * (PAD_HEAD_MS/1000.0)), dtype=np.float32)
    tail = np.zeros(int(sr * (PAD_TAIL_MS/1000.0)), dtype=np.float32)
    out  = np.concatenate([head, samples_f32, tail])

    # Enforce minimum duration (prevents truncation on very short lines)
    need = int(max(0.0, MIN_DURATION_SEC - (out.shape[0] / sr)) * sr)
    if need > 0:
        out = np.concatenate([out, np.zeros(need, dtype=np.float32)])

    sf.write(buf, out, sr, subtype="ULAW", format="WAV")

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

def last_punct(s: str) -> str:
    m = re.search(r'([.!?。！？…~])\s*$', s)
    return m.group(1) if m else ''

def pause_ms_for(sentence: str, wpm_hint: float | None = None) -> int:
    """Return a natural pause length based on trailing punctuation (ms)."""
    p = last_punct(sentence)
    # Base pauses tuned for 44.1k model audio -> still fine when later downsampled
    if p in ('?', '？', '!','！'):
        base = 260
    elif p in ('…',):
        base = 300
    elif p in ('.', '。'):
        base = 170
    elif p in ('~',):
        base = 180
    else:
        base = 120  # soft boundary (no explicit punctuation)

    # Optional: slow/fast speaker hint (~words per minute) -> adjust ±15%
    if wpm_hint:
        # 180 wpm -> -10%; 120 wpm -> +10% (clamped)
        scale = np.clip(1.5 - (wpm_hint / 200.0), 0.85, 1.15)
        base = int(base * scale)
    return base

# ---- in your synthesize() helper path (used by /synthesize) ----
def synth_one_model_sr(eng, s: str, speaker_wav: str | None = None, **gen):
    t = tts_prepare(s)

    max_req = int(gen.get("max_new_tokens", 128))
    want = 192 if len(t) > 60 else 128
    cap = min(max_req, want)

    temperature        = float(gen.get("temperature", 0.9))
    top_p              = float(gen.get("top_p", 0.8))
    repetition_penalty = float(gen.get("repetition_penalty", 1.2))
    chunk_length       = int(gen.get("chunk_length", 128))
    use_memory_cache   = bool(gen.get("use_memory_cache", False))

    # pick which reference to use
    ref_path = speaker_wav if speaker_wav else reference_audio

    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        audio_f32, model_sr = synthesize_with_guard(
            eng,
            text=t,
            speaker_wav=ref_path,
            temperature=max(0.0, min(1.0, temperature)),
            top_p=max(0.0, min(1.0, top_p)),
            repetition_penalty=repetition_penalty,
            max_new_tokens=cap,
            chunk_length=chunk_length,
            use_memory_cache=use_memory_cache,
        )
    a = np.asarray(audio_f32, dtype=np.float32)
    if a.ndim > 1: a = a.mean(axis=0)
    m = float(np.max(np.abs(a))) if a.size else 1.0
    if m > 1e-6: a = np.clip(a / max(1.0, m), -1.0, 1.0)
    return a, model_sr


def equal_power_xfade(a: np.ndarray, b: np.ndarray, sr: int, fade_ms: int = 24) -> np.ndarray:
    """Cross-fade end of a into start of b with equal-power window."""
    n = max(1, int(sr * fade_ms / 1000.0))
    n = min(n, len(a), len(b))
    if n <= 1:
        return np.concatenate([a, b], axis=0)
    # tails/heads
    a_tail = a[-n:].astype(np.float32)
    b_head = b[:n].astype(np.float32)
    # equal-power ramps
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    a_w = np.sqrt(1.0 - t)   # fade-out
    b_w = np.sqrt(t)         # fade-in
    mixed = a_tail * a_w + b_head * b_w
    return np.concatenate([a[:-n], mixed, b[n:]], axis=0)

def join_with_pause_and_xfade(chunks: list[np.ndarray], sr: int, pauses_ms: list[int], fade_ms: int = 24) -> np.ndarray:
    """Join chunks with a bit of silence and a micro cross-fade around each junction."""
    out = chunks[0].astype(np.float32)
    for i in range(1, len(chunks)):
        pause = np.zeros(int(sr * pauses_ms[i-1] / 1000.0), dtype=np.float32)
        # fade-out previous, insert pause, fade-in next
        bridge = np.concatenate([pause, chunks[i][:0]], axis=0)  # placeholder (keeps type)
        # do a tiny fade to zero at end of previous and from zero at start of next to avoid clicks
        # (xfade is applied directly between audio segments; pause is true silence between them)
        out = equal_power_xfade(out, np.concatenate([pause, chunks[i]], axis=0), sr, fade_ms=fade_ms)
    return out

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

def synthesize_with_guard(engine, text: str, **kwargs) -> tuple[np.ndarray, int]:
    """
    Wrapper around engine.synthesize that retries once with a prosody nudge
    for Korean polite endings that sometimes early-stop. Returns (audio, sr).
    Accepts and forwards arbitrary kwargs (speaker_wav, sr, temperature, etc.).
    """
    # 1st pass
    audio, model_sr = engine.synthesize(text=tts_prepare(text), **kwargs)
    dur = (len(audio) / float(model_sr)) if model_sr else 0.0

    # Heuristic: endings that sometimes early-stop
    risky = ("겠습니다" in text) or ("드리겠습니다" in text)

    if risky and dur < 1.5:
        # Retry with a soft tail if not already present
        retry_text = text if text.endswith(("…", "~")) else (text + "…")
        try:
            audio2, sr2 = engine.synthesize(text=retry_text, **kwargs)
            dur2 = (len(audio2) / float(sr2)) if sr2 else 0.0
            if dur2 > dur:  # keep the better (longer) one
                return audio2, sr2
        except Exception as e:
            # Don't fail the request if retry has issues—just fall back to first pass
            print(f"[WARN] synthesize_with_guard retry failed: {e}")

    return audio, model_sr


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
                text=tts_prepare("안녕하세요."),
                references=[],
                reference_id=None,
                max_new_tokens=256,
                chunk_length=chunk_length,
                top_p=0.7,
                repetition_penalty=1.2,
                temperature=0.9,
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

        llama_ckpt   = cfg.get("llama_checkpoint_path", "/home/work/VALL-E/fishspeech/fish-speech/checkpoints/openaudio-s1-mini")
        decoder_cfg  = cfg.get("decoder_config_name", "modded_dac_vq")
        decoder_ckpt = cfg.get("decoder_checkpoint_path", "/home/work/VALL-E/fishspeech/fish-speech/checkpoints/openaudio-s1-mini/codec.pth")
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
        clean = tts_prepare(text)
        # req = ServeTTSRequest(
        #     text=clean,
        #     references=[],
        #     reference_id=None,
        #     reference_audio=speaker_wav,
        #     reference_text=None,
        #     max_new_tokens=int(params.get("max_new_tokens", 0)),
        #     chunk_length=chunk_length, #int(params.get("chunk_length", 300)),
        #     top_p=float(params.get("top_p", 0.8)),
        #     repetition_penalty=float(params.get("repetition_penalty", 1.1)),
        #     temperature=float(params.get("temperature", 0.8)),
        #     seed=int(params.get("seed", 42)),
        #     use_memory_cache=umc_str,  # 'on' | 'off'
        #     format="wav",
        #     stream=False,
        # )
        temp = _clip(params.get("temperature", 0.8), 0.0, 1.0, 0.8)
        topp = _clip(params.get("top_p", 0.8),       0.0, 1.0, 0.8)
        rpen = _clip(params.get("repetition_penalty", 1.25), 0.5, 2.0, 1.25)
        
        req = ServeTTSRequest(
            text=clean,
            references=[],
            reference_id=None,
            reference_audio=speaker_wav,
            reference_text=None,
            max_new_tokens=int(params.get("max_new_tokens", 0)),
            chunk_length=chunk_length,
            top_p=topp,
            repetition_penalty=rpen,
            temperature=temp,
            seed=int(params.get("seed", 42)),
            use_memory_cache=umc_str,
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

def _load_engine_once():
    global ENGINE
    if ENGINE is None:
        ENGINE = get_engine()
    return ENGINE

def _load_target_se_once():
    """Load cached embedding (if available)"""
    global TGT_SE
    if TGT_SE is not None:
        return TGT_SE
    emb_path = os.getenv("FISHSPEECH_TGT_SE_PTH", "/home/work/VALL-E/audio_samples/cached_ref.pt")
    if os.path.isfile(emb_path):
        TGT_SE = torch.load(emb_path, map_location="cpu")
        logger.info(f"[FishSpeech] Loaded target SE from {emb_path}")
    else:
        logger.warning("[FishSpeech] No target SE found, warmup may be slower.")
        TGT_SE = None
    return TGT_SE

# Detect a leading style tag block like "(friendly)" or "(happy, bright)"
STYLE_BLOCK_RE = re.compile(r'^\s*\(([^)]+)\)\s*', flags=re.IGNORECASE)

def extract_style_block(text: str) -> tuple[str, str]:
    """
    If text starts with a (style, tag) block, return (style_block_with_parens, remainder_without_it).
    Otherwise return ("", text).
    """
    m = STYLE_BLOCK_RE.match(text or "")
    if not m:
        return "", text
    # keep original parentheses as-is so model sees the same tokens
    start, end = m.span()
    return text[start:end].strip(), text[end:].lstrip()

def has_leading_style_block(s: str) -> bool:
    return bool(STYLE_BLOCK_RE.match(s or ""))

def apply_style_to_each_sentence(sentences: list[str], style_block: str) -> list[str]:
    """
    Prepend style_block to each sentence unless it already has a leading style tag.
    """
    if not style_block:
        return sentences
    out = []
    for s in sentences:
        s2 = s.strip()
        if not has_leading_style_block(s2):
            s2 = f"{style_block} {s2}"
        out.append(s2)
    return out


# ===================== FastAPI =====================
app = FastAPI()
from pydantic import BaseModel, Field, conint, confloat

class SynthesizeIn(BaseModel):
    text: str
    do_endpoint: bool = False
    endpoint: Optional[str] = None
    key_prefix: Optional[str] = None
    sample_rate: Optional[int] = None
    speaker_wav: Optional[str] = None
    ref_id: Optional[str] = None

    # Keep these in-bounds for ServeTTSRequest
    temperature: Optional[confloat(ge=0.0, le=1.0)] = Field(0.8, description="0..1")
    top_p:       Optional[confloat(ge=0.0, le=1.0)] = Field(0.8, description="0..1")
    seed:        Optional[int] = 42
    repetition_penalty: Optional[confloat(ge=0.5, le=2.0)] = Field(1.25, description="0.5..2.0")

    max_new_tokens: Optional[conint(ge=0, le=1024)] = 128
    chunk_length:   Optional[conint(ge=32, le=512)] = chunk_length
    use_memory_cache: Optional[bool] = False

def _check_auth(authorization: Optional[str]):
    if API_TOKEN and authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="unauthorized")

def _clip(x, lo, hi, default):
    try:
        v = float(x if x is not None else default)
    except Exception:
        v = float(default)
    return max(lo, min(hi, v))


@app.post("/synthesize/warmup")
async def warmup():
    logger.info("Loading FishSpeech engine at startup…")
    # Do the warmup in a background task so startup doesn't fail if warmup errors
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _warm_once)
    logger.info("Engine warmup done.")
    return {"ok": True}

def _cuda_warm_once():
    import torch
    if torch.cuda.is_available():
        _ = torch.randn(1, device="cuda")
        torch.cuda.synchronize()

def _warmup_fishspeech():
    global WARMED
    if WARMED:
        return
    logger.info("[FishSpeech] Starting warmup...")

    eng = _load_engine_once()
    tgt_se = _load_target_se_once()

    try:
        dummy_text = "안녕하세요. 테스트입니다."
        logger.info("[FishSpeech] Running dummy inference for warmup...")
        with torch.inference_mode(), torch.amp.autocast(dtype=torch.float16, device_type='cuda'):
            # FishEngineAdapter.synthesize(...) exists in your class
            audio, sr = eng.synthesize(
                tts_prepare(dummy_text),
                speaker_wav=reference_audio,  # embedding is already loaded
                sr=8000,
                temperature=0.9,
                top_p=0.9,
                repetition_penalty=1.1,
                max_new_tokens=128,
                chunk_length=chunk_length,
                use_memory_cache=False,
            )
        logger.info(f"[FishSpeech] Warmup complete. Dummy audio length: {len(audio)} samples at {sr} Hz")
        WARMED = True
    except Exception as e:
        logger.warning(f"[FishSpeech] Warmup failed: {e}")

@app.on_event("startup")
def _startup_event():
    _warmup_fishspeech()
    
@app.get("/health")
def health():
    return {"ok": True, "bucket": TTS_BUCKET, "region": AWS_REGION}

@app.post("/reference")
async def upload_reference(ref_wav: UploadFile = File(...), authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    
    # Save the uploaded file
    saved_path = await _save_upload_to_disk(ref_wav, REF_DIR)
    ref_id = uuid.uuid4().hex
    
    logger.info(f"[REF] Saved upload to {saved_path}")

    # Store in memory
    REF_AUDIO[ref_id] = str(saved_path)

    # Create the .pt embedding file
    pt_path = REF_DIR / f"{ref_id}.pt"
    try:
        logger.info(f"[REF] Encoding reference to .pt file...")
        wav_to_prompt_pt_via_dac_cli(str(saved_path), str(pt_path))
        logger.info(f"[REF] Successfully created {pt_path}")
    except Exception as e:
        logger.exception(f"[REF] Failed to build .pt from {saved_path}: {e}")
        # Don't fail the request - the engine might be able to encode on-the-fly
        logger.warning(f"[REF] Continuing without .pt file, will try runtime encoding")

    return {
        "ref_id": ref_id,
        "path": str(saved_path),
        "has_pt": pt_path.exists(),
    }


def get_dac_model():
    """Lazy-load DAC once, reuse for every ref conversion."""
    global _dac_model
    if _dac_model is None:
        from fish_speech.models.dac.inference import load_model
        model = load_model(str(DAC_CKPT_PATH))
        model.eval()
        _dac_model = model
    return _dac_model


def wav_to_prompt_pt(wav_path: str, pt_path: str, target_sr: int = 24000) -> str:
    """
    1. load wav
    2. resample to DAC sample rate
    3. run DAC encode -> codes
    4. wrap as {prompt_tokens: [codes_tensor], prompt_texts: [""]}  (like convert_tokens.py)
    5. save to pt_path
    """
    import torch
    import numpy as np

    model = get_dac_model()

    wav, sr = torchaudio.load(wav_path)
    # mono
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # resample if needed
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    with torch.no_grad():
        encoded = model.encode(wav)

    # model.encode(...) sometimes returns dict, sometimes tensor – copy run_fish_pipeline.py’s pattern :contentReference[oaicite:2]{index=2}
    if isinstance(encoded, dict) and "codes" in encoded:
        codes = encoded["codes"].cpu()
    else:
        codes = encoded.cpu()

    # we want a 2D long tensor, then a list of length 1 – just like convert_tokens.py :contentReference[oaicite:3]{index=3}
    codes = codes.to(torch.long)
    prompt_tokens = [codes]          # list of 1 tensor
    prompt_texts = [""]              # list of 1 string

    torch.save(
        {
            "prompt_tokens": prompt_tokens,
            "prompt_texts": prompt_texts,
        },
        pt_path,
    )
    return pt_path

def synthesize_with_optional_ref(
    adapter,
    text: str,
    ref_pt_path: str | None = None,
    ref_wav_path: str | None = None,
    **gen_params,
):
    import io, soundfile as sf

    # 🔍 DEBUG: Log what we received
    logger.info(f"[SYNTH] ref_wav_path={ref_wav_path}")
    logger.info(f"[SYNTH] ref_pt_path={ref_pt_path}")
    
    # Check if files actually exist
    if ref_wav_path:
        exists = os.path.exists(ref_wav_path)
        logger.info(f"[SYNTH] ref_wav exists={exists}")
        if exists:
            logger.info(f"[SYNTH] ref_wav size={os.path.getsize(ref_wav_path)} bytes")
    
    max_new_tokens = int(gen_params.get("max_new_tokens", 128))
    temperature = float(gen_params.get("temperature", 0.9))
    top_p = float(gen_params.get("top_p", 0.95))
    repetition_penalty = float(gen_params.get("repetition_penalty", 1.2))
    chunk_length = int(gen_params.get("chunk_length", 128))
    use_memory_cache_bool = bool(gen_params.get("use_memory_cache", False))
    use_memory_cache = "on" if use_memory_cache_bool else "off"
    seed = gen_params.get("seed", None)

    req_kwargs = dict(
        text=text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        chunk_length=chunk_length,
        use_memory_cache=use_memory_cache,
        streaming=False,
        format="wav",
        seed=seed,
    )

    # 🎯 IMPORTANT: use the WAV, not the .pt
    if ref_wav_path is not None:
        logger.info(f"[SYNTH] Using reference_audio: {ref_wav_path}")
        req = ServeTTSRequest(
            **req_kwargs,
            references=[],
            reference_id=None,
            reference_audio=ref_wav_path,  # ← This is the key!
        )
    else:
        logger.info("[SYNTH] No reference audio, using default")
        req = ServeTTSRequest(
            **req_kwargs,
            references=[],
            reference_id=None,
            reference_audio=None,
        )

    # run inference
    results = adapter.engine.inference(req)
    sr = None
    audio_np = None
    for item in results:
        if item.code == "final":
            sr, audio_np = item.audio
            break

    if audio_np is None:
        raise RuntimeError("no audio generated")

    buf = io.BytesIO()
    sf.write(buf, audio_np, sr, format="WAV", subtype="PCM_16")
    return sr, buf.getvalue()

@app.post("/synthesize")
def synthesize(req: SynthesizeIn, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    t0 = time.time()

    target_sr = int(req.sample_rate or 8000)
    eng = get_engine()

    # 0) figure out which reference audio to use
    chosen_ref = None
    emb_obj = None
    if req.ref_id and req.ref_id in REF_EMB:
        emb_obj = REF_EMB[req.ref_id]
        chosen_ref = None  # we will NOT pass a wav
    else:
        # old behavior
        if req.ref_id:
            chosen_ref = REF_AUDIO.get(req.ref_id) or ...
        elif req.speaker_wav:
            chosen_ref = req.speaker_wav
        else:
            chosen_ref = reference_audio

    # Extract a leading style block once (e.g., "(friendly)" or "(happy, bright)")
    style_block, remainder = extract_style_block(req.text or "")
    
    # Split sentences on the remainder to avoid losing the tag
    sentences = split_ko_sentences(remainder) or [remainder.strip()]
    
    # Re-apply the style block to EVERY sentence so sentence-by-sentence still carries tone
    sentences = apply_style_to_each_sentence(sentences, style_block)
    
    # (Optional but recommended) keep your generation args together
    gen = dict(
        temperature=req.temperature,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        max_new_tokens=(req.max_new_tokens or 128),
        chunk_length=(req.chunk_length or chunk_length),
        use_memory_cache=bool(req.use_memory_cache),
    )
    
    pieces_model = []
    model_sr_ref = None
    for s in sentences:
        a, sr_m = synth_one_model_sr(eng, s, chosen_ref, **gen)
        model_sr_ref = model_sr_ref or sr_m
        pieces_model.append(a)


    # 2) Natural pauses per punctuation
    pauses_ms = [pause_ms_for(s) for s in sentences[:-1]]

    # 3) Join with micro cross-fade (at model_sr), then add the real pause
    if len(pieces_model) == 1:
        joined_model = pieces_model[0]
    else:
        # do: [a] + (pause + xfade) + [b] + ...
        out = pieces_model[0]
        for i in range(1, len(pieces_model)):
            pause = np.zeros(int(model_sr_ref * pauses_ms[i-1] / 1000.0), dtype=np.float32)
            # cross-fade into the pause+next to soften the boundary
            out = equal_power_xfade(
                out,
                np.concatenate([pause, pieces_model[i]], axis=0),
                model_sr_ref,
                fade_ms=6,  # was 24
            )
        joined_model = out

    # 4) Resample once to target_sr
    float_mono, _ = _to_float_mono(joined_model, model_sr_ref, target_sr)

    # 5) Gentle endpointing
    # replace in synthesize()
    speech_end = _voiced_endpoint(
        float_mono,
        target_sr,
        frame_ms=20,
        rms_thr=0.0010,   # ↓ lower threshold = keep quieter tail
        hang_ms=480,      # ↑ longer hang to keep more of the last syllable
    )
    trimmed = float_mono[:max(speech_end, int(target_sr * 0.30))] if speech_end else float_mono


    # 6) μ-law WAV to S3 (unchanged)
    buf = io.BytesIO()
    write_ulaw_wav(buf, trimmed, target_sr)
    buf.seek(0)
    key_prefix = (req.key_prefix or KEY_PREFIX_DEF).rstrip("/")
    key = f"{key_prefix}/{uuid.uuid4().hex}.wav"
    _put_ulaw_wav(buf, key)
    url = s3.generate_presigned_url("get_object", Params={"Bucket": TTS_BUCKET, "Key": key}, ExpiresIn=PRESIGN_EXPIRES)

    logger.info(f"[TTS OUT] text_len={len(req.text)} out_sr={target_sr} key={key}")
    return {"bucket": TTS_BUCKET, "key": key, "url": url, "s3_url": url,
            "latency_ms": int((time.time() - t0) * 1000), "sample_rate": target_sr, "text": req.text}

@app.post("/synthesize2")
async def synthesize2(req: SynthesizeIn, background_tasks: BackgroundTasks):
    t0 = time.time()
    eng = get_engine()
    adapter = eng.engine

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        # ---- 1) Style + sentence splitting ----
        style_block, bare_text = extract_style_block(text)
        sentences = split_ko_sentences(bare_text)
        if not sentences:
            raise HTTPException(status_code=400, detail="No sentences parsed")

        # Apply style (e.g. "(happy)") to each sentence
        sentences = apply_style_to_each_sentence(sentences, style_block)

        # We will later resample to 8k; work at model/native SR then downsample
        target_sr = int(req.sample_rate or 8000)

        # ---- 2) Shared TTS request kwargs (except text) ----
        req_kwargs = dict(
            max_new_tokens=req.max_new_tokens or 128,
            temperature=_clip(req.temperature, 0.0, 1.0, 0.8),
            top_p=_clip(req.top_p, 0.0, 1.0, 0.8),
            repetition_penalty=_clip(req.repetition_penalty, 0.5, 2.0, 1.25),
            chunk_length=req.chunk_length or chunk_length,
            use_memory_cache="on" if req.use_memory_cache else "off",
            streaming=False,
            format="wav",
            seed=getattr(req, "seed", None),
        )

        use_ref_id = bool(getattr(req, "ref_id", None))
        use_speaker_wav = bool(getattr(req, "speaker_wav", None))
        use_reference = use_ref_id or use_speaker_wav

        pieces: list[np.ndarray] = []
        pause_list: list[int] = []

        # ---- 3) Generate per-sentence audio ----
        if use_reference:
            # Reference-mode: call adapter.inference per sentence, with same ref_id / speaker_wav
            for s in sentences:
                s = s.strip()
                if not s:
                    continue

                this_kwargs = dict(req_kwargs)
                this_kwargs["text"] = tts_prepare(s)

                serve_req = ServeTTSRequest(
                    **this_kwargs,
                    references=[],
                    reference_id=req.ref_id if use_ref_id else None,
                    reference_audio=req.speaker_wav if use_speaker_wav else None,
                )

                stream = adapter.inference(serve_req)
                items = list(stream)
                if not items:
                    continue
                out = items[-1]
                audio_seg, sr_model = _extract_audio_sr(out, eng.sample_rate)
                if audio_seg is None or len(audio_seg) == 0:
                    continue

                pieces.append(audio_seg.astype(np.float32))
                pause_list.append(pause_ms_for(s))

            if not pieces:
                raise HTTPException(status_code=500, detail="No audio generated with reference")

            work_sr = sr_model  # model's SR

        else:
            # Non-reference: use synth_one (your existing helper)
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                seg = synth_one(eng, s, target_sr)  # float32 mono at target_sr
                if seg is None or len(seg) == 0:
                    continue
                pieces.append(seg.astype(np.float32))
                pause_list.append(pause_ms_for(s))

            if not pieces:
                raise HTTPException(status_code=500, detail="No audio chunks synthesized")

            work_sr = target_sr

        # ---- 4) Join sentences with pauses (for BOTH ref and non-ref) ----
        if len(pieces) == 1:
            audio_np = pieces[0]
        else:
            full = pieces[0]
            for i in range(1, len(pieces)):
                ms = pause_list[i - 1]
                pause = np.zeros(int(work_sr * ms / 1000.0), dtype=np.float32)
                full = np.concatenate([full, pause, pieces[i]], axis=0)
            audio_np = full

        sr = work_sr

        # ---- 5) Apply tail fade-out for more natural "요~/까요?" ----
        FADE_OUT_MS = 200  # you can tune 180–220ms
        n_fade = int(sr * FADE_OUT_MS / 1000.0)
        if audio_np.size > n_fade:
            fade = np.linspace(1.0, 0.0, n_fade, dtype=np.float32)
            audio_np[-n_fade:] *= fade

        # ---- 6) Shared POST-PROCESSING ----

        # optional endpoint trimming (uses current sr)
        if getattr(req, "do_endpoint", False) or getattr(req, "endpoint", None) == "voiced":
            audio_np = _voiced_endpoint(audio_np, sr)

        # resample to 8k mono if needed
        if sr != 8000:
            audio_np, sr = _to_float_mono(audio_np, sr, 8000)
            sr = 8000

        # convert to μ-law WAV bytes
        buf = io.BytesIO()
        write_ulaw_wav(buf, audio_np, sr)
        ulaw_bytes = buf.getvalue()

        # upload to S3
        key = f"bridge/fishspeech/{uuid.uuid4().hex}.wav"
        s3.put_object(
            Bucket=TTS_BUCKET,
            Key=key,
            Body=ulaw_bytes,
            ContentType="audio/wav",
        )

        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": TTS_BUCKET, "Key": key},
            ExpiresIn=PRESIGN_EXPIRES,
        )

        logger.info(
            f"[TTS OUT] text_len={len(req.text)} out_sr={sr} key={key} ref_id={getattr(req, 'ref_id', None)}"
        )

        return {
            "bucket": TTS_BUCKET,
            "key": key,
            "url": url,
            "s3_url": url,
            "latency_ms": int((time.time() - t0) * 1000),
            "sample_rate": sr,
            "text": req.text,
        }

    except Exception as e:
        logger.exception(f"synthesize2 failed: {e}")
        return {
            "error": str(e),
            "latency_ms": int((time.time() - t0) * 1000),
        }


# def split_ko_sentences(text: str):
#     import re
#     # first split on sentence finals
#     sents = re.findall(r'.+?(?:[.!?。！？…]+|\n+|$)', text.strip(), flags=re.S)
#     sents = [s.strip() for s in sents if s.strip()]

#     refined = []
#     for s in sents:
#         if len(s) > 40 and ',' in s:
#             parts = [p.strip() for p in s.split(',') if p.strip()]
#             buf = []
#             for p in parts:
#                 if buf and (len(buf[-1]) + len(p) < 40):
#                     buf[-1] = f"{buf[-1]}, {p}"
#                 else:
#                     buf.append(p)
#             refined.extend(buf)
#         else:
#             refined.append(s)
#     return refined

def split_ko_sentences(text: str):
    import re
    sents = re.findall(r'.+?(?:[.!?。！？…]+|\n+|$)', text.strip(), flags=re.S)
    sents = [s.strip() for s in sents if s.strip()]
    return sents

def synth_one(eng, s, target_sr):
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        audio_f32, model_sr = eng.synthesize(
            text=tts_prepare(s),              # hard-stop sanitization (no ellipsis)
            temperature=0.8, top_p=0.8,
            repetition_penalty=1.25,
            max_new_tokens=128,               # cap (96–128 works best)
            chunk_length=128,
            use_memory_cache=False,
        )
    # to mono at target_sr once per sentence
    return _to_float_mono(audio_f32, model_sr, target_sr)[0]


@app.post("/synthesize_stream")
def synthesize_stream(req: SynthesizeIn, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    eng = get_engine()
    target_sr = int(req.sample_rate or 8000)

    job_id = uuid.uuid4().hex
    keys: list[str] = []

    req_obj = ServeTTSRequest(
        text=tts_prepare(req.text),
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

@app.post("/synthesize_stream_start")
def synthesize_stream_start(req: SynthesizeIn, background: BackgroundTasks,
                            authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    import uuid, time

    job_id = uuid.uuid4().hex
    target_sr = int(req.sample_rate or 8000)

    # Kick off chunked uploader (generate->upload per chunk)
#     text = normalize_text(req.text)
    background.add_task(_run_stream_job_chunked, job_id, g2p(req.text), target_sr)
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

from fastapi import Header, HTTPException


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

@app.get("/download_unified")
def download_unified(
    ref_id: Optional[str] = None,
    bucket: Optional[str] = None,
    key: Optional[str] = None,
    s3_url: Optional[str] = None,
    convert: bool = True,
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)

    # 1) local reference by ref_id (what you uploaded to /reference)
    if ref_id:
        path = REF_AUDIO.get(ref_id)
        
        # Try disk if not in memory (maybe after restart)
        if not path:
            for ext in (".wav", ".m4a", ".mp3"):
                candidate = REF_DIR / f"{ref_id}{ext}"
                if candidate.exists():
                    path = str(candidate)
                    break

        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Reference not found")
        filename = os.path.basename(path)
        return FileResponse(path, filename=filename, media_type="application/octet-stream")

    # 2) fetch from a presigned S3 URL (the one your synth returns)
    if s3_url:
        import requests
        resp = requests.get(s3_url, timeout=20)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Failed to fetch from s3_url: {resp.status_code}")
        mulaw_bytes = resp.content

        if convert:
            pcm_bytes = mulaw_wav_to_pcm16_wav_bytes(mulaw_bytes)
            return StreamingResponse(
                io.BytesIO(pcm_bytes),
                media_type="audio/wav",
                headers={"Content-Disposition": 'attachment; filename="tts_playable.wav"'},
            )
        else:
            return StreamingResponse(
                io.BytesIO(mulaw_bytes),
                media_type="audio/wav",
                headers={"Content-Disposition": 'attachment; filename="tts_raw_mulaw.wav"'},
            )

    # 3) S3 bucket + key path (only works if your server IAM can GetObject)
    if bucket and key:
        obj = s3.get_object(Bucket=bucket, Key=key)
        mulaw_bytes = obj["Body"].read()

        if convert:
            pcm_bytes = mulaw_wav_to_pcm16_wav_bytes(mulaw_bytes)
            filename = key.split("/")[-1].rsplit(".", 1)[0] + "_pcm.wav"
            return StreamingResponse(
                io.BytesIO(pcm_bytes),
                media_type="audio/wav",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        else:
            filename = key.split("/")[-1]
            return StreamingResponse(
                io.BytesIO(mulaw_bytes),
                media_type="audio/wav",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )

    raise HTTPException(status_code=400, detail="Provide either ref_id or s3_url or bucket+key")

# ensure we warm up even when mounted
try:
    _warmup_fishspeech()
except Exception as e:
    print(f"[FishSpeech] warmup (import-time) failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)