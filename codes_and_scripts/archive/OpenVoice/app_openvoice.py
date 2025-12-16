import torch, os, io, uuid, base64, re, time, tempfile, json, sys, math
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")  # PyTorch 2.x
torch.backends.cudnn.benchmark = False
from typing import Optional, Tuple, List, Dict, Any, Union
import numpy as np
import logging, asyncio, warnings
import hashlib, urllib.parse, urllib.request, tempfile
import yaml

from functools import lru_cache
import soundfile as sf
from scipy import signal
from scipy.signal import fftconvolve
import audioop
from pydantic import BaseModel
import threading, queue
from threading import Event

import subprocess, boto3
from botocore.client import Config as BotoConfig
from fastapi import Request, FastAPI, Header, HTTPException, BackgroundTasks

from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS
from dsp_spectral_gate import spectral_denoise
sys.path.append("/home/work/VALL-E/hifi-gan")
from models import Generator as HifiGAN
from env import AttrDict

warnings.filterwarnings(
    "ignore",
    message="torchaudio._backend.list_audio_backends has been deprecated"
)
warnings.filterwarnings(
    "ignore", 
    category=FutureWarning, 
    message=".*torch.nn.utils.weight_norm.*"
)

app = FastAPI()


STREAM_MAX_CHARS=100
_SAFE_MAX_CHARS = 150  # good default for Melo
_SENT_SPLIT = re.compile(r'(?<=[\.!\?？！。])\s+|[;；]\s+|[…]{1,}|$')


# ===================== env/config =====================
USE_COMPILE = os.getenv("TORCH_COMPILE", "0") == "1"
CFG_PATH        = os.getenv("FISH_CONFIG", "/home/work/VALL-E/OpenVoice/config.yaml")  # kept for compatibility; not used by OpenVoice
AWS_REGION      = os.getenv("AWS_REGION", "ap-northeast-2")
TTS_BUCKET      = os.getenv("TTS_BUCKET", "tts-bucket-250810")
KEY_PREFIX_DEF  = os.getenv("KEY_PREFIX", "sessions/demo")
PRESIGN_EXPIRES = int(os.getenv("PRESIGN_EXPIRES", "600"))
KMS_KEY_ARN     = os.getenv("KMS_KEY_ARN")
API_TOKEN       = os.getenv("API_TOKEN", "")  # optional bearer
S3_REGION       = AWS_REGION

# OpenVoice-specific and related envs
ENGINE = None
TARGET_SE = None
WARMED = False
MEL_QUEUE_MAX = 8
PCM_QUEUE_MAX = 8
speaker_wav = '/home/work/VALL-E/audio_samples/speaker.wav'
HIFIGAN_CFG  = os.getenv("HIFIGAN_CONFIG", "/home/work/VALL-E/hifigan_ckpt/config.json")
HIFIGAN_CKPT = os.getenv("HIFIGAN_CKPT",  "/home/work/VALL-E/hifigan_ckpt/g_02500000.pth")
OV_CONVERTER_DIR = os.getenv("OV_CONVERTER_DIR", "checkpoints_v2/converter")
OV_BASE_SES_DIR  = os.getenv("OV_BASE_SES_DIR",  "checkpoints_v2/base_speakers/ses")
OV_LANG          = os.getenv("OV_LANG", "KR")    # "KR" or "KO" depending on build
OV_SPEED         = float(os.getenv("OV_SPEED", "1.20"))  # default speaking rate
OV_TGT_SE_DIR = os.getenv("OV_TGT_SE_DIR", "checkpoints_v2/target_speakers/ses")
OV_DEFAULT_REF   = speaker_wav  # path to .wav
OV_WARMUP_REF   = speaker_wav  # path to .wav
OV_TGT_SE_PTH    = os.getenv("OV_TGT_SE_PTH")  # path to .pth (precomputed target SE)
os.makedirs(OV_TGT_SE_DIR, exist_ok=True)

PAD_MS = 30  # tail padding for wav writing
FRAME_S = 0.02
SAMPLE_RATE = 8000
SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_S)
CANCEL_EVENTS: Dict[str, Event] = {}
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "100"))
chunk_length = 64  # retained for API compatibility

TTS_TOP_P       = float(os.getenv("TTS_TOP_P", "0.9"))   # kept for API symmetry
TTS_TEMP        = float(os.getenv("TTS_TEMP",  "0.7"))
TTS_REP         = float(os.getenv("TTS_REP",   "1.0"))

HTTP_BASE = os.getenv("HTTP_BASE", "https://honest-trivially-buffalo.ngrok-free.app")


s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    config=BotoConfig(signature_version="s3v4", max_pool_connections=50),
)
# Optional pretty logger if available
try:
    from loguru import logger
except Exception:
    logger = logging.getLogger("openvoice-app")
    logging.basicConfig(level=logging.INFO)    
    

def load_hifigan() -> HifiGAN:
    # load JSON → AttrDict (what jik876 Generator expects)
    with open(HIFIGAN_CFG, "r", encoding="utf-8") as f:
        h = AttrDict(json.load(f))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    g = HifiGAN(h).to(device)
    sd = torch.load(HIFIGAN_CKPT, map_location="cpu")
    sd = sd.get("generator", sd)  # some checkpoints store under 'generator'
    g.load_state_dict(sd, strict=False)
    try:
        g.remove_weight_norm()
    except Exception:
        pass
    g.eval()
    if device == "cuda":
        g = g.half()
    return g



class MelAdapter(torch.nn.Module):
    def __init__(self, n_mels: int):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1, n_mels, 1))
        self.bias  = torch.nn.Parameter(torch.zeros(1, n_mels, 1))
    def forward(self, mel):
        # mel expected in log-mel domain; if your acoustic model outputs linear, take log first
        return mel * self.scale + self.bias


try:
    hfg = load_hifigan()
except Exception as e:
    hfg = None
    print(f"[HiFi-GAN] failed to load: {e}")
    
mel_adapter = MelAdapter(n_mels=80).to("cuda").half()
mel_adapter.eval()  
    
def to_logmel_if_needed(mel):
    # If OpenVoice returns linear mel (power), convert to log:
    # guard from zeros
    eps = 1e-5
    if mel.max() > 10:  # crude heuristic; adjust for your case
        mel = torch.log(mel.clamp_min(eps))
    return mel

# --- Add this helper (waveform -> log-mel) ---
def wav_to_logmel_for_vocoder(
    wav_np: np.ndarray,
    sr_in: int,
    target_sr: int = 22050,
    n_fft: int = 1024,
    hop: int = 256,
    win: int = 1024,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: float = 8000.0,
    norm: str = "slaney",   # or None
    mel_scale: str = "slaney",  # or 'htk'
):
    # ensure mono float32 in [-1,1]
    x = np.asarray(wav_np, dtype=np.float32).reshape(-1)
    mx = float(np.max(np.abs(x))) if x.size else 1.0
    if mx > 0:
        x = x / mx

    xt = torch.from_numpy(x).float().unsqueeze(0)  # [1,T]

    # resample if needed
    if sr_in != target_sr:
        try:
            import torchaudio
            xt = torchaudio.functional.resample(xt, sr_in, target_sr)
        except Exception:
            # linear fallback
            import numpy as np
            ylen = int(round(x.shape[0] * (target_sr / float(sr_in))))
            t_src = np.linspace(0.0, 1.0, num=x.shape[0], endpoint=False, dtype=np.float64)
            t_dst = np.linspace(0.0, 1.0, num=ylen,       endpoint=False, dtype=np.float64)
            y = np.interp(t_dst, t_src, x).astype(np.float32)
            xt = torch.from_numpy(y).unsqueeze(0)
            
    pre = 0.97
    x_np = xt.squeeze(0).numpy()
    x_np[1:] = x_np[1:] - pre * x_np[:-1]
    xt = torch.from_numpy(x_np).unsqueeze(0)
    # STFT
    window = torch.hann_window(win)
    spec = torch.stft(
        xt, n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, center=True, pad_mode="reflect", return_complex=True
    )  # [1, F, T]
    mag = spec.abs().clamp_min(1e-7).squeeze(0)  # [F,T]

    # Mel filter
    try:
        import torchaudio
        mel_fb = torchaudio.functional.create_fb_matrix(
            n_fft // 2 + 1, target_sr, fmin, fmax, n_mels,
            norm=(norm if norm in ("slaney",) else None),
            mel_scale=(mel_scale if mel_scale in ("slaney","htk") else "slaney"),
        )
        mel = torch.matmul(mel_fb, mag)  # [n_mels,T]
    except Exception:
        import librosa, numpy as np
        mel_fb = librosa.filters.mel(
            sr=target_sr, n_fft=n_fft, n_mels=n_mels,
            fmin=fmin, fmax=fmax,
            htk=(mel_scale=="htk"), norm=(norm if norm=="slaney" else None)
        )
        mel = torch.from_numpy(mel_fb).float().mm(mag.float())

    logmel = torch.log(mel + 1e-6).unsqueeze(0)  # [1,n_mels,T]
    m = logmel.mean(dim=(1,2), keepdim=True)
    s = logmel.std(dim=(1,2), keepdim=True).clamp_min(1e-4)
    logmel = (logmel - m) / s
    
    return logmel, target_sr


def _load_engine_once():
    global ENGINE
    if ENGINE is not None:
        return ENGINE
    # TODO: replace with your actual load function
    # e.g., ENGINE = OpenVoiceEngine.from_ckpt(os.getenv("OV_CKPT"))
    ENGINE = load_openvoice_engine()  # your existing factory
    return ENGINE

def _load_target_se_once():
    global TARGET_SE
    if TARGET_SE is not None:
        return TARGET_SE
    # Prefer a precomputed SE file if you have it
    se_pth = os.getenv("OV_TGT_SE_PTH")
    if se_pth and os.path.exists(se_pth):
        TARGET_SE = torch.load(se_pth, map_location="cpu")
    else:
        ref = os.getenv("OV_DEFAULT_REF")  # fallback to ref.wav → cache SE internally
        if ref and os.path.exists(ref):
            TARGET_SE = compute_target_se_from_ref(ref)  # your existing helper
    return TARGET_SE

def _cuda_warm_once():
    import torch
    if torch.cuda.is_available():
        _ = torch.randn(1, device="cuda")
        torch.cuda.synchronize()

def _tts_warm_inference():
    """Run a tiny synth once to trigger model + converter + vocoder init."""
    try:
        eng = get_engine()  # <-- your existing factory
        # Prefer an explicit ref if available
        ref = speaker_wav
        ref = ref if (ref and os.path.isfile(ref)) else None
        with torch.inference_mode():
            _ = eng.synthesize("안녕하세요.", speaker_wav=ref)
    except Exception as e:
        logger.warning(f"warmup failed: {e}")

def _clean_text(txt: str) -> str:
    # Normalize ellipses and weird punctuation that can explode the flow
    txt = txt.replace("…", "...")
    txt = re.sub(r"[“”]", '"', txt)
    txt = re.sub(r"[‘’]", "'", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def _split_safe(txt: str, max_chars: int = _SAFE_MAX_CHARS):
    parts = []
    for sent in filter(None, _SENT_SPLIT.split(txt)):
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) <= max_chars:
            parts.append(sent)
        else:
            # hard wrap long sentences
            for i in range(0, len(sent), max_chars):
                parts.append(sent[i:i+max_chars])
    return parts


def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _embedding_path_for_ref(local_wav: str) -> str:
    return os.path.join(OV_TGT_SE_DIR, _sha1_file(local_wav) + ".pth")

def _materialize_ref(ref: str) -> str:
    # Local path -> as-is; S3/HTTP -> download to temp
    if ref and os.path.isfile(ref):
        return ref
    if not ref:
        return ""
    if ref.startswith("s3://"):
        u = urllib.parse.urlparse(ref)
        bkt, key = u.netloc, u.path.lstrip("/")
        fd, tmp = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        s3.download_file(bkt, key, tmp)
        return tmp
    if ref.startswith("http://") or ref.startswith("https://"):
        fd, tmp = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        urllib.request.urlretrieve(ref, tmp)
        return tmp
    return ref


def maybe_compile(m):
    return torch.compile(m, mode="reduce-overhead", fullgraph=False) if USE_COMPILE else m


# ---- chunker helpers (unchanged) ----
def _split_into_chunks(text: str, max_chars: int = 220):
    sents = re.split(r'(?<=[.!?。！？])\s+', text.strip())
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) + (1 if cur else 0) <= max_chars:
            cur = (cur + " " + s).strip() if cur else s
        else:
            if cur: chunks.append(cur)
            if len(s) <= max_chars:
                cur = s
            else:
                for i in range(0, len(s), max_chars):
                    chunks.append(s[i:i+max_chars])
                cur = ""
    if cur: chunks.append(cur)
    return chunks

#### Added ####
import soundfile as sf, os, uuid
#### Added ####

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
def _chunk_text(text: str, max_len: int = int(os.getenv("STREAM_MAX_CHARS", "110"))) -> list[str]:
# def _chunk_text(text: str, max_len: int = 140) -> list[str]:
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
            chunks.append(buf.strip()); buf = s
    if buf: chunks.append(buf.strip())
    return chunks or [text.strip()]



# ===================== S3 + audio helpers =====================

# put near your audio utils
def apply_emotion_preset(wav: np.ndarray, sr: int, emotion: Optional[str]) -> np.ndarray:
    if not emotion or emotion == "neutral":
        return wav

    try:
        import librosa
    except ImportError:
        # Fallback: simple speed/gain/tilt via numpy only (no pitch)
        if emotion == "happy":
            # 6% faster
            idx = np.round(np.arange(0, len(wav), 1/1.06)).astype(int)
            wav = wav[np.clip(idx, 0, len(wav)-1)]
            wav = wav * 1.15
            wav = wav.astype(np.float32)
        elif emotion == "sad":
            idx = np.round(np.arange(0, len(wav), 1/0.92)).astype(int)
            wav = wav[np.clip(idx, 0, len(wav)-1)]
            wav = wav * 0.9
        elif emotion == "angry":
            idx = np.round(np.arange(0, len(wav), 1/1.10)).astype(int)
            wav = wav[np.clip(idx, 0, len(wav)-1)]
            wav = np.tanh(1.2 * wav)  # soft saturation
        return wav.astype(np.float32)

    # With librosa: do proper time-stretch + small pitch shift (fast)
    if emotion == "happy":
        y = librosa.effects.time_stretch(wav, 1.06)
        y = librosa.effects.pitch_shift(y, sr, n_steps=+2)      # ~ +2 semitones
        y = y * 1.15
        return y.astype(np.float32)
    elif emotion == "sad":
        y = librosa.effects.time_stretch(wav, 0.92)
        y = librosa.effects.pitch_shift(y, sr, n_steps=-1)      # ~ -1 semitone
        # gentle low-shelf feel: tilt EQ (very rough)
        y = (0.95 * y + 0.05 * np.convolve(y, np.ones(64)/64, mode="same")).astype(np.float32)
        return y
    elif emotion == "angry":
        y = librosa.effects.time_stretch(wav, 1.10)
        y = librosa.effects.pitch_shift(y, sr, n_steps=+1)
        # presence/brightness + soft clip
        y = np.tanh(1.25 * y).astype(np.float32)
        return y
    return wav


def pre_emphasis(x: np.ndarray, a: float = 0.85) -> np.ndarray:
    if a <= 0: return x
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - a * x[:-1]
    return y

def biquad_highpass(sr, f=120, q=0.707):
    w0 = 2*np.pi*f/sr
    alpha = np.sin(w0)/(2*q)
    cosw0 = np.cos(w0)
    b0 =  (1+cosw0)/2
    b1 = -(1+cosw0)
    b2 =  (1+cosw0)/2
    a0 =   1+alpha
    a1 =  -2*cosw0
    a2 =   1-alpha
    return np.array([b0/a0,b1/a0,b2/a0]), np.array([1,a1/a0,a2/a0])

def peaking_eq(sr, f=3200, q=0.7, gain_db=4.0):
    A = 10**(gain_db/40.0)
    w0 = 2*np.pi*f/sr
    alpha = np.sin(w0)/(2*q)
    cosw0 = np.cos(w0)
    b0 =   1 + alpha*A
    b1 =  -2*cosw0
    b2 =   1 - alpha*A
    a0 =   1 + alpha/A
    a1 =  -2*cosw0
    a2 =   1 - alpha/A
    return np.array([b0/a0,b1/a0,b2/a0]), np.array([1,a1/a0,a2/a0])

def apply_biquad(x, b, a):
    return signal.lfilter(b, a, x).astype(np.float32)

def soft_compand(x, gain_db=6.0):
    # gentle compand-ish: apply make-up gain then soft clip
    g = 10**(gain_db/20.0)
    y = np.tanh(g * x)
    return (y / max(1.0, np.max(np.abs(y))+1e-6)).astype(np.float32)


def encode_ulaw_wav(part_f32_mono: np.ndarray, sr: int) -> bytes:
    pad = np.zeros(int(sr * 0.08), dtype=np.float32)
    data = np.concatenate([part_f32_mono.astype(np.float32), pad])
    buf = io.BytesIO()
    sf.write(buf, data, sr, subtype="ULAW", format="WAV")
    buf.seek(0)
    info = sf.info(buf)
    if info.subtype.upper() != "ULAW" or info.samplerate != sr or info.channels != 1:
        raise RuntimeError(f"ULAW encode failed: subtype={info.subtype}, sr={info.samplerate}, ch={info.channels}")
    return buf.getvalue()

def write_ulaw_wav(buf, samples_f32, sr):
    pad = np.zeros(int(sr * (PAD_MS/1000.0)), dtype=np.float32)
    sf.write(buf, np.concatenate([samples_f32, pad]), sr, subtype="ULAW", format="WAV")

def lowpass_aa(x: np.ndarray, sr: int, fc: float = 3600.0, order: int = 6) -> np.ndarray:
    """Anti-alias low-pass using Butterworth, zero-phase filter."""
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * sr
    Wn = min(max(fc / nyq, 1e-4), 0.999)
    b, a = butter(order, Wn, btype="low", analog=False, output="ba")
    y = filtfilt(b, a, x).astype(np.float32)
    return y
    
def _to_float_mono(audio: Union[np.ndarray, torch.Tensor], sr_src: int, sr_dst: int) -> tuple[np.ndarray, int]:
    # tensor -> numpy
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().float().numpy()
    audio = np.asarray(audio, dtype=np.float32)

    # to mono
    if audio.ndim == 2:
        if 2 in audio.shape:
            audio = audio.mean(axis=0 if audio.shape[0] == 2 else 1)
        else:
            audio = audio.squeeze()

    # resample (use linear to avoid heavy deps)
    if sr_src != sr_dst:
        src_t = np.linspace(0.0, 1.0, num=max(1, audio.size), endpoint=False, dtype=np.float64)
        dst_len = int(round(max(1, audio.size) * (sr_dst / float(sr_src))))
        dst_t = np.linspace(0.0, 1.0, num=max(1, dst_len), endpoint=False, dtype=np.float64)
        audio = np.interp(dst_t, src_t, audio).astype(np.float32)

    maxv = float(np.max(np.abs(audio))) if audio.size else 1.0
    if maxv < 1e-6: maxv = 1.0
    audio = np.clip(audio / maxv, -1.0, 1.0).astype(np.float32)
    return audio, sr_dst

def _apply_tel_profile(x: np.ndarray, sr: int, profile: str) -> np.ndarray:
    profile = (profile or "bright").lower()
    if profile == "flat":
        # no EQ/companding; normalize a touch for safety
        peak = float(np.max(np.abs(x))) + 1e-6
        return (0.9 * x / peak).astype(np.float32)

    # shared: gentle HPF to clear mud
    b,a = biquad_highpass(sr, f=120, q=0.707)
    y = apply_biquad(x, b, a)

    if profile == "bright":
        # stronger presence, a bit more compand
        b,a = peaking_eq(sr, f=3200, q=0.8, gain_db=3.0)
        y = apply_biquad(y, b, a)
        b,a = peaking_eq(sr, f=350, q=1.1, gain_db=-2.5)
        y = apply_biquad(y, b, a)
        y = soft_compand(y, gain_db=2.5)
    else:
        # "tame" (default): subtle, safe
        b,a = peaking_eq(sr, f=3100, q=0.9, gain_db=2.0)
        y = apply_biquad(y, b, a)
        b,a = peaking_eq(sr, f=350, q=1.1, gain_db=-2.0)
        y = apply_biquad(y, b, a)
        y = soft_compand(y, gain_db=1.5)

    # final safety headroom
    peak = float(np.max(np.abs(y))) + 1e-6
    return (0.9 * y / peak).astype(np.float32)



_PAUSE_TOKENS = {
    "soft":   " …",
    "medium": " — …",
    "strong": " 　…  …",   # ideographic space + double ellipsis
}
_CLAUSE_TOKENS = {
    "soft":   " …",
    "medium": " …",
    "strong": " — …",
}

_SENT_END_RE   = re.compile(r"\s*([.!?]+)\s*")
_CLAUSE_END_RE = re.compile(r"\s*([,;:])\s*")
_TILDES_RE     = re.compile(r"~+")  # one or more tildes

def normalize_text(
    text: str,
    lang: str = "ko",
    pause_strength: str = "strong",
    tilde_behavior: str = "drag",  # "drag" or "stop"
) -> str:
    """Normalize symbols and inject pause/drag markers for OpenVoice-style TTS."""
    strength = pause_strength if pause_strength in _PAUSE_TOKENS else "medium"
    sent_pause   = _PAUSE_TOKENS[strength]
    clause_pause = _CLAUSE_TOKENS[strength]

    # 1) Basic symbol expansions
    repl = {
        "%": " 퍼센트" if lang == "ko" else " percent",
        "&": " 앤드" if lang == "ko" else " and",
        "$": " 달러" if lang == "ko" else " dollars",
    }
    for k, v in repl.items():
        text = text.replace(k, v)

    # 2) Normalize whitespace first
    text = re.sub(r"\s+", " ", text).strip()

    # 3) Tilde behavior
    #    - drag: convert ~~~ to progressively longer “draggy” sequences
    #    - stop: convert any ~~~ to a period + strong pause
    def _tilde_sub(m: re.Match) -> str:
        n = len(m.group(0))
        if tilde_behavior == "stop":
            return f".{sent_pause} "
        # drag behavior: scale by length of run
        # n=1 → " — …"
        # n=2 → " — … …"
        # n>=3 → " — … … …"
        ellipses = " …" * min(1 + (n - 1), 3)
        return f" -{ellipses} "

    text = _TILDES_RE.sub(_tilde_sub, text)

    # 4) Stronger pauses after sentence enders / clauses
    text = _SENT_END_RE.sub(lambda m: f"{m.group(1)}{sent_pause} ", text)
    text = _CLAUSE_END_RE.sub(lambda m: f"{m.group(1)}{clause_pause} ", text)

    # 5) Compress any runaway dots into a single ellipsis
    text = re.sub(r"\.{4,}", "…", text)

    return text.strip()


def fir_lowpass_linear(x: np.ndarray, sr: int, fc: float = 3600.0, taps: int = 101) -> np.ndarray:
    """Linear-phase FIR LPF (windowed-sinc). taps: odd number (e.g., 101/151)."""
    nyq = 0.5 * sr
    wc = fc / nyq
    N = taps
    n = np.arange(N) - (N - 1) / 2.0
    # sinc kernel
    h = np.sinc(wc * n)
    # Hamming window
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (np.arange(N)) / (N - 1))
    h = h * w
    h /= np.sum(h) + 1e-12
    y = fftconvolve(x.astype(np.float32), h.astype(np.float32), mode="same")
    return y.astype(np.float32)


def pre_emphasis(x: np.ndarray, a: float = 0.97) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    y = np.empty_like(x, dtype=np.float32)
    y[0] = x[0]
    y[1:] = x[1:] - a * x[:-1]
    return y

def _put_ulaw_wav(buf: io.BytesIO, key: str):
    extra = {"ContentType": "audio/wav", "CacheControl": "no-cache"}
    if KMS_KEY_ARN:
        extra.update({"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": KMS_KEY_ARN})
    s3.upload_fileobj(buf, TTS_BUCKET, key, ExtraArgs=extra)

def _presign(key: str, expires: int = 3600) -> str:
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": TTS_BUCKET, "Key": key},
        ExpiresIn=expires,
    )

def _read_yaml(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def _ensure_mono(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 2:
        if a.shape[1] == 1:
            a = a[:, 0]
        else:
            a = np.mean(a, axis=1)
    return a

# ===================== OpenVoice adapter (kept class/function names) =====================
def _speaker_key_from_path(path: str) -> Optional[str]:
    base = os.path.basename(path)
    if base.endswith(".pth"):
        return base[:-4]
    return None

def _find_base_ses() -> Dict[str, str]:
    if not os.path.isdir(OV_BASE_SES_DIR):
        return {}
    paths = [os.path.join(OV_BASE_SES_DIR, fn) for fn in os.listdir(OV_BASE_SES_DIR) if fn.endswith(".pth")]
    out: Dict[str, str] = {}
    for p in paths:
        key = _speaker_key_from_path(p)
        if key:
            out[key] = p
    return out

@lru_cache(maxsize=32)
def _target_se_from_ref(path: str):
    # simple cache key by path; extraction happens in adapter to access converter
    return path

class FishEngineAdapter:
    # Keeps the original class name, but internally uses OpenVoice:
    #   - Melo TTS (text -> src wav)
    #   - ToneColorConverter (src wav + base SE -> target SE from ref -> converted wav)
    # .synthesize(...) returns (float32 mono np.ndarray, sample_rate)
    def __init__(self, cfg: dict):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.language = cfg.get("openvoice_language", OV_LANG)
        self.speed = float(cfg.get("openvoice_speed", OV_SPEED))

        # Tone color converter
        conv_dir = cfg.get("openvoice_converter_dir", OV_CONVERTER_DIR)
        conv_cfg = os.path.join(conv_dir, "config.json")
        conv_ckpt = os.path.join(conv_dir, "checkpoint.pth")
        self.converter = ToneColorConverter(conv_cfg, device=self.device)
        self.converter.load_ckpt(conv_ckpt)

        # TTS
        self.tts = TTS(language=self.language, device=self.device)
        try:
            self.sample_rate = int(getattr(self.tts.hps.data, "sampling_rate", 22050))
        except Exception:
            self.sample_rate = 22050

        # base speaker embeddings
        self.base_ses = _find_base_ses()
        if not self.base_ses:
            logger.warning(f"No base speaker embeddings found in {OV_BASE_SES_DIR}")
        self.default_base_key = next(iter(self.base_ses.keys()), None)

        # Melo speakers map
        self.spk2id = self.tts.hps.data.spk2id
        
        self.default_target_se = None
        # 1) Prefer explicit .pth if provided
        if OV_TGT_SE_PTH and os.path.isfile(OV_TGT_SE_PTH):
            try:
                self.default_target_se = torch.load(OV_TGT_SE_PTH, map_location=self.device)
                logger.info(f"[OpenVoice] Loaded default target SE from {OV_TGT_SE_PTH}")
            except Exception as e:
                logger.warning(f"[OpenVoice] Failed to load OV_TGT_SE_PTH: {e}")

        # 2) Otherwise compute from a default reference WAV (and cache to disk)
        if self.default_target_se is None and OV_DEFAULT_REF:
            local_ref = _materialize_ref(OV_DEFAULT_REF)
            if os.path.isfile(local_ref):
                try:
                    cache_pth = _embedding_path_for_ref(local_ref)
                    if os.path.isfile(cache_pth):
                        self.default_target_se = torch.load(cache_pth, map_location=self.device)
                        logger.info(f"[OpenVoice] Loaded default target SE from cache: {cache_pth}")
                    else:
                        self.default_target_se, _ = se_extractor.get_se(local_ref, self.converter, vad=True)
                        try:
                            torch.save(self.default_target_se.cpu(), cache_pth)
                            logger.info(f"[OpenVoice] Cached default target SE -> {cache_pth}")
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(f"[OpenVoice] Failed to compute default target SE: {e}")


        logger.info(f"[OpenVoice] device={self.device} lang={self.language} speed={self.speed} sr={self.sample_rate}")
        logger.info(f"[OpenVoice] base_ses={list(self.base_ses.keys())} default={self.default_base_key}")

    def _get_source_se(self, speaker_key: Optional[str] = None):
        key = speaker_key or self.default_base_key
        if not key or key not in self.base_ses:
            raise RuntimeError(f"No valid base speaker key. Available: {list(self.base_ses.keys())}")
        return torch.load(self.base_ses[key], map_location=self.device)

    def _get_target_se(self, ref: Optional[str]):
        # a) No ref provided -> use preloaded default if available
        if not ref:
            if self.default_target_se is not None:
                return self.default_target_se
            raise RuntimeError("No speaker reference available (set OV_DEFAULT_REF or OV_TGT_SE_PTH).")

        # b) Accept cached embedding directly
        if ref.endswith(".pth") and os.path.isfile(ref):
            return torch.load(ref, map_location=self.device)

        # c) Treat as WAV (local or URL) with on-disk cache
        local = _materialize_ref(ref)
        if not os.path.isfile(local):
            raise RuntimeError(f"speaker_wav path not found: {ref}")
        se_pth = _embedding_path_for_ref(local)
        if os.path.isfile(se_pth):
            return torch.load(se_pth, map_location=self.device)

        tgt_se, _ = se_extractor.get_se(local, self.converter, vad=True)
        try:
            torch.save(tgt_se.cpu(), se_pth)
            logger.info(f"[OpenVoice] Cached target SE -> {se_pth}")
        except Exception:
            pass
        return tgt_se


    def synthesize(
        self,
        text: str,
        speaker_wav: Optional[str] = speaker_wav,
        sr: Optional[int] = None,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        chunk_length: Optional[int] = None,
        use_memory_cache: Optional[bool] = None,
        use_ext_vocoder: Optional[bool] = None, 
    ) -> Tuple[np.ndarray, int]:
        if not speaker_wav:
            raise RuntimeError("OpenVoice requires 'speaker_wav' (reference voice) for tone conversion.")
#         print(text)
        # Solution 1a: Clean and validate input text
        text = _clean_text(text)
        if not text or len(text.strip()) == 0:
            # Return silence for empty text
            silence = np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
            return silence, self.sample_rate

        # Solution 1b: Limit text length to avoid numerical issues
        if len(text) > 500:
            text = text[:500] + "."
            logger.warning(f"[OpenVoice] Text truncated to 500 chars to avoid numerical issues")

        try:
            spk_id = next(iter(self.spk2id.values()))
        except Exception:
            spk_id = 0

        with tempfile.TemporaryDirectory() as td:
            src_wav = os.path.join(td, "src.wav")
            out_wav = os.path.join(td, "out.wav")

            # Solution 1c: Add retry logic with different parameters
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # TTS -> src wav with numerical stability improvements
                    with torch.inference_mode():
                        # Solution 1d: Use more conservative inference settings
                        if torch.cuda.is_available():
                            # Force full precision for stability
                            with torch.amp.autocast('cuda', enabled=False):
                                self.tts.tts_to_file(
                                    text, 
                                    spk_id, 
                                    src_wav, 
                                    speed=min(self.speed, OV_SPEED),  # Limit speed to avoid instability
                                    pbar=False,
                                    format="wav"
                                )
                        else:
                            self.tts.tts_to_file(
                                text, 
                                spk_id, 
                                src_wav, 
                                speed=min(self.speed, OV_SPEED),
                                pbar=False,
                                format="wav"
                            )
                    break  # Success, exit retry loop

                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"[OpenVoice] TTS attempt {attempt + 1} failed: {error_msg}")

                    if "discriminant" in error_msg or "assert" in error_msg:
                        if attempt < max_retries - 1:
                            # Solution 1e: Try with fallback parameters
                            logger.info(f"[OpenVoice] Retrying with fallback settings...")

                            # Reduce text complexity for retry
                            if attempt == 1:
                                # Second attempt: simplify text
                                text = re.sub(r'[^\w\s가-힣.,!?]', '', text)
                                logger.info(f"[OpenVoice] Simplified text: {text[:50]}...")
                            elif attempt == 2:
                                # Third attempt: use very basic text
                                text = "안녕하세요. 음성을 생성합니다."
                                logger.info(f"[OpenVoice] Using fallback text")

                            time.sleep(0.1)  # Brief delay before retry
                            continue
                        else:
                            # Final attempt failed, generate silence
                            logger.error(f"[OpenVoice] All TTS attempts failed, returning silence")
                            silence = np.zeros(int(self.sample_rate * 1.0), dtype=np.float32)
                            return silence, self.sample_rate
                    else:
                        # Non-discriminant error, re-raise immediately
                        raise

            # Continue with tone conversion...
            source_se = self._get_source_se()
            target_se = None
            try:
                target_se = self._get_target_se(speaker_wav)
            except Exception as e:
                logger.warning(f"[OpenVoice] No target SE; falling back to base voice. Reason: {e}")

            # If we have a target SE -> do conversion; else serve base TTS
            if target_se is not None:
                try:
                    self.converter.convert(
                        audio_src_path=src_wav,
                        src_se=source_se,
                        tgt_se=target_se,
                        output_path=out_wav,
                        message="@MyShell",
                    )
                    audio, sr2 = sf.read(out_wav, dtype="float32")
                except Exception as conv_error:
                    logger.warning(f"[OpenVoice] Tone conversion failed, using base TTS: {conv_error}")
                    audio, sr2 = sf.read(src_wav, dtype="float32")
            else:
                audio, sr2 = sf.read(src_wav, dtype="float32")

            audio = _ensure_mono(audio).astype(np.float32)
            
            # Optional: external vocoder A/B (use global hfg)
            flag = use_ext_vocoder if use_ext_vocoder is not None else (os.getenv("OV_USE_VOCODER", "0") == "1")
            if flag and hfg is not None:
                try:
                    # Build log-mel from the current wideband waveform
                    mel, mel_sr = wav_to_logmel_for_vocoder(
                        wav_np=audio, sr_in=int(sr2),
                        target_sr=int(os.getenv("MEL_SR", "22050")),
                        n_fft=int(os.getenv("MEL_NFFT", "1024")),
                        hop=int(os.getenv("MEL_HOP", "256")),
                        win=int(os.getenv("MEL_WIN", "1024")),
                        n_mels=int(os.getenv("MEL_NMELS", "80")),
                        fmin=float(os.getenv("MEL_FMIN", "0")),
                        fmax=float(os.getenv("MEL_FMAX", "8000")),
                        norm=os.getenv("MEL_NORM", "slaney").lower(),
                        mel_scale=("htk" if os.getenv("MEL_NORM","slaney").lower()=="htk" else "slaney"),
                    )

                    # (optional) mel adapter if you created one
                    try:
                        mel = mel.to(next(hfg.parameters()).device)
                        if next(hfg.parameters()).dtype == torch.half:
                            mel = mel.half()
                        mel = mel_adapter(mel)  # safe to leave; remove if you didn’t define it
                    except Exception:
                        pass

                    with torch.inference_mode():
                        y = hfg(mel).squeeze(0)  # [T]
                    y = y.float().cpu().numpy()

                    # normalize with a little headroom
                    peak = float(np.max(np.abs(y))) + 1e-6
                    audio = (0.95 * y / peak).astype(np.float32)
                    sr2 = mel_sr
                except Exception as e:
                    logger.warning(f"[Vocoder] external path failed, using original waveform: {e}")

        return audio, int(sr2)
    

def wiener_denoise(x: np.ndarray, mysize: int = 29) -> np.ndarray:
    # SciPy’s Wiener works 1D; choose modest window to avoid smearing
    try:
        from scipy.signal import wiener
        y = wiener(x.astype(np.float32), mysize=mysize)
        return y.astype(np.float32)
    except Exception:
        return x.astype(np.float32)

def rms(x: np.ndarray) -> float:
    x = x.astype(np.float32)
    return float(np.sqrt(np.mean(x*x) + 1e-12))

def noise_gate(x: np.ndarray, thresh_db: float = -42.0, ratio: float = 4.0) -> np.ndarray:
    """
    Simple downward expander: attenuate frames below threshold.
    thresh_db is absolute (rel. to full-scale), good for TTS.
    """
    sr = 24000  # this runs before downsample; keep at wideband stage
    hop = 240    # 10 ms @ 24k
    win = 480    # 20 ms
    th = 10**(thresh_db/20.0)
    y = x.copy().astype(np.float32)

    for i in range(0, len(x), hop):
        j = min(i+win, len(x))
        frame = x[i:j]
        lvl = max(np.max(np.abs(frame)), 1e-6)
        if lvl < th:
            # attenuation grows as we go below threshold
            att = (lvl / th)**(ratio-1.0)
            y[i:j] *= att
    return y

@lru_cache(maxsize=1)
def get_engine() -> FishEngineAdapter:
    cfg = _read_yaml(CFG_PATH)
    eng = FishEngineAdapter(cfg or {})
    return eng

# ===================== streaming helpers (sentence-chunked using synthesize) =====================
def _synthesize_chunk_to_key(chunk: str, target_sr: int, job_id: str, idx: int, speaker_wav: Optional[str] = speaker_wav) -> str:
    eng = get_engine()
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=False):
        audio_f32, model_sr = eng.synthesize(text=normalize_text(chunk, lang="ko"), speaker_wav=speaker_wav, sr=target_sr)
    float_mono, _ = _to_float_mono(audio_f32, model_sr, target_sr)
    buf = io.BytesIO()
    write_ulaw_wav(buf, float_mono, target_sr)
    buf.seek(0)
    key = f"connect_chunk/{job_id}/part{idx}.wav"
    _put_ulaw_wav(buf, key)
    logger.info(f"[CHUNK] uploaded {key} dur≈{len(float_mono)/target_sr:.2f}s len={len(chunk)}")
    return key

def _split_into_sentences(text: str):
    if not text:
        return []
    pieces = re.findall(r'.+?(?:[.!?。！？…]+|\n+|$)', text.strip(), flags=re.S)
    return [p.strip() for p in pieces if p.strip()]

def _run_stream_job_chunked(job_id: str, text: str, target_sr: int, start_idx: int = 0, speaker_wav: Optional[str] = speaker_wav):
    sentences = _split_into_sentences(text)
    for idx, sent in enumerate(sentences[start_idx:], start=start_idx):
        if _is_cancelled(job_id):
            tail = np.zeros(int(target_sr * 0.2), dtype=np.float32)
            buf = io.BytesIO(); write_ulaw_wav(buf, tail, target_sr); buf.seek(0)
            _put_ulaw_wav(buf, f"connect_chunk/{job_id}/final.wav")
            logger.info(f"[CHUNK] job connect_chunk/{job_id} cancelled; uploaded final.wav")
            return
        try:
            _synthesize_chunk_to_key(sent, target_sr, job_id, idx, speaker_wav=speaker_wav)
            logger.info(f"[CHUNK] uploaded connect_chunk/{job_id}/part{idx}.wav")
        except Exception:
            logger.exception(f"[CHUNK] sentence {idx} failed; skipping")

    tail = np.zeros(int(target_sr * 0.2), dtype=np.float32)
    buf = io.BytesIO(); write_ulaw_wav(buf, tail, target_sr); buf.seek(0)
    _put_ulaw_wav(buf, f"connect_chunk/{job_id}/final.wav")
    logger.info(f"[CHUNK] uploaded connect_chunk/{job_id}/final.wav")

def _run_stream_job_from_idx(job_id: str, text: str, target_sr: int, start_idx: int, speaker_wav: Optional[str] = speaker_wav):
    chunks = _chunk_text(text, max_len=160)
    for idx, chunk in enumerate(chunks[start_idx:], start=start_idx):
        try:
            _synthesize_chunk_to_key(chunk, target_sr, job_id, idx, speaker_wav=speaker_wav)
            logger.info(f"[STREAM] uploaded connect_chunk/{job_id}/part{idx}.wav")
        except Exception:
            logger.exception(f"chunk {idx} failed; skipping")

    tail = np.zeros(int(target_sr * 0.2), dtype=np.float32)
    buf = io.BytesIO(); write_ulaw_wav(buf, tail, target_sr); buf.seek(0)
    _put_ulaw_wav(buf, f"connect_chunk/{job_id}/final.wav")
    logger.info(f"[STREAM] uploaded connect_chunk/{job_id}/final.wav")

# Warmup using synthesize (kept API)
def _warm_once():
    eng = get_engine()
    try:
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=False):
            ref = os.getenv("OV_WARMUP_REF") or os.getenv("OPENVOICE_WARMUP_REF") or ""
            if ref and os.path.isfile(ref):
                eng.synthesize("짧은 테스트입니다.", speaker_wav=ref)
            else:
                logger.warning("OpenVoice warmup skipped (set OV_WARMUP_REF to a reference wav).")
    except Exception as e:
        logger.warning(f"warmup failed: {e}")

# ===================== FastAPI =====================
app = FastAPI()

class SynthesizeIn(BaseModel):
    text: str
    key_prefix: Optional[str] = None
    sample_rate: Optional[int] = None
    speaker_wav: Optional[str] = speaker_wav  # path to ref wav (required for OpenVoice)
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    seed: Optional[int] = 42
    repetition_penalty: Optional[float] = 1.1
    max_new_tokens: Optional[int] = 0
    chunk_length: Optional[int] = 300
    use_memory_cache: Optional[bool] = True
    encode: Optional[str] = "mulaw"
    tel_enhance: Optional[bool] = False
    emotion: Optional[str] = None
    tel_profile: Optional[str] = "flat"
    use_ext_vocoder: Optional[bool] = None

def _check_auth(authorization: Optional[str]):
    if API_TOKEN and authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="unauthorized")

@app.on_event("startup")
def _warm_once():
    global WARMED
    if WARMED:
        return
    _cuda_warm_once()
    _tts_warm_inference()
    WARMED = True

# @app.on_event("startup")
# async def _warm_engine():
#     logger.info("Loading OpenVoice engine at startup…")
    
#     # Solution 4a: Set environment variables for numerical stability
#     os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For better error reporting
#     os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
#     # Solution 4b: Set PyTorch settings for stability
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False  # Disable benchmark for stability
#     torch.set_float32_matmul_precision("high")  # Keep high precision
    
#     loop = asyncio.get_running_loop()
#     await loop.run_in_executor(None, _warm_once)
#     logger.info("Engine warmup done.")

@app.get("/healthz")
def healthz():
    return {"ok": True, "warmed": bool(WARMED)}

@app.post("/synthesize/warmup")
def synth_warmup():
    _warm_once()
    return {"ok": True, "warmed": bool(WARMED)}

@app.on_event("startup")
def _warm_once():
    global WARMED
    if WARMED:
        return
    _load_engine_once()
    _load_target_se_once()
    _cuda_warm_once()
    _tts_warm_inference()
    WARMED = True
    
# Solution 5: Alternative fallback TTS method (if you have other TTS models available)
def _synthesize_with_fallback(self, text: str, speaker_wav: str = None):
    """Synthesize with automatic fallback to simpler methods"""
    
    # Try primary method
    try:
        return self.synthesize(text, speaker_wav=speaker_wav)
    except Exception as e:
        logger.warning(f"[OpenVoice] Primary synthesis failed: {e}")
        
        # Fallback 1: Try with very simple text
        try:
            simple_text = "안녕하세요."
            return self.synthesize(simple_text, speaker_wav=speaker_wav)
        except Exception as e2:
            logger.error(f"[OpenVoice] Fallback synthesis also failed: {e2}")
            
            # Fallback 2: Return pre-generated silence
            silence = np.zeros(int(self.sample_rate * 2.0), dtype=np.float32)
            return silence, self.sample_rate

@app.get("/health")
def health():
    eng = get_engine()
    return {
        "ok": True,
        "bucket": TTS_BUCKET,
        "region": AWS_REGION,
        "device": eng.device,
        "ov_lang": eng.language,
        "ov_speed": eng.speed,
        "base_speakers": list(eng.base_ses.keys()),
    }

@app.post("/synthesize")
def synthesize(req: SynthesizeIn, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
#     print(req.text)
    # Validate and clean text
    cleaned_text = _clean_text(req.text)
#     print(cleaned_text)
    if not cleaned_text:
        raise HTTPException(400, "Invalid or empty text input")
    
    # Limit text length to prevent numerical instability
    if len(cleaned_text) > 500:
        cleaned_text = cleaned_text[:500] + "."
    
    req.text = cleaned_text
    # Only error if we have neither a request ref nor a preloaded default
    if not req.speaker_wav and get_engine().default_target_se is None:
        raise HTTPException(400, "No speaker reference available. Provide speaker_wav or set OV_DEFAULT_REF / OV_TGT_SE_PTH.")


    t0 = time.time()
    target_sr = int(req.sample_rate or 8000)

    eng = get_engine()
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=False):
        audio_f32, model_sr = eng.synthesize(
            text=normalize_text(req.text, lang="ko"),
            speaker_wav=speaker_wav,
            sr=req.sample_rate,
            temperature=req.temperature,
            top_p=req.top_p,
            seed=req.seed,
            repetition_penalty=req.repetition_penalty,
            max_new_tokens=req.max_new_tokens,
            chunk_length=req.chunk_length,
            use_memory_cache=req.use_memory_cache,
            use_ext_vocoder=req.use_ext_vocoder,
        )
    
    wide_sr = int(req.sample_rate or 24000)
    float_mono, _ = _to_float_mono(audio_f32, model_sr, wide_sr)

    # apply optional emotion preset
    float_mono = apply_emotion_preset(float_mono, wide_sr, req.emotion)
#     float_mono, _ = _to_float_mono(audio_f32, model_sr, target_sr)
    
    #### Added ####
    # after you have float_mono and target_sr
    if os.getenv("LOCAL_ONLY", "0") == "1":
        os.makedirs("out_test", exist_ok=True)
        out_ref = f"out_test/ref_{uuid.uuid4().hex}_24k.wav"
        sf.write(out_ref, float_mono, 24000)  # wideband reference
        return {"file": out_ref, "sample_rate": 24000}
    #### Added ####

    buf = io.BytesIO()
    encode = (req.encode or "mulaw").lower()

    if encode == "pcm":
        target_sr = int(req.sample_rate or 24000)
    else:
        target_sr = 8000

    float_mono, _ = _to_float_mono(audio_f32, model_sr, 24000)  # keep 24k for DSP
#     if np.var(float_mono) > 1e-8:
#         try:
#             float_mono = wiener_denoise(float_mono, mysize=29)
#         except Exception as e:
#             logger.warning(f"Wiener denoise skipped: {e}")
    float_mono = spectral_denoise(
        float_mono, sr=wide_sr,
        noise_ms=60,
        reduce_db=3.0,
        floor_db=-15.0,
        time_smooth=0,
        freq_smooth=1
    )
#     float_mono = noise_gate(float_mono, thresh_db=-42.0, ratio=3.0)
#     float_mono = noise_gate(float_mono, thresh_db=-48.0, ratio=2.5)
    
    if encode == "mulaw":
        # respect tel_enhance if you want an on/off; otherwise always apply with profile
        
        if getattr(req, "tel_enhance", True):
            float_mono = _apply_tel_profile(float_mono, wide_sr, req.tel_profile)
#         float_mono = lowpass_aa(float_mono, sr=24000, fc=3800.0, order=4)
        float_mono = fir_lowpass_linear(float_mono, sr=24000, fc=3650.0, taps=121)
#         float_mono = pre_emphasis(float_mono, a=0.97)
        
        # 1) Gentle high-pass to clear mud
#         b,a = biquad_highpass(24000, f=120, q=0.707)
        # NEW: shave a hair of nasal mud (~280 Hz) – very gentle
#         b,a = peaking_eq(24000, f=280, q=1.2, gain_db=-1.0); float_mono = apply_biquad(float_mono, b, a)

#         # keep (or nudge) presence boost around 3.1 kHz
#         b,a = peaking_eq(24000, f=3100, q=0.9, gain_db=3.0); float_mono = apply_biquad(float_mono, b, a)
#         float_mono = apply_biquad(float_mono, b, a)

#         # 2) Subtle presence lift (was +4 dB, now +2 dB)
#         b,a = peaking_eq(24000, f=3100, q=0.9, gain_db=2.0)
#         float_mono = apply_biquad(float_mono, b, a)

#         # 3) Mild low-mid scoop at 350 Hz
#         b,a = peaking_eq(24000, f=350, q=1.1, gain_db=-2.0)
#         float_mono = apply_biquad(float_mono, b, a)

#         # 4) Softer compand (was +6 dB, now +3 dB)
#         float_mono = soft_compand(float_mono, gain_db=3.0)

        # 5) Normalize to safe headroom
        peak = np.max(np.abs(float_mono)) + 1e-6
        float_mono = (0.9 * float_mono / peak).astype(np.float32)

    # single resample to target_sr
    float_mono, _ = _to_float_mono(float_mono, 24000, target_sr)

    buf = io.BytesIO()
    if encode == "pcm":
        sf.write(buf, float_mono, target_sr, format="WAV", subtype="PCM_16")
    else:
        write_ulaw_wav(buf, float_mono, target_sr)
    buf.seek(0)


    key_prefix = (req.key_prefix or KEY_PREFIX_DEF).rstrip("/")
    suffix = f"{encode}_{target_sr}hz"
    key = f"{key_prefix}/{uuid.uuid4().hex}_{suffix}.wav"

    # If your uploader is named _put_ulaw_wav but just does a put-object, keep it.
    # If it actually re-encodes, swap to a generic uploader that just uploads bytes.
    _put_ulaw_wav(buf, key)

    url = s3.generate_presigned_url("get_object",
        Params={"Bucket": TTS_BUCKET, "Key": key}, ExpiresIn=PRESIGN_EXPIRES)

    return {
        "bucket": TTS_BUCKET, "key": key, "url": url,
        "sample_rate": target_sr, "encode": encode, "text": req.text,
    }


@app.post("/synthesize_stream")
def synthesize_stream(req: SynthesizeIn, authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    if not req.speaker_wav and get_engine().default_target_se is None:
        raise HTTPException(400, "No speaker reference available. Provide speaker_wav or set OV_DEFAULT_REF / OV_TGT_SE_PTH.")
    eng = get_engine()
    target_sr = int(req.sample_rate or 8000)

    job_id = uuid.uuid4().hex
    keys: list[str] = []

    # synchronous sentence-chunk streaming (generate → upload per sentence)
    sentences = _split_into_sentences(req.text)
    cur_part = 0
    for sent in sentences:
        audio_f32, model_sr = eng.synthesize(text=normalize_text(sent, lang="ko"), speaker_wav=req.speaker_wav, sr=req.sample_rate)
        float_mono, _ = _to_float_mono(audio_f32, model_sr, target_sr)

        buf = io.BytesIO(); write_ulaw_wav(buf, float_mono, target_sr); buf.seek(0)
        key = f"{job_id}/part{cur_part}.wav"
        _put_ulaw_wav(buf, key)
        keys.append(key)
        cur_part += 1

    # final sentinel
    tail = np.zeros(int(target_sr * 0.2), dtype=np.float32)
    b = io.BytesIO(); write_ulaw_wav(b, tail, target_sr); b.seek(0)
    key = f"{job_id}/final.wav"
    _put_ulaw_wav(b, key)
    keys.append(key)

    return {"keys": keys, "bucket": TTS_BUCKET, "region": S3_REGION}


# --- S3 listing helpers used by streaming status/batch ---
_PART_RE = re.compile(r".*/part(\d+)\.wav$")

def _list_ready_parts_from_s3(job_id: str):
    prefix = f"{job_id}/"
    client = s3
    part_indices: List[int] = []
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

def _presign(key: str, expires: int = 3600) -> str:
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": TTS_BUCKET, "Key": key},
        ExpiresIn=expires,
    )

@app.post("/synthesize_stream_start")
def synthesize_stream_start(req: SynthesizeIn, background: BackgroundTasks,
                            authorization: Optional[str] = Header(None)):
    _check_auth(authorization)
    if not req.speaker_wav and get_engine().default_target_se is None:
        raise HTTPException(400, "No speaker reference available. Provide speaker_wav or set OV_DEFAULT_REF / OV_TGT_SE_PTH.")

    job_id = uuid.uuid4().hex
    target_sr = int(req.sample_rate or 8000)

#     background.add_task(_run_stream_job_chunked, job_id, req.text, target_sr, 0, req.speaker_wav)
    background.add_task(_run_stream_job_from_idx, job_id, req.text, target_sr, 0, req.speaker_wav)

    wait_ms   = int(os.getenv("FIRST_URL_WAIT_MS", "800"))
    poll_ms   = int(os.getenv("FIRST_URL_POLL_MS", "50"))
    pres_ttl  = int(os.getenv("PRESIGN_TTL_SEC", "120"))

    first_key = None
    first_url = None

    if wait_ms > 0:
        deadline = time.time() + (wait_ms / 1000.0)
        while time.time() < deadline:
            part_indices, _ = _list_ready_parts_from_s3(job_id)
            if 0 in part_indices:
                first_key = f"{job_id}/part0.wav"
                print(first_key)
                first_url = _presign(first_key, expires=pres_ttl)
                print(first_url)
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

def _clean_text(txt: str) -> str:
    """Enhanced text cleaning for better MeLo stability"""
    if not txt:
        return ""
    
    # Normalize Unicode
    txt = txt.strip()
    
    # Remove problematic characters that can cause numerical issues
    txt = re.sub(r'[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ.,!?;:%&$\-\'"()[\]{}]', ' ', txt)
#     txt = re.sub(
#        r'[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ.,!?;:%&$~…—\-\'"()[\]{}]',
#        ' ',
#        txt
#    )
    
    # Normalize ellipses and weird punctuation
    txt = txt.replace("…", "...")
    txt = re.sub(r'["""]', '"', txt)  # Fixed: use single quotes to wrap the pattern
    txt = re.sub(r"[''']", "'", txt)
    
    # Collapse whitespace
    txt = re.sub(r'\s+', ' ', txt).strip()
    
    # Ensure proper sentence ending
    if txt and not txt[-1] in '.!?':
        txt += '.'
    
    return txt

# Solution 3: Add model reinitialization method for severe cases
def reinitialize_model(self):
    """Reinitialize the TTS model if it gets into a bad state"""
    logger.warning("[OpenVoice] Reinitializing TTS model due to numerical instability")
    try:
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        # Reinitialize TTS model
        self.tts = TTS(language=self.language, device=self.device)
        
        logger.info("[OpenVoice] Model reinitialized successfully")
        return True
    except Exception as e:
        logger.error(f"[OpenVoice] Model reinitialization failed: {e}")
        return False
    
    
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
