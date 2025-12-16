import argparse
import os
import time
import uuid
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import yaml
import boto3
from botocore.client import Config as BotoConfig

# ---- FishSpeech imports (your stack) ----
import torch
import io
from loguru import logger
from fish_speech.inference_engine.utils import InferenceResult
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest

# ===================== config helpers =====================
def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _extract_audio_sr(obj, fallback_sr: int):
    """
    Normalize FishSpeech outputs into (np.float32 array, int sample_rate).

    Handles:
      - (audio, sr) and (sr, audio) tuples
      - dicts with audio or wav_path
      - InferenceResult with .audio being (sr, ndarray) or other variants
    """
    import numpy as _np
    import soundfile as _sf

    # ---- A) direct tuple ----
    if isinstance(obj, tuple) and len(obj) == 2:
        first, second = obj
        # (sr, ndarray)
        if isinstance(first, (int, _np.integer)) and isinstance(second, _np.ndarray):
            return _np.asarray(second, dtype=_np.float32), int(first)
        # (ndarray, sr)
        if isinstance(first, _np.ndarray) and isinstance(second, (int, _np.integer)):
            return _np.asarray(first, dtype=_np.float32), int(second)
        # bytes in either slot → read WAV
        if isinstance(first, (bytes, bytearray)):
            audio, sr2 = _sf.read(io.BytesIO(first), dtype="float32")
            return _np.asarray(audio, dtype=_np.float32), int((second if isinstance(second, (int, _np.integer)) else sr2) or fallback_sr)
        if isinstance(second, (bytes, bytearray)):
            audio, sr2 = _sf.read(io.BytesIO(second), dtype="float32")
            return _np.asarray(audio, dtype=_np.float32), int((first if isinstance(first, (int, _np.integer)) else sr2) or fallback_sr)
        # last resort: treat one side as a file path
        for p in (first, second):
            try:
                audio, sr = _sf.read(str(p), dtype="float32")
                return _np.asarray(audio, dtype=_np.float32), int(sr or fallback_sr)
            except Exception:
                pass

    # ---- B) dict-like ----
    if isinstance(obj, dict):
        if "audio" in obj:
            a = obj["audio"]
            sr = int(obj.get("sample_rate") or obj.get("sr") or fallback_sr)
            if isinstance(a, (bytes, bytearray)):
                audio, sr2 = _sf.read(io.BytesIO(a), dtype="float32")
                return _np.asarray(audio, dtype=_np.float32), int(sr or sr2 or fallback_sr)
            return _np.asarray(a, dtype=_np.float32), sr
        if "wav_path" in obj:
            audio, sr = _sf.read(obj["wav_path"], dtype="float32")
            return _np.asarray(audio, dtype=_np.float32), int(sr)

    # ---- C) InferenceResult (streaming API) ----
    if isinstance(obj, InferenceResult):
        a = getattr(obj, "audio", None)
        if a is not None:
            # common: (sr, ndarray)
            if isinstance(a, (list, tuple)) and len(a) == 2:
                first, second = a
                if isinstance(first, (int, _np.integer)) and isinstance(second, _np.ndarray):
                    return _np.asarray(second, dtype=_np.float32), int(first)
                if isinstance(first, _np.ndarray) and isinstance(second, (int, _np.integer)):
                    return _np.asarray(first, dtype=_np.float32), int(second)
                if isinstance(first, (bytes, bytearray)):
                    audio, sr2 = _sf.read(io.BytesIO(first), dtype="float32")
                    return _np.asarray(audio, dtype=_np.float32), int((second if isinstance(second, (int, _np.integer)) else sr2) or fallback_sr)
                if isinstance(second, (bytes, bytearray)):
                    audio, sr2 = _sf.read(io.BytesIO(second), dtype="float32")
                    return _np.asarray(audio, dtype=_np.float32), int((first if isinstance(first, (int, _np.integer)) else sr2) or fallback_sr)
            # pure bytes/ndarray/list
            if isinstance(a, (bytes, bytearray)):
                audio, sr = _sf.read(io.BytesIO(a), dtype="float32")
                return _np.asarray(audio, dtype=_np.float32), int(sr or fallback_sr)
            if isinstance(a, _np.ndarray):
                return _np.asarray(a, dtype=_np.float32), int(fallback_sr)
            if isinstance(a, list):
                return _np.asarray(a, dtype=_np.float32), int(fallback_sr)

    raise RuntimeError(f"Unsupported inference output type: {type(obj)}")



# ===================== engine loading =====================
class FishEngineAdapter:
    """Thin adapter so the rest of the script can call a stable synthesize().

    It builds your TTSEngine (llama + decoder) once and exposes synthesize(text,...)
    returning (audio_f32, sample_rate).
    """
    def __init__(self, cfg: dict):
        device = cfg.get("device", "cuda")
        half = bool(cfg.get("half", False))
        precision = torch.half if half else torch.bfloat16
        compile_flag = bool(cfg.get("compile", False))
        load_embeddings = bool(cfg.get("load_embeddings", True))

        llama_ckpt = cfg.get("llama_checkpoint_path", "/home/work/VALL-E/fish-speech/checkpoints/openaudio-s1-mini")
        decoder_cfg = cfg.get("decoder_config_name", "modded_dac_vq")
        decoder_ckpt = cfg.get("decoder_checkpoint_path", "/home/work/VALL-E/fish-speech/checkpoints/openaudio-s1-mini/codec.pth")
        self.sample_rate = int(cfg.get("audio", {}).get("sample_rate", 22050))

        logger.info("Launching LLM queue…")
        self.llama_queue = launch_thread_safe_queue(
            checkpoint_path=llama_ckpt,
            device=device,
            precision=precision,
            compile=compile_flag,
        )

        logger.info("Loading decoder model (%s)…", decoder_cfg)
        self.decoder_model = load_decoder_model(
            config_name=decoder_cfg,
            checkpoint_path=decoder_ckpt,
            device=device,
        )

        logger.info("Constructing TTSInferenceEngine…")
        self.engine = TTSInferenceEngine(
            llama_queue=self.llama_queue,
            decoder_model=self.decoder_model,
            compile=compile_flag,
            precision=precision,
            load_embeddings=load_embeddings,
            embedding_path=cfg.get("embedding_path", "/home/work/VALL-E/fish-speech/cached_ref.pt"),
        )

        # No Gradio/WebUI wrapper — call engine.infer() directly
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
        use_memory_cache: Optional[object] = None,
    ) -> Tuple[np.ndarray, int]:
        """Synthesize audio with explicit defaults matching FishSpeech schema."""
        params = {**self.default_gen}
        if temperature is not None: params["temperature"] = temperature
        if top_p is not None: params["top_p"] = top_p
        if seed is not None: params["seed"] = seed
        if repetition_penalty is not None: params["repetition_penalty"] = repetition_penalty
        if max_new_tokens is not None: params["max_new_tokens"] = max_new_tokens
        if chunk_length is not None: params["chunk_length"] = chunk_length
        if use_memory_cache is not None: params["use_memory_cache"] = use_memory_cache

        # Normalize use_memory_cache to the schema's expected literal strings ('on'|'off')
        # BEFORE creating ServeTTSRequest:
        # inside FishEngineAdapter.synthesize(...)
        umc_val = params.get("use_memory_cache", False)
        umc_str = "on" if (str(umc_val).lower() == "on" or bool(umc_val)) else "off"

        req = ServeTTSRequest(
            text=text,
            references=[],                # align with webui defaults
            reference_id=None,
            reference_audio=speaker_wav,
            reference_text=None,
            max_new_tokens=int(params.get("max_new_tokens", 0)),
            chunk_length=int(params.get("chunk_length", 300)),
            top_p=float(params.get("top_p", 0.8)),
            repetition_penalty=float(params.get("repetition_penalty", 1.1)),
            temperature=float(params.get("temperature", 0.8)),
            seed=int(params.get("seed", 42)),
            use_memory_cache=umc_str,     # 'on' | 'off'
            format="wav",
            stream=False,
        )

        # NOTE: your engine is streaming-only; collect the final result
        # inside FishEngineAdapter.synthesize(...)
        umc_val = params.get("use_memory_cache", False)
        umc_str = "on" if (str(umc_val).lower() == "on" or bool(umc_val)) else "off"

        req = ServeTTSRequest(
            text=text,
            references=[],                # align with webui defaults
            reference_id=None,
            reference_audio=speaker_wav,
            reference_text=None,
            max_new_tokens=int(params.get("max_new_tokens", 0)),
            chunk_length=int(params.get("chunk_length", 300)),
            top_p=float(params.get("top_p", 0.8)),
            repetition_penalty=float(params.get("repetition_penalty", 1.1)),
            temperature=float(params.get("temperature", 0.8)),
            seed=int(params.get("seed", 42)),
            use_memory_cache=umc_str,     # 'on' | 'off'
            format="wav",
            stream=False,
        )

        # NOTE: your engine is streaming-only; collect the final result
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
def load_engine(cfg_path: str) -> FishEngineAdapter:
    cfg = _read_yaml(cfg_path)
    return FishEngineAdapter(cfg)


# ===================== audio / s3 helpers =====================
def _ensure_int16(audio: np.ndarray) -> np.ndarray:
    if audio.dtype != np.float32 and audio.dtype != np.float64:
        audio = audio.astype(np.float32)
    max_val = float(np.max(np.abs(audio))) if audio.size else 1.0
    max_val = max(max_val, 1e-6)
    audio = audio / max_val
    return np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)


def s3_upload_and_presign(local_path: str, bucket: str, key_prefix: str, expires: int = 3600) -> dict:
    s3 = boto3.client("s3", config=BotoConfig(signature_version="s3v4"))
    key_prefix = key_prefix.rstrip("/")
    key = f"{key_prefix}/{os.path.basename(local_path)}"
    s3.upload_file(local_path, bucket, key, ExtraArgs={"ContentType": "audio/wav"})
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )
    return {"bucket": bucket, "key": key, "url": url}


# ===================== CLI entrypoint =====================
def main():
    ap = argparse.ArgumentParser(description="FishSpeech inference → WAV → S3 (presigned URL)")
    ap.add_argument("--config", default="./fishspeech_infer/config.yaml", help="Path to config.yaml")
    ap.add_argument("--text", default='안녕하세요, 자동차 보험 비교 가입 도와드리는, 차집사 다이렉트, 차은하 팀장입니다. 잠시 통화 가능하실까요? 지금 이용하고 계신 업체 있으실 텐데요, 저희가 이번에, 보험사 연도 대상자 출신들로 팀을 재구성하면서, 수수료 7%의 조건으로 진행을 하고 있어서, 안내차 연락드렸습니다. 사고 건이 많거나 해서, 다이렉트 가입이 안 되시는 고객님들도, 오프라인으로 가입 가능하게 해드리고 있으며, 오프라인, 텔레마케팅, 비교사이트 가입 시 모두, 7% 수수료를 익일 오후에 바로 지급해드리고 있습니다. 수수료 조건도 좋고, 체결율도 95% 이상이라, 많은 분들이 함께하고 계신데요, 앞으로 딜러님, 사장님 담당은 제가 할 거라, 인사차 연락드렸습니다. 제 번호 저장해두셨다가, 견적 문의 있으실 때 연락주시면, 저희가 빠르게 진행 도와드리겠습니다. 명함, 문자로 남겨드릴게요. 감사합니다.')
    ap.add_argument("--speaker-wav", default=None, help="Optional reference WAV for voice/style")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--bucket", default='seoul-bucket-65432')
    ap.add_argument("--key-prefix", default="tts-out")
    ap.add_argument("--expires", type=int, default=None, help="Presign expiry seconds (override config)")
    ap.add_argument("--sr", type=int, default=None, help="Desired output sample rate (no resample by default)")
    # Generation params (explicit to avoid the Gradio wrapper requirements)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--max-new-tokens", type=int, default=0)
    ap.add_argument("--chunk-length", type=int, default=300)
    ap.add_argument("--use-memory-cache", action="store_true")
    args = ap.parse_args()

    cfg = _read_yaml(args.config)
    expires = args.expires or cfg.get("presign_expires", 3600)

    engine = load_engine(args.config)
    audio_f32, sr = engine.synthesize(
        text=args.text,
        speaker_wav=args.speaker_wav,
        sr=args.sr or cfg.get("audio", {}).get("sample_rate"),
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
        chunk_length=args.chunk_length,
        use_memory_cache=args.use_memory_cache,
    )

    os.makedirs(args.outdir, exist_ok=True)
    fname = f"tts_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav"
    local_wav = os.path.join(args.outdir, fname)
    sf.write(local_wav, _ensure_int16(audio_f32), sr, subtype="PCM_16")

    result = s3_upload_and_presign(
        local_path=local_wav,
        bucket=args.bucket,
        key_prefix=args.key_prefix or cfg.get("s3", {}).get("key_prefix", "tts-out"),
        expires=expires,
    )

    print(result)


if __name__ == "__main__":
    main()