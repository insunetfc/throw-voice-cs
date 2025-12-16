#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate short neutral filler audios via your TTS /synthesize endpoint.
- Minimal: no FAQ, no categories, no streaming.
- Transcodes to 8k/mono/µ-law WAV for telephony.
- Saves under outputs/<s3_prefix>/NN.wav and (optionally) uploads to S3.

Requires: requests, boto3, soundfile, ffmpeg in PATH
"""

import os, io, csv, argparse, subprocess
import requests
import boto3
import soundfile as sf
import unicodedata
import re

# ---------- Defaults (override via CLI) ----------
DEFAULT_TTS_URL = os.getenv("TTS_URL", "https://ef85da6954e5.ngrok-free.app/synthesize").rstrip("/")
DEFAULT_REGION  = os.getenv("AWS_REGION", "ap-northeast-2")
DEFAULT_BUCKET  = os.getenv("AUDIO_BUCKET", "tts-bucket-250810")

# A compact, safe default list (edit or pass --fillers-file)
DEFAULT_FILLERS = [
    # --- existing ones ---
    "(friendly) 네, 잠시만요.",
    "(friendly) 확인 중입니다…",
    "(friendly) 잠시만 기다려 주세요.",
    "(friendly) 지금 확인하고 있습니다.",
    "(friendly) 조금만 기다려 주세요.",
    "(friendly) 네, 확인해드리겠습니다.",
    "(friendly) 곧 안내드리겠습니다.",
    "(friendly) 처리 중입니다, 잠시만요.",
    "(friendly) 확인 완료 후 바로 안내드리겠습니다.",
    "(friendly) 잠시 후 결과를 알려드리겠습니다.",

    # --- new “thinking / acknowledgment” fillers ---
    "(friendly) 네, 이해했습니다.",
    "(friendly) 아, 네. 잠시만요.",
    "(friendly) 어… 잠시만요, 확인하겠습니다.",
    "(friendly) 네네, 지금 바로 확인 중이에요.",
    "(friendly) 네. 조금만 기다려 주세요.",
    "(friendly) 좋습니다, 잠시만요.",
    "(friendly) 네, 한 순간만요.",
    "(friendly) 확인해보겠습니다, 잠시만 기다려 주세요.",
    "(friendly) 네, 곧 말씀드리겠습니다.",
    "(friendly) 잠시만요, 확인이 완료되면 안내드릴게요.",

    "(friendly) 여보세요, 고객님? 혹시 들리고 계신가요?"
]

# ---------- Helpers ----------
def has_ffmpeg() -> bool:
    import shutil
    return shutil.which("ffmpeg") is not None

def to_https(bucket: str, region: str, key: str) -> str:
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"

def transcode_to_mulaw_8k_mono(in_wav_bytes: bytes, sr_out: int = 8000) -> bytes:
    if not has_ffmpeg():
        raise RuntimeError("ffmpeg not found in PATH; please install it.")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "wav", "-i", "pipe:0",
        "-ar", str(sr_out), "-ac", "1", "-c:a", "pcm_mulaw",
        "-af", "apad=pad_dur=0.5",  # add ~0.5s tail pad to avoid abrupt cutoff
        "-f", "wav", "pipe:1"
    ]
    proc = subprocess.run(cmd, input=in_wav_bytes, capture_output=True, check=True)
    return proc.stdout

def prepare_for_tts(text: str) -> str:
    """Light KR cleanup so TTS won’t truncate or misread symbols."""
    s = unicodedata.normalize("NFKC", text).strip()
    s = s.replace("**","").replace("__","").replace("*","").replace("_","").replace("`","")
    s = re.sub(r"https?://\S+|www\.\S+", " 링크 ", s, flags=re.I)
    s = re.sub(r"(\d+)\s*%", r"\1 퍼센트", s)
    s = re.sub(r"\s+", " ", s)
    if not re.search(r"[.!?…~다요]$", s):
        s += "…"
    return s

def synthesize_short(tts_url: str, text: str, key_prefix: str, sr: int = 8000, token: str = "") -> dict:
    """POST /synthesize → expect JSON with {bucket, key} of a temporary WAV."""
    payload = {"text": text, "sample_rate": sr, "key_prefix": key_prefix}
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.post(tts_url, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()

def fetch_s3_bytes(s3_client, bucket: str, key: str) -> bytes:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()

def write_wav(path: str, wav_bytes: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(wav_bytes)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Generate short neutral filler audio via /synthesize.")
    ap.add_argument("--tts-url", default=DEFAULT_TTS_URL, help="Your TTS /synthesize URL")
    ap.add_argument("--token", default=os.getenv("TTS_TOKEN", ""), help="Bearer token (optional)")
    ap.add_argument("--sample-rate", type=int, default=8000, help="Target sample rate for TTS request (default 8000)")
    ap.add_argument("--s3-prefix", default="neutral/fillers/", help="Folder prefix to embed in output paths (e.g., neutral/simple/)")
    ap.add_argument("--outdir", default="outputs", help="Local base output dir")
    ap.add_argument("--out-csv", default="neutral_fillers_index.csv", help="Where to write an index CSV")
    ap.add_argument("--fillers-file", help="UTF-8 text file (one filler per line) instead of defaults")
    ap.add_argument("--region", default=DEFAULT_REGION, help="AWS region for uploading")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET, help="S3 bucket for uploading")
    ap.add_argument("--upload", action="store_true", help="If set, upload generated WAVs to s3://<bucket>/<s3-prefix>")
    args = ap.parse_args()

    # Load filler lines
    if args.fillers_file and os.path.exists(args.fillers_file):
        with open(args.fillers_file, "r", encoding="utf-8") as f:
            fillers = [ln.strip() for ln in f if ln.strip()]
    else:
        fillers = DEFAULT_FILLERS

    # AWS clients (only if uploading or pulling temp objects from your TTS bucket)
    s3 = boto3.client("s3", region_name=args.region)

    rows = []
    for i, text in enumerate(fillers, start=1):
        safe_text = prepare_for_tts(text)

        # 1) Call your TTS server (single-shot)
        # key_prefix is only for your server-side temp organization (not the final S3 prefix)
        info = synthesize_short(args.tts_url if hasattr(args, "tts-url") else args.tts_url,
                                safe_text, key_prefix="neutral_simple", sr=args.sample_rate, token=args.token)

        # Expecting {"bucket": "...", "key": "..."} from your TTS
        temp_bucket = info.get("bucket")
        temp_key    = info.get("key")
        if not temp_bucket or not temp_key:
            print(f"[ERROR] TTS did not return bucket/key for item #{i}: {info}")
            continue

        # 2) Download temp WAV from TTS bucket, then transcode to µ-law 8k mono
        raw = fetch_s3_bytes(s3, temp_bucket, temp_key)
        final_wav = transcode_to_mulaw_8k_mono(raw, sr_out=8000)

        # 3) Local save under outputs/<s3_prefix>/NN.wav
        rel_key = f"{args.s3_prefix}{i:02d}.wav"
        out_path = os.path.join(args.outdir, rel_key)
        write_wav(out_path, final_wav)

        # 4) Optional: upload to S3 final location
        final_url = None
        if args.upload:
            s3.put_object(Bucket=args.bucket, Key=rel_key, Body=final_wav, ContentType="audio/wav")
            final_url = to_https(args.bucket, args.region, rel_key)

        rows.append({"id": f"neutral_{i:02d}", "text": text, "local_path": out_path,
                     "s3_key": rel_key, "s3_url": final_url or ""})

        # Best-effort: delete temp object from TTS bucket (optional)
        try:
            s3.delete_object(Bucket=temp_bucket, Key=temp_key)
        except Exception:
            pass

        print(f"[OK] #{i:02d} → {out_path}" + (f"  (uploaded: s3://{args.bucket}/{rel_key})" if args.upload else ""))

    # 5) Write an index CSV you can keep with the artifacts
    if rows:
        idx_path = os.path.join(args.outdir, args.out_csv)
        os.makedirs(os.path.dirname(idx_path), exist_ok=True)
        with open(idx_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["id","text","local_path","s3_key","s3_url"])
            w.writeheader(); w.writerows(rows)
        print(f"\n✅ Wrote index: {idx_path}")
    else:
        print("No rows generated.")

if __name__ == "__main__":
    main()
