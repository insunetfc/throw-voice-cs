#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bulk synthesize filler phrases via app.py (/synthesize), store in S3 by category,
and rename to stable numeric filenames (01.wav..05.wav), ensuring WAV/8kHz/mono/µ-law.

Requirements:
  pip install requests boto3 python-dotenv

Also requires `ffmpeg` binary in PATH (for µ-law transcode).

Env (or CLI) you will likely set:
  TTS_URL=http://localhost:8000/synthesize
  TTS_TOKEN=your_server_token_if_any
  AWS_REGION=ap-northeast-2
  FULL_SAMPLE_RATE=8000
"""

import os, csv, time, argparse, subprocess, shutil, io, struct
import requests
import boto3
import re
import soundfile as sf
from botocore.exceptions import ClientError

# -------- Configuration --------
DEFAULT_TTS_URL = os.getenv("TTS_URL", "http://localhost:8000/synthesize").rstrip("/")
TTS_TOKEN       = os.getenv("TTS_TOKEN", "").strip()
SAMPLE_RATE     = int(os.getenv("FULL_SAMPLE_RATE", "8000"))
AWS_REGION      = os.getenv("AWS_REGION", "ap-northeast-2")
MAX_RETRIES     = 3
RETRY_SLEEP_SEC = 1.0

# Categories → phrases
CATEGORIES = {
    "확인": [
        "네, 고객님 맞으십니다.",
        "네, 말씀하신 내용 확인했습니다.",
        "네, 그렇게 진행하겠습니다.",
        "알겠습니다. 그대로 처리할게요.",
        "확인되었습니다, 잠시만 기다려 주세요.",
        "네, 지금 바로 도와드릴게요.",
        "네, 안내드린 대로 진행하겠습니다.",
        "네, 요청하신 부분 접수했습니다.",
        "네, 문제 없습니다.",
        "네, 이어서 진행하겠습니다.",
    ],
    "설명": [
        "안내드리자면, 이런 절차로 진행됩니다.",
        "간단히 말씀드리면요…",
        "먼저 한 가지 확인 후 설명드릴게요.",
        "순서대로 설명드리겠습니다.",
        "요약해서 말씀드리면요…",
        "자세히 안내드릴게요.",
        "참고로, 이 부분은 이렇게 이해하시면 됩니다.",
        "조금 더 구체적으로 설명드릴게요.",
        "핵심만 짚어서 말씀드리겠습니다.",
        "이어서 추가 설명 드릴게요.",
    ],
    "공감": [
        "아, 그러셨군요. 많이 불편하셨겠어요.",
        "네, 그 마음 이해합니다.",
        "말씀 주셔서 감사합니다.",
        "아, 그런 상황이면 답답하셨겠습니다.",
        "네, 그렇게 느끼실 수 있어요.",
        "공감합니다. 더 신경 쓰겠습니다.",
        "불편을 드려 죄송합니다.",
        "네, 충분히 이해했습니다.",
        "의견 주셔서 감사합니다. 반영해 보겠습니다.",
        "걱정되실 수 있겠습니다.",
    ],
    "시간벌기형": [
        "잠시만 기다려 주세요, 바로 확인하겠습니다.",
        "지금 조회 중입니다…",
        "곧 결과 안내드릴게요…",
        "확인까지 1~2초만 더 부탁드립니다…",
        "자료를 불러오는 중입니다…",
        "금방 연결해 드리겠습니다…",
        "처리 중입니다, 잠시만요…",
        "이어서 준비 중입니다…",
        "조금만 더 기다려 주세요…",
        "확인이 완료되는 대로 말씀드릴게요…",
    ],
    "test": [
        "안녕하세요~ 자동차 보험 비교 가입 도와드리는 차집사 다이렉트 차은하 팀장입니다. 잠시 통화 가능하실까요? 지금 이용하고 계신 업체 있으실텐데 저희가 이번에 보험사 연도대상자 출신들로 팀을 재구성 하면서 수수료 7%프로의 조건으로 진행을 하고 있어서 안내차 연락드렸습니다. 사고건이 많거나 해서 다이렉트 가입이 안되시는 고객님들도 OFF 라인으로 가입 가능하게 해드리고 OFF, TM, CM 가입시 모두 7%수수료를 익일오후에 바로 지급 해드리고 있습니다. 수수료 조건도 좋은데 체결율도 95% 이상이라 많은 분들이 함께 하고 계신데 앞으로 딜러님(사장님) 담당은 제가 할꺼라 인사차 연락드렸구요. 제 번호 저장해 두셨다가 견적문의 있으실때 연락주시면 저희가 빠르게 진행 도와드리겠습니다. 명함 문자로 남겨드릴게요~ 감사합니다.",
    ],
    
}

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


# -------- Helpers --------

def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def synthesize_once(tts_url: str, text: str, key_prefix: str, sr: int, token: str = ""):
    """POST /synthesize -> returns dict with bucket, key, url, regional_url, sample_rate, text."""
    payload = {"text": text, "sample_rate": sr, "key_prefix": key_prefix}
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(tts_url, json=payload, headers=headers, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(RETRY_SLEEP_SEC * attempt)

    raise RuntimeError(f"/synthesize failed after {MAX_RETRIES} attempts: {last_err}")

def to_regional_url(bucket: str, region: str, key: str) -> str:
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"

def parse_wav_fmt_sr_ch(wav_bytes: bytes):
    """
    Minimal RIFF/WAV parser to find 'fmt ' chunk and read:
      - audio_format (1=PCM, 7=µ-law)
      - num_channels
      - sample_rate
    Returns (fmt_code, channels, sample_rate) or (None, None, None) if not WAV.
    """
    bio = io.BytesIO(wav_bytes)
    if bio.read(4) != b'RIFF':
        return (None, None, None)
    _ = bio.read(4)  # chunk size
    if bio.read(4) != b'WAVE':
        return (None, None, None)

    # Scan chunks until 'fmt '
    while True:
        hdr = bio.read(8)
        if len(hdr) < 8:
            return (None, None, None)
        chunk_id, chunk_sz = hdr[:4], struct.unpack("<I", hdr[4:8])[0]
        if chunk_id == b'fmt ':
            fmt_data = bio.read(chunk_sz)
            if len(fmt_data) < 16:
                return (None, None, None)
            fmt_code   = struct.unpack("<H", fmt_data[0:2])[0]
            channels   = struct.unpack("<H", fmt_data[2:4])[0]
            sample_rate= struct.unpack("<I", fmt_data[4:8])[0]
            return (fmt_code, channels, sample_rate)
        else:
            # skip this chunk (align to even boundary)
            bio.seek(chunk_sz + (chunk_sz % 2), io.SEEK_CUR)

def is_mulaw_8k_mono(wav_bytes: bytes) -> bool:
    fmt_code, ch, sr = parse_wav_fmt_sr_ch(wav_bytes)
    return (fmt_code == 7 and ch == 1 and sr == 8000)

def transcode_to_mulaw_8k_mono(in_wav_bytes: bytes, sr_out: int = 8000) -> bytes:
    """
    Use ffmpeg via pipes:
      stdin: WAV (any codec)
      stdout: WAV pcm_mulaw, 8 kHz, mono
    """
    if not has_ffmpeg():
        raise RuntimeError("ffmpeg not found in PATH; please install it to transcode to µ-law.")

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "wav", "-i", "pipe:0",
        "-ar", str(sr_out), "-ac", "1", "-c:a", "pcm_mulaw",
        "-f", "wav", "pipe:1"
    ]
    proc = subprocess.run(cmd, input=in_wav_bytes, capture_output=True, check=True)
    return proc.stdout

def fetch_s3_bytes(s3, bucket: str, key: str) -> bytes:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()

def put_s3_bytes(s3, bucket: str, key: str, data: bytes, content_type: str = "audio/wav"):
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)

# -------- Main --------

def main():
    parser = argparse.ArgumentParser(description="Bulk synthesize filler phrases into S3 folders (µ-law ensured).")
    parser.add_argument("--tts-url", default=DEFAULT_TTS_URL, help="app.py /synthesize URL")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE, help="target sample rate (default 8000)")
    parser.add_argument("--token", default=TTS_TOKEN, help="Bearer token if server requires auth")
    parser.add_argument("--region", default=AWS_REGION, help="AWS region of your S3 bucket")
    parser.add_argument("--keep-original", action="store_true", help="Do not delete the UUID-named source object")
    parser.add_argument("--out-csv", default="filler_index.csv", help="Where to write the index CSV")
    parser.add_argument("--force-transcode", action="store_true",
                        help="Always transcode to µ-law even if file already looks µ-law/8k/mono")
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=args.region)

    rows = []
    total = sum(len(v) for v in CATEGORIES.values())
    i = 0

    print(f"Starting synthesis of {total} items → {args.tts_url}")
    if not has_ffmpeg():
        print("[WARN] ffmpeg not found; files that are not already µ-law/8k/mono will fail. Install ffmpeg.", flush=True)

    for category, phrases in CATEGORIES.items():
        for idx, text in enumerate(phrases, start=1):
            i += 1
            print(f"[{i}/{total}] {category} #{idx}: \"{text}\"")

            # 1) Synthesize (server writes to S3 using key_prefix=category)
            info = synthesize_once(args.tts_url, normalize_text(text, lang="ko", pause_strength="strong", tilde_behavior="stop"), key_prefix=category, sr=args.sample_rate, token=args.token)

            bucket = info.get("bucket")
            src_key = info.get("key")  # UUID-based name chosen by the server
            url = info.get("regional_url") or info.get("url") or to_regional_url(bucket, args.region, src_key)

            if not bucket or not src_key:
                raise RuntimeError(f"Server did not return bucket/key for {category} #{idx}: {info}")

            # 2) Final desired key (e.g., 확인/01.wav)
            dst_key = f"{category}/{idx:02d}.wav"
            print(dst_key)

            # 3) Download, check/convert to µ-law, then upload to final key
            try:
                raw = fetch_s3_bytes(s3, bucket, src_key)
            except ClientError as e:
                raise RuntimeError(f"S3 get failed for {src_key}: {e}")

            must_transcode = args.force_transcode or (not is_mulaw_8k_mono(raw))
            if must_transcode:
                try:
                    raw = transcode_to_mulaw_8k_mono(raw, sr_out=8000)
                    print("    -> transcoded to µ-law/8k/mono")
                except Exception as e:
                    raise RuntimeError(f"Transcode failed for {src_key}: {e}")
            else:
                print("    -> already µ-law/8k/mono; no transcode")

            local_path = f"/home/work/VALL-E/OpenVoice/filler/{dst_key}"
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            with open(local_path, "wb") as f:
                f.write(raw) 
            # Upload µ-law WAV to final key
            try:
                put_s3_bytes(s3, bucket, dst_key, raw, content_type="audio/wav")
            except ClientError as e:
                raise RuntimeError(f"S3 put failed for {dst_key}: {e}")

            # Optionally delete original
            if not args.keep_original:
                try:
                    s3.delete_object(Bucket=bucket, Key=src_key)
                except ClientError as e:
                    print(f"[WARN] Could not delete original {src_key}: {e}")

            final_url = to_regional_url(bucket, args.region, dst_key)
            print(f"    -> s3://{bucket}/{dst_key}")
            print(f"    -> URL: {final_url}")

            rows.append({
                "category": category,
                "index": idx,
                "text": text,
                "bucket": bucket,
                "final_key": dst_key,
                "final_url": final_url,
                "src_key": src_key,
                "server_url": url,
                "sample_rate": info.get("sample_rate", args.sample_rate),
                "transcoded": "yes" if must_transcode else "no",
            })

    # 4) Write index CSV
    if rows:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nDone. Wrote index CSV: {args.out_csv}")
    else:
        print("No rows to write.")

if __name__ == "__main__":
    main()
