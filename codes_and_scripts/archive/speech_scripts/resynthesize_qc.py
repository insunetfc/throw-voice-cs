#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resynthesize UtteranceCache RESPONSE rows with QC + endpoint trim
and print progress lines including the (prepared) TTS text.

Flow per item:
  - Read cached_text (logical text; hashing/DB unchanged)
  - POST /synthesize with use_memory_cache=False (server does preprocess internally)
  - Download temp μ-law WAV from S3
  - Decode, endpoint-trim, pad, write μ-law/8k WAV to deterministic key
    {DEST_PREFIX}/{locale}/{response_hash}.wav
  - Update DDB audio_s3_uri
  - Delete temp key
  - Append metrics to CSV report
  - Print: [ok] <hash8> <locale>: s3://<bucket>/<final_key> | text=<prepared_tts_text>

Env (override via flags as needed):
  AWS_REGION=ap-northeast-2
  UTT_CACHE_TABLE=UtteranceCache
  TTS_URL=http://localhost:8000/synthesize
  API_TOKEN=...
  DEST_PREFIX=ko-KR
  BUCKET= (if empty, use the bucket that /synthesize returns)

Usage:
  python resynthesize_qc_trim.py --locale ko-KR --limit 50 --dry-run
  python resynthesize_qc_trim.py --locale ko-KR --skip-existing --verbose
"""

import os, io, re, csv, time, json, argparse, unicodedata
import requests, boto3
import numpy as np
import soundfile as sf
from botocore.config import Config as BotoConfig
from collections import defaultdict

AWS_REGION     = os.getenv("AWS_REGION", "ap-northeast-2")
TABLE_NAME     = os.getenv("UTT_CACHE_TABLE", "UtteranceCache")
TTS_URL        = os.getenv("TTS_URL", "http://localhost:8000/synthesize").rstrip("/")
API_TOKEN      = os.getenv("API_TOKEN", "")
DEST_PREFIX    = os.getenv("DEST_PREFIX", "ko-KR")
DEFAULT_SR     = int(os.getenv("DEFAULT_SR", "8000"))
DEFAULT_LOCALE = os.getenv("DEFAULT_LOCALE", "ko-KR")
DEFAULT_BUCKET = os.getenv("BUCKET", "")

# Endpointing & padding
RMS_THR      = float(os.getenv("RMS_THR", "0.025"))
HANG_MS      = int(os.getenv("HANG_MS", "120"))
MIN_TOTALS   = float(os.getenv("MIN_TOTAL_SEC", "0.30"))  # >= 300ms total
PAD_HEAD_MS  = int(os.getenv("PAD_HEAD_MS", "40"))
PAD_TAIL_MS  = int(os.getenv("PAD_TAIL_MS", "80"))
REPORT_CSV   = os.getenv("REPORT_CSV", "tts_resynthesis_report.csv")

ddb = boto3.resource("dynamodb", region_name=AWS_REGION)
tbl = ddb.Table(TABLE_NAME)
s3  = boto3.client("s3", region_name=AWS_REGION, config=BotoConfig(max_pool_connections=32))

# ---- helpers ----

def headers():
    h = {"Content-Type": "application/json"}
    if API_TOKEN:
        h["Authorization"] = f"Bearer {API_TOKEN}"
    return h

def synthesize(text: str, sr: int):
    payload = {"text": text, "sample_rate": sr, "use_memory_cache": False}
    r = requests.post(TTS_URL, headers=headers(), data=json.dumps(payload), timeout=120)
    r.raise_for_status()
    return r.json()  # expects {"bucket","key",...}

def fetch_s3_bytes(bucket: str, key: str) -> bytes:
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()

def put_s3_bytes(bucket: str, key: str, data: bytes):
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType="audio/wav")

def _to_float_mono(wav_bytes: bytes):
    a, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    return a, sr

def _voiced_endpoint(samples: np.ndarray, sr: int, frame_ms: int = 20,
                     rms_thr: float = 0.025, hang_ms: int = 120):
    if samples.ndim > 1:
        samples = samples.mean(axis=0)
    frame = max(1, int(sr * (frame_ms / 1000.0)))
    hang  = max(1, int(sr * (hang_ms / 1000.0)))
    last_voiced = 0
    for i in range(0, len(samples) - frame, frame):
        rms = float(np.sqrt(np.mean(np.square(samples[i:i+frame])) + 1e-9))
        if rms > rms_thr:
            last_voiced = i + frame
    return min(len(samples), last_voiced + hang)

def write_ulaw_wav(buf, samples_f32, sr, pad_head_ms=PAD_HEAD_MS,
                   pad_tail_ms=PAD_TAIL_MS, min_total_sec=MIN_TOTALS):
    head = np.zeros(int(sr * (pad_head_ms/1000.0)), dtype=np.float32)
    tail = np.zeros(int(sr * (pad_tail_ms/1000.0)), dtype=np.float32)
    out  = np.concatenate([head, samples_f32, tail])
    need = int(max(0.0, min_total_sec - (out.shape[0] / sr)) * sr)
    if need > 0:
        out = np.concatenate([out, np.zeros(need, dtype=np.float32)])
    sf.write(buf, out, sr, subtype="ULAW", format="WAV")

def build_final_key(prefix: str, locale: str, key_hash: str) -> str:
    prefix = (prefix or "").strip("/")
    if not prefix:
        return f"{locale}/{key_hash}.wav"
    if prefix.split("/")[-1] == locale:
        return f"{prefix}/{key_hash}.wav"
    return f"{prefix}/{locale}/{key_hash}.wav"

# ---- TTS preview helpers (for logging only; server still does its own prep) ----
_ZWJ = "\u2060"  # WORD JOINER

def sanitize_markup_and_urls(text: str) -> str:
    s = unicodedata.normalize("NFKC", text).strip()
    s = s.replace("**","").replace("__","").replace("*","").replace("_","").replace("`","")
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
    return s

def tts_prepare_preview(text: str) -> str:
    s = sanitize_markup_and_urls(text)
    s = s.replace("켇", "켜")
    s = re.sub(r"(\d+)\s*%", r"\1 퍼센트", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"겠(?=(다|습니다|어요|에요|지요|네요|군요|고요|죠))", "겠"+_ZWJ, s)
    if not re.search(r"[.!?…~다요]$", s):
        s += "…"
    return s

def _oneline_preview(s: str, max_len: int = 160) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return (s if len(s) <= max_len else s[:max_len-1] + "…")

def parse_s3_uri(uri: str):
    if not uri or not uri.startswith("s3://"):
        return None, None
    rest = uri[5:]
    i = rest.find("/")
    if i < 0: return None, None
    return rest[:i], rest[i+1:]

def tg_notify(token: str, chat_id: str, text: str):
    if not token or not chat_id:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat_id, "text": text}, timeout=10)
    except Exception as e:
        print(f"[tg ] notify failed: {e}", flush=True)

# ---- main ----

def main():
    ap = argparse.ArgumentParser()
    # Telegram
    ap.add_argument("--tg-token", default=os.getenv("TG_BOT_TOKEN"))
    ap.add_argument("--tg-chat",  default=os.getenv("TG_CHAT_ID"))
    ap.add_argument("--tg-every", default=200, type=int, help="Send Telegram progress every N items (0=off)")
    ap.add_argument("--tg-on-error", action="store_true", help="Telegram ping for each error/reject")
    # Core
    ap.add_argument("--locale", default=DEFAULT_LOCALE)
    ap.add_argument("--force-if-zwj", action="store_true",
                    help="If cached_text matches the 겠+ending pattern, force re-synth even if target key exists")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", default=REPORT_CSV)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--key-type", default="response", choices=["response","utterance"])
    # Skip logic (default ON; use --no-skip-existing to force re-synth)
    ap.add_argument("--skip-existing", default=True, dest="skip_existing", action="store_true",
                    help="Skip if target final object already exists")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.set_defaults(skip_existing=True)

    args = ap.parse_args()

    processed = 0
    counts = defaultdict(int)
    lek = None
    seen = set()

    # CSV report
    fout = open(args.report, "w", encoding="utf-8", newline="")
    writer = csv.writer(fout)
    writer.writerow(["utterance_hash","locale","dur_in_sec","speech_end_sec",
                     "dur_trim_sec","final_key","status","note"])

    while True:
        # DDB scan (alias reserved 'status' -> #st)
        scan_kwargs = {
            "FilterExpression": (
                "(attribute_not_exists(key_type) OR key_type = :kt) "
                "AND locale = :loc "
                "AND attribute_exists(cached_text)"
            ),
            "ExpressionAttributeValues": {
                ":kt": args.key_type,
                ":loc": args.locale,
            },
            "ExpressionAttributeNames": {"#st": "status"},
            "ProjectionExpression": "utterance_hash, locale, key_type, approval_type, #st, audio_s3_uri, cached_text",
            "Limit": args.batch,
        }
        if lek:
            scan_kwargs["ExclusiveStartKey"] = lek

        resp = tbl.scan(**scan_kwargs)
        items = resp.get("Items", [])

        for it in items:
            cached = it.get("cached_text") or ""
            if not cached:
                continue

            pk_hash = it["utterance_hash"]
            key_run = (args.locale, pk_hash)
            if key_run in seen:
                if args.verbose:
                    print(f"[skip] {pk_hash[:8]} {args.locale}: already processed in this run", flush=True)
                continue
            seen.add(key_run)

            final_key = build_final_key(DEST_PREFIX, args.locale, pk_hash)
            preview = _oneline_preview(tts_prepare_preview(cached))

            # Skip-existing (only if existing key EXACTLY matches target final_key and object exists)
            need_zwj_fix = bool(re.search(r"겠(다|습니다|어요|에요|지요|네요|군요|고요|죠)", cached))
            if args.skip_existing and not (args.force_if_zwj and need_zwj_fix):
                existing_uri = it.get("audio_s3_uri", "")
                eb, ek = parse_s3_uri(existing_uri)
                should_skip = False
                if eb and ek:
                    try:
                        s3.head_object(Bucket=eb, Key=ek)
                        if ek == final_key:
                            should_skip = True
                        else:
                            if args.verbose:
                                why = []
                                if "//" in ek: why.append("double-slash")
                                if "/ko-KR/ko-KR/" in ek: why.append("duplicated-locale")
                                if ek != final_key: why.append("key-mismatch")
                                print(f"[redo] {pk_hash[:8]} {args.locale}: existing={existing_uri} "
                                      f"-> target=s3://{(DEFAULT_BUCKET or eb)}/{final_key} "
                                      f"({', '.join(why) or 'outdated'})", flush=True)
                    except Exception:
                        pass  # HEAD failed; treat as missing

                if should_skip:
                    tag = "[skip-dry]" if args.dry_run else "[skip]"
                    if args.verbose or args.dry_run:
                        print(f"{tag} {pk_hash[:8]} {args.locale}: {existing_uri} | text={preview}", flush=True)
                    counts["skip-existing"] += 1
                    writer.writerow([pk_hash, args.locale, "", "", "", final_key, "skip-existing", existing_uri])
                    processed += 1
                    if args.tg_every and (processed % args.tg_every == 0):
                        tg_notify(args.tg_token, args.tg_chat,
                                  f"Resynth {args.locale}: processed={processed} ok={counts['ok']} "
                                  f"skip={counts['skip-existing']} rej={counts['reject']} err={counts['error']}")
                    if args.limit and processed >= args.limit:
                        print(f"[fin] limit reached: {processed}", flush=True)
                        msg = (f"Resynth {args.locale} FINISHED (limit): processed={processed} "
                               f"ok={counts['ok']} skip={counts['skip-existing']} "
                               f"reject={counts['reject']} error={counts['error']}")
                        tg_notify(args.tg_token, args.tg_chat, msg)
                        fout.close()
                        return
                    continue

            # Dry run
            if args.dry_run:
                print(f"[dry-run] {pk_hash[:8]} {args.locale}: s3://{(DEFAULT_BUCKET or '<bucket-from-app>')}/{final_key} | text={preview}", flush=True)
                writer.writerow([pk_hash, args.locale, "", "", "", final_key, "dry-run", preview])
                processed += 1
                if args.tg_every and (processed % args.tg_every == 0):
                    tg_notify(args.tg_token, args.tg_chat,
                              f"Resynth {args.locale}: processed={processed} ok={counts['ok']} "
                              f"skip={counts['skip-existing']} rej={counts['reject']} err={counts['error']}")
                if args.limit and processed >= args.limit:
                    print(f"[fin] limit reached: {processed}", flush=True)
                    msg = (f"Resynth {args.locale} FINISHED (limit): processed={processed} "
                           f"ok={counts['ok']} skip={counts['skip-existing']} "
                           f"reject={counts['reject']} error={counts['error']}")
                    tg_notify(args.tg_token, args.tg_chat, msg)
                    fout.close()
                    return
                continue

            # Guard (belt-and-suspenders)
            if it.get("key_type") != args.key_type:
                if args.verbose:
                    print(f"[skip] {pk_hash[:8]} {args.locale}: key_type={it.get('key_type')} != {args.key_type}", flush=True)
                continue

            if args.verbose:
                print(f"[go  ] {pk_hash[:8]} {args.locale}: synthesize...", flush=True)

            # Synthesize
            try:
                info = synthesize(cached, DEFAULT_SR)
            except Exception as e:
                msg = f"synthesize: {e}"
                counts["error"] += 1
                if args.tg_on_error:
                    tg_notify(args.tg_token, args.tg_chat, f"Resynth {args.locale}: error synth ({pk_hash[:8]})")
                if args.verbose:
                    print(f"[err ] {pk_hash[:8]} {args.locale}: {msg}", flush=True)
                writer.writerow([pk_hash, args.locale, "", "", "", "", "error", msg])
                processed += 1
                if args.tg_every and (processed % args.tg_every == 0):
                    tg_notify(args.tg_token, args.tg_chat,
                              f"Resynth {args.locale}: processed={processed} ok={counts['ok']} "
                              f"skip={counts['skip-existing']} rej={counts['reject']} err={counts['error']}")
                if args.limit and processed >= args.limit:
                    print(f"[fin] limit reached: {processed}", flush=True)
                    msg2 = (f"Resynth {args.locale} FINISHED (limit): processed={processed} "
                            f"ok={counts['ok']} skip={counts['skip-existing']} "
                            f"reject={counts['reject']} error={counts['error']}")
                    tg_notify(args.tg_token, args.tg_chat, msg2)
                    fout.close()
                    return
                continue

            tmp_bucket = info.get("bucket") or DEFAULT_BUCKET
            tmp_key    = info.get("key")
            if not (tmp_bucket and tmp_key):
                counts["error"] += 1
                if args.tg_on_error:
                    tg_notify(args.tg_token, args.tg_chat, f"Resynth {args.locale}: error no-key ({pk_hash[:8]})")
                if args.verbose:
                    print(f"[err ] {pk_hash[:8]} {args.locale}: no bucket/key", flush=True)
                writer.writerow([pk_hash, args.locale, "", "", "", "", "error", "no bucket/key"])
                processed += 1
                if args.tg_every and (processed % args.tg_every == 0):
                    tg_notify(args.tg_token, args.tg_chat,
                              f"Resynth {args.locale}: processed={processed} ok={counts['ok']} "
                              f"skip={counts['skip-existing']} rej={counts['reject']} err={counts['error']}")
                if args.limit and processed >= args.limit:
                    print(f"[fin] limit reached: {processed}", flush=True)
                    msg2 = (f"Resynth {args.locale} FINISHED (limit): processed={processed} "
                            f"ok={counts['ok']} skip={counts['skip-existing']} "
                            f"reject={counts['reject']} error={counts['error']}")
                    tg_notify(args.tg_token, args.tg_chat, msg2)
                    fout.close()
                    return
                continue

            # Fetch temp WAV
            try:
                wav = fetch_s3_bytes(tmp_bucket, tmp_key)
            except Exception as e:
                counts["error"] += 1
                if args.tg_on_error:
                    tg_notify(args.tg_token, args.tg_chat, f"Resynth {args.locale}: error fetch ({pk_hash[:8]})")
                msg = f"fetch temp: {e}"
                if args.verbose:
                    print(f"[err ] {pk_hash[:8]} {args.locale}: {msg}", flush=True)
                writer.writerow([pk_hash, args.locale, "", "", "", "", "error", msg])
                processed += 1
                if args.tg_every and (processed % args.tg_every == 0):
                    tg_notify(args.tg_token, args.tg_chat,
                              f"Resynth {args.locale}: processed={processed} ok={counts['ok']} "
                              f"skip={counts['skip-existing']} rej={counts['reject']} err={counts['error']}")
                if args.limit and processed >= args.limit:
                    print(f"[fin] limit reached: {processed}", flush=True)
                    msg2 = (f"Resynth {args.locale} FINISHED (limit): processed={processed} "
                            f"ok={counts['ok']} skip={counts['skip-existing']} "
                            f"reject={counts['reject']} error={counts['error']}")
                    tg_notify(args.tg_token, args.tg_chat, msg2)
                    fout.close()
                    return
                continue

            # Decode
            try:
                float_mono, sr_in = _to_float_mono(wav)
            except Exception as e:
                counts["reject"] += 1
                if args.tg_on_error:
                    tg_notify(args.tg_token, args.tg_chat, f"Resynth {args.locale}: reject decode ({pk_hash[:8]})")
                if args.verbose:
                    print(f"[rej ] {pk_hash[:8]} {args.locale}: decode failed: {e}", flush=True)
                writer.writerow([pk_hash, args.locale, "", "", "", "", "reject", f"decode: {e}"])
                try: s3.delete_object(Bucket=tmp_bucket, Key=tmp_key)
                except Exception: pass
                processed += 1
                if args.tg_every and (processed % args.tg_every == 0):
                    tg_notify(args.tg_token, args.tg_chat,
                              f"Resynth {args.locale}: processed={processed} ok={counts['ok']} "
                              f"skip={counts['skip-existing']} rej={counts['reject']} err={counts['error']}")
                if args.limit and processed >= args.limit:
                    print(f"[fin] limit reached: {processed}", flush=True)
                    msg2 = (f"Resynth {args.locale} FINISHED (limit): processed={processed} "
                            f"ok={counts['ok']} skip={counts['skip-existing']} "
                            f"reject={counts['reject']} error={counts['error']}")
                    tg_notify(args.tg_token, args.tg_chat, msg2)
                    fout.close()
                    return
                continue

            sr_eff = sr_in or DEFAULT_SR
            dur_in = len(float_mono) / float(sr_eff)
            if dur_in < 0.10:
                counts["reject"] += 1
                if args.tg_on_error:
                    tg_notify(args.tg_token, args.tg_chat, f"Resynth {args.locale}: reject short ({pk_hash[:8]})")
                if args.verbose:
                    print(f"[rej ] {pk_hash[:8]} {args.locale}: too short ({dur_in:.2f}s)", flush=True)
                writer.writerow([pk_hash, args.locale, f"{dur_in:.3f}", "", "", "", "reject", "too-short"])
                try: s3.delete_object(Bucket=tmp_bucket, Key=tmp_key)
                except Exception: pass
                processed += 1
                if args.tg_every and (processed % args.tg_every == 0):
                    tg_notify(args.tg_token, args.tg_chat,
                              f"Resynth {args.locale}: processed={processed} ok={counts['ok']} "
                              f"skip={counts['skip-existing']} rej={counts['reject']} err={counts['error']}")
                if args.limit and processed >= args.limit:
                    print(f"[fin] limit reached: {processed}", flush=True)
                    msg2 = (f"Resynth {args.locale} FINISHED (limit): processed={processed} "
                            f"ok={counts['ok']} skip={counts['skip-existing']} "
                            f"reject={counts['reject']} error={counts['error']}")
                    tg_notify(args.tg_token, args.tg_chat, msg2)
                    fout.close()
                    return
                continue

            if args.verbose:
                print(f"[wav ] {pk_hash[:8]} {args.locale}: dur_in={dur_in:.2f}s", flush=True)

            # Endpointing
            end_idx = _voiced_endpoint(float_mono, sr_eff, frame_ms=20, rms_thr=RMS_THR, hang_ms=HANG_MS)
            trim_end = max(end_idx, int(sr_eff * MIN_TOTALS))
            trimmed = float_mono[:trim_end]
            dur_trim = len(trimmed) / float(sr_eff)
            if args.verbose:
                print(f"[trim] {pk_hash[:8]} {args.locale}: end={end_idx/sr_eff:.2f}s -> {dur_trim:.2f}s", flush=True)

            # Upload final μ-law WAV
            dest_bucket = DEFAULT_BUCKET or tmp_bucket
            buf = io.BytesIO()
            write_ulaw_wav(buf, trimmed, sr_eff)
            buf.seek(0)
            try:
                put_s3_bytes(dest_bucket, final_key, buf.read())
            except Exception as e:
                counts["error"] += 1
                if args.tg_on_error:
                    tg_notify(args.tg_token, args.tg_chat, f"Resynth {args.locale}: error put ({pk_hash[:8]})")
                msg = f"put final: {e}"
                if args.verbose:
                    print(f"[err ] {pk_hash[:8]} {args.locale}: {msg}", flush=True)
                writer.writerow([pk_hash, args.locale, f"{dur_in:.3f}", f"{end_idx/sr_eff:.3f}",
                                 f"{dur_trim:.3f}", final_key, "error", msg])
                processed += 1
                if args.tg_every and (processed % args.tg_every == 0):
                    tg_notify(args.tg_token, args.tg_chat,
                              f"Resynth {args.locale}: processed={processed} ok={counts['ok']} "
                              f"skip={counts['skip-existing']} rej={counts['reject']} err={counts['error']}")
                if args.limit and processed >= args.limit:
                    print(f"[fin] limit reached: {processed}", flush=True)
                    msg2 = (f"Resynth {args.locale} FINISHED (limit): processed={processed} "
                            f"ok={counts['ok']} skip={counts['skip-existing']} "
                            f"reject={counts['reject']} error={counts['error']}")
                    tg_notify(args.tg_token, args.tg_chat, msg2)
                    fout.close()
                    return
                continue

            # Update DDB
            try:
                tbl.update_item(
                    Key={"utterance_hash": pk_hash, "locale": args.locale},
                    UpdateExpression="SET audio_s3_uri = :u, updated_at = :t",
                    ExpressionAttributeValues={
                        ":u": f"s3://{dest_bucket}/{final_key}",
                        ":t": int(time.time()),
                    },
                )
                print(f"[ok  ] {pk_hash[:8]} {args.locale}: s3://{dest_bucket}/{final_key} | text={preview}", flush=True)
                counts["ok"] += 1
                writer.writerow([pk_hash, args.locale, f"{dur_in:.3f}", f"{end_idx/sr_eff:.3f}",
                                 f"{dur_trim:.3f}", final_key, "ok", preview])
            except Exception as e:
                counts["error"] += 1
                if args.tg_on_error:
                    tg_notify(args.tg_token, args.tg_chat, f"Resynth {args.locale}: error ddb ({pk_hash[:8]})")
                msg = f"ddb update: {e}"
                if args.verbose:
                    print(f"[err ] {pk_hash[:8]} {args.locale}: {msg}", flush=True)
                writer.writerow([pk_hash, args.locale, f"{dur_in:.3f}", f"{end_idx/sr_eff:.3f}",
                                 f"{dur_trim:.3f}", final_key, "error", msg])

            # Cleanup temp
            try:
                s3.delete_object(Bucket=tmp_bucket, Key=tmp_key)
            except Exception:
                pass

            # Progress + optional Telegram heartbeat
            processed += 1
            if args.tg_every and (processed % args.tg_every == 0):
                tg_notify(args.tg_token, args.tg_chat,
                          f"Resynth {args.locale}: processed={processed} ok={counts['ok']} "
                          f"skip={counts['skip-existing']} rej={counts['reject']} err={counts['error']}")
            if args.limit and processed >= args.limit:
                print(f"[fin] limit reached: {processed}", flush=True)
                msg = (f"Resynth {args.locale} FINISHED (limit): processed={processed} "
                       f"ok={counts['ok']} skip={counts['skip-existing']} "
                       f"reject={counts['reject']} error={counts['error']}")
                tg_notify(args.tg_token, args.tg_chat, msg)
                fout.close()
                return

        lek = resp.get("LastEvaluatedKey")
        if not lek:
            print(f"[fin] scan complete. processed={processed}", flush=True)
            msg = (f"Resynth {args.locale} FINISHED: processed={processed} "
                   f"ok={counts['ok']} skip={counts['skip-existing']} "
                   f"reject={counts['reject']} error={counts['error']}")
            tg_notify(args.tg_token, args.tg_chat, msg)
            break

    fout.close()

if __name__ == "__main__":
    main()
