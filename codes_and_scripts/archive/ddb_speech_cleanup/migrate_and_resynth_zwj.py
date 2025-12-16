#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
migrate_and_resynth_zwj.py

Three-fold job:
1) Move legacy S3 audio under .../approved/responses/<response_hash>/<file>.wav
   into canonical key: {DEST_PREFIX}/{locale}/{response_hash}.wav,
   and update DynamoDB's audio_s3_uri in-place (no TTS).
2) Scan rows (post-move) for the 겠 + ending pattern in any of:
   cached_text, chatbot_response, original_utterance.
3) Re-synthesize ONLY those rows needing the ZWJ fix using the same
   pipeline: POST /synthesize -> fetch temp WAV -> endpoint trim/pad
   -> write μ-law 8k WAV to canonical key -> update DDB -> delete temp.

Safe defaults:
- DRY-RUN by default (prints plan; no writes). Use --apply to perform changes.
- Locale default = ko-KR
- Includes rows even if key_type is missing. (You may optionally filter by key_type with --require-key-type.)

Usage examples:
  Dry run migration plan only:
    python migrate_and_resynth_zwj.py --locale ko-KR --phase move

  Dry run: move + detect ZWJ + resynth plan (no writes):
    python migrate_and_resynth_zwj.py --locale ko-KR --phase all

  Apply move only:
    python migrate_and_resynth_zwj.py --locale ko-KR --phase move --apply

  Apply all phases (move + detect + resynth). Limit TTS to 300 rows:
    python migrate_and_resynth_zwj.py --locale ko-KR --phase all --apply --tts-limit 300

  Parallel scan (4 workers): run this script 4x with --total-segments 4 and segment 0..3
"""

import argparse, io, json, os, re, sys, time
from collections import defaultdict

import boto3
from boto3.dynamodb.conditions import Attr
import numpy as np
import requests
import soundfile as sf
from botocore.config import Config as BotoConfig

# -------------------- Config (env overridable) --------------------

AWS_REGION     = os.getenv("AWS_REGION", "ap-northeast-2")
TABLE_NAME     = os.getenv("UTT_CACHE_TABLE", "UtteranceCache")
TTS_URL        = os.getenv("TTS_URL", "http://localhost:8000/synthesize").rstrip("/")
API_TOKEN      = os.getenv("API_TOKEN", "")
DEST_PREFIX    = os.getenv("DEST_PREFIX", "ko-KR")
DEFAULT_LOCALE = os.getenv("DEFAULT_LOCALE", "ko-KR")
DEFAULT_BUCKET = os.getenv("BUCKET", "")  # if empty, use bucket from /synthesize response

# Endpointing & padding
RMS_THR      = float(os.getenv("RMS_THR", "0.025"))
HANG_MS      = int(os.getenv("HANG_MS", "120"))
MIN_TOTALS   = float(os.getenv("MIN_TOTAL_SEC", "0.30"))
PAD_HEAD_MS  = int(os.getenv("PAD_HEAD_MS", "40"))
PAD_TAIL_MS  = int(os.getenv("PAD_TAIL_MS", "80"))

# -------------------- Clients --------------------
ddb = boto3.resource("dynamodb", region_name=AWS_REGION)
tbl = ddb.Table(TABLE_NAME)
s3  = boto3.client("s3", region_name=AWS_REGION, config=BotoConfig(max_pool_connections=32))

# -------------------- Helpers --------------------

def headers():
    h = {"Content-Type": "application/json"}
    if API_TOKEN:
        h["Authorization"] = f"Bearer {API_TOKEN}"
    return h

def synthesize(text: str, sr: int = 8000):
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
    if hasattr(a, "ndim") and a.ndim > 1:
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

def parse_s3_uri(uri: str):
    if not uri or not uri.startswith("s3://"):
        return None, None
    rest = uri[5:]
    i = rest.find("/")
    if i < 0: return None, None
    return rest[:i], rest[i+1:]

def final_key_for(locale: str, response_hash: str) -> str:
    prefix = (DEST_PREFIX or "").strip("/")
    if prefix.endswith("/" + locale) or prefix.split("/")[-1] == locale:
        return f"{prefix}/{response_hash}.wav"
    return f"{prefix}/{locale}/{response_hash}.wav"

ZWJ_PATTERN = re.compile(r"겠(다|습니다|어요|에요|지요|네요|군요|고요|죠)")

def needs_zwj_fix(*texts: str) -> bool:
    for t in texts:
        if not t:
            continue
        if ZWJ_PATTERN.search(t):
            return True
    return False

# -------------------- Core ops --------------------

def phase_move(locale: str, apply: bool, limit: int = 0, total_segments: int = 1, segment: int = 0):
    """
    Move legacy approved/responses audio to canonical key and update DDB.
    """
    fe = (Attr("locale").eq(locale) &
          Attr("response_hash").exists() &
          Attr("audio_s3_uri").contains("approved/responses/"))
    proj = "utterance_hash, locale, response_hash, audio_s3_uri, updated_at, chatbot_response, cached_text, original_utterance"

    print(f"[move] scanning legacy rows… locale={locale} seg={segment}/{total_segments-1}")
    planned = done = scanned = 0
    lek = None
    while True:
        scan_kwargs = {"FilterExpression": fe, "ProjectionExpression": proj}
        if lek:
            scan_kwargs["ExclusiveStartKey"] = lek
        if total_segments and total_segments > 1:
            scan_kwargs["TotalSegments"] = total_segments
            scan_kwargs["Segment"] = segment
        resp = tbl.scan(**scan_kwargs)
        items = resp.get("Items", [])
        for it in items:
            scanned += 1
            uh = it["utterance_hash"]; loc = it["locale"]; rh = it["response_hash"]
            old_uri = it.get("audio_s3_uri") or ""
            b_old, k_old = parse_s3_uri(old_uri)
            if not (b_old and k_old):
                continue

            k_new = final_key_for(loc, rh)
            b_new = DEFAULT_BUCKET or b_old

            # Skip if already canonical path exists and matches
            exists = False
            try:
                s3.head_object(Bucket=b_new, Key=k_new)
                exists = True
            except Exception:
                exists = False

            planned += 1
            print(f"[plan] {rh} :: {old_uri} -> s3://{b_new}/{k_new}")

            if apply:
                if not exists:
                    s3.copy_object(Bucket=b_new, Key=k_new,
                                   CopySource={"Bucket": b_old, "Key": k_old},
                                   ContentType="audio/wav",
                                   MetadataDirective="REPLACE")
                    # a tiny delay so S3 read-after-write settles
                    time.sleep(0.01)

                # conditional update to avoid races
                tbl.update_item(
                    Key={"utterance_hash": uh, "locale": loc},
                    UpdateExpression="SET audio_s3_uri = :u, updated_at = :t",
                    ConditionExpression=Attr("audio_s3_uri").eq(old_uri),
                    ExpressionAttributeValues={
                        ":u": f"s3://{b_new}/{k_new}",
                        ":t": int(time.time())
                    }
                )
                done += 1

            if limit and done >= limit:
                break

        if limit and done >= limit:
            break
        lek = resp.get("LastEvaluatedKey")
        if not lek:
            break

    print(f"[move] scanned={scanned}, planned={planned}, {'migrated='+str(done) if apply else 'dry-run'}")

def phase_detect(locale: str, total_segments: int = 1, segment: int = 0):
    """
    Return a list of items (minimal info) that need ZWJ fix after move.
    """
    fe = (Attr("locale").eq(locale) & Attr("response_hash").exists())
    proj = "utterance_hash, locale, response_hash, audio_s3_uri, chatbot_response, cached_text, original_utterance"

    targets = []
    scanned = 0
    lek = None
    print(f"[detect] scanning rows… locale={locale} seg={segment}/{total_segments-1}")
    while True:
        scan_kwargs = {"FilterExpression": fe, "ProjectionExpression": proj}
        if lek:
            scan_kwargs["ExclusiveStartKey"] = lek
        if total_segments and total_segments > 1:
            scan_kwargs["TotalSegments"] = total_segments
            scan_kwargs["Segment"] = segment
        resp = tbl.scan(**scan_kwargs)
        items = resp.get("Items", [])
        for it in items:
            scanned += 1
            if needs_zwj_fix(it.get("cached_text",""), it.get("chatbot_response",""), it.get("original_utterance","")):
                targets.append({
                    "utterance_hash": it["utterance_hash"],
                    "locale": it["locale"],
                    "response_hash": it["response_hash"],
                    "audio_s3_uri": it.get("audio_s3_uri",""),
                    "cached_text": it.get("cached_text","")
                })
        lek = resp.get("LastEvaluatedKey")
        if not lek:
            break

    print(f"[detect] scanned={scanned}, need_zwj={len(targets)}")
    return targets

def phase_resynth(locale: str, targets, apply: bool, tts_limit: int = 0):
    """
    Re-synthesize a list of target rows (dicts) needing ZWJ fix.
    """
    total = 0
    ok = err = rej = 0
    for it in targets:
        if tts_limit and total >= tts_limit:
            break
        total += 1

        uh = it["utterance_hash"]
        rh = it["response_hash"]
        loc = it["locale"]
        text = it.get("cached_text","").strip()
        if not text:
            # If cached_text missing, fallback to chatbot_response or original_utterance is possible,
            # but we deliberately stick to cached_text to keep hashing stable.
            print(f"[skip] {uh[:8]} no cached_text")
            continue

        k_final = final_key_for(loc, rh)

        print(f"[go  ] {uh[:8]} {loc}: synthesize for rh={rh}")
        if not apply:
            # dry-run
            print(f"[dry] would synth -> s3://{DEFAULT_BUCKET or '<bucket-from-app>'}/{k_final}")
            continue

        # 1) POST /synthesize
        try:
            info = synthesize(text, 8000)
        except Exception as e:
            print(f"[err] synth ({uh[:8]}): {e}")
            err += 1
            continue

        tmp_bucket = info.get("bucket") or DEFAULT_BUCKET
        tmp_key    = info.get("key")
        if not (tmp_bucket and tmp_key):
            print(f"[err] no bucket/key from TTS for {uh[:8]}")
            err += 1
            continue

        # 2) Fetch temp WAV
        try:
            wav = fetch_s3_bytes(tmp_bucket, tmp_key)
        except Exception as e:
            print(f"[err] fetch temp ({uh[:8]}): {e}")
            err += 1
            continue

        # 3) Decode
        try:
            float_mono, sr_in = _to_float_mono(wav)
        except Exception as e:
            print(f"[rej] decode failed ({uh[:8]}): {e}")
            rej += 1
            try: s3.delete_object(Bucket=tmp_bucket, Key=tmp_key)
            except Exception: pass
            continue

        sr_eff = sr_in or 8000
        dur_in = len(float_mono) / float(sr_eff)
        if dur_in < 0.10:
            print(f"[rej] too short ({uh[:8]}): {dur_in:.2f}s")
            rej += 1
            try: s3.delete_object(Bucket=tmp_bucket, Key=tmp_key)
            except Exception: pass
            continue

        # 4) Endpointing + write μ-law WAV
        end_idx = _voiced_endpoint(float_mono, sr_eff, frame_ms=20, rms_thr=RMS_THR, hang_ms=HANG_MS)
        trim_end = max(end_idx, int(sr_eff * MIN_TOTALS))
        trimmed = float_mono[:trim_end]

        dest_bucket = DEFAULT_BUCKET or tmp_bucket
        buf = io.BytesIO()
        write_ulaw_wav(buf, trimmed, sr_eff)
        buf.seek(0)
        try:
            put_s3_bytes(dest_bucket, k_final, buf.read())
        except Exception as e:
            print(f"[err] put final ({uh[:8]}): {e}")
            err += 1
            continue

        # 5) Update DDB to canonical key (even if already set)
        try:
            tbl.update_item(
                Key={"utterance_hash": uh, "locale": loc},
                UpdateExpression="SET audio_s3_uri = :u, updated_at = :t",
                ExpressionAttributeValues={
                    ":u": f"s3://{dest_bucket}/{k_final}",
                    ":t": int(time.time()),
                },
            )
        except Exception as e:
            print(f"[err] ddb update ({uh[:8]}): {e}")
            err += 1
            continue

        # 6) Delete temp
        try:
            s3.delete_object(Bucket=tmp_bucket, Key=tmp_key)
        except Exception:
            pass

        print(f"[ok ] {uh[:8]} {loc}: s3://{dest_bucket}/{k_final}")
        ok += 1

    print(f"[resynth] total={total} ok={ok} rej={rej} err={err} ({'dry-run' if not apply else 'applied'})")

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locale", default=DEFAULT_LOCALE)
    ap.add_argument("--phase", default="all", choices=["move", "detect", "resynth", "all"],
                    help="Which phase to run: move only, detect only, resynth only, or all (move+detect+resynth)")
    ap.add_argument("--require-key-type", default="", help="Optionally require key_type to equal this value")
    ap.add_argument("--apply", action="store_true", help="Apply changes (default dry-run)")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of migrated rows (phase move)")
    ap.add_argument("--tts-limit", type=int, default=0, help="Limit number of rows to re-synthesize (phase resynth)")
    ap.add_argument("--total-segments", type=int, default=1, help="Parallel scan: total segments")
    ap.add_argument("--segment", type=int, default=0, help="Parallel scan: this worker segment id (0..N-1)")
    args = ap.parse_args()

    # (Optional) Key-type filter is enforced by an additional scan pass in each phase
    require_key_type = args.require_key_type.strip() or None

    # PHASE 1: MOVE
    if args.phase in ("move", "all"):
        phase_move(args.locale, apply=args.apply, limit=args.limit,
                   total_segments=args.total_segments, segment=args.segment)

    # PHASE 2: DETECT
    targets = None
    if args.phase in ("detect", "all", "resynth"):
        targets = phase_detect(args.locale,
                               total_segments=args.total_segments,
                               segment=args.segment)
        # If user required key_type, filter in-memory
        if require_key_type:
            # Fetch key_type via on-demand get_item (cheaper than rescan):
            filtered = []
            for t in targets:
                try:
                    g = tbl.get_item(Key={"utterance_hash": t["utterance_hash"], "locale": t["locale"]}).get("Item", {})
                    if g.get("key_type") == require_key_type:
                        filtered.append(t)
                except Exception:
                    pass
            print(f"[detect] key_type={require_key_type} filtered {len(targets)} -> {len(filtered)}")
            targets = filtered

    # PHASE 3: RESYNTH
    if args.phase in ("resynth", "all"):
        phase_resynth(args.locale, targets or [], apply=args.apply, tts_limit=args.tts_limit)

if __name__ == "__main__":
    sys.exit(main())
