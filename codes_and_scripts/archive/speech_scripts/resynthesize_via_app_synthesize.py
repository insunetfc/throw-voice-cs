#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resynthesize RESPONSE rows by calling your running app's /synthesize endpoint.

Flow per row:
  1) Take logical cached_text (no TTS preprocessing here).
  2) Hash = sha256(normalize_utt(cached_text))  # must match backfill/Lambda
  3) POST /synthesize {text, sample_rate=8000}  # your app applies preprocess_korean_text internally
  4) Response JSON: {"bucket","key","url",...}  # temp key (uuid) already in S3 as μ-law 8k WAV
  5) GET S3 bytes for temp key, PUT to final key: {DEST_PREFIX}/{locale}/{hash}.wav
  6) Update DDB UtteranceCache[hash, locale].audio_s3_uri -> s3://{bucket}/{final_key}
  7) Delete temp key.

Env:
  AWS_REGION=ap-northeast-2
  UTT_CACHE_TABLE=UtteranceCache
  TTS_URL=http://localhost:8000/synthesize
  API_TOKEN=...                        (optional Bearer for app auth)
  DEST_PREFIX=ko-KR                    (default; final S3 prefix root)
  BUCKET=tts-bucket-250810             (if omitted, uses the bucket returned by /synthesize)
"""

import os, re, time, json, argparse, unicodedata, hashlib, sys, io
import boto3, requests
from botocore.exceptions import ClientError
from botocore.config import Config as BotoConfig

AWS_REGION   = os.getenv("AWS_REGION", "ap-northeast-2")
TABLE_NAME   = os.getenv("UTT_CACHE_TABLE", "UtteranceCache")
TTS_URL      = os.getenv("TTS_URL", "http://localhost:8000/synthesize").rstrip("/")
API_TOKEN    = os.getenv("API_TOKEN", "")
DEST_PREFIX  = os.getenv("DEST_PREFIX", "ko-KR")
DEFAULT_SR   = int(os.getenv("DEFAULT_SR", "8000"))
DEFAULT_LOCALE = os.getenv("DEFAULT_LOCALE", "ko-KR")
DEFAULT_BUCKET = os.getenv("BUCKET")  # optional override

ddb = boto3.resource("dynamodb", region_name=AWS_REGION)
tbl = ddb.Table(TABLE_NAME)
s3  = boto3.client("s3", region_name=AWS_REGION, config=BotoConfig(max_pool_connections=32))

# -------- Hashing (must match backfill/Lambda) --------
def normalize_utt(text: str) -> str:
    s = unicodedata.normalize("NFKC", text).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\.!\?]+$", "", s)  # strip trailing punctuation often unstable in ASR
    return s

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def headers():
    h = {"Content-Type": "application/json"}
    if API_TOKEN: h["Authorization"] = f"Bearer {API_TOKEN}"
    return h

def synthesize(text: str, sr: int):
    payload = {"text": text, "sample_rate": sr}
    r = requests.post(TTS_URL, headers=headers(), data=json.dumps(payload), timeout=120)
    r.raise_for_status()
    return r.json()  # expects {"bucket","key","url",...}

def fetch_s3_bytes(bucket: str, key: str) -> bytes:
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()

def put_s3_bytes(bucket: str, key: str, data: bytes):
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType="audio/wav")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locale", default="ko-KR", help="Locale to process (default ko-KR)")
    ap.add_argument("--limit", type=int, default=0, help="Max rows to process (0=all)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    processed = 0
    lek = None
    while True:
        scan_kwargs = {
            "FilterExpression": "key_type = :resp AND locale = :loc",
            "ExpressionAttributeValues": {":resp": "response", ":loc": args.locale},
            "ProjectionExpression": "utterance_hash, locale, cached_text",
        }
        if lek:
            scan_kwargs["ExclusiveStartKey"] = lek
        resp = tbl.scan(**scan_kwargs)
        items = resp.get("Items", [])

        for it in items:
            cached = it.get("cached_text") or ""
            if not cached:
                continue
            resp_hash = sha256_hex(normalize_utt(cached))
            pk_hash = it["utterance_hash"]
            if pk_hash != resp_hash:
                print(f"[warn] PK hash!=recomputed for {pk_hash[:8]} (recomp {resp_hash[:8]}); using PK.")
            key_hash = pk_hash

            if args.dry_run:
                print(f"[dry-run] {key_hash[:8]} {args.locale}: POST /synthesize → copy to {DEST_PREFIX}/{args.locale}/{key_hash}.wav → DDB update")
                processed += 1
                if args.limit and processed >= args.limit:
                    print(f"[fin] limit reached: {processed}")
                    return
                continue

            # 1) Call your running app; it will apply preprocess_korean_text internally.
            info = synthesize(cached, DEFAULT_SR)
            tmp_bucket = info.get("bucket") or DEFAULT_BUCKET
            tmp_key    = info.get("key")
            if not (tmp_bucket and tmp_key):
                print(f"[ERR] synthesize returned no bucket/key for {key_hash[:8]}")
                continue

            # 2) Fetch the μ-law WAV uploaded by the app
            try:
                wav = fetch_s3_bytes(tmp_bucket, tmp_key)
            except Exception as e:
                print(f"[ERR] fetch temp s3://{tmp_bucket}/{tmp_key} failed: {e}")
                continue

            # 3) Put to final deterministic key
            dest_bucket = DEFAULT_BUCKET or tmp_bucket
            final_key = f"{DEST_PREFIX}/{args.locale}/{key_hash}.wav"
            try:
                put_s3_bytes(dest_bucket, final_key, wav)
            except Exception as e:
                print(f"[ERR] put final s3://{dest_bucket}/{final_key} failed: {e}")
                continue

            # 4) Update DDB
            try:
                tbl.update_item(
                    Key={"utterance_hash": key_hash, "locale": args.locale},
                    UpdateExpression="SET audio_s3_uri = :u, updated_at = :t",
                    ExpressionAttributeValues={
                        ":u": f"s3://{dest_bucket}/{final_key}",
                        ":t": int(time.time()),
                    },
                )
                print(f"[ok] {key_hash[:8]} {args.locale}: s3://{dest_bucket}/{final_key}")
            except Exception as e:
                print(f"[ERR] DDB update failed {key_hash[:8]}: {e}")
                # best effort; continue

            # 5) Cleanup temp
            try:
                s3.delete_object(Bucket=tmp_bucket, Key=tmp_key)
            except Exception:
                pass

            processed += 1
            if args.limit and processed >= args.limit:
                print(f"[fin] limit reached: {processed}")
                return

        lek = resp.get("LastEvaluatedKey")
        if not lek:
            print(f"[fin] scan complete. processed={processed}")
            break

if __name__ == "__main__":
    main()
