#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (See docstring in the earlier message)

import argparse
import io
import sys
import time
import concurrent.futures as futures
from typing import List, Tuple, Optional

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

try:
    import soundfile as sf
except Exception:
    sf = None
import wave
import contextlib
import json
import re

def wav_is_playable(wav_bytes: bytes, strict_rate: int = 8000, allow_pcm16: bool = True) -> Tuple[bool, str]:
    if sf is not None:
        try:
            with sf.SoundFile(io.BytesIO(wav_bytes)) as f:
                sr = f.samplerate
                ch = f.channels
                subtype = f.subtype
                fmt = f.format
                ok_subtype = (subtype == "ULAW") or (allow_pcm16 and subtype == "PCM_16")
                ok = (sr == strict_rate) and (ch == 1) and ok_subtype and (fmt == "WAV")
                return ok, f"sf: fmt={fmt} subtype={subtype} sr={sr} ch={ch}"
        except Exception as e:
            pass
    try:
        with contextlib.closing(wave.open(io.BytesIO(wav_bytes), 'rb')) as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            ok = (sr == strict_rate) and (ch == 1)
            return ok, f"wave: sr={sr} ch={ch}"
    except Exception as e:
        return False, f"wave-error: {e!r}"

def s3_list_prefix(s3, bucket: str, prefix: str) -> List[str]:
    keys = []
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for obj in resp.get("Contents", []):
            keys.append(obj["Key"])
        token = resp.get("NextContinuationToken")
        if not token:
            break
    return keys

def s3_get_bytes(s3, bucket: str, key: str) -> Optional[bytes]:
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:
        return None

def s3_copy(s3, bucket: str, src_key: str, dst_key: str) -> bool:
    try:
        s3.copy_object(Bucket=bucket, Key=dst_key, CopySource={"Bucket": bucket, "Key": src_key},
                       ContentType="audio/wav", MetadataDirective="REPLACE")
        return True
    except Exception:
        return False

def s3_delete(s3, bucket: str, key: str) -> bool:
    try:
        s3.delete_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False

def ddb_find_text_for_hash(ddb, table_name: str, response_hash: str, locale: str) -> Optional[str]:
    tbl = ddb.Table(table_name)
    from boto3.dynamodb.conditions import Attr
    fe = Attr("locale").eq(locale) & Attr("response_hash").eq(response_hash)
    proj = "cached_text, chatbot_response, original_utterance"
    lek = None
    while True:
        kw = {"FilterExpression": fe, "ProjectionExpression": proj}
        if lek:
            kw["ExclusiveStartKey"] = lek
        page = tbl.scan(**kw)
        for it in page.get("Items", []):
            text = it.get("cached_text") or it.get("chatbot_response") or it.get("original_utterance")
            if text:
                return text
        lek = page.get("LastEvaluatedKey")
        if not lek:
            break
    return None

def tts_synthesize(tts_url: str, text: str, sample_rate: int = 8000, token: str = "") -> Optional[bytes]:
    import requests
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = {"text": text, "sample_rate": sample_rate, "use_memory_cache": False}
    r = requests.post(tts_url.rstrip("/"), headers=headers, data=json.dumps(payload), timeout=120)
    r.raise_for_status()
    info = r.json()
    bucket = info.get("bucket")
    key = info.get("key")
    if not (bucket and key):
        return None
    s3 = boto3.client("s3")
    try:
        b = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        try:
            s3.delete_object(Bucket=bucket, Key=key)
        except Exception:
            pass
        return b
    except Exception:
        return None

def process_hash(s3, ddb, args, response_hash: str) -> tuple[str, str]:
    prefix = args.prefix.strip("/")
    exact = f"{prefix}/{response_hash}.wav"
    siblings = s3_list_prefix(s3, args.bucket, f"{prefix}/{response_hash}")
    candidates = [k for k in siblings if k != exact]

    exact_bytes = s3_get_bytes(s3, args.bucket, exact)
    exact_ok, exact_desc = (False, "missing")
    if exact_bytes is not None:
        exact_ok, exact_desc = wav_is_playable(exact_bytes)

    if exact_ok:
        deleted = 0
        for k in candidates:
            if args.apply:
                s3_delete(s3, args.bucket, k)
            deleted += 1
        return response_hash, f"EXACT_OK [{exact_desc}] deleted_dups={deleted}"

    best_key = None
    best_bytes = None
    best_desc = ""
    for k in candidates:
        b = s3_get_bytes(s3, args.bucket, k)
        if b is None:
            continue
        ok, desc = wav_is_playable(b)
        if ok:
            best_key = k
            best_bytes = b
            best_desc = desc
            break

    if best_key:
        if args.apply:
            if not s3_copy(s3, args.bucket, best_key, exact):
                return response_hash, f"PROMOTE_FAIL from={best_key} reason=copy_failed"
            for k in candidates:
                s3_delete(s3, args.bucket, k)
        return response_hash, f"PROMOTED {best_key} -> {exact} [{best_desc}]"

    if args.tts_url and args.table and args.locale:
        text = ddb_find_text_for_hash(ddb, args.table, response_hash, args.locale)
        if text:
            wav = tts_synthesize(args.tts_url, text, sample_rate=8000, token=args.api_token or "")
            if wav:
                ok, desc = wav_is_playable(wav)
                if ok and args.apply:
                    s3.put_object(Bucket=args.bucket, Key=exact, Body=wav, ContentType="audio/wav")
                    for k in candidates:
                        s3_delete(s3, args.bucket, k)
                return response_hash, f"RESYNTH {'APPLIED' if args.apply else 'DRY'} [{desc}]"
            return response_hash, "RESYNTH_FAIL (tts/no-audio)"
        return response_hash, "RESYNTH_SKIP (no text in DDB)"
    else:
        return response_hash, "NEEDS_ATTENTION (no playable; provide --tts-url/--table/--locale)"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default='tts-bucket-250810')
    ap.add_argument("--prefix", default="ko-KR/")
    ap.add_argument("--hash", action="append", dest="hashes")
    ap.add_argument("--scan-all", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--table", default="UtteranceCache")
    ap.add_argument("--locale", default="ko-KR")
    ap.add_argument("--region", default='ap-northeast-2')
    ap.add_argument("--tts-url", default="")
    ap.add_argument("--api-token", default="")
    args = ap.parse_args()

    cfg = BotoConfig(max_pool_connections=64, retries={"max_attempts": 6, "mode": "standard"})
    s3 = boto3.client("s3", region_name=args.region, config=cfg)
    ddb = boto3.resource("dynamodb", region_name=args.region, config=cfg) if args.table else None

    todo = []
    if args.hashes:
        todo = list(dict.fromkeys(args.hashes))
    elif args.scan_all:
        keys = s3_list_prefix(s3, args.bucket, args.prefix.strip("/"))
        prefix = args.prefix.strip("/")
        pat = re.compile(rf"^{prefix}/([0-9a-f]{{16,64}})(?:.*)\.wav$", re.IGNORECASE)
        seen = set()
        for k in keys:
            m = pat.match(k)
            if m:
                seen.add(m.group(1))
        todo = sorted(seen)
    else:
        print("Provide --hash <hash> (repeatable) or --scan-all to scan prefix")
        sys.exit(2)

    print(f"[start] bucket={args.bucket} prefix={args.prefix} hashes={len(todo)} apply={args.apply}")
    t0 = time.time()
    results = []
    with futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futs = [ex.submit(process_hash, s3, ddb, args, h) for h in todo]
        for fut in futures.as_completed(futs):
            try:
                h, msg = fut.result()
                print(f"[{h}] {msg}")
                results.append((h, msg))
            except Exception as e:
                print(f"[error] {e!r}")

    dt = time.time() - t0
    print(f"[done] processed={len(results)} elapsed={dt:.1f}s apply={args.apply}")

if __name__ == "__main__":
    main()
