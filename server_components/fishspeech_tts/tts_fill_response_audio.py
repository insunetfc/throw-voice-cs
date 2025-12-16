
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tts_fill_response_audio_v3_1.py
- Based on v3 (concurrency-safe) plus regeneration controls:
  * regenerate_existing: generate even if S3 object already exists
  * overwrite_upload: when regenerating, overwrite the original <hash>.wav key
  * version_tag: if not overwriting, write to <hash>_<tag>.wav (defaults to timestamp)
  * no_update_ddb: don't touch audio_s3_uri; append to alt_audio_s3 list instead
  * audio_prev_uri: when updating DDB, preserve previous audio_s3_uri

Reads config from tts_config.json (same dir), then env, then optional CLI flags.
"""

import os
import io
import sys
import time
import json
import argparse
import threading
from typing import Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import re

import boto3
from botocore.exceptions import ClientError
import requests

# -------- Config helpers --------
def load_json_config(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")
        return {}

def to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in ("1","true","yes","y","on")

# -------- Core helpers --------
def s3_key_for(locale: str, response_hash: str, prefix: str = "", version_tag: str = None) -> str:
    suffix = f"_{version_tag}" if version_tag else ""
    parts = [p for p in [prefix.strip("/"), locale, f"{response_hash}{suffix}.wav"] if p]
    return "/".join(parts)

def s3_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise

def upload_wav(s3, bucket: str, key: str, wav_bytes: bytes) -> str:
    s3.put_object(Bucket=bucket, Key=key, Body=wav_bytes, ContentType="audio/wav", ACL="private")
    return f"s3://{bucket}/{key}"

class RateLimiter:
    def __init__(self, qps: float):
        self.qps = max(0.0, float(qps))
        self.last = 0.0
        self.lock = threading.Lock()
    def wait(self):
        if self.qps <= 0:
            return
        with self.lock:
            now = time.time()
            min_interval = 1.0 / self.qps
            delta = now - self.last
            if delta < min_interval:
                time.sleep(min_interval - delta)
            self.last = time.time()

GLOBAL_SEM = None
VOICE_LOCKS: Dict[str, threading.Lock] = {}
VOICE_LOCKS_LOCK = threading.Lock()
RATE = None

def with_retries(fn, *, retries=3, base_delay=0.8, max_delay=5.0, verbose=False):
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:
            attempt += 1
            if attempt > retries:
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            if verbose:
                print(f"   retry {attempt}/{retries} after error: {e} (sleep {delay:.2f}s)")
            time.sleep(delay)

def call_tts(tts_url: str, text: str, voice: str, sample_rate: int, token: str = "", timeout: int = 60) -> Tuple[str, Optional[bytes]]:
    headers = {"Accept": "application/json, audio/wav", "Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = {"text": text, "voice": voice, "sample_rate": sample_rate}
    url = f"{tts_url.rstrip('/')}/synthesize"
    RATE.wait()
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)

    ctype = resp.headers.get("Content-Type", "")
    if "audio/wav" in ctype or "octet-stream" in ctype:
        wav_bytes = resp.content
        if resp.status_code != 200 or not wav_bytes:
            raise RuntimeError(f"TTS binary response bad status={resp.status_code} len={len(wav_bytes)}")
        return ("", wav_bytes)

    try:
        data = resp.json()
    except Exception:
        raise RuntimeError(f"TTS returned non-JSON and non-audio: status={resp.status_code}, ctype={ctype}, text={resp.text[:200]}")

    if resp.status_code != 200:
        raise RuntimeError(f"TTS error {resp.status_code}: {data}")

    bucket = data.get("bucket")
    key = data.get("key")
    if bucket and key:
        return (f"s3://{bucket}/{key}", None)
    raise RuntimeError(f"TTS JSON did not include bucket/key. Payload: {data}")

def update_ddb_replace(table, response_hash: str, new_uri: str, prev_uri: str = None):
    now = int(time.time())
    expr = "SET audio_s3_uri = :uri, tts_generated = :tg, updated_at = :ts"
    attrs = {":uri": new_uri, ":tg": True, ":ts": now}
    if prev_uri:
        expr += ", audio_prev_uri = :prev"
        attrs[":prev"] = prev_uri
    table.update_item(Key={"response_hash": response_hash}, UpdateExpression=expr, ExpressionAttributeValues=attrs)

def update_ddb_append_alt(table, response_hash: str, alt_uri: str):
    now = int(time.time())
    # append to alt_audio_s3 list (create if missing)
    table.update_item(
        Key={"response_hash": response_hash},
        UpdateExpression="SET alt_audio_s3 = list_append(if_not_exists(alt_audio_s3, :empty), :one), updated_at = :ts",
        ExpressionAttributeValues={
            ":empty": [],
            ":one": [alt_uri],
            ":ts": now
        }
    )

def process_item(item: Dict[str, Any], cfg, table, s3, counters, verbose=False) -> Dict[str, Any]:
    response_hash = item.get("response_hash")
    response_text = item.get("response_text") or ""

    # Always prefix a friendly style tag unless there's already a leading style block
    if not re.match(r'^\s*\(', response_text):
        response_text = f"(friendly) {response_text}"
    locale = cfg["locale"]
    bucket = cfg["bucket"]
    prefix = cfg["prefix"]
    voice = cfg["voice"]

    # existing key and uri
    base_key = s3_key_for(locale, response_hash, prefix)
    exists = s3_exists(s3, bucket, base_key)
    prev_uri = item.get("audio_s3_uri")

    if verbose:
        print(f"[ITEM] hash={response_hash} exists={exists} only_missing={cfg['only_missing']} force={cfg['force']} regen={cfg['regenerate_existing']} overwrite={cfg['overwrite_upload']} no_update_ddb={cfg['no_update_ddb']}")

    # Skip if exists and we're not forcing/regenerating
    if exists and cfg["only_missing"] and not (cfg["force"] or cfg["regenerate_existing"]):
        counters["skipped"] += 1
        if verbose:
            print("       -> skip (exists & only-missing, no force/regen)")
        return {"hash": response_hash, "skipped": True, "reason": "exists"}

    # Concurrency controls
    if GLOBAL_SEM:
        GLOBAL_SEM.acquire()
    try:
        lock = None
        if cfg["per_voice_serial"]:
            with VOICE_LOCKS_LOCK:
                lock = VOICE_LOCKS.setdefault(voice, threading.Lock())
        if lock:
            lock.acquire()
        try:
            def _call():
                return call_tts(cfg["tts_url"], response_text, voice, int(cfg["sample_rate"]), cfg["tts_token"], int(cfg["timeout"]))
            s3_uri, wav_bytes = with_retries(_call, retries=int(cfg["retries"]), base_delay=cfg["retry_base_delay"], max_delay=cfg["retry_max_delay"], verbose=verbose)

            # Decide target key
            if s3_uri:
                new_uri = s3_uri
            else:
                if exists and not cfg["overwrite_upload"]:
                    # versioned upload
                    tag = cfg["version_tag"] or datetime.utcnow().strftime("%Y%m%d%H%M%S")
                    key = s3_key_for(locale, response_hash, prefix, version_tag=tag)
                else:
                    # overwrite or base
                    key = base_key
                new_uri = upload_wav(s3, bucket, key, wav_bytes)

            # DDB update policy
            if cfg["no_update_ddb"]:
                update_ddb_append_alt(table, response_hash, new_uri)
            else:
                update_ddb_replace(table, response_hash, new_uri, prev_uri=prev_uri)

            counters["ok"] += 1
            if verbose:
                print(f"[OK#{counters['ok']}] {response_hash} -> {new_uri}")
            return {"hash": response_hash, "audio_s3_uri": new_uri, "ok": True}
        finally:
            if lock:
                lock.release()
    finally:
        if GLOBAL_SEM:
            GLOBAL_SEM.release()

def main():
    # Load config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.getenv("TTS_CFG", os.path.join(script_dir, "tts_config.json"))
    cfg_file = load_json_config(cfg_path)

    defaults = {
        "region": cfg_file.get("region", os.getenv("AWS_REGION", "ap-northeast-2")),
        "table": cfg_file.get("table", os.getenv("RESPONSE_TABLE", "ResponseAudio")),
        "bucket": cfg_file.get("bucket", os.getenv("RESPONSE_AUDIO_BUCKET", "tts-bucket-250810")),
        "locale": cfg_file.get("locale", os.getenv("RESPONSE_AUDIO_LOCALE", "ko-KR")),
        "prefix": cfg_file.get("prefix", os.getenv("RESPONSE_AUDIO_PREFIX", "")),
        "tts_url": cfg_file.get("tts_url", os.getenv("TTS_URL", "http://localhost:8000")),
        "tts_token": cfg_file.get("tts_token", os.getenv("TTS_TOKEN", "")),
        "voice": cfg_file.get("voice", os.getenv("TTS_VOICE", "Seoyeon")),
        "sample_rate": int(cfg_file.get("sample_rate", os.getenv("TTS_SAMPLE_RATE", 8000))),
        "concurrency": int(cfg_file.get("concurrency", os.getenv("TTS_CONCURRENCY", 1))),
        "global_semaphore": int(cfg_file.get("global_semaphore", os.getenv("TTS_GLOBAL_SEMAPHORE", 1))),
        "per_voice_serial": to_bool(cfg_file.get("per_voice_serial", os.getenv("TTS_PER_VOICE_SERIAL", "true"))),
        "qps": float(cfg_file.get("qps", os.getenv("TTS_QPS", 0.5))),
        "only_missing": to_bool(cfg_file.get("only_missing", os.getenv("TTS_ONLY_MISSING", "true"))),
        "force": to_bool(cfg_file.get("force", os.getenv("TTS_FORCE", "false"))),
        "timeout": int(cfg_file.get("timeout", os.getenv("TTS_HTTP_TIMEOUT", 60))),
        "verbose": to_bool(cfg_file.get("verbose", os.getenv("TTS_VERBOSE", "true"))),
        "limit": cfg_file.get("limit", None),
        "retries": int(cfg_file.get("retries", os.getenv("TTS_RETRIES", 3))),
        "retry_base_delay": float(cfg_file.get("retry_base_delay", os.getenv("TTS_RETRY_BASE_DELAY", 0.8))),
        "retry_max_delay": float(cfg_file.get("retry_max_delay", os.getenv("TTS_RETRY_MAX_DELAY", 5.0))),
        # New regeneration controls
        "regenerate_existing": to_bool(cfg_file.get("regenerate_existing", os.getenv("TTS_REGENERATE_EXISTING", "false"))),
        "overwrite_upload": to_bool(cfg_file.get("overwrite_upload", os.getenv("TTS_OVERWRITE_UPLOAD", "false"))),
        "version_tag": cfg_file.get("version_tag", os.getenv("TTS_VERSION_TAG", "")),
        "no_update_ddb": to_bool(cfg_file.get("no_update_ddb", os.getenv("TTS_NO_UPDATE_DDB", "false"))),
    }

    # Optional CLI overrides
    ap = argparse.ArgumentParser(description="TTS filler with regeneration controls (config-driven)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-only-missing", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int)
    ap.add_argument("--global-semaphore", type=int)
    ap.add_argument("--qps", type=float)
    ap.add_argument("--no-per-voice-serial", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    # new toggles
    ap.add_argument("--regenerate-existing", action="store_true")
    ap.add_argument("--overwrite-upload", action="store_true")
    ap.add_argument("--version-tag")
    ap.add_argument("--no-update-ddb", action="store_true")
    args = ap.parse_args()

    if args.force: defaults["force"] = False
    if args.no_only_missing: defaults["only_missing"] = True
    if args.limit is not None: defaults["limit"] = args.limit
    if args.concurrency is not None: defaults["concurrency"] = args.concurrency
    if args.global_semaphore is not None: defaults["global_semaphore"] = args.global_semaphore
    if args.qps is not None: defaults["qps"] = args.qps
    if args.no_per_voice_serial: defaults["per_voice_serial"] = False
    if args.verbose: defaults["verbose"] = True
    if args.regenerate_existing: defaults["regenerate_existing"] = True
    if args.overwrite_upload: defaults["overwrite_upload"] = True
    if args.version_tag: defaults["version_tag"] = args.version_tag
    if args.no_update_ddb: defaults["no_update_ddb"] = True

    # Print effective config
    show = {k: v for k, v in defaults.items() if k != "tts_token"}
    print("Effective config:", json.dumps(show, ensure_ascii=False, indent=2))

    # Init AWS
    ddb = boto3.resource("dynamodb", region_name=defaults["region"])
    s3 = boto3.client("s3", region_name=defaults["region"])
    table = ddb.Table(defaults["table"])

    # Setup controls
    global GLOBAL_SEM, RATE
    GLOBAL_SEM = threading.Semaphore(int(defaults["global_semaphore"])) if int(defaults["global_semaphore"]) > 0 else None
    RATE = RateLimiter(float(defaults["qps"]))

    # Scan
    items = []
    scan_kwargs = {}
    while True:
        resp = table.scan(**scan_kwargs)
        items.extend(resp.get("Items", []))
        lek = resp.get("LastEvaluatedKey")
        if not lek:
            break
        scan_kwargs["ExclusiveStartKey"] = lek

    if defaults["limit"]:
        items = items[: int(defaults["limit"])]
    total = len(items)
    print(f"Loaded {total} items from {defaults['table']}")

    counters = {"ok": 0, "skipped": 0, "failed": 0}
    results = []

    if int(defaults["concurrency"]) <= 1:
        for it in items:
            results.append(process_item(it, defaults, table, s3, counters, verbose=defaults["verbose"]))
    else:
        with ThreadPoolExecutor(max_workers=int(defaults["concurrency"])) as ex:
            futmap = {ex.submit(process_item, it, defaults, table, s3, counters, defaults["verbose"]): it for it in items}
            for fut in as_completed(futmap):
                results.append(fut.result())

    print("\n=== SUMMARY ===")
    print(f"Total:   {total}")
    print(f"OK:      {counters['ok']}")
    print(f"Skipped: {counters['skipped']}")
    print(f"Failed:  {counters['failed']}")

if __name__ == "__main__":
    main()
