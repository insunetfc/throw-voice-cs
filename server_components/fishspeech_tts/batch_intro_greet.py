#!/usr/bin/env python3
import os, re, csv, sys, json, time, uuid, argparse
import urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

# ---------- Config (env or CLI) ----------
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
TTS_BUCKET = os.getenv("TTS_BUCKET", "tts-bucket-250810")
TTS_URL = os.getenv("TTS_URL", "https://ef85da6954e5.ngrok-free.app")
TTS_TOKEN = os.getenv("TTS_TOKEN", "")
DDB_TABLE = os.getenv("CALLER_TABLE", "PhoneIntro")
KEY_PREFIX_GREETING = os.getenv("KEY_PREFIX_GREETING", "greetings/intro")
FULL_SAMPLE_RATE = 8000  # matches Lambda
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1"))

# ---------- Helpers copied from Lambda semantics ----------
def _normalize_phone(e164_or_raw: str) -> str:
    if not e164_or_raw:
        return ""
    s = re.sub(r"[^\d+]", "", e164_or_raw)
    if not s.startswith("+") and s.startswith("0"):
        s = "+82" + s[1:]
    return s  # Lambda behavior: E.164-ish (+82...)  # :contentReference[oaicite:7]{index=7}

def _format_addressee(display_name: str | None) -> str:
    n = (display_name or "").strip()
    if not n or n == "고객님":
        return "고객님"
    n = re.sub(r"\s+", "", n)
    if n.endswith(("고객님", "님")):
        return n
    return f"{n}고객님"  # :contentReference[oaicite:8]{index=8}

def _make_intro_text(display_name: str) -> str:
    addressee = _format_addressee(display_name)
    return f"(friendly) 안녕하세요 {addressee}. 반갑습니다."  # :contentReference[oaicite:9]{index=9}

def _s3_regional_url(bucket: str, key: str, region: str = AWS_REGION) -> str:
    host = f"s3.{region}.amazonaws.com.cn" if region.startswith("cn-") else f"s3.{region}.amazonaws.com"
    return f"https://{bucket}.{host}/{key.lstrip('/')}"  # :contentReference[oaicite:10]{index=10}

def _tts_short_to_s3(text: str, key_hint_prefix: str = KEY_PREFIX_GREETING):
    payload = {"text": text, "sample_rate": FULL_SAMPLE_RATE, "key_prefix": key_hint_prefix}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{TTS_URL.rstrip('/')}/synthesize",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {TTS_TOKEN}"},
    )  # :contentReference[oaicite:11]{index=11}

    audio_url = ""
    result = {}
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            result = json.loads(resp.read().decode())
            audio_url = result.get("url") or ""
            if not audio_url:
                b = result.get("bucket")
                k = result.get("key")
                if b and k:
                    audio_url = _s3_regional_url(b, k, AWS_REGION)
            else:
                # Canonicalize regional host if it looks like us-east-1 style
                m = re.match(r"https://([^./]+)\.s3\.amazonaws\.com/(.+)", audio_url)
                if m:
                    b, k = m.group(1), m.group(2)
                    audio_url = _s3_regional_url(b, k, AWS_REGION)
    except Exception as e:
        print(json.dumps({"msg": "urllib synth failed", "error": str(e)}))  # :contentReference[oaicite:12]{index=12}

    if not audio_url:
        # Fallback: construct a deterministic S3 URL path (service is expected to upload itself)
        key = f"{key_hint_prefix}/{uuid.uuid4().hex}.wav"
        audio_url = _s3_regional_url(TTS_BUCKET, key, AWS_REGION)
        print(json.dumps({"msg": "fallback_url_used", "key": key}))  # :contentReference[oaicite:13]{index=13}
    else:
        key = result.get("key") or f"{key_hint_prefix}/{uuid.uuid4().hex}.wav"

    return audio_url, key

def _ddb_update(ddb_table, phone: str, display_name: str, audio_key: str):
    # mirrors: SET display_name, greeting_audio_s3, updated_at  # :contentReference[oaicite:14]{index=14}
    ddb_table.update_item(
        Key={"phone_number": phone},
        UpdateExpression="SET display_name=:n, greeting_audio_s3=:g, updated_at=:t",
        ExpressionAttributeValues={":n": display_name, ":g": f"{TTS_BUCKET}/{audio_key}", ":t": int(time.time())},
    )

# ---------- Batch worker ----------
def process_row(ddb_table, phone_raw, name_override, force_refresh=False, allow_same_name_cache=True):
    phone = _normalize_phone(phone_raw)
    if not phone:
        return {"phone": phone_raw, "name": name_override or "", "status": "skip_no_phone"}

    # Try cache
    display_name = None
    cached_audio = None
    try:
        resp = ddb_table.get_item(Key={"phone_number": phone}, ConsistentRead=False)
        item = resp.get("Item") or {}
        display_name = item.get("display_name")
        cached_audio = item.get("greeting_audio_s3")
    except Exception as e:
        print(json.dumps({"msg": "ddb_get_failed", "phone": phone, "error": str(e)}))

    # If we provided a name_override and it's identical to stored name, treat as "no override"
    same_name = (name_override is None) or (display_name is not None and name_override == display_name)
    effective_override_present = (name_override is not None) and not same_name

    # Lambda cache gate: cached_audio and not force_refresh and not override_name  # :contentReference[oaicite:15]{index=15}
    if cached_audio and (not force_refresh) and (not effective_override_present):
        try:
            if cached_audio.startswith("s3://"):
                parsed = urllib.parse.urlparse(cached_audio)
                bucket, key = parsed.netloc, parsed.path.lstrip("/")
            else:
                bucket, key = cached_audio.split("/", 1)
            regional_url = _s3_regional_url(bucket, key, AWS_REGION)
            return {"phone": phone, "name": display_name or name_override or "고객님", "audio_url": regional_url, "cache_hit": True, "status": "ok"}
        except Exception as e:
            print(json.dumps({"msg": "cache_url_failed", "phone": phone, "error": str(e)}))

    # No hit → generate
    final_name = name_override or display_name or "고객님"
    text = _make_intro_text(final_name)
    audio_url, audio_key = _tts_short_to_s3(text, key_hint_prefix=KEY_PREFIX_GREETING)
    try:
        _ddb_update(ddb_table, phone, final_name, audio_key)
    except Exception as e:
        print(json.dumps({"msg": "db_update_failed", "phone": phone, "error": str(e)}))

    return {"phone": phone, "name": final_name, "audio_url": audio_url, "cache_hit": False, "status": "ok"}

# ---------- CLI ----------
def main():
    global AWS_REGION, DDB_TABLE, TTS_BUCKET, TTS_URL, TTS_TOKEN, KEY_PREFIX_GREETING
    parser = argparse.ArgumentParser(description="Batch intro_greet TTS generator")
    parser.add_argument("--csv", default="/home/work/VALL-E/fishspeech/fish-speech/PhoneNumbers.csv", help="Path to CSV with columns: phone_number,display_name")
    parser.add_argument("--region", default="ap-northeast-2")
    parser.add_argument("--ddb-table", default=DDB_TABLE)
    parser.add_argument("--bucket", default=TTS_BUCKET)
    parser.add_argument("--tts-url", default=TTS_URL)
    parser.add_argument("--tts-token", default=TTS_TOKEN)
    parser.add_argument("--key-prefix", default=KEY_PREFIX_GREETING)
    parser.add_argument("--force-refresh", action="store_true", help="Force re-TTS even if cached")
    parser.add_argument("--allow-same-name-cache", action="store_true", help="If override == stored name, allow cache hit")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--out", default="batch_intro_results.csv")
    args = parser.parse_args()

    # propagate config (so helpers use them)    
    AWS_REGION = args.region
    DDB_TABLE = args.ddb_table
    TTS_BUCKET = args.bucket
    TTS_URL = args.tts_url
    TTS_TOKEN = args.tts_token
    KEY_PREFIX_GREETING = args.key_prefix

    # AWS clients
    ddb = boto3.resource("dynamodb", region_name=AWS_REGION)
    table = ddb.Table(DDB_TABLE)

    # Read CSV
    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Accept flexible headers
        for r in reader:
            phone = r.get("phone_number") or r.get("phone") or r.get("number") or ""
            name  = r.get("display_name") or r.get("name") or r.get("Name") or None
            rows.append((phone, name))

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(process_row, table, p, n, args.force_refresh, args.allow_same_name_cache) for (p, n) in rows]
        for fut in as_completed(futures):
            results.append(fut.result())

    # Write output CSV
    fieldnames = ["phone", "name", "audio_url", "cache_hit", "status"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # Also print a compact summary
    hit = sum(1 for r in results if r.get("cache_hit"))
    miss = sum(1 for r in results if r.get("status") == "ok" and not r.get("cache_hit"))
    print(json.dumps({
        "total": len(results), "cache_hits": hit, "cache_misses": miss,
        "out_csv": os.path.abspath(args.out)
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
