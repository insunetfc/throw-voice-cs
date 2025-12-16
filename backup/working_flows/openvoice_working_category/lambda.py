# lambda_connect_tts.py
# Single-Lambda router for Amazon Connect:
#   action=start  -> kick off async TTS, return job_bucket/job_key
#   action=check  -> poll S3 for object; when ready, return presigned URL
#   action=worker -> (self-invoked) do slow TTS + upload to the exact key
# Uses urllib.request (no 'requests'), structured logs, and your existing helpers.

import os, json, uuid, time, logging, re
import os, botocore
import re
import json
import uuid
import base64
import logging
import urllib.request
import urllib.error
from typing import Tuple, Optional
import time
from urllib.parse import urlparse, unquote
import boto3
from botocore.exceptions import ClientError, BotoCoreError
import hashlib, unicodedata, re, json
import re
from typing import Optional, Dict, Any

CACHE_TABLE = os.getenv("CACHE_TABLE", "ConnectPromptCache")
# ========= Env vars =========
CONNECT_REGION = os.getenv("CONNECT_REGION", "ap-northeast-2")
CONNECT_INSTANCE_ID = os.getenv("CONNECT_INSTANCE_ID", "5b83741e-7823-4d70-952a-519d1ac05e63")
PROMPT_NAME_PREFIX = os.getenv("PROMPT_NAME_PREFIX", "dyn-tts-")

connect = boto3.client("connect", region_name=CONNECT_REGION)

FILLER_BUCKET = os.getenv("FILLER_BUCKET", os.getenv("TTS_BUCKET", "tts-bucket-250810"))  # bucket where filler WAVs live
FILLER_REGION = os.getenv("FILLER_REGION", os.getenv("AWS_REGION", "ap-northeast-2"))
FILLER_PREFIX = os.getenv("FILLER_PREFIX", "")  # e.g., "fillers/" or "" if categories are at root
FULL_SAMPLE_RATE = int(os.getenv("FULL_SAMPLE_RATE", "8000"))
SYNTH_HTTP_TIMEOUT_SEC = int(os.getenv("SYNTH_HTTP_TIMEOUT_SEC", "45"))
DEFAULT_FILLER_CATEGORY = "시간벌기형" 
CHAT_URL   = os.getenv("CHAT_URL", "http://15.165.60.45:5000/chat")  # optional
CHAT_TOKEN = os.getenv("CHAT_TOKEN", "")  # optional bearer
TTS_URL    = os.getenv("TTS_URL", "https://honest-trivially-buffalo.ngrok-free.app/synthesize")
TTS_TOKEN  = os.getenv("TTS_TOKEN", "")  # optional bearer
COMPANY_BUCKET  = os.getenv("COMPANY_BUCKET", "tts-bucket-250810")
COMPANY_REGION  = os.getenv("COMPANY_BUCKET_REGION", "ap-northeast-2")
KEY_PREFIX      = os.getenv("KEY_PREFIX", "connect/sessions")
TTS_BUCKET = COMPANY_BUCKET
S3_REGION = os.getenv("S3_REGION", "ap-northeast-2")
USE_PRESIGN = os.getenv("USE_PRESIGN", "1") == "1"   # default: presign URLs
_s3 = boto3.client("s3", region_name=S3_REGION)
FORCE_REUPLOAD  = os.getenv("FORCE_REUPLOAD", "0") == "1"
BATCH = int(os.getenv("STREAM_BATCH", "3"))
CHAT_MODE = "echo"
PREVIEW_COPY = 1
TTS_TEMP = 0.7,
TTS_TOP_P = 0.95,
TTS_REP = 1.0
chunk_length = 64
DISABLE_DDB = '1'

KYES   = r"(네|예|맞(아|습니다)|그렇(습)?니다|좋습니다)"
KASK   = r"(왜|무엇|뭐(야|예요)|어떻게|언제|어디|얼마|가능|되는지|설명|자세히)"
KCONFQ = r"(맞(나요|습니까)|괜찮(나요|습니까)|되(나요|겠습니까)|이거(로)? (할까요|진행할까요)|확인(해)?주시겠어요)"
KTHANK = r"(감사|고맙)"
KFRUS  = r"(느리|답답|짜증|문제|안돼|안 되|에러|오류|힘들|헷갈|복잡)"

import json, random

def _state_get(event):
    """Extract per-call filler state (JSON) from Connect attributes."""
    params = (event.get("Details", {}) or {}).get("Parameters", {}) or {}
    st_raw = params.get("FillerState") or params.get("filler_state")
    try:
        return json.loads(st_raw) if st_raw else {}
    except Exception:
        return {}

def _state_put(resp, state_dict):
    """Attach updated state to Lambda response so Connect can SetAttribute."""
    # Keep both cases for convenience (camel + snake)
    resp["FillerState"] = json.dumps(state_dict, ensure_ascii=False)
    resp["filler_state"] = resp["FillerState"]
    return resp

def _pick_index_no_repeat(num_items, used_list):
    """
    Choose an index not in used_list. If exhausted, reset.
    Returns: idx, new_used_list
    """
    used = list(used_list or [])
    all_idx = list(range(num_items))
    unused = [i for i in all_idx if i not in used]
    if not unused:
        # reset when we've used everything
        used = []
        unused = all_idx
    idx = random.choice(unused)
    used.append(idx)
    return idx, used

def _choose_key_no_repeat(category_slug, man, state, prefer_engine=None, prefer_tone=None):
    items = list(man.get(category_slug, []))
    if not items:
        return None, state

    # Optional filtering by engine/tone (keep if you already had this)
    if prefer_engine or prefer_tone:
        filt = []
        for rel in items:
            ok = True
            if prefer_engine and not rel.startswith(f"{prefer_engine}/"):
                ok = False
            if prefer_tone and (f"/{prefer_tone}/" not in rel):
                ok = False
            if ok: filt.append(rel)
        if filt: items = filt

    # Fetch used indices for this category from state
    used_map = state.get("used", {})
    used_list = used_map.get(category_slug, [])

    idx, new_used = _pick_index_no_repeat(len(items), used_list)
    rel = items[idx]

    # persist back
    used_map[category_slug] = new_used
    state["used"] = used_map
    return f"fillers/{category_slug}/{rel}", state


def _choose_category(ctx: Dict[str, Any]) -> Optional[str]:
    """
    Returns one of {'시간벌기형','확인','설명','공감'} or None.
    ctx keys we read if present:
      user_text, asr_confidence, waiting_ms, about_to_explain,
      asked_confirmation, slots_complete, last_used_category,
      cooldown_on_repeat (bool), barge_in_happened, sentiment_score (-1..1)
    """
    t = (ctx.get("user_text") or "").strip()
    asr = float(ctx.get("asr_confidence") or 1.0)
    wait = int(ctx.get("waiting_ms") or 0)
    about_to_explain = bool(ctx.get("about_to_explain"))
    asked_conf = bool(ctx.get("asked_confirmation"))
    slots_complete = bool(ctx.get("slots_complete"))
    barged = bool(ctx.get("barge_in_happened"))
    sentiment = ctx.get("sentiment_score")
    last_cat = ctx.get("last_used_category")
    avoid_repeat = bool(ctx.get("cooldown_on_repeat", True))

    def ok(cat):
        return (cat != last_cat) if (avoid_repeat and last_cat) else True

    # 1) Stall
    if barged or wait >= 500 or asr < 0.80:
        return "시간벌기형" if ok("시간벌기형") else None

    # 2) Confirm
    if asked_conf or slots_complete or re.search(KCONFQ, t):
        return "확인" if ok("확인") else None
    if re.fullmatch(rf"{KYES}[.!…]*", t):
        return "확인" if ok("확인") else None

    # 3) Explain
    if about_to_explain or re.search(KASK, t):
        return "설명" if ok("설명") else None

    # 4) Empathy
    if (sentiment is not None and sentiment < -0.2) or re.search(KFRUS, t) or re.search(KTHANK, t):
        return "공감" if ok("공감") else None

    return None

def _ctx_from_event(event: dict) -> Dict[str, Any]:
    p = (event.get("Details", {}) or {}).get("Parameters", {}) or {}
    return {
        "user_text": p.get("user_text") or p.get("lex_input_transcript") or p.get("text") or "",
        "asr_confidence": float(p.get("asr_confidence") or 1.0),
        "waiting_ms": int(p.get("waiting_ms") or 0),
        "about_to_explain": str(p.get("about_to_explain") or "false").lower() in ("1","true","y","yes"),
        "asked_confirmation": str(p.get("asked_confirmation") or "false").lower() in ("1","true","y","yes"),
        "slots_complete": str(p.get("slots_complete") or "false").lower() in ("1","true","y","yes"),
        "barge_in_happened": str(p.get("barge_in_happened") or "false").lower() in ("1","true","y","yes"),
        "sentiment_score": float(p["sentiment_score"]) if "sentiment_score" in p else None,
        "last_used_category": p.get("last_used_category"),
        "cooldown_on_repeat": str(p.get("cooldown_on_repeat") or "true").lower() in ("1","true","y","yes"),
    }


if DISABLE_DDB:
    def ddb_put_pending_if_absent(*args, **kwargs): 
        return False
    def ddb_get(*args, **kwargs): 
        return None
    def ddb_put_pending(*args, **kwargs): 
        return None
    def ddb_mark_ready(*args, **kwargs): 
        return None
    
    # Also stub the DDB resource/table to prevent any accidental access
    ddb = None
    
    print(f"[INIT] DynamoDB DISABLED - using S3-only mode")
else:
    # Only create DDB resources when enabled
    CACHE_TABLE = os.getenv("CACHE_TABLE", "ConnectPromptCache")
    COMPANY_REGION = os.getenv("COMPANY_BUCKET_REGION", "ap-northeast-2")
    ddb = boto3.resource("dynamodb", region_name=COMPANY_REGION).Table(CACHE_TABLE)
    print(f"[INIT] DynamoDB ENABLED - using table {CACHE_TABLE}")

ASSUME_ROLE_ARN = os.getenv("ASSUME_ROLE_ARN", "")
ASSUME_ROLE_EXTERNAL_ID = os.getenv("ASSUME_ROLE_EXTERNAL_ID", "")
prefetch = int(os.getenv("BATCH_LIMIT", "1"))

import os, json, urllib.request, urllib.error

TTS_BASE = os.environ.get("TTS_BASE_URL")  # e.g., http://tts.internal:8000

def _ping_warmup():
    if not TTS_BASE:
        return
    url = f"{TTS_BASE.rstrip('/')}/synthesize/warmup"
    data = json.dumps({}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3) as _:
            pass
    except Exception:
        # Don't crash cold start; just log if you want
        pass

_ping_warmup()

import subprocess, tempfile

def presign(bucket, key, ttl=900):
    s3 = boto3.client("s3", region_name="ap-northeast-2")
    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=ttl,
    )

def s3_url(bucket: str, key: str, expires: int = 300) -> str:
    """Return a URL Connect can fetch (presigned by default)."""
    if USE_PRESIGN:
        return _s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires,
        )
    return f"https://{bucket}.s3.{S3_REGION}.amazonaws.com/{key}"

def make_pcm16_preview_from_ulaw_wav(data: bytes) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
        tmp_in.write(data)
        tmp_in.flush()
        tmp_out = tmp_in.name + "_pcm16.wav"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-y", "-i", tmp_in.name,
                "-ar", "8000", "-ac", "1", "-c:a", "pcm_s16le",
                tmp_out
            ],
            check=True
        )
        with open(tmp_out, "rb") as f:
            return f.read()



# Use ONE env var for self-invoke
ASYNC_FUNCTION_NAME = os.getenv("ASYNC_FUNCTION_NAME", "InvokeBotLambda")  # set to THIS Lambda's function name
BYPASS_CHAT = 0
ddb = boto3.resource("dynamodb", region_name=COMPANY_REGION).Table(CACHE_TABLE)
WARMED = False
_PART_RE = re.compile(r"/?part(\d+)\.wav$")
filler_text = "NIPA 클라우드는 일정 시간 미사용 시 세션이 재시작되므로, 전체 과정을 2분 이내에 재실행할 수 있는 스크립트를 작성하여 재사용성을 확보했습니다. 또한 전화 통화에서는 FishSpeech 시스템이 사용자 발화를 성공적으로 인식하고 챗봇 응답을 받은 뒤 오디오를 생성하여 S3에 업로드하는 데 성공했습니다. 다만, 오디오 재생에서 권한 및 형식 문제로 인해 일부 문제가 발생하였으며, 현재 이를 해결하기 위해 디버깅 중입니다. 흐름이 정상적으로 동작하도록 마무리하면 기본 시스템 프로토타입이 완성될 예정입니다."

# --- Warm-up guards (module scope) ---
_IS_WARM = False
_WARM_TS = 0.0
PRESERVE_ORIGINAL = False

import random

FILLER_CATS = {"확인":5, "설명":5, "공감":5, "시간벌기형":5}


def _filler_key_url(category: str | None = None, index: int | None = None):
    cats = list(FILLER_CATS.keys())
    cat = category if category in FILLER_CATS else random.choice(cats)
    n = FILLER_CATS[cat]
    idx = index if (isinstance(index, int) and 1 <= index <= n) else random.randint(1, n)
    
    key = f"{(FILLER_PREFIX + '/') if FILLER_PREFIX and not FILLER_PREFIX.endswith('/') else FILLER_PREFIX}{cat}/{idx:02d}.wav"
    
    # Use presigned URL instead of direct S3 URL
    url = _s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": FILLER_BUCKET, "Key": key},
        ExpiresIn=3600
    )
    
    return cat, idx, key, url

def _tts_full_with_filler(event):
    # 1) Get input text
    params = (event.get("Details", {}) or {}).get("Parameters", {}) or {}
    text = (params.get("text") or event.get("text") or _extract_user_speech_input(event) or "").strip()
    if not text:
        text = " "
    sr = int(params.get("sample_rate") or FULL_SAMPLE_RATE)

    # 2) Compute cache key
    pk = make_pk(text=text, sr=sr)

    if not DISABLE_DDB:
        try:
            ddb_put_pending_if_absent(pk, bucket=os.getenv("TTS_BUCKET",""), key="", input_text=text)
        except Exception:
            pass

    # Choose a deterministic final key so the checker can HEAD it later
    job_id = uuid.uuid4().hex
    target_key = f"{KEY_PREFIX.rstrip('/')}/full/{pk}.wav"
    job_bucket = COMPANY_BUCKET

    # 4) Fire-and-forget worker to synthesize full audio
    payload = {
        "action": "tts_full_worker",
        "text": text,
        "pk": pk,
        "sample_rate": sr,
        "target_key": target_key,
        "job_id": job_id,
        "job_bucket": job_bucket
    }
    # FIX #2: use the client you actually created
    lambda_client.invoke(
        FunctionName=os.environ["AWS_LAMBDA_FUNCTION_NAME"],
        InvocationType="Event",
        Payload=json.dumps(payload).encode("utf-8"),
    )

    # 5) Pick a filler and return it immediately
    cat = params.get("filler_category")
    idx = params.get("filler_index")
    _, _, filler_key, filler_url = _filler_key_url(cat, int(idx) if idx else None)

    attrs = {
        "ready": "true",
        "AudioS3Url0": filler_url,
        "AudioS3UrlCount": "1",
        "BatchCount": "1",
        "HasMore": "true",
        "NextIndexOut": "0",
        "pk": pk,
        "mode": "full_with_filler",
        "job_bucket": job_bucket,
        # FIX #3: return target_key, not undefined job_key
        "job_key": target_key
    }
    return {"setAttributes": {k: (v if isinstance(v, str) else str(v)) for k, v in attrs.items()}}


def _tts_full_worker(event):
    try:
        text = (event.get("text") or "").strip()
        pk = event.get("pk") or ""
        sr = int(event.get("sample_rate") or FULL_SAMPLE_RATE)
        target_key = event.get("target_key")  # may be None if older caller
        job_bucket = event.get("job_bucket") or COMPANY_BUCKET

        res = _http_post_json(
            TTS_URL.rstrip("/"),  # should point to .../synthesize
            {"text": text, "sample_rate": sr, "key_prefix": KEY_PREFIX},
            token=TTS_TOKEN,
            timeout=SYNTH_HTTP_TIMEOUT_SEC,
        )
        audio_url = res.get("url") or res.get("audio_url") or res.get("s3_url")
        if not audio_url:
            _log("tts_full_worker: missing audio URL", resp=list(res.keys()))
            return {"ok": False}

        # Ensure Connect-playable audio and force our final location if target_key provided
        presigned_url, final_key = _ensure_connect_playable(audio_url, target_key=target_key)
        bucket = job_bucket
        key = final_key or target_key

        # Optionally keep prompt parity if you use Connect prompts elsewhere:
        # prompt_id, prompt_arn = _ensure_connect_prompt_for_key(bucket, key, "full")

        ddb_mark_ready(pk, bucket=bucket, key=key, final_text=res.get("text",""), prompt_arn="")

        return {"ok": True, "bucket": bucket, "key": key}
    except Exception as e:
        _log("tts_full_worker failed", error=str(e))
        return {"ok": False, "error": str(e)}

from botocore.exceptions import ClientError

def _check_full(event):
    params = (event.get("Details", {}) or {}).get("Parameters", {}) or {}

    # ---------- S3-ONLY MODE ----------
    if DISABLE_DDB:
        bucket = params.get("job_bucket") or COMPANY_BUCKET
        key    = params.get("job_key") or ""
        if not key:
            return {"setAttributes": {"ready":"false", "Error":"missing_job_key"}}

        try:
            _s3.head_object(Bucket=bucket, Key=key)
            url = f"https://{bucket}.s3.{S3_REGION}.amazonaws.com/{key}"
            return {"setAttributes": {
                "ready":"true",
                "AudioS3Url0": url,
                "AudioS3UrlCount": "1",
                "BatchCount": "1",
                "HasMore": "false",
                "job_bucket": bucket,
                "job_key": key
            }}
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("404","NoSuchKey","NotFound"):
                return {"setAttributes": {
                    "ready":"false",
                    "job_bucket": bucket,
                    "job_key": key
                }}
            # Unexpected S3 error
            return {"setAttributes": {
                "ready":"false",
                "Error": f"s3_head_error:{code}",
                "job_bucket": bucket,
                "job_key": key
            }}

    # ---------- DDB MODE (only when enabled) ----------
    # If you re-enable DDB later, you can call your existing DDB-based check here.
    return _check(event)

def _post_with_retry(url, payload, token=None, timeout=6, backoff=0.3, tries=2):
    # Use your existing _http_post_json if you have it; otherwise:
    for i in range(tries):
        try:
            return _http_post_json(url, payload, token=token, timeout=timeout)
        except Exception:
            if i + 1 == tries:
                raise
            time.sleep(backoff)

def _warm_once(force: bool = False) -> dict:
    """
    Warm both: (a) this Lambda container imports/clients, (b) your TTS engine.
    Returns a tiny attribute dict for logging/visibility; caller can ignore it.
    """
    global _IS_WARM, _WARM_TS

    # Skip if recently warmed (default TTL 10 min) unless forced
    ttl = int(os.getenv("WARM_TTL_SEC", "600"))
    now = time.time()
    if _IS_WARM and not force and (now - _WARM_TS) < ttl:
        return {"Warmed": "skip", "AgeSec": str(int(now - _WARM_TS))}

    # 1) Ensure clients are created at module scope elsewhere (boto3, http Session, etc.)
    # 2) Warm the TTS engine with a tiny request (no need to wait for audio)
    warm_text = os.getenv("WARMUP_TEXT", "ping")
    start_url = f"{TTS_URL.rstrip('/').replace('synthesize', 'synthesize_stream_start')}"
    payload   = {"text": warm_text, "sample_rate": 8000}
    # If your server supports a warmup hint, add it (ignored otherwise):
    payload["warmup"] = True

    try:
        _ = _post_with_retry(
            start_url, payload, token=TTS_TOKEN,
            timeout=int(os.getenv("START_HTTP_TIMEOUT_SEC", "8")),
            backoff=0.3, tries=2
        )
    except Exception as e:
        # Log and move on; we don't want to block the call
        print(f"[warmup] start call failed: {e}")

    _IS_WARM = True
    _WARM_TS = now
    attrs = {"Warmed": "true", "AgeSec": "0"}
    return {"setAttributes": {k: str(v) for k, v in attrs.items()}}

from botocore.exceptions import ClientError

def _head_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=COMPANY_BUCKET, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        # 404 / NoSuchKey -> not ready yet
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise  # any other error should bubble up

def _build_batch(
    job_id: str,
    next_idx: int,
    limit: int = 1,
    include_prompts: bool = False,
    bucket: Optional[str] = None,
    prompt_name_prefix: Optional[str] = None,
) -> dict:
    bucket = bucket or (globals().get("COMPANY_BUCKET") or globals().get("TTS_BUCKET"))
    prefix_for_name = prompt_name_prefix or job_id

    keys = _list_parts_from_s3(job_id)  # ["<job_id>/part0.wav", ..., maybe "final.wav"]
    n = len(keys)
    has_final = bool(keys) and keys[-1].endswith("final.wav")  # we sort final last

    # 0 parts yet → poll
    if n == 0:
        return {
            "AudioS3UrlCount": "0",
            "BatchCount": "0",
            "HasMore": "true",           # keep polling
            "NextIndexOut": str(next_idx),
        }

    # Caller has consumed all keys currently present
    if next_idx >= n:
        return {
            "AudioS3UrlCount": "0",
            "BatchCount": "0",
            "HasMore": "false" if has_final else "true",  # keep polling until final
            "NextIndexOut": str(next_idx),
        }

    end = min(next_idx + max(1, int(limit)), n)
    batch_keys = keys[next_idx:end]

    resp = {}
    for i, key in enumerate(batch_keys):
        resp[f"AudioS3Url{i}"] = _url_for_key(key)
        if include_prompts:
            try:
                m = _PART_RE.search(key)
                prompt_name = f"{prefix_for_name}-part{m.group(1)}" if m else (
                    f"{prefix_for_name}-final" if key.endswith("final.wav") else f"{prefix_for_name}"
                )
                _, arn = _ensure_connect_prompt_for_key(bucket, key, prompt_name)
                resp[f"PromptARN{i}"] = arn
            except Exception as e:
                _log("Prompt create/get failed", key=key, error=str(e))

    count_str = str(len(batch_keys))
    resp["AudioS3UrlCount"] = count_str
    resp["BatchCount"] = count_str
    resp["NextIndexOut"] = str(end)
    # More if there are unconsumed keys *or* the stream hasn't finished yet
    resp["HasMore"] = "true" if (end < n or not has_final) else "false"

    # stringify
    resp = {k: (v if isinstance(v, str) else str(v)) for k, v in resp.items()}
    _log("Batch response built", job_id=job_id, next_idx=next_idx, end=end,
         has_more=resp["HasMore"], keys=len(batch_keys))
    return resp


def _build_batch_with_prompts(job_id: str, next_index: int):
    """Build batch response with Connect prompt ARNs instead of raw URLs"""
    urls = []
    prompt_arns = []
    i = next_index
    
    _log("Building batch with prompts", job_id=job_id, next_index=next_index)
    
    while len(urls) < BATCH:
        key = f"{job_id}/part{i}.wav"
        if _head_exists(key):
            url = to_regional_url(key)
            urls.append(url)
            
            # Create/get Connect prompt for this audio file
            try:
                prompt_id, prompt_arn = _ensure_connect_prompt_for_key(
                    COMPANY_BUCKET, key, f"{job_id}-part{i}"
                )
                prompt_arns.append(prompt_arn)
                _log("Created prompt for part", key=key, prompt_arn=prompt_arn[:60] + "...")
            except Exception as e:
                _log("Failed to create prompt for part", key=key, error=str(e))
                prompt_arns.append("")
            
            i += 1
        else:
            _log("Part not found", key=key)
            break

    # Check for final.wav if no parts found
    if not urls and _head_exists(f"{job_id}/final.wav"):
        final_key = f"{job_id}/final.wav"
        url = to_regional_url(final_key)
        
        try:
            prompt_id, prompt_arn = _ensure_connect_prompt_for_key(
                COMPANY_BUCKET, final_key, f"{job_id}-final"
            )
            
            _log("Created final prompt", key=final_key, prompt_arn=prompt_arn[:60] + "...")
            
            return {
                "AudioS3Url0": url,
                "PromptARN0": prompt_arn,  # Use this in flow instead
                "BatchCount": "1",
                "AudioS3UrlCount": "1",
                "HasMore": "false",
                "NextIndexOut": str(next_index),
            }
        except Exception as e:
            _log("Failed to create final prompt", key=final_key, error=str(e))
            return {
                "AudioS3Url0": url,
                "BatchCount": "1", 
                "AudioS3UrlCount": "1",
                "HasMore": "false",
                "NextIndexOut": str(next_index),
            }

    batch_count = str(len(urls))
    next_index_out = str(i if urls else next_index)
    
    response = {
        "BatchCount": batch_count,
        "AudioS3UrlCount": batch_count, 
        "HasMore": "true" if urls else "false",
        "NextIndexOut": next_index_out,
    }
    
    # Add both URLs and prompt ARNs
    for j, (url, prompt_arn) in enumerate(zip(urls, prompt_arns)):
        response[f"AudioS3Url{j}"] = url
        if prompt_arn:
            response[f"PromptARN{j}"] = prompt_arn
    
    _log("Batch response built", response_keys=list(response.keys()))
    return response

def _infer_job_id_from_keys(keys):
    # keys look like "<job_id>/part0.wav"
    for k in keys:
        if "/" in k:
            return k.split("/", 1)[0]
    return None

def _list_ready_parts(job_id: str):
    """
    Returns (sorted_part_indices, has_final).
    part indices are ints for partN.wav present; final is boolean for final.wav.
    """
    prefix = f"{job_id}/"
    resp = s3.list_objects_v2(Bucket=COMPANY_BUCKET, Prefix=prefix, MaxKeys=1000) 
    contents = resp.get("Contents", [])
    part_indices = []
    has_final = False
    for obj in contents:
        key = obj["Key"]
        base = key.rsplit("/", 1)[-1]  # "partN.wav" or "final.wav"
        if base.startswith("part") and base.endswith(".wav"):
            n_str = base[len("part"):-len(".wav")]
            if n_str.isdigit():
                part_indices.append(int(n_str))
        elif base == "final.wav":
            has_final = True
    part_indices.sort()
    return part_indices, has_final

def _list_parts_from_s3(job_id: str) -> list[str]:
    """Return all *.wav keys under <job_id>/ sorted by part index; final.wav last."""
    prefix = f"{job_id}/"
    keys: list[str] = []
    token = None
    while True:
        if token:
            resp = _s3.list_objects_v2(Bucket=TTS_BUCKET, Prefix=prefix, ContinuationToken=token)
        else:
            resp = _s3.list_objects_v2(Bucket=TTS_BUCKET, Prefix=prefix)
        for obj in resp.get("Contents", []):
            k = obj["Key"]
            if k.endswith(".wav"):
                keys.append(k)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break

    def sort_key(k: str):
        m = _PART_RE.search(k)
        if m:
            return (0, int(m.group(1)))  # parts in numeric order
        # put final.wav at the very end
        return (1, 10**9 if k.endswith("final.wav") else 10**8)

    keys.sort(key=sort_key)
    return keys

def _url_for_key(key: str, expires: int = 300) -> str:
    """Return HTTPS URL to the object (presigned if needed)."""
    if USE_PRESIGN:
        return _s3.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": TTS_BUCKET,
                "Key": key,
                # helps some players infer type
                "ResponseContentType": "audio/wav",
            },
            ExpiresIn=expires,
        )
    # public-bucket path
    return f"https://{TTS_BUCKET}.s3.{S3_REGION}.amazonaws.com/{key}"


def _diag_url(event):
    url = (event.get("url") or
           event.get("Details", {}).get("Parameters", {}).get("url") or
           event.get("Details", {}).get("Parameters", {}).get("PromptLocation"))
    if not url:
        return {"setAttributes": {"DiagError": "no_url"}}

    p = urlparse(url)
    # bucket.s3.region.amazonaws.com
    bucket = p.netloc.split(".")[0]
    key = unquote(p.path.lstrip("/"))
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
        return {"setAttributes": {
            "DiagContentType": head.get("ContentType", ""),
            "DiagSSE": head.get("ServerSideEncryption", ""),
            "DiagKmsKey": head.get("SSEKMSKeyId", ""),
            "DiagLen": str(head.get("ContentLength", 0)),
        }}
    except Exception as e:
        return {"setAttributes": {"DiagError": str(e)}}

def ddb_put_pending_if_absent(pk: str, bucket: str, key: str, input_text: str, ttl_days=14) -> bool:
    """
    Try to create a 'pending' item only if it doesn't exist yet.
    Returns True if we created it, False if it already existed.
    """
    now = int(time.time())
    try:
        ddb.put_item(
            Item={
                "pk": pk,
                "state": "pending",
                "bucket": bucket,
                "key": key,
                "input_text": _canon(input_text),
                "updated_at": now,
                "expires_at": now + ttl_days*24*3600,
            },
            ConditionExpression="attribute_not_exists(pk)"
        )
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            return False
        raise


def _canon(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text))
    t = re.sub(r'^\.+', '', t).strip()
    t = re.sub(r'\s+', ' ', t)
    return t

def make_pk(text: str, *, voice="Jihye", sr=8000, lang="ko-KR", chat=False) -> str:
    payload = {"t": _canon(text), "v": voice, "sr": sr, "lang": lang, "chat": bool(chat)}
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

def ddb_get(pk: str):
    r = ddb.get_item(Key={"pk": pk})
    return r.get("Item")

def ddb_put_pending(pk: str, bucket: str, key: str, input_text: str, ttl_days=14):
    ddb.put_item(Item={
        "pk": pk,
        "state": "pending",
        "bucket": bucket,
        "key": key,
        "input_text": _canon(input_text),
        "updated_at": int(time.time()),
        "expires_at": int(time.time()) + ttl_days*24*3600
    })

def ddb_mark_ready(pk: str, *, prompt_arn: str, final_text: str, bucket: str, key: str):
    ddb.update_item(
        Key={"pk": pk},
        UpdateExpression=(
            "SET #s = :s, "
            "#p = :p, "
            "#b = :b, "
            "#k = :k, "
            "#f = :f, "
            "#u = :u"
        ),
        ExpressionAttributeNames={
            "#s": "state",
            "#p": "prompt_arn",
            "#b": "bucket",     # <-- alias reserved word
            "#k": "key",        # <-- 'key' is also safer to alias
            "#f": "final_text",
            "#u": "updated_at",
        },
        ExpressionAttributeValues={
            ":s": "ready",
            ":p": prompt_arn,
            ":b": bucket,
            ":k": key,
            ":f": final_text,
            ":u": int(time.time()),
        },
    )


def canonicalize(text: str) -> str:
    # Unicode normalize (good for ko-KR too), strip leading dots/spaces, collapse whitespace
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r'^\.+', '', t).strip()
    t = re.sub(r'\s+', ' ', t)
    return t

def cache_key(text: str, voice="Jihye", lang="ko-KR", tts_sr=8000, chat_on=False) -> str:
    payload = {"t": canonicalize(text), "v": voice, "l": lang, "sr": tts_sr, "chat": bool(chat_on)}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()  # e.g., '2f6a…'

# ========= S3 readiness helper =========
def wait_s3_ready(bucket: str, key: str, timeout: float = 3.0, interval: float = 0.2) -> bool:
    """Head the exact S3 object until it exists with non-zero length."""
    end = time.time() + timeout
    while time.time() < end:
        try:
            head = _s3.head_object(Bucket=bucket, Key=key)
            if head.get("ContentLength", 0) > 0:
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False

# ========= Logging =========
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='{"level":"%(levelname)s","message":"%(message)s","logger":"%(name)s","aws_request_id":"%(aws_request_id)s"}'
)
logging.getLogger("botocore").setLevel(logging.DEBUG)
logging.getLogger("boto3").setLevel(logging.DEBUG)
logger = logging.getLogger("connect-tts")

class RequestIdFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.aws_request_id = "-"

    def set_request_id(self, req_id: str):
        self.aws_request_id = req_id or "-"

    def filter(self, record: logging.LogRecord) -> bool:
        record.aws_request_id = self.aws_request_id
        return True

_req_filter = RequestIdFilter()
logger.addFilter(_req_filter)

def _log(msg: str, **fields):
    try:
        logger.info(msg + " " + json.dumps(fields, ensure_ascii=False))
    except Exception:
        logger.info(msg)

def _redact(s: Optional[str], keep: int = 8) -> str:
    if not s:
        return ""
    return (s[:keep] + "...") if len(s) > keep else "***"

# ========= AWS clients =========
def _ensure_connect_prompt_for_key(bucket: str, key: str, job_id: str) -> tuple[str, str]:
    """
    Create or update a Connect Prompt pointing at s3://bucket/key.
    Returns (prompt_id, prompt_arn).
    """
    if not CONNECT_INSTANCE_ID:
        raise RuntimeError("CONNECT_INSTANCE_ID env is required to create prompts")

    s3_uri = f"s3://{bucket}/{key}"
    name   = f"{PROMPT_NAME_PREFIX}{job_id}"[:127]

    # Try create; if name already used, update that prompt
    try:
        resp = connect.create_prompt(
            InstanceId=CONNECT_INSTANCE_ID,
            Name=name,
            Description=f"Dynamic TTS {job_id}",
            S3Uri=s3_uri,
        )
        return resp["PromptId"], resp["PromptARN"]
    except connect.exceptions.DuplicateResourceException:
        srch = connect.search_prompts(
            InstanceId=CONNECT_INSTANCE_ID,
            MaxResults=1,
            SearchCriteria={"StringCondition": {"FieldName": "Name", "Value": name, "ComparisonType": "EXACT"}}
        )
        if not srch.get("Prompts"):
            # Rare race: just try create again with a slightly different name
            alt = f"{name}-{uuid.uuid4().hex[:6]}"
            resp = connect.create_prompt(
                InstanceId=CONNECT_INSTANCE_ID,
                Name=alt,
                Description=f"Dynamic TTS {job_id}",
                S3Uri=s3_uri,
            )
            return resp["PromptId"], resp["PromptARN"]

        pid = srch["Prompts"][0]["PromptId"]
        arn = srch["Prompts"][0]["PromptARN"]
        connect.update_prompt(
            InstanceId=CONNECT_INSTANCE_ID,
            PromptId=pid,
            Name=name,
            Description=f"Dynamic TTS {job_id}",
            S3Uri=s3_uri,
        )
        return pid, arn

def _get_prompt(event):
    params = event.get("Details", {}).get("Parameters", {})
    # Either pass pk directly, or compute it from text + settings
    pk = params.get("pk")
    if not pk:
        text = (params.get("user_input") or params.get("text") or "").strip()
        chat_on = params.get("chat_on")
        if chat_on is None:
            chat_on = os.getenv("BYPASS_CHAT","0") != "1"
        else:
            chat_on = str(chat_on).lower() in ("1","true","yes","y")
        # match the same settings used originally
        pk = make_pk(text, sr=8000, lang="ko-KR", voice="Jihye", chat=chat_on)

    item = ddb_get(pk)
    if item and item.get("state") == "ready" and item.get("prompt_arn"):
        return {
            "ready": True,
            "prompt_arn": item["prompt_arn"],
            "setAttributes": {
                "ready": "true",
                "prompt_arn": item["prompt_arn"],
                "final_text": item.get("final_text","")
            }
        }
    return {
        "ready": False,
        "error": "not_ready_or_missing",
        "setAttributes": {"ready":"false","prompt_arn":""}
    }


def _s3_client():
    if ASSUME_ROLE_ARN:
        sts = boto3.client("sts", region_name=COMPANY_REGION)
        params = {"RoleArn": ASSUME_ROLE_ARN, "RoleSessionName": "connect-tts"}
        if ASSUME_ROLE_EXTERNAL_ID:
            params["ExternalId"] = ASSUME_ROLE_EXTERNAL_ID
        creds = sts.assume_role(**params)["Credentials"]
        _log("Assumed role", role=_redact(ASSUME_ROLE_ARN, 12))
        return boto3.client(
            "s3",
            region_name=COMPANY_REGION,
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
        )
    return boto3.client("s3", region_name=COMPANY_REGION)

s3 = _s3_client()
lambda_client = boto3.client("lambda", region_name=os.getenv("AWS_REGION", COMPANY_REGION))

# ========= Utils =========
def _make_key(job_id: Optional[str] = None) -> str:
    return f"{KEY_PREFIX.rstrip('/')}/{(job_id or uuid.uuid4().hex)}.wav"

# ========= Audio/S3 helpers =========
def _http_post_json(url: str, obj: dict, token: str = "", timeout: int = 30) -> dict:
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
            try:
                return json.loads(payload.decode("utf-8"))
            except json.JSONDecodeError:
                return json.loads(payload)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        logger.exception("HTTPError on POST")
        raise RuntimeError(f"POST {url} failed: {e.code} {e.reason} – {body}") from e
    except urllib.error.URLError as e:
        logger.exception("URLError on POST")
        raise RuntimeError(f"POST {url} connection error: {e}") from e

def _http_get_bytes(url: str, timeout: int = 60) -> Tuple[bytes, str]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read()
            ctype = resp.headers.get("Content-Type", "application/octet-stream")
            return content, ctype
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        logger.exception("HTTPError on GET")
        raise RuntimeError(f"GET {url} failed: {e.code} {e.reason} – {body}") from e
    except urllib.error.URLError as e:
        logger.exception("URLError on GET")
        raise RuntimeError(f"GET {url} connection error: {e}") from e

def _ensure_connect_playable(url_or_data: str, target_key: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Process audio URL/data to make it Connect-compatible
    """
    if url_or_data.startswith("data:"):
        m = re.match(r"data:(.*?);base64,(.*)", url_or_data)
        if not m:
            raise ValueError("Bad data URL from TTS")
        content_type = m.group(1) or "audio/wav"
        body = base64.b64decode(m.group(2))
        _log("Handling data URL", content_type=content_type, bytes=len(body))
        return _upload_bytes_to_s3(body, content_type, key=target_key)

    if url_or_data.startswith("http"):
        _log("Handling http(s) audio URL", src=_redact(url_or_data, 40))
        if target_key:  # Always re-upload if target_key specified
            content, ctype = _http_get_bytes(url_or_data)
            return _upload_bytes_to_s3(content, ctype or "audio/wav", key=target_key)
        return url_or_data, None

    raise ValueError("Unsupported audio format (expected data: or http(s) URL)")

def _upload_bytes_to_s3(data: bytes, content_type: str = "audio/wav", key: Optional[str] = None) -> Tuple[str, str]:
    put_key = key or f"{KEY_PREFIX.rstrip('/')}/{uuid.uuid4().hex}.wav"
    extra = {"ContentType": content_type}
    if os.getenv("CROSS_ACCOUNT_ACL", "0") == "1":
        extra["ACL"] = "bucket-owner-full-control"
    _s3.put_object(Bucket=COMPANY_BUCKET, Key=put_key, Body=data, **extra)
    url = _s3.generate_presigned_url(
        "get_object", Params={"Bucket": COMPANY_BUCKET, "Key": put_key}, ExpiresIn=300
    )
    _log("Uploaded audio to S3", bucket=COMPANY_BUCKET, key=put_key, content_type=content_type)

    # ✅ Add preview generation here
    try:
        if os.getenv("PREVIEW_COPY", "0") == "1":
            preview_bytes = make_pcm16_preview_from_ulaw_wav(data)
            preview_key = put_key.replace("/full/", "/preview/").replace("/full_cache/", "/preview/")
            _s3.put_object(
                Bucket=COMPANY_BUCKET,
                Key=preview_key,
                Body=preview_bytes,
                ContentType="audio/wav"
            )
            _log("Preview written", preview_key=preview_key, size=len(preview_bytes))
    except Exception as e:
        _log("Preview generation failed", error=str(e))

    return url, put_key


# ========= Input extraction =========
def _extract_text(event: dict) -> str:
    d = event.get("Details", {}) or {}
    params = d.get("Parameters", {}) or {}
    contact = d.get("ContactData", {}) or {}
    attrs = contact.get("Attributes", {}) or {}

    # ▶ If Slots is a dict, pull a likely slot like "UserInput" (avoid "{...}")
    slots = d.get("Lex", {}).get("Slots", {})
    if isinstance(slots, dict):
        for k in ("UserInput", "user_input", "utterance", "Transcript"):
            v = slots.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

    candidates = [
        params.get("input_text"),
        params.get("user_input"),
        params.get("CustomerInput"),             # ▶ added
        params.get("Details.Parameters.user_input"),
        params.get("Details.Parameters.input_text"),
        attrs.get("user_input"),
        attrs.get("CustomerInput"),
        params.get("lex_input_transcript"),
        d.get("Lex", {}).get("InputTranscript"),
        event.get("InputTranscript"),
        event.get("inputTranscript"),
        params.get("text"),
        event.get("text"),
    ]

    for c in candidates:
        if c is None:
            continue
        s = re.sub(r'^\.+', '', str(c)).strip()
        if s:
            return s
    return ""

# ========= Core TTS step (shared) =========
def _session_id_from_event(event: dict, fallback: str) -> str:
    return (
        event.get("Details", {}).get("ContactData", {}).get("ContactId")
        or event.get("contact_id")
        or fallback
    )

def _get_ready_items(event):
    """Get all items where state='ready' - requires GSI on 'state'"""
    try:
        # This requires a GSI named 'state-index' on the 'state' attribute
        response = ddb.query(
            IndexName='state-index',  # You'll need to create this GSI
            KeyConditionExpression='#state = :ready',
            ExpressionAttributeNames={'#state': 'state'},
            ExpressionAttributeValues={':ready': 'ready'},
            Limit=50  # Adjust as needed
        )
        
        items = response.get('Items', [])
        return {
            "success": True,
            "count": len(items),
            "items": items,
            "setAttributes": {
                "ready_count": str(len(items))
            }
        }
    except Exception as e:
        logger.exception("Failed to query ready items")
        return {
            "success": False,
            "error": str(e),
            "setAttributes": {"ready_count": "0"}
        }

def _scan_ready_items(event):
    """Alternative: Scan for ready items (less efficient but no GSI needed)"""
    try:
        response = ddb.scan(
            FilterExpression='#state = :ready',
            ExpressionAttributeNames={'#state': 'state'},
            ExpressionAttributeValues={':ready': 'ready'},
            Limit=50
        )
        
        items = response.get('Items', [])
        return {
            "success": True,
            "count": len(items),
            "items": items,
            "setAttributes": {
                "ready_count": str(len(items))
            }
        }
    except Exception as e:
        logger.exception("Failed to scan ready items")
        return {
            "success": False,
            "error": str(e),
            "setAttributes": {"ready_count": "0"}
        }

def _check_ready_by_pk(event):
    """Check if a specific item is ready by its pk"""
    params = event.get("Details", {}).get("Parameters", {}) or {}
    pk = params.get("pk")
    
    if not pk:
        return {
            "ready": "false",
            "error": "pk_required",
            "setAttributes": {
                "ready": "false",
                "prompt_arn": "",
                "final_text": "",
                "error": "pk_required"
            }
        }

    _log("Checking ready status by pk", pk=pk[:12] + "...")

    item = ddb_get(pk)
    if not item:
        return {
            "ready": "false",
            "error": "item_not_found",
            "setAttributes": {
                "ready": "false",
                "prompt_arn": "",
                "final_text": "",
                "error": "item_not_found"
            }
        }

    # ✅ Define these from the DDB item before branching
    state      = str(item.get("state", "")).lower()
    prompt_arn = item.get("prompt_arn", "")
    final_text = item.get("final_text", "")
    bucket     = item.get("bucket", COMPANY_BUCKET)
    key        = item.get("key", "")

    if state == "ready":
        return {
            "ready": "true",
            "state": "ready",
            "prompt_arn": prompt_arn,
            "final_text": final_text,
            "bucket": bucket,
            "key": key,
            "pk": pk,
            "setAttributes": {
                "ready": "true",
                "prompt_arn": prompt_arn,
                "final_text": final_text,
                "job_bucket": bucket,
                "job_key": key,
                "key": key,
                "pk": pk
            }
        }

    elif state == "pending":
        return {
            "ready": "false",
            "state": "pending",
            "bucket": bucket,
            "key": key,
            "pk": pk,
            "setAttributes": {
                "ready": "false",
                "prompt_arn": "",
                "final_text": "",
                "job_bucket": bucket,
                "job_key": key,
                "key": key,
                "pk": pk
            }
        }

    else:
        _log("Item in unknown state", pk=pk[:12] + "...", state=item.get("state"))
        return {
            "ready": "false",
            "state": item.get("state", "unknown"),
            "error": f"unknown_state_{item.get('state', 'none')}",
            "setAttributes": {
                "ready": "false",
                "prompt_arn": "",
                "final_text": "",
                "error": f"unknown_state_{item.get('state', 'none')}"
            }
        }

def _chat_and_stream(event):
    """
    ONLY responsible for:
    1. Extract user input
    2. Call chatbot
    3. Start TTS streaming job
    4. Return job metadata (NO audio polling)
    """
    _cancel_previous_job_if_any(event)
    
    user_speech = _extract_user_speech_input(event)
    if not user_speech:
        _log("No user speech input found, using default")
        user_speech = "안녕하세요"
    
    session_id = _session_id_from_event(event, str(uuid.uuid4()))
    
    _log("Chat and stream - starting TTS job only", 
         session_id=session_id[:8],
         user_speech_preview=user_speech[:100])
    
    try:
        # Get chatbot response
        chatbot_response = call_chatbot(user_speech, session_id)
        
        # Start TTS streaming job (NO POLLING HERE)
        start_url = f"{TTS_URL.rstrip('/').replace('synthesize', 'synthesize_stream_start')}"
        res = _http_post_json(
            start_url,
            {"text": chatbot_response, "sample_rate": 8000},
            token=TTS_TOKEN,
            timeout=int(os.getenv("START_HTTP_TIMEOUT_SEC", "6"))
        )

        job_id = res.get("job_id")
        if not job_id:
            return {"setAttributes": {"Error": "no_job_id_from_server", "ready": "false"}}

        # Return ONLY job metadata - no audio polling
        response_attrs = {
            "JobId": job_id,
            "ready": "true",
            "ChatAnswer": chatbot_response,
            "UserInput": user_speech,
            "NextIndex": "0",  # Start from beginning
            "HasMore": "true"  # Assume there will be audio
        }
        
        _log("Chat and stream - TTS job started", 
             job_id=job_id,
             user_input=user_speech[:50],
             bot_response=chatbot_response[:50])
        
        return {"setAttributes": response_attrs}
        
    except Exception as e:
        _log("Chat and stream failed", error=str(e))
        return {"setAttributes": {"Error": f"chat_stream_error: {str(e)}", "ready": "false"}}

def _get_next_batch(event):
    """
    ONLY responsible for:
    1. Get job_id and next_index from Connect
    2. Poll S3 for audio parts
    3. Return batch of audio URLs
    4. Handle streaming completion
    """
    params = event.get("Details", {}).get("Parameters", {}) or {}
    job_id = params.get("JobId") or event.get("JobId")
    next_idx = int(params.get("NextIndex") or event.get("NextIndex") or "0")
    limit = int(params.get("Limit") or os.getenv("STREAM_BATCH", "1"))

    if not job_id:
        return {"setAttributes": {"Error": "Missing JobId", "ready": "false"}}

    _log("Get next batch", 
         job_id=job_id, 
         next_index=next_idx, 
         limit=limit)

    try:
        # Get batch of audio URLs from S3
        # batch_attrs = _build_batch(job_id, next_idx, limit=limit, include_prompts=False)
        batch_attrs = _build_batch(job_id, next_idx, limit=prefetch, include_prompts=False)
        
        # Add job metadata
        batch_attrs["JobId"] = job_id
        
        # Log what we found
        audio_count = int(batch_attrs.get("AudioS3UrlCount", 0))
        has_more = str(batch_attrs.get("HasMore", "false")).lower() == "true"
        
        _log("Get next batch result", 
             job_id=job_id,
             next_index=next_idx,
             audio_count=audio_count,
             has_more=has_more)
        
        # Stringify for Connect
        batch_attrs = {k: (v if isinstance(v, str) else str(v)) for k, v in batch_attrs.items()}
        return {"setAttributes": batch_attrs}
        
    except Exception as e:
        _log("Get next batch failed", job_id=job_id, error=str(e))
        return {"setAttributes": {"Error": f"batch_error: {str(e)}", "ready": "false"}}

def _enhanced_chat_and_stream(event):
    """Enhanced handler for chat_and_stream action with detailed logging"""
    _cancel_previous_job_if_any(event)
    # Extract user speech input with detailed logging
    user_speech = _extract_user_speech_input(event)
    _log("DEBUG: User speech extracted", 
         user_speech=user_speech, 
         length=len(user_speech) if user_speech else 0)
    
    if not user_speech:
        _log("No user speech input found, using default")
        user_speech = "안녕하세요"
    
    # Get session ID  
    session_id = _session_id_from_event(event, str(uuid.uuid4()))
    
    _log("Chat and stream starting", 
         session_id=session_id[:8],
         user_speech_preview=user_speech[:100])
    
    try:
        # Debug: Check chatbot configuration
        _log("DEBUG: Chatbot config", 
             chat_url=_redact(CHAT_URL, 30), 
             bypass_chat=BYPASS_CHAT,
             has_token=bool(CHAT_TOKEN))
        
        # Process through chatbot and get TTS - USE YOUR ORIGINAL WORKING FUNCTION
        chatbot_response = call_chatbot(user_speech, session_id)
        
        _log("DEBUG: Chatbot response", 
             input=user_speech[:50],
             output=chatbot_response[:50],
             same_as_input=(chatbot_response == user_speech))
        
        # If chatbot response is the same as input, something's wrong
        if chatbot_response == user_speech:
            _log("WARNING: Chatbot returned same as input - check BYPASS_CHAT or chatbot URL")
        
        # Start streaming TTS for the chatbot response
        start_url = f"{TTS_URL.rstrip('/').replace('synthesize', 'synthesize_stream_start')}"
        _log("DEBUG: Starting TTS", 
             url=_redact(start_url, 40),
             text_preview=chatbot_response[:100])
        
        res = _http_post_json(
            start_url,
            {"text": chatbot_response, "sample_rate": 8000},
            token=TTS_TOKEN,
            timeout=int(os.getenv("START_HTTP_TIMEOUT_SEC", "6"))
        )

        job_id = res.get("job_id")
        if not job_id:
            return {"setAttributes": {"Error": "no_job_id_from_server", "ready": "false"}}

        _log("DEBUG: TTS job started", job_id=job_id)

        # Quick poll for first part
        deadline = time.time() + float(os.getenv("FIRST_PART_WAIT_SEC", "1.0"))
        interval = float(os.getenv("FIRST_PART_POLL_MS", "100")) / 1000.0

        batch_attrs = {"AudioS3UrlCount": "0", "BatchCount": "0", "HasMore": "true", "NextIndexOut": "0"}
        poll_count = 0
        while time.time() < deadline:
            batch_attrs = _build_batch(job_id, 0, limit=prefetch, include_prompts=False)
            # batch_attrs = _build_batch(job_id, 0, limit=1, include_prompts=False)
            poll_count += 1
            _log("DEBUG: Polling for audio", 
                 attempt=poll_count, 
                 audio_count=batch_attrs.get("AudioS3UrlCount"))
            
            if str(batch_attrs.get("AudioS3UrlCount")) == "1":
                break
            time.sleep(interval)

        if str(batch_attrs.get("AudioS3UrlCount")) != "1":
            part0_key = f"{job_id}/part0.wav"
            if wait_s3_ready(TTS_BUCKET, part0_key, timeout=1.5, interval=0.2):
                region = os.getenv("AWS_REGION", "ap-northeast-2")
                s3_url0 = f"https://{TTS_BUCKET}.s3.{region}.amazonaws.com/{part0_key}"
                batch_attrs = {"BatchCount":"1","AudioS3UrlCount":"1","HasMore":"true","NextIndexOut":"1","AudioS3Url0":s3_url0}

        # Add response metadata
        batch_attrs["JobId"] = job_id
        batch_attrs["ready"] = "true"
        batch_attrs["ChatAnswer"] = chatbot_response
        batch_attrs["UserInput"] = user_speech
        
        # Stringify all values
        batch_attrs = {k: (v if isinstance(v, str) else str(v)) for k, v in batch_attrs.items()}
        
        _log("Chat and stream complete", 
             job_id=job_id,
             user_input=user_speech[:50],
             bot_response=chatbot_response[:50],
             audio_ready=(str(batch_attrs.get("AudioS3UrlCount")) == "1"))
        
        return {"setAttributes": batch_attrs}
        
    except Exception as e:
        _log("Chat and stream failed", error=str(e), error_type=type(e).__name__)
        import traceback
        _log("Full error traceback", traceback=traceback.format_exc())
        return {"setAttributes": {"Error": f"chat_stream_error: {str(e)}", "ready": "false"}}

# Make sure your call_chatbot function is exactly as it was working before:
def call_chatbot(user_input: str, session_id: str = None) -> str:
    """
    Chatbot integration function - EXACTLY as it was working before
    """
    if not CHAT_URL:
        _log("No CHAT_URL configured, returning original input", input=user_input[:50])
        return user_input
    
    if BYPASS_CHAT:
        _log("BYPASS_CHAT is enabled, returning original input")
        return user_input
    
    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Prepare payload matching your chatbot_connect.py format
    payload = {
        "session_id": session_id,
        "question": user_input.strip()
    }
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    # Add authorization if token is provided
    if CHAT_TOKEN:
        headers['Authorization'] = f'Bearer {CHAT_TOKEN}'
    
    try:
        _log("Calling chatbot", 
             url=_redact(CHAT_URL, 30), 
             session_id=session_id[:8], 
             input_length=len(user_input),
             payload_preview=str(payload)[:200])
        
        # Make the HTTP request
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(CHAT_URL, data=data, headers=headers, method="POST")
        
        with urllib.request.urlopen(req, timeout=30) as response:
            response_data = response.read()
            chat_response = json.loads(response_data.decode("utf-8"))
        
        _log("Raw chatbot response", raw_response=str(chat_response)[:500])
        
        # Extract the answer from response
        chatbot_answer = extract_chatbot_answer(chat_response)
        
        if chatbot_answer and chatbot_answer.strip():
            _log("Chatbot response received", 
                 input_preview=user_input[:50], 
                 output_preview=chatbot_answer[:50],
                 changed=(chatbot_answer.strip() != user_input.strip()))
            return chatbot_answer.strip()
        else:
            _log("Empty chatbot response, returning original input")
            return user_input
            
    except Exception as e:
        _log("Chatbot call failed, returning original input", 
             error=str(e), 
             error_type=type(e).__name__)
        return user_input

def extract_chatbot_answer(chat_response) -> str:
    """
    Extract the answer from various chatbot response formats
    """
    if isinstance(chat_response, str):
        return chat_response.strip()
    
    if not isinstance(chat_response, dict):
        return ""
    
    # Try common response keys
    for key in ["answer", "text", "reply", "response", "message", "content", "output"]:
        if key in chat_response and isinstance(chat_response[key], str):
            answer = chat_response[key].strip()
            if answer:
                return answer
    
    # Try nested structures
    if "data" in chat_response:
        data = chat_response["data"]
        if isinstance(data, dict):
            for key in ["answer", "text", "reply", "response"]:
                if key in data and isinstance(data[key], str):
                    answer = data[key].strip()
                    if answer:
                        return answer
    
    return ""

def _extract_user_speech_input(event: dict) -> str:
    """Enhanced function to extract user speech input from Connect/Lex"""
    d = event.get("Details", {}) or {}
    params = d.get("Parameters", {}) or {}
    contact = d.get("ContactData", {}) or {}
    attrs = contact.get("Attributes", {}) or {}
    
    # Handle Lex data properly
    lex_data = d.get("Lex", {})
    
    candidates = [
        # Most reliable for speech input
        lex_data.get("InputTranscript"),
        
        # Parameters from Connect
        params.get("user_input"),
        params.get("text"),
        params.get("Details.Parameters.user_input"),
        params.get("Details.Parameters.text"),
        
        # Contact attributes
        attrs.get("user_input"),
        attrs.get("UserInput"),
        
        # Direct event properties
        event.get("InputTranscript"),
        event.get("text"),
        
        # Fallback to existing _extract_text function
        _extract_text(event)
    ]
    
    for candidate in candidates:
        if candidate is None:
            continue
        
        text = str(candidate).strip()
        text = re.sub(r'^\.+', '', text)  # Remove leading dots
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Skip empty strings and system messages
        if text and len(text) > 0 and not text.startswith("{") and text != "null":
            _log("Found user speech input", source="extracted", length=len(text), preview=text[:100])
            return text
    
    _log("No user speech input found in event")
    return ""

def _safe_get_lex_slot(event: dict, slot_name: str) -> Optional[str]:
    """
    Safely extract Lex slot values
    """
    try:
        lex_data = event.get("Details", {}).get("Lex", {})
        slots = lex_data.get("Slots", {})
        if isinstance(slots, dict) and slot_name in slots:
            return slots[slot_name]
        return None
    except Exception:
        return None

def _run_enhanced_chat_and_tts(user_speech: str, session_id: str = None, target_key: Optional[str] = None) -> Tuple[str, Optional[str], str]:
    """
    Enhanced function that:
    1. Takes user speech input
    2. Calls chatbot to get response
    3. Runs TTS on chatbot response
    4. Returns audio URL, S3 key, and final text
    """
    if not user_speech or not user_speech.strip():
        user_speech = "안녕하세요. 무엇을 도와드릴까요?"
    
    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
    
    _log("Processing user speech through chatbot and TTS", 
         session_id=session_id[:8], 
         input_length=len(user_speech),
         input_preview=user_speech[:100])
    
    # Step 1: Get chatbot response
    chatbot_response = call_chatbot(user_speech, session_id)
    
    # Step 2: Run TTS on chatbot response
    if not TTS_URL:
        raise RuntimeError("TTS_URL env is required")
    
    # Randomize TTS seed per job
    job_seed = int(time.time_ns() % 2_147_483_647)
    
    payload = {
        "text": chatbot_response,
        "key_prefix": KEY_PREFIX,
        "sample_rate": 8000,
        "seed": job_seed,
        "use_memory_cache": False
    }
    
    _log("Calling TTS with chatbot response", 
         url=_redact(TTS_URL, 24), 
         seed=job_seed,
         text_preview=chatbot_response[:100])
    
    tts_resp = _http_post_json(TTS_URL, payload, token=TTS_TOKEN, timeout=60)
    
    audio_url = tts_resp.get("url") or tts_resp.get("audio_url") or tts_resp.get("s3_url")
    if not audio_url:
        _log("TTS response missing audio URL", keys=list(tts_resp.keys()))
        raise RuntimeError(f"TTS response missing URL: {list(tts_resp.keys())}")
    
    _log("TTS returned audio for chatbot response", url=_redact(audio_url, 40))
    
    # Step 3: Ensure Connect-playable URL
    presigned_url, s3_key = _ensure_connect_playable(audio_url, target_key)
    
    _log("Enhanced processing complete", 
         user_input_preview=user_speech[:50],
         chatbot_response_preview=chatbot_response[:50],
         audio_url=_redact(presigned_url, 60))
    
    return presigned_url, (s3_key or target_key), chatbot_response

def _run_tts_and_prepare_url(text: str, target_key: Optional[str] = None) -> Tuple[str, Optional[str]]:
    final_text = text

    # Build a unique session id
    # NOTE: pass event/session id in args if you want; simplest is to make it global per invocation:
    # you can store it in a threadlocal or pass as an argument; for brevity, derive from target_key/job id:
    sess_id = (target_key or uuid.uuid4().hex).split("/")[-1].replace(".wav","")

    if CHAT_URL and os.getenv("BYPASS_CHAT","0") != "1":
        _log("Calling CHAT", url=_redact(CHAT_URL, 24), session_id=sess_id)
        chat_resp = _http_post_json(
            CHAT_URL, {"session_id": sess_id, "question": text},
            token=CHAT_TOKEN, timeout=20
        )
        final_text = chat_resp.get("answer") or chat_resp.get("text") or text
        _log("CHAT response", changed=(final_text != text))

    # Randomize TTS seed per job (diversify prosody)
    import time
    job_seed = int(time.time_ns() % 2_147_483_647)

    if not TTS_URL:
        raise RuntimeError("TTS_URL env is required")
    payload = {
        "text": final_text,
        "key_prefix": KEY_PREFIX,
        "sample_rate": 8000,   # use 8000 if your server supports it
        "seed": job_seed,
        "use_memory_cache": False
    }
    _log("Calling TTS", url=_redact(TTS_URL, 24), seed=job_seed)
    tts_resp = _http_post_json(TTS_URL, payload, token=TTS_TOKEN, timeout=60)

    audio_url = tts_resp.get("url") or tts_resp.get("audio_url") or tts_resp.get("s3_url")
    if not audio_url:
        _log("TTS response missing audio URL", keys=list(tts_resp.keys()))
        raise RuntimeError(f"TTS response missing URL: {list(tts_resp.keys())}")
    _log("TTS returned audio", url=_redact(audio_url, 40))

    # Ensure Connect-playable URL; if target_key provided, upload into that exact key
    presigned_url, s3_key = _ensure_connect_playable(audio_url, target_key)
    _log("Prepared Connect audio", presigned=_redact(presigned_url, 60), s3_key=s3_key or target_key or "")
    return presigned_url, (s3_key or target_key), final_text

# ========= Router sub-handlers =========
def _pick_answer(chat_res) -> str:
    """Extract a non-empty answer from common response shapes."""
    if isinstance(chat_res, str):
        # Sometimes services return a raw string
        return chat_res.strip()
    if not isinstance(chat_res, (dict, list)):
        return ""

    # flat keys (most common)
    for k in ("answer", "text", "reply", "response", "message", "content", "output", "output_text"):
        v = chat_res.get(k) if isinstance(chat_res, dict) else None
        if isinstance(v, str) and v.strip():
            return v.strip()

    # nested / OpenAI-like
    def g(o, *path):
        cur = o
        for p in path:
            if isinstance(p, int):
                if isinstance(cur, list) and len(cur) > p:
                    cur = cur[p]
                else:
                    return None
            else:
                cur = cur.get(p) if isinstance(cur, dict) else None
            if cur is None:
                return None
        return cur

    candidates = [
        g(chat_res, "data", "answer"),
        g(chat_res, "data", "text"),
        g(chat_res, "result", "answer"),
        g(chat_res, "result", "text"),
        g(chat_res, "choices", 0, "message", "content"),  # OpenAI format
        g(chat_res, "choices", 0, "text"),
    ]
    for v in candidates:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _start(event):
    params = event.get("Details", {}).get("Parameters", {})
    chat_on = params.get("chat_on")
    if chat_on is None:
        chat_on = os.getenv("BYPASS_CHAT","0") != "1"
    else:
        chat_on = str(chat_on).lower() in ("1","true","yes","y")

    text = _extract_text(event) or " "
    pk = make_pk(text, sr=8000, chat=chat_on)

    # 1) Cache hit → return immediately with cache_hit flag
    item = ddb_get(pk)
    if item:
        if item.get("state") == "ready" and item.get("prompt_arn"):
            _log("Cache hit - ready item found", pk=pk[:12] + "...")
            return {
                "job_id": "",
                "bucket": item["bucket"],
                "key": item["key"],
                "pk": pk,
                "ready": "true",
                "cache_hit": "true",  # ← New flag for flow logic
                "prompt_arn": item["prompt_arn"],
                "setAttributes": {
                    "ready": "true",
                    "cache_hit": "true",  # ← Flow can check this
                    "prompt_arn": item["prompt_arn"],
                    "final_text": item.get("final_text", ""),
                    "job_bucket": item["bucket"],
                    "job_key": item["key"],
                    "key": item["key"],
                    "pk": pk
                }
            }
        if item.get("state") == "pending":
            _log("Cache hit - pending item found", pk=pk[:12] + "...")
            return {
                "job_id": "",
                "bucket": item["bucket"],
                "key": item["key"],
                "pk": pk,
                "ready": "false",
                "cache_hit": "true",  # ← Still a cache hit, just not ready
                "prompt_arn": "",
                "setAttributes": {
                    "ready": "false",
                    "cache_hit": "true",
                    "job_bucket": item["bucket"],
                    "job_key": item["key"],
                    "key": item["key"],
                    "pk": pk
                }
            }

    # 2) No item → create pending and start worker (cache miss)
    _log("Cache miss - creating new job", pk=pk[:12] + "...")
    job_id = uuid.uuid4().hex
    key = _make_key(job_id)

    created = ddb_put_pending_if_absent(pk, COMPANY_BUCKET, key, text)
    if DISABLE_DDB:
        created = True  # force worker start in S3-only mode
    if not created:
        # Race condition: someone else just created it
        item = ddb_get(pk) or {}
        return {
            "job_id": "",
            "bucket": item.get("bucket", COMPANY_BUCKET),
            "key": item.get("key", key),
            "pk": pk,
            "prompt_arn": "",
            "ready": "false",
            "cache_hit": "true",  # ← Race condition is still a cache scenario
            "setAttributes": {
                "ready": "false",
                "cache_hit": "true",
                "job_bucket": item.get("bucket", COMPANY_BUCKET),
                "job_key": item.get("key", key),
                "key": item.get("key", key),
                "pk": pk
            }
        }

    # Fire-and-forget worker
    lambda_client.invoke(
        FunctionName=ASYNC_FUNCTION_NAME,
        InvocationType="Event",
        Payload=json.dumps({
            "action":"worker","job_id":job_id,"text":text,
            "bucket":COMPANY_BUCKET,"key":key,"pk":pk,
            "chat_on": chat_on
        }).encode("utf-8")
    )
    
    return {
        "job_id": job_id,
        "bucket": COMPANY_BUCKET,
        "key": key,
        "pk": pk,
        "prompt_arn": "", 
        "ready": "false",
        "cache_hit": "false",  # ← New job, not a cache hit
        "setAttributes": {
            "job_bucket": COMPANY_BUCKET, 
            "job_key": key,
            "key": key,
            "ready": "false", 
            "cache_hit": "false",  # ← Flow can branch on this
            "pk": pk
        }
    }


def _check(event):
    """
    Check if a TTS job is ready by:
    1. Getting the pk from event parameters 
    2. Checking DynamoDB state (not just S3 existence)
    3. Only returning ready=true when DDB state is 'ready'
    """
    params = event.get("Details", {}).get("Parameters", {})
    
    # Get pk - either directly provided or compute from original text
    pk = params.get("pk")
    if not pk:
        # If no pk provided, try to compute it from the original request
        text = _extract_text(event)
        if not text:
            return {
                "ready": "false", 
                "error": "no_pk_or_text",
                "setAttributes": {"ready": "false", "prompt_arn": "", "final_text": ""}
            }
        
        chat_on = params.get("chat_on")
        if chat_on is None:
            chat_on = os.getenv("BYPASS_CHAT", "0") != "1"
        else:
            chat_on = str(chat_on).lower() in ("1", "true", "yes", "y")
        
        pk = make_pk(text, sr=8000, chat=chat_on)
    
    _log("Checking job status", pk=pk[:12] + "...")
    
    if DISABLE_DDB:
        params = event.get("Details", {}).get("Parameters", {}) or {}
        bucket = params.get("job_bucket") or COMPANY_BUCKET
        key    = params.get("job_key")    or params.get("key") or ""
        if not key:
            return {"setAttributes": {"ready":"false","Error":"missing_job_key"}}
        try:
            s3.head_object(Bucket=bucket, Key=key)
            url = f"https://{bucket}.s3.{os.getenv('AWS_REGION', COMPANY_REGION)}.amazonaws.com/{key}"
            return {"setAttributes":{
                "ready":"true","job_bucket":bucket,"job_key":key,"AudioS3Url0":url,
                "AudioS3UrlCount":"1","BatchCount":"1","HasMore":"false"
            }}
        except Exception:
            return {"setAttributes":{"ready":"false","job_bucket":bucket,"job_key":key}}
    item = ddb_get(pk)
    if not item:
        _log("No DDB item found", pk=pk[:12] + "...")
        return {
            "ready": "false",
            "error": "item_not_found", 
            "setAttributes": {"ready": "false", "prompt_arn": "", "final_text": ""}
        }
    
    state = item.get("state", "").lower()
    _log("DDB item state", pk=pk[:12] + "...", state=state)
    
    if state == "ready":
        prompt_arn = item.get("prompt_arn", "")
        final_text = item.get("final_text", "")
        bucket = item.get("bucket", COMPANY_BUCKET)
        key = item.get("key", "")
        
        # Double-check S3 object exists (optional safety check)
        if key:
            try:
                s3.head_object(Bucket=bucket, Key=key)
                _log("S3 object confirmed", bucket=bucket, key=key)
            except Exception as e:
                _log("S3 object missing despite ready state", bucket=bucket, key=key, error=str(e))
                # Still return ready since DDB says it's ready - the prompt_arn should work
        
        return {
            "ready": "true",
            "state": "ready",
            "prompt_arn": prompt_arn,
            "final_text": final_text,
            "bucket": bucket,
            "key": key,
            "setAttributes": {
                "ready": "true", 
                "prompt_arn": prompt_arn, 
                "final_text": final_text,
                "job_bucket": bucket,
                "job_key": key
            }
        }
    
    elif state == "pending":
        bucket = item.get("bucket", COMPANY_BUCKET)
        key = item.get("key", "")
        _log("Job still pending", pk=pk[:12] + "...")
        
        return {
            "ready": "false",
            "state": "pending", 
            "bucket": bucket,
            "key": key,
            "setAttributes": {
                "ready": "false", 
                "prompt_arn": "", 
                "final_text": "",
                "job_bucket": bucket,
                "job_key": key
            }
        }
    
    else:
        _log("Unknown state", pk=pk[:12] + "...", state=state)
        return {
            "ready": "false",
            "state": state,
            "error": f"unknown_state_{state}",
            "setAttributes": {"ready": "false", "prompt_arn": "", "final_text": ""}
        }


# Alternative: Replace your existing _check with _check_ready_by_pk entirely
# since _check_ready_by_pk already does exactly what you need
def _check_v2(event):
    """Simplified version - just call the existing _check_ready_by_pk function"""
    return _check_ready_by_pk(event)

def _worker(event):
    chat_on = event.get("chat_on")
    if chat_on is None:
        chat_on = os.getenv("BYPASS_CHAT","0") != "1"
    job_id = event["job_id"]
    text   = event["text"]
    key    = event.get("key") or _make_key(job_id)
    pk     = event.get("pk")

    presigned, _, final_text = _run_tts_and_prepare_url(text, target_key=key)

    # Create/Update Connect prompt pointing at the exact s3://bucket/key
    prompt_id, prompt_arn = _ensure_connect_prompt_for_key(COMPANY_BUCKET, key, job_id)

    # Persist artifacts for the checker (and auditing)
    s3.put_object(Bucket=COMPANY_BUCKET, Key=key + ".txt",
                  Body=final_text.encode("utf-8"), ContentType="text/plain")
    s3.put_object(Bucket=COMPANY_BUCKET, Key=key + ".prompt",
                  Body=prompt_arn.encode("utf-8"), ContentType="text/plain")

    # Flip DynamoDB to ready
    if pk:
        ddb_mark_ready(pk, prompt_arn=prompt_arn, final_text=final_text, bucket=COMPANY_BUCKET, key=key)

    return {"ok": True, "prompt_arn": prompt_arn}

def _start_stream(text: str, filler_text: str = "", *, include_prompts: bool = False) -> dict:
    """
    Kick off streaming TTS and return contact attributes for immediate playback.
    - If the TTS server returns first_url/first_key, use it.
    - Otherwise, quick-poll for part0 for a short time window.
    - Returns: batch_attrs dict (convert to {"setAttributes": batch_attrs} at the call site)
    """
    # Build the /synthesize_stream_start endpoint from TTS_URL env
    start_url = f"{TTS_URL.rstrip('/').replace('synthesize', 'synthesize_stream_start')}"

    payload   = {"text": (text or filler_text), "sample_rate": 8000}
    timeout_s = int(os.getenv("START_HTTP_TIMEOUT_SEC", "6"))

    res = _http_post_json(start_url, payload, token=TTS_TOKEN, timeout=timeout_s)

    job_id = res.get("job_id")
    if not job_id:
        # Return attributes so the flow can branch/log gracefully
        return {"Error": "no_job_id_from_server", "ready": "false"}

    # Prefer an immediate first clip if the server provides one
    first_url = res.get("first_url")
    first_key = res.get("first_key")

    if first_url or first_key:
        if not first_url and first_key:
            # Convert key -> URL using your existing helper (presigned or public)
            first_url = _url_for_key(first_key)
        batch_attrs = {
            "AudioS3Url0":   first_url,
            "AudioS3UrlCount":"1",
            "BatchCount":    "1",
            "NextIndexOut":  "1",   # we just consumed part0
            "HasMore":       "true",
        }
    else:
        # Fallback: short quick-poll for part0 to keep UX snappy
        deadline = time.time() + float(os.getenv("FIRST_PART_WAIT_SEC", "1.0"))
        interval = float(os.getenv("FIRST_PART_POLL_MS", "100")) / 1000.0

        # Start pessimistic; will be overwritten if a clip appears
        batch_attrs = {
            "AudioS3UrlCount": "0",
            "BatchCount":      "0",
            "HasMore":         "true",
            "NextIndexOut":    "0",
        }
        while time.time() < deadline:
            # Use S3 direct by default (include_prompts=False). Flip to True if you want PromptARNs.
            # ba = _build_batch(job_id, 0, limit=1, include_prompts=include_prompts)
            ba = _build_batch(job_id, 0, limit=prefetch, include_prompts=include_prompts)
            # We only accept a “play now” if we actually have 1 item ready
            if str(ba.get("AudioS3UrlCount")) == "1" or (include_prompts and str(ba.get("PromptARNCount")) == "1"):
                batch_attrs = ba
                break
            time.sleep(interval)

    # Common fields
    batch_attrs["JobId"] = job_id
    batch_attrs["ready"] = "true"
    # Coerce all values to strings for Connect contact attributes
    batch_attrs = {k: (v if isinstance(v, str) else str(v)) for k, v in batch_attrs.items()}
    return batch_attrs

# Check your environment variables:
def debug_environment():
    """Debug function to check environment variables"""
    _log("Environment check",
         chat_url=_redact(CHAT_URL, 30),
         chat_token_set=bool(CHAT_TOKEN),
         tts_url=_redact(TTS_URL, 30),
         bypass_chat=BYPASS_CHAT,
         bypass_chat_env=os.getenv("BYPASS_CHAT", "not_set"))
from urllib.parse import urlparse, quote

def _warmup_url(base: str) -> str:
    # if base endswith '/synthesize', post to '/synthesize/warmup', else '/warmup'
    parsed = urlparse(base)
    if parsed.path.rstrip("/").endswith("/synthesize"):
        return base.rstrip("/") + "/warmup"
    return base.rstrip("/") + "/synthesize/warmup" 

def to_regional_url(key: str) -> str:
    # Properly encode the key for URLs
    encoded_key = quote(key, safe='/')
    return f"https://{COMPANY_BUCKET}.s3.{COMPANY_REGION}.amazonaws.com/{encoded_key}"

def _is_lex_code_hook(event: dict) -> bool:
    # Lex V2 code hook events always have sessionState + bot + invocationSource
    if not isinstance(event, dict):
        return False
    if not isinstance(event.get("sessionState"), dict):
        return False
    if not isinstance(event.get("bot"), dict):
        return False
    inv_src = event.get("invocationSource")
    return inv_src in ("DialogCodeHook", "FulfillmentCodeHook", "InitializationAndValidation")

def _lex_delegate_with_interrupt_attrs(event: dict) -> dict:
    session_state = event.get("sessionState", {}) or {}
    session_attributes = session_state.get("sessionAttributes", {}) or {}

    # Make barge-in + audio capture snappy
    session_attributes.update({
        "x-amz-lex:allow-interrupt:*:*": "true",
        "x-amz-lex:audio:start-timeout-ms:*:*": "1200",
        "x-amz-lex:audio:end-timeout-ms:*:*": "400",
        "x-amz-lex:audio:max-length-ms:*:*": "12000",
    })

    # Minimal, safe Lex response
    return {
        "sessionState": {
            "sessionAttributes": session_attributes,
            "dialogAction": {"type": "Delegate"},
            "intent": session_state.get("intent", {}) or {}
        }
    }

def _unified_chat_and_stream(event):
    """
    Unified function that combines caching logic with streaming TTS.
    
    Behavior:
    1. Cancel any previous streaming job
    2. Extract user input and compute cache key (pk)
    3. Check DynamoDB cache:
       - If ready: return immediate playback with cached audio
       - If pending: start new streaming job (don't recreate worker)
       - If miss: create pending item, start worker for caching, AND start streaming
    4. Always return streaming attributes for immediate audio playback
    """
    
    # 0) Cancel any previous streaming job to avoid overlap
    _cancel_previous_job_if_any(event)
    
    # 1) Extract user input and determine chat settings
    user_speech = _extract_user_speech_input(event)
    if not user_speech:
        _log("No user speech input found, using default")
        user_speech = "안녕하세요"
    
    params = event.get("Details", {}).get("Parameters", {}) or {}
    
    # Determine if chat is enabled
    chat_on = params.get("chat_on")
    if chat_on is None:
        chat_on = os.getenv("BYPASS_CHAT", "0") != "1"
    else:
        chat_on = str(chat_on).lower() in ("1", "true", "yes", "y")
    
    # Generate cache key based on input + settings
    pk = make_pk(user_speech, sr=8000, lang="ko-KR", voice="Jihye", chat=chat_on)
    session_id = _session_id_from_event(event, str(uuid.uuid4()))
    
    _log("Unified chat and stream starting", 
         session_id=session_id[:8],
         pk=pk[:12] + "...",
         chat_enabled=chat_on,
         user_input_preview=user_speech[:100])
    
    # 2) Check DynamoDB cache
    item = ddb_get(pk)
    
    # 3) CACHE HIT - Ready item exists
    if item and item.get("state") == "ready" and item.get("key"):
        bucket = item.get("bucket", COMPANY_BUCKET)
        key = item["key"]
        prompt_arn = item.get("prompt_arn", "")
        final_text = item.get("final_text", "")
        
        # Ensure we have a Connect prompt ARN
        if not prompt_arn:
            try:
                _, prompt_arn = _ensure_connect_prompt_for_key(bucket, key, pk[:12])
                # Update DDB with the prompt ARN for next time
                ddb_mark_ready(pk, prompt_arn=prompt_arn, final_text=final_text, bucket=bucket, key=key)
            except Exception as e:
                _log("Failed to create prompt ARN for cached item", error=str(e))
                prompt_arn = ""
        
        # Return immediate playback from cache with both prompt ARN and S3 URL
        s3_url = to_regional_url(key)
        _log("Cache hit - immediate playback", pk=pk[:12] + "...", key=key)
        
        return {"setAttributes": {
            "AudioS3Url0": s3_url,
            "PromptARN0": prompt_arn,
            "CachedAudioUrl": s3_url,  # Direct S3 URL for cached audio
            "CachedPromptArn": prompt_arn,  # Cached prompt ARN
            "AudioS3UrlCount": "1",
            "BatchCount": "1",
            "HasMore": "false",
            "NextIndexOut": "1",
            "JobId": "",  # No active streaming job
            "ready": "true",
            "cache_hit": "true",
            "UserInput": user_speech,
            "ChatAnswer": final_text,
            "job_bucket": bucket,
            "job_key": key,
            "pk": pk
        }}
    
    # 4) CACHE MISS or PENDING - Need to generate audio
    
    # Get chatbot response for streaming (this is what we'll stream immediately)
    try:
        chatbot_response = call_chatbot(user_speech, session_id)
        _log("Chatbot response received", 
             input_preview=user_speech[:50],
             output_preview=chatbot_response[:50])
    except Exception as e:
        _log("Chatbot call failed, using original input", error=str(e))
        chatbot_response = user_speech
    
    # If this is a complete cache miss, set up the caching infrastructure
    cache_bucket = COMPANY_BUCKET
    cache_key = None
    
    if not item:
        # Create new cache entry
        cache_key = _make_key(uuid.uuid4().hex)
        created = ddb_put_pending_if_absent(pk, cache_bucket, cache_key, user_speech)
        
        if created:
            _log("Cache miss - created pending entry", pk=pk[:12] + "...", cache_key=cache_key)
            
            # Start background worker to generate cached version
            try:
                lambda_client.invoke(
                    FunctionName=ASYNC_FUNCTION_NAME,
                    InvocationType="Event",
                    Payload=json.dumps({
                        "action": "worker",
                        "job_id": uuid.uuid4().hex,
                        "text": user_speech,  # Use original input for caching
                        "bucket": cache_bucket,
                        "key": cache_key,
                        "pk": pk,
                        "chat_on": chat_on
                    }).encode("utf-8")
                )
                _log("Background caching worker started", pk=pk[:12] + "...")
            except Exception as e:
                _log("Failed to start caching worker", error=str(e))
        else:
            # Race condition - someone else created it, get their key
            item = ddb_get(pk) or {}
            cache_key = item.get("key", cache_key)
            _log("Cache miss race condition", pk=pk[:12] + "...")
    else:
        # Item exists but is pending
        cache_bucket = item.get("bucket", COMPANY_BUCKET)
        cache_key = item.get("key")
        _log("Cache pending - continuing with stream", pk=pk[:12] + "...")
    
    # 5) Start streaming TTS for immediate playback (using chatbot response)
    try:
        # Start the streaming job
        start_url = f"{TTS_URL.rstrip('/').replace('synthesize', 'synthesize_stream_start')}"
        
        stream_res = _http_post_json(
            start_url,
            {"text": chatbot_response, "sample_rate": 8000},
            token=TTS_TOKEN,
            timeout=int(os.getenv("START_HTTP_TIMEOUT_SEC", "6"))
        )
        
        stream_job_id = stream_res.get("job_id")
        if not stream_job_id:
            raise RuntimeError("No job_id returned from streaming TTS")
        
        _log("Streaming TTS job started", 
             stream_job_id=stream_job_id,
             text_preview=chatbot_response[:100])
        
        # Quick poll for first audio part
        deadline = time.time() + float(os.getenv("FIRST_PART_WAIT_SEC", "1.0"))
        interval = float(os.getenv("FIRST_PART_POLL_MS", "100")) / 1000.0
        
        batch_attrs = {"AudioS3UrlCount": "0", "BatchCount": "0", "HasMore": "true", "NextIndexOut": "0"}
        poll_count = 0
        
        while time.time() < deadline:
            batch_attrs = _build_batch(stream_job_id, 0, limit=prefetch, include_prompts=False)
            poll_count += 1
            
            if str(batch_attrs.get("AudioS3UrlCount")) == "1":
                _log("First audio part ready", attempts=poll_count)
                break
            time.sleep(interval)
        
        # If polling didn't work, try direct S3 check
        if str(batch_attrs.get("AudioS3UrlCount")) != "1":
            part0_key = f"{stream_job_id}/part0.wav"
            if wait_s3_ready(TTS_BUCKET, part0_key, timeout=1.5, interval=0.2):
                s3_url0 = to_regional_url(part0_key)
                batch_attrs = {
                    "BatchCount": "1",
                    "AudioS3UrlCount": "1", 
                    "HasMore": "true",
                    "NextIndexOut": "1",
                    "AudioS3Url0": s3_url0
                }
                _log("First audio part found via direct S3 check")
        
        # Add metadata for the streaming response
        batch_attrs.update({
            "JobId": stream_job_id,
            "ready": "true",
            "cache_hit": "false" if not item else "true",  # True if item existed (even if pending)
            "UserInput": user_speech,
            "ChatAnswer": chatbot_response,
            "job_bucket": cache_bucket,
            "job_key": cache_key or "",
            "pk": pk
        })
        
        # Stringify all values for Connect
        batch_attrs = {k: (v if isinstance(v, str) else str(v)) for k, v in batch_attrs.items()}
        
        _log("Unified streaming response ready", 
             stream_job_id=stream_job_id,
             cache_setup=(cache_key is not None),
             audio_ready=(str(batch_attrs.get("AudioS3UrlCount")) == "1"))
        
        return {"setAttributes": batch_attrs}
        
    except Exception as e:
        _log("Streaming TTS failed", error=str(e), error_type=type(e).__name__)
        import traceback
        _log("Streaming error traceback", traceback=traceback.format_exc())
        
        return {"setAttributes": {
            "Error": f"streaming_error: {str(e)}",
            "ready": "false",
            "cache_hit": "false",
            "UserInput": user_speech,
            "pk": pk
        }}


# Also add this helper function for job cancellation with better error handling
def _cancel_previous_job_if_any(event):
    """Cancel any previous TTS streaming job to avoid overlap"""
    try:
        # Try to get previous job ID from contact attributes
        attrs = (event.get("Details", {}) or {}).get("ContactData", {}).get("Attributes", {}) or {}
        prev_job_id = attrs.get("JobId")
        
        if not prev_job_id:
            return
        
        cancel_url = f"{TTS_URL.rstrip('/').replace('synthesize', 'synthesize_stream_cancel')}"
        
        try:
            _http_post_json(
                cancel_url, 
                {"job_id": prev_job_id}, 
                token=TTS_TOKEN, 
                timeout=3
            )
            _log("Cancelled previous TTS job", prev_job_id=prev_job_id[:8] + "...")
        except Exception as cancel_error:
            # Don't fail the whole request if cancellation fails
            _log("Cancel previous TTS job failed (continuing)", 
                 prev_job_id=prev_job_id[:8] + "...",
                 error=str(cancel_error))
                 
    except Exception as e:
        _log("Error in cancel previous job", error=str(e))


def _synthesize_full_with_filler(event):
    """
    Enhanced full synthesis that plays filler audio immediately while generating the full response.
    Fixed to properly respect DISABLE_DDB flag.
    """
    # Cancel any previous job
    _cancel_previous_job_if_any(event)
    
    # Extract user speech input
    user_speech = _extract_user_speech_input(event)
    if not user_speech:
        _log("No user speech input found, using default")
        user_speech = "안녕하세요"
    
    session_id = _session_id_from_event(event, str(uuid.uuid4()))
    
    _log("Synthesize full with filler starting", 
         session_id=session_id[:8],
         user_speech_preview=user_speech[:100],
         ddb_disabled=DISABLE_DDB)
    
    try:
        # Get chatbot response (this will be used for the full audio generation)
        chatbot_response = call_chatbot(user_speech, session_id)
        
        _log("DEBUG: Chatbot response for full synthesis with filler", 
             input=user_speech[:50],
             output=chatbot_response[:50])
        
        # Generate cache key and check for existing audio
        if DISABLE_DDB:
            # S3-only mode: use content-based deterministic key
            import hashlib
            content_hash = hashlib.sha256(chatbot_response.encode('utf-8')).hexdigest()[:16]
            cache_key = f"{KEY_PREFIX.rstrip('/')}/full_cache/{content_hash}.wav"
            params = (event.get("Details", {}) or {}).get("Parameters", {}) or {}
            raw_cat = params.get("filler_category") or DEFAULT_FILLER_CATEGORY
            raw_idx = params.get("filler_index")
            try:
                idx_int = int(raw_idx) if raw_idx is not None and str(raw_idx).strip() != "" else None
            except Exception:
                idx_int = None

            # Choose filler (returns canonical cat/index and a ready-to-play URL)
            cat, idx, filler_key, filler_url = _filler_key_url(raw_cat, idx_int)
            
            _log("S3-only mode: checking cache", cache_key=cache_key)
            
            # Check if this audio already exists in S3
            try:
                head_response = _s3.head_object(Bucket=COMPANY_BUCKET, Key=cache_key)
                
                if head_response.get("ContentLength", 0) > 0:
                    # Cache hit - return the cached audio immediately
                    s3_url = f"https://{COMPANY_BUCKET}.s3.{S3_REGION}.amazonaws.com/{cache_key}"
                    
                    _log("S3 cache hit for full synthesis - returning immediately", cache_key=cache_key)
                    
                    attrs = {
                        "ready": "true",
                        "AudioS3Url0": s3_url(COMPANY_BUCKET, f"{raw_cat}/{raw_idx:02d}.wav"),
                        "AudioS3UrlCount": "1",
                        "BatchCount": "1",
                        "HasMore": "true",                 # we will poll for final
                        "NextIndexOut": "1",
                        "mode": "full_with_filler",
                        "filler_category": raw_cat,
                        "filler_index": str(raw_idx),
                        "job_bucket": COMPANY_BUCKET,
                        "job_key": key,             # e.g. "connect/sessions/full_cache/<id>.wav"
                        "cache_hit": "false"
                    }
                    return {"setAttributes": attrs}
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code not in ("404", "NoSuchKey", "NotFound"):
                    _log("Unexpected S3 error during cache check", error=str(e))
                # Cache miss - continue with generation
                _log("S3 cache miss - will generate new audio", cache_key=cache_key)
        
        else:
            # DDB mode - only if DDB is actually enabled
            pk = make_pk(chatbot_response, sr=8000, lang="ko-KR", voice="Jihye", chat=False)
            
            _log("DDB mode: checking cache", pk=pk[:12] + "...")
            
            item = ddb_get(pk)  # This should be safe now because ddb_get is stubbed when DISABLE_DDB=True
            if item and item.get("state") == "ready" and item.get("key"):
                # Cache hit
                bucket = item.get("bucket", COMPANY_BUCKET)
                key = item["key"]
                s3_url = f"https://{bucket}.s3.{S3_REGION}.amazonaws.com/{key}"
                
                _log("DDB cache hit for full synthesis - returning immediately", pk=pk[:12] + "...")
                
                return {
                    "ready": "true",
                    "AudioS3Url0": s3_url,
                    "AudioS3UrlCount": "1",
                    "BatchCount": "1",
                    "HasMore": "false",
                    "NextIndexOut": "1",
                    "ChatAnswer": chatbot_response,
                    "UserInput": user_speech,
                    "final_text": chatbot_response,
                    "cache_hit": "true",
                    "pk": pk,
                    "mode": "full_cached_ddb",
                    "job_bucket": bucket,
                    "job_key": key
                }
            
            # Set cache key for DDB mode
            cache_key = _make_key(uuid.uuid4().hex)
        
        # Cache miss - need to generate audio
        # 1. Choose and return filler audio immediately
        params = event.get("Details", {}).get("Parameters", {}) or {}
        raw_cat = params.get("filler_category")
        raw_idx = params.get("filler_index")

        if not raw_cat:
            raw_cat = DEFAULT_FILLER_CATEGORY

        try:
            idx_int = int(raw_idx) if raw_idx is not None and str(raw_idx).strip() != "" else None
        except Exception:
            idx_int = None

        cat, idx, filler_key, filler_url = _filler_key_url(raw_cat, idx_int)
        
        _log("Selected filler audio", category=cat, index=idx, key=filler_key)
        
        # 2. Set up cache entry for background generation
        if not DISABLE_DDB:
            # Only create DDB entry if DDB is enabled
            created = ddb_put_pending_if_absent(pk, COMPANY_BUCKET, cache_key, chatbot_response)
            if not created:
                # Race condition - someone else is generating it
                item = ddb_get(pk) or {}
                cache_key = f"{KEY_PREFIX.rstrip('/')}/full_cache/{content_hash}.wav"
                _log("Full audio generation race condition", pk=pk[:12] + "...")
        
        # 3. Start background worker to generate the full audio
        try:
            worker_payload = {
                "action": "synthesize_full_worker",
                "text": chatbot_response,
                "bucket": COMPANY_BUCKET,
                "key": cache_key,
                "sample_rate": 8000,
                "disable_ddb": DISABLE_DDB
            }
            
            # Only add pk if using DDB
            if not DISABLE_DDB:
                worker_payload["pk"] = pk
            
            lambda_client.invoke(
                FunctionName=ASYNC_FUNCTION_NAME,
                InvocationType="Event",
                Payload=json.dumps(worker_payload).encode("utf-8")
            )
            _log("Background full synthesis worker started", 
                 cache_key=cache_key,
                 mode="s3_only" if DISABLE_DDB else "ddb")
        except Exception as e:
            _log("Failed to start full synthesis worker", error=str(e))
        
        # 4. Return filler audio immediately for playback
        response_attrs = {
            "ready": "true",
            "AudioS3Url0": filler_url,  # This is now a presigned URL
            "AudioS3UrlCount": "1",
            "BatchCount": "1",
            "HasMore": "true",
            "NextIndexOut": "1",
            "ChatAnswer": chatbot_response,
            "UserInput": user_speech,
            "mode": "full_with_filler",
            "job_bucket": COMPANY_BUCKET,
            "job_key": cache_key,
            "cache_hit": "false"
        }
        
        # Add pk only if using DDB
        if not DISABLE_DDB:
            response_attrs["pk"] = pk
        
        # Stringify all values
        response_attrs = {k: (v if isinstance(v, str) else str(v)) for k, v in response_attrs.items()}
        
        _log("Synthesize full with filler - returning filler audio", 
             filler_url=_redact(filler_url, 40),
             cache_key=cache_key,
             category=cat,
             index=idx,
             mode="s3_only" if DISABLE_DDB else "ddb")
        
        return response_attrs
        
    except Exception as e:
        _log("Synthesize full with filler failed", error=str(e), error_type=type(e).__name__)
        import traceback
        _log("Full synthesis with filler error traceback", traceback=traceback.format_exc())
        
        return {
            "ready": "false",
            "Error": f"synthesize_full_filler_error: {str(e)}",
            "ErrorDetail": str(e),
            "UserInput": user_speech,
            "AudioS3UrlCount": "0",
            "BatchCount": "0",
            "HasMore": "false"
        }


def _synthesize_full_worker(event):
    """
    Background worker that generates the full audio using the non-streaming /synthesize endpoint.
    Now supports both DDB and S3-only modes with enhanced debugging.
    """
    try:
        text = (event.get("text") or "").strip()
        bucket = event.get("bucket") or COMPANY_BUCKET
        key = event.get("key") or ""
        sr = int(event.get("sample_rate") or 8000)
        disable_ddb = event.get("disable_ddb", DISABLE_DDB)
        
        # Only require pk if DDB is enabled
        if not disable_ddb:
            pk = event.get("pk") or ""
            if not pk:
                _log("synthesize_full_worker: missing pk for DDB mode")
                return {"ok": False}

        if not text or not key:
            _log("synthesize_full_worker: missing required parameters", 
                 has_text=bool(text), has_key=bool(key))
            return {"ok": False}

        _log("synthesize_full_worker starting", 
             key=key,
             text_preview=text[:100],
             mode="s3_only" if disable_ddb else "ddb")

        # Call the NON-STREAMING FastAPI /synthesize endpoint
        base_url = TTS_URL.rstrip("/")
        if base_url.endswith("/synthesize_stream_start"):
            base_url = base_url.replace("/synthesize_stream_start", "/synthesize")
        elif not base_url.endswith("/synthesize"):
            base_url = f"{base_url}/synthesize"

        # Use the exact key we want for the final audio
        key_prefix = "/".join(key.split("/")[:-1]) if "/" in key else "connect/sessions"
        
        payload = {
            "text": text,
            "sample_rate": sr,
            "key_prefix": key_prefix,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.0,
            "max_new_tokens": 512,
            "chunk_length": 64,
            "use_memory_cache": True
        }

        _log("synthesize_full_worker calling TTS", url=_redact(base_url, 40))
        
        res = _http_post_json(
            base_url,
            payload,
            token=TTS_TOKEN,
            timeout=SYNTH_HTTP_TIMEOUT_SEC,
        )
        
        server_bucket = res.get("bucket") or bucket
        server_key = res.get("key")
        
        if not (server_bucket and server_key):
            _log("synthesize_full_worker: TTS response missing bucket/key", resp_keys=list(res.keys()))
            return {"ok": False}

        _log("synthesize_full_worker: TTS server response", 
             server_bucket=server_bucket, 
             server_key=server_key,
             target_key=key)

        # Check the server's audio file before copying
        try:
            server_head = s3.head_object(Bucket=server_bucket, Key=server_key)
            server_size = server_head.get("ContentLength", 0)
            server_type = server_head.get("ContentType", "")
            
            _log("synthesize_full_worker: server audio info", 
                 size=server_size, 
                 content_type=server_type)
            
            if server_size == 0:
                _log("synthesize_full_worker: server audio file is empty!")
                return {"ok": False, "error": "server_audio_empty"}
                
        except Exception as e:
            _log("synthesize_full_worker: failed to check server audio", error=str(e))
            return {"ok": False, "error": f"server_check_failed: {str(e)}"}

        # If the server used a different key than we requested, copy it to our desired key
        if server_key != key:
            _log("synthesize_full_worker: copying to desired key", 
                 server_key=server_key, target_key=key)
            try:
                # Use the consistent 's3' client (not '_s3')
                s3.copy_object(
                    Bucket=bucket,  # Target bucket
                    CopySource={"Bucket": server_bucket, "Key": server_key},
                    Key=key,  # Target key
                    MetadataDirective='COPY'  # Preserve metadata
                )
                
                # Verify the copy worked
                copy_head = s3.head_object(Bucket=bucket, Key=key)
                copy_size = copy_head.get("ContentLength", 0)
                copy_type = copy_head.get("ContentType", "")
                
                _log("synthesize_full_worker: copy verification", 
                     original_size=server_size,
                     copied_size=copy_size,
                     size_match=(server_size == copy_size),
                     content_type=copy_type)
                
                if copy_size != server_size:
                    _log("synthesize_full_worker: WARNING - copy size mismatch!")
                
                # Optionally delete the server's temporary key
                if not PRESERVE_ORIGINAL:
                    try:
                        s3.delete_object(Bucket=server_bucket, Key=server_key)
                        _log("synthesize_full_worker: deleted original server file")
                    except Exception as del_error:
                        _log("synthesize_full_worker: failed to delete server file", error=str(del_error))
                        
            except Exception as copy_error:
                _log("synthesize_full_worker: S3 copy failed", error=str(copy_error))
                # Use the server's key as fallback
                key = server_key
                bucket = server_bucket
                _log("synthesize_full_worker: using server's original key as fallback")
        else:
            _log("synthesize_full_worker: server used our requested key directly")

        # Final verification of the audio file
        try:
            final_head = s3.head_object(Bucket=bucket, Key=key)
            final_size = final_head.get("ContentLength", 0)
            final_type = final_head.get("ContentType", "")
            
            _log("synthesize_full_worker: final audio verification", 
                 bucket=bucket,
                 key=key,
                 size=final_size,
                 content_type=final_type)
                 
            if final_size == 0:
                _log("synthesize_full_worker: ERROR - final audio file is empty!")
                return {"ok": False, "error": "final_audio_empty"}
                
        except Exception as e:
            _log("synthesize_full_worker: failed final verification", error=str(e))
            return {"ok": False, "error": f"final_verification_failed: {str(e)}"}

        # Create Connect prompt if needed
        prompt_arn = ""
        if os.getenv("USE_CONNECT_PROMPT", "0") == "1":
            try:
                prompt_id, prompt_arn = _ensure_connect_prompt_for_key(
                    bucket, key, key.split('/')[-1].replace('.wav', '')
                )
                _log("synthesize_full_worker: created Connect prompt", prompt_arn=prompt_arn[:60] + "...")
            except Exception as prompt_error:
                _log("synthesize_full_worker: Connect prompt creation failed", error=str(prompt_error))

        # Mark as ready (only if DDB is enabled)
        if not disable_ddb and 'pk' in event:
            try:
                ddb_mark_ready(pk, bucket=bucket, key=key, final_text=text, prompt_arn=prompt_arn)
                _log("synthesize_full_worker: marked ready in DDB", pk=pk[:12] + "...")
            except Exception as ddb_error:
                _log("synthesize_full_worker: DDB mark ready failed", error=str(ddb_error))
        else:
            _log("synthesize_full_worker: skipping DDB (disabled or no pk)")

        _log("synthesize_full_worker completed successfully", 
             bucket=bucket, 
             key=key,
             final_size=final_size,
             mode="s3_only" if disable_ddb else "ddb")

        return {"ok": True, "bucket": bucket, "key": key, "prompt_arn": prompt_arn, "size": final_size}
        
    except Exception as e:
        _log("synthesize_full_worker failed", error=str(e))
        import traceback
        _log("synthesize_full_worker error traceback", traceback=traceback.format_exc())
        return {"ok": False, "error": str(e)}


# Fix 2: Add a diagnostic function to check audio files
def _diagnose_audio(event):
    """
    Diagnostic function to check audio file properties
    """
    params = event.get("Details", {}).get("Parameters", {}) or {}
    bucket = params.get("job_bucket") or COMPANY_BUCKET
    key = params.get("job_key") or params.get("key") or ""
    
    if not key:
        return {"setAttributes": {"DiagError": "missing_key"}}
    
    try:
        # Get object metadata
        head_response = s3.head_object(Bucket=bucket, Key=key)
        
        # Get object content for analysis (first 1KB)
        get_response = s3.get_object(Bucket=bucket, Key=key, Range="bytes=0-1023")
        content_sample = get_response['Body'].read()
        
        # Analyze the content
        is_wav = content_sample.startswith(b'RIFF') and b'WAVE' in content_sample[:12]
        is_mp3 = content_sample.startswith(b'ID3') or content_sample.startswith(b'\xff\xfb')
        
        # Extract WAV header info if it's a WAV file
        wav_info = {}
        if is_wav and len(content_sample) >= 44:
            # Basic WAV header parsing
            file_size = int.from_bytes(content_sample[4:8], 'little')
            fmt_chunk = content_sample[12:16]
            audio_format = int.from_bytes(content_sample[20:22], 'little')
            channels = int.from_bytes(content_sample[22:24], 'little')
            sample_rate = int.from_bytes(content_sample[24:28], 'little')
            byte_rate = int.from_bytes(content_sample[28:32], 'little')
            
            wav_info = {
                "format": audio_format,
                "channels": channels,
                "sample_rate": sample_rate,
                "byte_rate": byte_rate
            }
        
        return {"setAttributes": {
            "DiagBucket": bucket,
            "DiagKey": key,
            "DiagSize": str(head_response.get("ContentLength", 0)),
            "DiagContentType": head_response.get("ContentType", ""),
            "DiagLastModified": str(head_response.get("LastModified", "")),
            "DiagETag": head_response.get("ETag", ""),
            "DiagIsWAV": str(is_wav),
            "DiagIsMP3": str(is_mp3),
            "DiagWAVChannels": str(wav_info.get("channels", "")),
            "DiagWAVSampleRate": str(wav_info.get("sample_rate", "")),
            "DiagWAVFormat": str(wav_info.get("format", "")),
            "DiagHeaderHex": content_sample[:32].hex() if content_sample else ""
        }}
        
    except Exception as e:
        return {"setAttributes": {"DiagError": str(e)}}


# Fix 3: Add audio format validation to the main synthesis function
def _validate_audio_format(bucket: str, key: str) -> dict:
    """
    Validate that the audio file is properly formatted
    """
    try:
        # Check if file exists and get metadata
        head_response = s3.head_object(Bucket=bucket, Key=key)
        content_length = head_response.get("ContentLength", 0)
        content_type = head_response.get("ContentType", "")
        
        if content_length == 0:
            return {"valid": False, "error": "empty_file", "size": 0}
        
        # Read first few bytes to check format
        get_response = s3.get_object(Bucket=bucket, Key=key, Range="bytes=0-100")
        header_bytes = get_response['Body'].read()
        
        # Check for valid WAV header
        is_wav = header_bytes.startswith(b'RIFF') and b'WAVE' in header_bytes[:12]
        
        if not is_wav:
            return {
                "valid": False, 
                "error": "invalid_wav_header", 
                "size": content_length,
                "header_hex": header_bytes.hex()
            }
        
        return {
            "valid": True, 
            "size": content_length, 
            "content_type": content_type
        }
        
    except Exception as e:
        return {"valid": False, "error": str(e), "size": 0}


def _check_full_ready(event):
    """
    Check if the full audio is ready. Now supports both DDB and S3-only modes.
    Fixed to use the correct S3 client with proper permissions.
    """
    params = event.get("Details", {}).get("Parameters", {}) or {}
    
    if DISABLE_DDB:
        # S3-only mode: check by job_bucket and job_key
        bucket = params.get("job_bucket") or COMPANY_BUCKET
        key = params.get("job_key") or params.get("key") or ""

        try:
            _s3.head_object(Bucket=bucket, Key=key)
            # Final is ready → return playable URL and stop polling
            final_url = s3_url(bucket, key)  # presigned by default
            return {
                "setAttributes": {
                    "ready": "true",
                    "AudioS3Url0": final_url,   # <<—— play this
                    "AudioS3UrlCount": "1",
                    "BatchCount": "1",
                    "HasMore": "false",         # <<—— stop loop
                    "job_bucket": bucket,
                    "job_key": key
                }
            }
        except _s3.exceptions.NoSuchKey:
            pass
        except Exception:
            # if head_object transient error, fall through to not-ready

            pass
        
        if not key:
            return {
                "ready": "false",
                "error": "missing_job_key",
                "setAttributes": {
                    "ready": "false",
                    "AudioS3Url0": "",
                    "prompt_arn": "",
                    "final_text": "",
                    "error": "missing_job_key"
                }
            }

        _log("Checking full audio readiness (S3-only mode)", bucket=bucket, key=key)

        try:
            # Use the s3 client (not _s3) which has proper permissions/role assumption
            head_response = s3.head_object(Bucket=bucket, Key=key)
            content_length = head_response.get("ContentLength", 0)
            
            _log("S3 head_object successful", bucket=bucket, key=key, content_length=content_length)
            
            if content_length > 0:
                # Audio is ready
                s3_url = f"https://{bucket}.s3.{S3_REGION}.amazonaws.com/{key}"
                
                # Try to create Connect prompt if needed
                prompt_arn = ""
                try:
                    prompt_id, prompt_arn = _ensure_connect_prompt_for_key(bucket, key, key.split('/')[-1].replace('.wav', ''))
                    _log("Created Connect prompt for full audio", prompt_arn=prompt_arn[:60] + "...")
                except Exception as prompt_error:
                    _log("Failed to create Connect prompt", error=str(prompt_error))
                
                return {
                    "ready": "true",
                    "state": "ready",
                    "setAttributes": {
                        "ready": "true",
                        "AudioS3Url0": s3_url,
                        "PromptARN0": prompt_arn,
                        "AudioS3UrlCount": "1",
                        "BatchCount": "1",
                        "HasMore": "false",
                        "NextIndexOut": "1",
                        "prompt_arn": prompt_arn,
                        "final_text": "",  # We don't have the original text in S3-only mode
                        "job_bucket": bucket,
                        "job_key": key,
                        "mode": "full_ready_s3"
                    }
                }
            else:
                # Object exists but has no content (still processing)
                _log("S3 object exists but empty - still processing", bucket=bucket, key=key)
                return {
                    "ready": "false",
                    "state": "pending",
                    "setAttributes": {
                        "ready": "false", 
                        "AudioS3Url0": "",
                        "prompt_arn": "",
                        "final_text": "",
                        "HasMore": "true",
                        "job_bucket": bucket,
                        "job_key": key
                    }
                }
                
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            error_message = e.response.get("Error", {}).get("Message", "")
            
            _log("S3 ClientError during full audio check", 
                 error_code=error_code,
                 error_message=error_message,
                 bucket=bucket,
                 key=key)
            
            if error_code in ("404", "NoSuchKey", "NotFound"):
                # Object doesn't exist yet (still processing)
                return {
                    "ready": "false",
                    "state": "pending",
                    "setAttributes": {
                        "ready": "false", 
                        "AudioS3Url0": "",
                        "prompt_arn": "",
                        "final_text": "",
                        "HasMore": "true",
                        "job_bucket": bucket,
                        "job_key": key
                    }
                }
            else:
                # Unexpected S3 error (like 403 Forbidden)
                return {
                    "ready": "false",
                    "error": f"s3_error_{error_code}",
                    "error_message": error_message,
                    "setAttributes": {
                        "ready": "false",
                        "error": f"s3_error_{error_code}",
                        "error_detail": error_message,
                        "job_bucket": bucket,
                        "job_key": key
                    }
                }
        except Exception as e:
            _log("Unexpected error during full audio check", error=str(e))
            return {
                "ready": "false",
                "error": f"unexpected_error",
                "setAttributes": {
                    "ready": "false",
                    "error": "unexpected_error",
                    "error_detail": str(e),
                    "job_bucket": bucket,
                    "job_key": key
                }
            }
    
    else:
        # DDB mode: use existing logic (unchanged)
        pk = params.get("pk") or event.get("pk") or ""
        
        if not pk:
            return {
                "ready": "false",
                "error": "pk_required",
                "setAttributes": {
                    "ready": "false",
                    "AudioS3Url0": "",
                    "prompt_arn": "",
                    "final_text": "",
                    "error": "pk_required"
                }
            }

        _log("Checking full audio readiness (DDB mode)", pk=pk[:12] + "...")

        item = ddb_get(pk)
        if not item:
            return {
                "ready": "false", 
                "error": "item_not_found",
                "setAttributes": {
                    "ready": "false",
                    "AudioS3Url0": "",
                    "prompt_arn": "",
                    "final_text": "",
                    "error": "item_not_found"
                }
            }

        state = str(item.get("state", "")).lower()
        
        if state == "ready":
            bucket = item.get("bucket", COMPANY_BUCKET)
            key = item.get("key", "")
            prompt_arn = item.get("prompt_arn", "")
            final_text = item.get("final_text", "")
            
            s3_url = f"https://{bucket}.s3.{S3_REGION}.amazonaws.com/{key}"
            
            return {
                "ready": "true",
                "state": "ready",
                "setAttributes": {
                    "ready": "true",
                    "AudioS3Url0": s3_url,
                    "PromptARN0": prompt_arn,
                    "AudioS3UrlCount": "1",
                    "BatchCount": "1",
                    "HasMore": "false",
                    "NextIndexOut": "1",
                    "prompt_arn": prompt_arn,
                    "final_text": final_text,
                    "job_bucket": bucket,
                    "job_key": key,
                    "pk": pk,
                    "mode": "full_ready_ddb"
                }
            }
        
        else:  # still pending
            return {
                "ready": "false",
                "state": "pending",
                "setAttributes": {
                    "ready": "false", 
                    "AudioS3Url0": "",
                    "prompt_arn": "",
                    "final_text": "",
                    "HasMore": "true",
                    "pk": pk
                }
            }


def _filler_key_url(category: str | None = None, index: int | None = None):
    """
    Generate S3 key and URL for filler audio.
    Returns (category, index, key, url) tuple.
    """
    cats = list(FILLER_CATS.keys())
    cat = category if category in FILLER_CATS else random.choice(cats)
    n = FILLER_CATS[cat]
    idx = index if (isinstance(index, int) and 1 <= index <= n) else random.randint(1, n)
    
    # Build the S3 key - matches what generate_preset.py created
    key = f"{(FILLER_PREFIX + '/') if FILLER_PREFIX and not FILLER_PREFIX.endswith('/') else FILLER_PREFIX}{cat}/{idx:02d}.wav"
    
    # Build the S3 URL
    url = f"https://{FILLER_BUCKET}.s3.{FILLER_REGION}.amazonaws.com/{key}"
    
    return cat, idx, key, url

def _synthesize_full(event):
    """
    Non-streaming, one-shot synthesis that calls the FastAPI /synthesize endpoint.
    Generates the entire speech at once instead of streaming chunks.
    """
    # Cancel any previous job
    _cancel_previous_job_if_any(event)
    
    # Extract user speech input with detailed logging
    user_speech = _extract_user_speech_input(event)
    _log("DEBUG: User speech extracted for full synthesis", 
         user_speech=user_speech, 
         length=len(user_speech) if user_speech else 0)
    
    if not user_speech:
        _log("No user speech input found, using default")
        user_speech = "안녕하세요"
    
    # Get session ID  
    session_id = _session_id_from_event(event, str(uuid.uuid4()))
    
    _log("Synthesize full starting", 
         session_id=session_id[:8],
         user_speech_preview=user_speech[:100])
    
    try:
        # Debug: Check chatbot configuration
        _log("DEBUG: Chatbot config for full synthesis", 
             chat_url=_redact(CHAT_URL, 30), 
             bypass_chat=BYPASS_CHAT,
             has_token=bool(CHAT_TOKEN))
        
        # Process through chatbot and get response
        chatbot_response = call_chatbot(user_speech, session_id)
        
        _log("DEBUG: Chatbot response for full synthesis", 
             input=user_speech[:50],
             output=chatbot_response[:50],
             same_as_input=(chatbot_response == user_speech))
        
        # If chatbot response is the same as input, something's wrong
        if chatbot_response == user_speech:
            _log("WARNING: Chatbot returned same as input for full synthesis")
        
        # Call the NON-STREAMING FastAPI /synthesize endpoint (not /synthesize_stream_start)
        if not TTS_URL:
            raise RuntimeError("TTS_URL env is required")
        
        # Make sure we use the base /synthesize endpoint, not the streaming one
        base_url = TTS_URL.rstrip("/")
        if base_url.endswith("/synthesize_stream_start"):
            base_url = base_url.replace("/synthesize_stream_start", "/synthesize")
        elif not base_url.endswith("/synthesize"):
            base_url = f"{base_url}/synthesize"
        
        # Use simpler payload that matches your /synthesize endpoint exactly
        payload = {
            "text": chatbot_response,
            "sample_rate": 8000,
            "key_prefix": KEY_PREFIX,
            # Remove the problematic parameters that might be causing tuple issues
            # The /synthesize endpoint should use defaults from your app.py
        }
        
        # Debug: Check the values before creating payload
        _log("DEBUG: TTS parameter values before payload creation",
             TTS_TEMP_raw=TTS_TEMP,
             TTS_TOP_P_raw=TTS_TOP_P,
             TTS_REP_raw=TTS_REP,
             TTS_TEMP_type=type(TTS_TEMP),
             TTS_TOP_P_type=type(TTS_TOP_P))
        
        # Robust type conversion function
        def safe_float_convert(value, default):
            try:
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    return float(value)
                elif isinstance(value, (list, tuple)) and len(value) > 0:
                    return float(value[0])
                else:
                    _log("WARNING: Unexpected type for TTS param", value=value, type=type(value))
                    return float(default)
            except (ValueError, TypeError) as e:
                _log("WARNING: Failed to convert TTS param", value=value, error=str(e))
                return float(default)
        
        # Safely convert all parameters
        temperature_val = safe_float_convert(TTS_TEMP, 0.7)
        top_p_val = safe_float_convert(TTS_TOP_P, 0.9)
        rep_penalty_val = safe_float_convert(TTS_REP, 1.0)
        
        payload = {
            "text": chatbot_response,
            "sample_rate": 8000,
            "key_prefix": KEY_PREFIX,
            "temperature": temperature_val,
            "top_p": top_p_val,
            "repetition_penalty": rep_penalty_val,
            "max_new_tokens": 512,
            "chunk_length": int(chunk_length),
            "use_memory_cache": True
        }
        
        _log("DEBUG: Final payload values", 
             temp_type=type(payload["temperature"]),
             top_p_type=type(payload["top_p"]),
             temp_val=payload["temperature"],
             top_p_val=payload["top_p"],
             rep_val=payload["repetition_penalty"])
        
        _log("DEBUG: Calling NON-STREAMING synthesize endpoint", 
             url=_redact(base_url, 40),
             text_preview=chatbot_response[:100])
        
        # Make HTTP call to your FastAPI app's /synthesize endpoint (NOT streaming)
        tts_resp = _http_post_json(
            base_url,  # Use the corrected URL
            payload,
            token=TTS_TOKEN,
            timeout=int(os.getenv("SYNTH_HTTP_TIMEOUT_SEC", "60"))
        )
        
        _log("DEBUG: FastAPI synthesize response", 
             keys=list(tts_resp.keys()),
             has_url=bool(tts_resp.get("url")))
        
        # Get the URL from the response
        audio_url = tts_resp.get("url") or tts_resp.get("s3_url") or tts_resp.get("regional_url")
        if not audio_url:
            _log("TTS response missing audio URL", keys=list(tts_resp.keys()))
            raise RuntimeError(f"TTS response missing URL: {list(tts_resp.keys())}")
        
        _log("DEBUG: Full synthesis complete", 
             audio_url=_redact(audio_url, 60))
        
        # The FastAPI endpoint already uploaded to S3 and returned a URL
        # Extract key from the response if available, or parse from URL
        s3_key = tts_resp.get("key", "")
        
        # Return response metadata (no streaming, single complete audio file)
        response_attrs = {
            "ready": "true",
            "AudioS3Url0": audio_url,
            "AudioS3UrlCount": "1",
            "BatchCount": "1",
            "HasMore": "false",  # No more parts - this is the complete audio
            "NextIndexOut": "1",
            "ChatAnswer": chatbot_response,
            "UserInput": user_speech,
            "final_text": chatbot_response,
            "sample_rate": str(tts_resp.get("sample_rate", 8000)),
            "job_bucket": TTS_BUCKET,
            "job_key": s3_key,
            "JobId": "",  # No streaming job ID needed
            "latency_ms": str(tts_resp.get("latency_ms", 0)),
        }
        
        # Stringify all values
        response_attrs = {k: (v if isinstance(v, str) else str(v)) for k, v in response_attrs.items()}
        
        _log("Synthesize full complete", 
             user_input=user_speech[:50],
             bot_response=chatbot_response[:50],
             audio_url=_redact(audio_url, 60))
        
        return response_attrs
        
    except Exception as e:
        _log("Synthesize full failed", error=str(e), error_type=type(e).__name__)
        import traceback
        _log("Full synthesis error traceback", traceback=traceback.format_exc())
        
        return {
            "ready": "false",
            "Error": f"synthesize_full_error: {str(e)}",
            "ErrorDetail": str(e),
            "UserInput": user_speech,
            "AudioS3UrlCount": "0",
            "BatchCount": "0",
            "HasMore": "false"
        }


# ========= Main handler (router) =========
def lambda_handler(event, context):
    if _is_lex_code_hook(event):
        return _lex_delegate_with_interrupt_attrs(event)

    # 1) Otherwise, this is your **Amazon Connect** invocation: keep your existing logic.
    session_state = event.get("sessionState", {})  # harmless if Connect calls you
    session_attributes = session_state.get("sessionAttributes", {}) or {}
    params = (event.get("Details", {}) or {}).get("Parameters", {}) or {}
    action = params.get("action") or event.get("action") or ""

    # (Optional) You can also keep these attrs on Connect side if you ever wire Lex->Connect->Lex loops
    session_attributes.update({
        "x-amz-lex:allow-interrupt:*:*": "true",
        "x-amz-lex:audio:start-timeout-ms:*:*": "1200",
        "x-amz-lex:audio:end-timeout-ms:*:*": "400",
        "x-amz-lex:audio:max-length-ms:*:*": "12000"
    })

    # --- your existing logging/warmup/router code follows unchanged ---
    _log("DEBUG Environment", 
         bypass_chat=os.getenv("BYPASS_CHAT", "not_set"),
         chat_url=CHAT_URL[:30] if CHAT_URL else "not_set")

    _log("RAW_EVENT", raw=json.dumps(event, indent=2, default=str))
    global WARMED
    if not WARMED:
        try:
            _http_post_json(_warmup_url(TTS_URL), {})
        except Exception as e:
            logging.warning(f"Warmup call failed: {e}")
        WARMED = True

    _req_filter.set_request_id(getattr(context, "aws_request_id", "-"))

    try:
        preview = json.dumps(event, ensure_ascii=False, default=str)[:2000]
    except Exception:
        preview = (str(event) if event is not None else "null")[:2000]
    _log("EVENT_IN", preview=preview)

    try:
        action = (
            event.get("Details", {}).get("Parameters", {}).get("action")
            or event.get("action") or "start"
        ).lower()
    except Exception:
        action = "start"
    _log("Enhanced handler routing", action=action)

    try:
        speech_input = _extract_user_speech_input(event)
        _log("EXTRACTED_SPEECH", length=len(speech_input), preview=speech_input[:200])
    except Exception as e:
        _log("SPEECH_EXTRACTION_ERROR", error=str(e))

    try:
        if action == "start":
            return _start(event)
        elif action == "warmup":
            _ping_warmup()
            return {"ok": True, "warmed": True}
            # return _warm_once(force=False)
        elif action == "chat_and_stream":
            return _enhanced_chat_and_stream(event)
        elif action == "start_stream":
            if not params.get("filler_category"):
                params = (event.get("Details", {}) or {}).get("Parameters", {}) or {}
                text = (params.get("text") or event.get("text") or "").strip() or filler_text
                attrs = _start_stream(text, filler_text, include_prompts=False)
            return {"setAttributes": attrs}
        elif action == "get_next_batch":
            return _get_next_batch(event)
        elif action == "diagnose_audio":
            return _diagnose_audio(event)
        elif action == "chat_and_stream_cached":
            return _unified_chat_and_stream(event) 
        elif action == "check":
            return _check_ready_by_pk(event)
        elif action in ("synthesize_full", "tts_full"):
            if not params.get("filler_category"):
                ctx = _ctx_from_event(event)
                auto_cat = _choose_category(_ctx_from_event(event))
                params["filler_category"] = auto_cat or "시간벌기형"
                event.setdefault("Details", {}).setdefault("Parameters", {}).update(params)
            return {"setAttributes": _synthesize_full(event)}
        elif action == "synthesize_full_with_filler":
            ctx = _ctx_from_event(event)
            auto_cat = _choose_category(_ctx_from_event(event))
            params["filler_category"] = auto_cat or "시간벌기형"
            event.setdefault("Details", {}).setdefault("Parameters", {}).update(params)
            return {"setAttributes": _synthesize_full_with_filler(event)}
        elif action == "synthesize_full_worker":  # Internal async call
            return _synthesize_full_worker(event)
        elif action == "check_full_ready":
            return _check_full_ready(event)
        elif action == "diag_url":
            return _diag_url(event)
        elif action == "worker":
            return _worker(event)
        elif action == "check_ready":
            return _check_ready_by_pk(event)
        elif action == "get_prompt":
            return _get_prompt(event)        
        elif action == "tts_full_with_filler":
            return _tts_full_with_filler(event)
        elif action == "tts_full_worker":
            return _tts_full_worker(event)
        elif action == "check_full":
            return _check_full(event)
        else:
            return {"error": "bad_action", "message": f"Unknown action: {action}"}
    except Exception as e:
        logger.exception("Invocation failed (router)")
        return {"error": "exception", "message": str(e)}
