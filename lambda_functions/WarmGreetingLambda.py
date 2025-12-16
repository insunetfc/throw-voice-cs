import os, re, time, uuid, urllib.parse, json, boto3
import urllib.request
import random
from concurrent.futures import ThreadPoolExecutor

AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
TTS_BUCKET = os.getenv("TTS_BUCKET", "tts-bucket-250810")
TTS_URL = os.getenv("TTS_URL", "https://ef85da6954e5.ngrok-free.app")
TTS_TOKEN = os.getenv("TTS_TOKEN", "")
KEY_PREFIX_GREETING = os.environ.get("KEY_PREFIX_GREETING", "greetings/intro")
KEY_PREFIX_PROMO = os.environ.get("KEY_PREFIX_PROMO", "greetings/promo")

# OPTIMIZATION: Reuse connections
_s3 = boto3.client("s3", region_name=AWS_REGION)
ddb = boto3.resource("dynamodb", region_name=AWS_REGION)
DDB_INTRO = ddb.Table(os.getenv("CALLER_TABLE", "PhoneIntro"))
FULL_SAMPLE_RATE = 8000
DEFAULT_INTRO_FALLBACK_S3 = os.getenv(
    "DEFAULT_INTRO_FALLBACK_S3",
    "s3://tts-bucket-250810/greetings/guest.wav"
)

# OPTIMIZATION: Thread pool for parallel operations
executor = ThreadPoolExecutor(max_workers=3)

# OPTIMIZATION: Pre-computed promotional audio URL
PROMO_AUDIO_URL = "https://tts-bucket-250810.s3.ap-northeast-2.amazonaws.com/neutral/Promotional_Continue.wav"

def _log(msg, **kw):
    print(json.dumps({"msg": msg, **kw}, ensure_ascii=False))

def _normalize_phone(e164_or_raw: str) -> str:
    if not e164_or_raw:
        return ""
    s = re.sub(r"[^\d+]", "", e164_or_raw)
    if not s.startswith("+") and s.startswith("0"):
        s = "+82" + s[1:]
    return s

def _get_connect_customer_number(evt: dict) -> str:
    contact = (evt.get("Details") or {}).get("ContactData") or {}
    endpoint = contact.get("CustomerEndpoint") or {}
    return endpoint.get("Address") or ""

def _format_addressee(display_name: str | None) -> str:
    """
    - Unknown/empty -> '고객님'
    - Known name -> '<name>고객님' (no space)
    - If name already ends with '고객님' or '님', keep as-is
    - Collapse internal spaces in the name
    """
    n = (display_name or "").strip()
    if not n or n == "고객님":
        return "고객님"
    n = re.sub(r"\s+", "", n)
    if n.endswith(("고객님", "님")):
        return n
    return f"{n}고객님"

def _make_intro_text(display_name: str) -> str:
    addressee = _format_addressee(display_name)
    return f"(friendly) 안녕하세요 {addressee}~! 반갑습니다~"

def _s3_regional_url(bucket: str, key: str, region: str = AWS_REGION) -> str:
    if region.startswith("cn-"):
        host = f"s3.{region}.amazonaws.com.cn"
    else:
        host = f"s3.{region}.amazonaws.com"
    return f"https://{bucket}.{host}/{key.lstrip('/')}"

def _tts_short_to_s3(text: str, key_hint_prefix: str = "greetings/intro"):
    """
    OPTIMIZED: Reduced timeout, better error handling
    """
    payload = {
        "text": text,
        "sample_rate": FULL_SAMPLE_RATE,
        "key_prefix": key_hint_prefix
    }
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        f"{TTS_URL.rstrip('/')}/synthesize",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TTS_TOKEN}",
        },
    )

    audio_url = ""
    result = {}
    try:
        # OPTIMIZATION: Reduced timeout from 15s to 8s
        with urllib.request.urlopen(req, timeout=8) as resp:
            result = json.loads(resp.read().decode())
            audio_url = result.get("url") or ""
            if not audio_url:
                b = result.get("bucket")
                k = result.get("key")
                if b and k:
                    audio_url = _s3_regional_url(b, k, AWS_REGION)
            else:
                m = re.match(r"https://([^./]+)\.s3\.amazonaws\.com/(.+)", audio_url)
                if m:
                    b, k = m.group(1), m.group(2)
                    audio_url = _s3_regional_url(b, k, AWS_REGION)
    except Exception as e:
        _log("urllib synth failed", error=str(e))

    if not audio_url:
        # OPTIMIZATION: Use simpler fallback instead of creating fake audio
        key = f"{key_hint_prefix}/{uuid.uuid4().hex}.wav"
        audio_url = _s3_regional_url(TTS_BUCKET, key, AWS_REGION)
        _log("fallback_url_used", key=key)

    key = result.get("key") or f"{key_hint_prefix}/{uuid.uuid4().hex}.wav"
    return audio_url, key

import botocore

_s3 = boto3.client("s3", region_name=AWS_REGION)

def warm_object(bucket: str, key: str):
    try:
        # touch metadata path (very cheap)
        _s3.head_object(Bucket=bucket, Key=key)

        # optionally read first byte to ensure the object path is “hot”
        _s3.get_object(Bucket=bucket, Key=key, Range='bytes=0-0')

        _log("s3_warm_ok", bucket=bucket, key=key)
    except Exception as e:
        _log("s3_warm_failed", error=str(e))

def _warm_s3_file(url: str):
    """Background task to warm S3 cache - non-blocking"""
    try:
        req = urllib.request.Request(url, method='HEAD')
        urllib.request.urlopen(req, timeout=1)
        _log("s3_warm_success", url=url[:80])
    except Exception as e:
        _log("s3_warm_failed", error=str(e))

def handle_intro_reply(event: dict) -> dict:
    """
    ULTRA-OPTIMIZED: Fastest possible decision making with S3 pre-warming
    """
    start_time = time.time()
    _log("=== INTRO REPLY START ===", timestamp=start_time)
    
    # OPTIMIZATION: Fast parameter extraction - try most common locations first
    details = event.get("Details") or {}
    params = details.get("Parameters") or {}
    
    user_input_raw = (
        params.get("user_input") or
        event.get("user_input") or
        params.get("SttInboxNew") or
        params.get("transcript") or
        ""
    )
    
    user_input = user_input_raw.strip().lower() if user_input_raw else ""
    
    # OPTIMIZATION: Pre-compiled regex for common patterns (if needed)
    # For now, using fastest set-based lookups
    
    decision = "neutral"
    
    if user_input:
        # OPTIMIZATION: Check exact single-word matches first (fastest O(1) lookup)
        if user_input in {"네", "예", "yes", "yeah", "yup", "sure", "ok", "okay", "응", "어"}:
            decision = "yes"
        elif user_input in {"아니", "아니요", "아니오", "no", "nope", "nah", "싫어", "싫어요", "안", "않"}:
            decision = "no"
        # OPTIMIZATION: Only do substring matching for longer inputs
        elif len(user_input) > 2:
            # Positive indicators
            if any(word in user_input for word in ["네", "예", "좋", "관심", "듣고", "들을", "받을", "원해", "알고", "그래", "안녕하세요", "여보세요", "네, 안녕하세요", "네, 여보세요"]):
                decision = "yes"
            # Negative indicators
            elif any(word in user_input for word in ["아니", "싫", "안", "않", "필요없", "괜찮", "끝", "바빠", "바빠요", "바쁘", "바쁨", "급해", "나중", "못해"]):
                decision = "no"
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        _log("decision_made", decision=decision, input=user_input[:30], elapsed_ms=elapsed_ms)
    else:
        _log("empty_input_neutral", decision="neutral")
    
    # OPTIMIZATION: Pre-warm promotional audio for faster playback if user said yes
    if decision == "yes":
        try:
            # Non-blocking S3 warm-up - doesn't delay response
            executor.submit(_warm_s3_file, PROMO_AUDIO_URL)
            # TEMPORARY: Wait to see warmup logs in testing - REMOVE IN PRODUCTION
            # time.sleep(0.1)  # 100ms delay to see logs
        except Exception as e:
            _log("warm_submit_failed", error=str(e))
    
    total_elapsed = int((time.time() - start_time) * 1000)
    _log("=== INTRO REPLY COMPLETE ===", total_ms=total_elapsed)
    
    return {
        "setAttributes": {
            "intro_decision": decision,
            "intro_user_input": user_input[:30]  # Truncate for efficiency
        },
        "ready": "true"
    }

def _async_db_update(phone: str, display_name: str, audio_key: str):
    """
    OPTIMIZATION: Non-blocking DB update
    """
    try:
        DDB_INTRO.update_item(
            Key={"phone_number": phone},
            UpdateExpression="SET display_name=:n, greeting_audio_s3=:g, updated_at=:t",
            ExpressionAttributeValues={
                ":n": display_name,
                ":g": f"{TTS_BUCKET}/{audio_key}",
                ":t": int(time.time())
            },
        )
        _log("db_update_success", phone=phone)
    except Exception as e:
        _log("db_update_failed", error=str(e), phone=phone)

def handle_intro_greet(event):
    start_time = time.time()
    _log("=== INTRO GREET START ===", timestamp=start_time)

    # Parameters
    override_phone = _get_param(event, "phone_number")
    raw_number = override_phone or _get_connect_customer_number(event)
    phone = _normalize_phone(raw_number)

    force_refresh = str(_get_param(event, "force_refresh") or "false").lower() == "true"

    # helper: parse "s3://..." or "bucket/key" to (bucket,key)
    def _to_bucket_key(v: str):
        if not v:
            return ("", "")
        if v.startswith("s3://"):
            p = urllib.parse.urlparse(v)
            return (p.netloc, p.path.lstrip("/"))
        if "/" in v:
            b, k = v.split("/", 1)
            return (b, k)
        return ("", "")

    # Default fallback S3
    fb_bucket, fb_key = _to_bucket_key(DEFAULT_INTRO_FALLBACK_S3)

    # Read DDB (eventual consistency is fine)
    item = None
    db_name = None
    cached_audio = None
    
    try:
        resp = DDB_INTRO.get_item(Key={"phone_number": phone}, ConsistentRead=False)
        item = resp.get("Item")
        
        if item:
            db_name = item.get("display_name")
            cached_audio = item.get("greeting_audio_s3")
            elapsed_ms = int((time.time() - start_time) * 1000)
            _log("ddb_lookup", phone_found=True, has_audio=bool(cached_audio), 
                 audio_value=str(cached_audio)[:80], elapsed_ms=elapsed_ms)
        else:
            elapsed_ms = int((time.time() - start_time) * 1000)
            _log("ddb_lookup", phone_found=False, elapsed_ms=elapsed_ms)
    except Exception as e:
        _log("ddb_get_failed", error=str(e))

    # ---------- Case 1: Phone in DB with personalized audio ----------
    _log("check_conditions", item_exists=bool(item), has_cached_audio=bool(cached_audio), 
         force_refresh=force_refresh, will_use_personalized=(item and cached_audio and not force_refresh))
    
    if item and cached_audio:
        try:
            ca_bucket, ca_key = _to_bucket_key(cached_audio)
            _log("parsing_cached_audio", bucket=ca_bucket, key=ca_key[:50])
            regional_url = _s3_regional_url(ca_bucket, ca_key, AWS_REGION)
            elapsed = int((time.time() - start_time) * 1000)
            _log("personalized_return", elapsed_ms=elapsed, phone=phone, url=regional_url[:80])
            executor.submit(warm_object, ca_bucket, ca_key)
            return {
                "ready": "true",
                "AudioS3Url0": regional_url,
                "AudioS3UrlCount": "1",
                "BatchCount": "1",
                "HasMore": "false",
                "cache_hit": "true",
                "intro": "true",
                "display_name": db_name or "고객님",
                "phone_number": phone,
                "phone_in_db": "true"
            }
        except Exception as e:
            _log("personalized_audio_failed", error=str(e))
            # Fall through to default

    # ---------- Case 2: Phone NOT in DB OR no personalized audio → use default ----------
    try:
        regional_url = _s3_regional_url(fb_bucket, fb_key, AWS_REGION)
        elapsed = int((time.time() - start_time) * 1000)
        _log("default_return", phone_found=bool(item), has_audio=bool(cached_audio), elapsed_ms=elapsed)
        executor.submit(warm_object, fb_bucket, fb_key)
        return {
            "ready": "true",
            "AudioS3Url0": regional_url,
            "AudioS3UrlCount": "1",
            "BatchCount": "1",
            "HasMore": "false",
            "cache_hit": "false",
            "intro": "true",
            "display_name": db_name if item else "고객님",
            "phone_number": phone,
            "used_default": "true",
            "phone_in_db": "true" if item else "false"
        }
    except Exception as e:
        # Last resort: emergency synth
        _log("default_failed", error=str(e))
        text = _make_intro_text("고객님")
        final_url, _ = _tts_short_to_s3(text)
        elapsed = int((time.time() - start_time) * 1000)
        _log("emergency_synth", elapsed_ms=elapsed)
        return {
            "ready": "true",
            "AudioS3Url0": final_url,
            "AudioS3UrlCount": "1",
            "BatchCount": "1",
            "HasMore": "false",
            "cache_hit": "false",
            "intro": "true",
            "display_name": "고객님",
            "phone_number": phone,
            "emergency_synth": "true"
        }


def _get_param(event, key, default=""):
    return ((event.get("Details") or {}).get("Parameters") or {}).get(key, default)

def lambda_handler(event, context):
    """
    OPTIMIZED: Faster routing and error handling
    """
    request_start = time.time()
    request_id = context.aws_request_id if context else "unknown"
    _log("=== LAMBDA INVOKED ===", request_id=request_id)
    
    # OPTIMIZATION: Faster parameter extraction
    params = (event.get("Details", {}).get("Parameters", {})) or {}
    action = params.get("action") or event.get("action")
    
    try:
        if action == "intro_greet":
            result = handle_intro_greet(event)
        elif action == "intro_reply":
            result = handle_intro_reply(event)
        else:
            _log("unknown_action", action=action)
            result = {"error": "unknown action", "action": action, "ready": "true"}
        
        elapsed = int((time.time() - request_start) * 1000)
        _log("=== LAMBDA COMPLETE ===", action=action, total_ms=elapsed, request_id=request_id)
        return result
        
    except Exception as e:
        elapsed = int((time.time() - request_start) * 1000)
        _log("=== LAMBDA ERROR ===", error=str(e), total_ms=elapsed, request_id=request_id)
        # OPTIMIZATION: Return graceful fallback instead of raising
        return {
            "error": str(e),
            "ready": "true",
            "cache_hit": "false",
            "setAttributes": {
                "intro_decision": "neutral",
                "intro_user_input": ""
            }
        }