# ========= Standard Library Imports =========
import os
import json
import uuid
import time
import logging
import re
import base64
import urllib.request
import urllib.error
import subprocess
import tempfile
import random
import hashlib
import unicodedata
from typing import Tuple, Optional, Dict, Any
from urllib.parse import urlparse, unquote, quote
from boto3.dynamodb.conditions import Key  # for lookup_cache_by_response
from typing import Optional
from uuid import uuid4 
import ssl

# ========= Third Party Imports =========
import boto3
from botocore.exceptions import ClientError, BotoCoreError

# ========= Environment Variables =========
# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
CONNECT_REGION = os.getenv("CONNECT_REGION", AWS_REGION)
CONNECT_INSTANCE_ID = os.getenv("CONNECT_INSTANCE_ID", "5b83741e-7823-4d70-952a-519d1ac05e63")

# S3 Configuration
TTS_BUCKET = os.getenv("TTS_BUCKET", "tts-bucket-250810")
TTS_BUCKET = os.getenv("TTS_BUCKET", TTS_BUCKET)
FILLER_REGION = os.getenv("FILLER_REGION", AWS_REGION)
FILLER_PREFIX = os.getenv("FILLER_PREFIX", "")
KEY_PREFIX = os.getenv("KEY_PREFIX", "connect/sessions")
NEGATIVE_KR = ["괜찮", "필요 없", "안 할", "안해요", "관심 없", "됐어요", "안 받"]
END_INTENTS = {"busy", "contact_namecard", "homepage", "phone", "contact_number"}

def is_not_interested(text: str) -> bool:
    t = (text or "").strip()
    return any(k in t for k in NEGATIVE_KR)

# DynamoDB Configuration
ddb = boto3.resource("dynamodb", region_name=AWS_REGION)
# CACHE_TABLE = os.getenv("CACHE_TABLE", "ConnectPromptCache")
CACHE_TABLE = os.getenv("CACHE_TABLE", "ResponseAudio")
RESPONSE_TABLE = os.getenv("RESPONSE_TABLE", "ResponseAudio")
_resp_table = ddb.Table(RESPONSE_TABLE) 
# Cache/table flags
USE_DDB = os.getenv("USE_DDB", "1").lower() in ("1","true","yes","y")
UTT_CACHE = boto3.resource("dynamodb", region_name=AWS_REGION).Table("ResponseAudio")
INBOX = boto3.resource("dynamodb", region_name=AWS_REGION).Table("SttInboxNew")

# External Services
# CHAT_URL = os.getenv("CHAT_URL", "http://15.165.60.45:5000/chat")
CHAT_URL = os.getenv("CHAT_URL", "https://honest-trivially-buffalo.ngrok-free.app/respond")
# CHAT_URL = os.getenv("CHAT_URL", "https://99c0ad659065.ngrok-free.app/respond")
CHAT_TOKEN = os.getenv("CHAT_TOKEN", "")
TTS_URL = os.getenv("TTS_URL", "https://ef85da6954e5.ngrok-free.app/synthesize")
TTS_TOKEN = os.getenv("TTS_TOKEN", "")

# TTS Parameters
FULL_SAMPLE_RATE = int(os.getenv("FULL_SAMPLE_RATE", "8000"))
SYNTH_HTTP_TIMEOUT_SEC = int(os.getenv("SYNTH_HTTP_TIMEOUT_SEC", "45"))
TTS_TEMP = 0.7
TTS_TOP_P = 0.95
TTS_REP = 1.0
chunk_length = 64

# Flow Control
USE_PRESIGN = os.getenv("USE_PRESIGN", "1") == "1"
FORCE_REUPLOAD = os.getenv("FORCE_REUPLOAD", "0") == "1"
BYPASS_CHAT = os.getenv("BYPASS_CHAT", "0").lower() in ("1","true","yes","y")
BATCH = int(os.getenv("STREAM_BATCH", "3"))
prefetch = int(os.getenv("BATCH_LIMIT", "1"))

# Lambda Configuration
ASYNC_FUNCTION_NAME = os.getenv("ASYNC_FUNCTION_NAME", "InvokeBotLambda")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Role Assumption
ASSUME_ROLE_ARN = os.getenv("ASSUME_ROLE_ARN", "")
ASSUME_ROLE_EXTERNAL_ID = os.getenv("ASSUME_ROLE_EXTERNAL_ID", "")

# Connect Prompts
PROMPT_NAME_PREFIX = os.getenv("PROMPT_NAME_PREFIX", "dyn-tts-")

# Miscellaneous
DEFAULT_FILLER_CATEGORY = "시간벌기형"
DEFAULT_INDEX_CATEGORY = 1
PREVIEW_COPY = 1
PRESERVE_ORIGINAL = False

# ========= Constants =========
FILLER_CATS = {"확인": 10, "설명": 10, "공감": 10, "시간벌기형": 10}

# put this near the top of your file
SPECIAL_KR_OVERRIDES = {
    # intent_name: (audio_s3_url, post_action)
    "about_company_homepage": (
        "https://tts-bucket-250810.s3.ap-northeast-2.amazonaws.com/greetings/custom/about_company_homepage.wav",
        "sms_end",   # or "end" if you don't want SMS flow
    ),
    "contact_namecard": (
        "https://tts-bucket-250810.s3.ap-northeast-2.amazonaws.com/greetings/custom/contact_namecard.wav",
        "sms_end",
    ),
    "closing_thanks_end": (
        "https://tts-bucket-250810.s3.ap-northeast-2.amazonaws.com/greetings/custom/thankyou_end.wav",
        "end",
    ),
}


def detect_special_intent_kr(text: str):
    """Return (intent_name, post_action) or (None, None)."""
    if not text:
        return None, None
    t = text.strip()

    # homepage / site
    if any(k in t for k in ["홈페이지", "사이트", "웹사이트", "링크", "주소"]):
        return "about_company_homepage", "sms_end"

    # name card / contact
    if any(k in t for k in ["명함", "연락처", "담당자", "번호 보내", "카톡"]):
        return "contact_namecard", "sms_end"

    # polite closing
    if any(k in t for k in ["감사합니다", "고맙습니다", "수고하세요", "네 알겠습니다"]):
        return "closing_thanks_end", "end"

    return None, None


# ========= AWS Clients =========
def _s3_client():
    if ASSUME_ROLE_ARN:
        sts = boto3.client("sts", region_name=CONNECT_REGION)
        params = {"RoleArn": ASSUME_ROLE_ARN, "RoleSessionName": "connect-tts"}
        if ASSUME_ROLE_EXTERNAL_ID:
            params["ExternalId"] = ASSUME_ROLE_EXTERNAL_ID
        creds = sts.assume_role(**params)["Credentials"]
        _log("Assumed role", role=_redact(ASSUME_ROLE_ARN, 12))
        return boto3.client(
            "s3",
            region_name=CONNECT_REGION,
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
        )
    return boto3.client("s3", region_name=CONNECT_REGION)

# Initialize AWS clients
_s3 = boto3.client("s3", region_name=AWS_REGION)
s3 = _s3_client()
lambda_client = boto3.client("lambda", region_name=AWS_REGION)


def normalize_utt(text: str) -> str:
    s = unicodedata.normalize("NFKC", text).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\.!\?]+$", "", s)  # strip trailing punctuation often unstable in ASR
    return s

def utt_hash(text: str, short=True) -> str:
    norm = normalize_utt(text)
    h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    return h[:16] if short else h


def ddb_get_cached_audio(utterance_or_response: str, locale: str) -> Optional[dict]:
    """
    Single cache lookup - always prefers response_audio_uri
    Replaces both ddb_get_cached_audio_with_context_fixed and 
    ddb_get_cached_audio_response_preferred
    """
    if not USE_DDB:
        return None
    
    uh = utt_hash(utterance_or_response, short=True)
    try:
        resp = UTT_CACHE.get_item(
            Key={"utterance_hash": uh, "locale": locale}
        )
        item = resp.get("Item")
        if not item or item.get("status") != "approved":
            return None
        
        # Always prefer response_audio_uri if available
        preferred_uri = item.get("audio_prev_uri") or item.get("audio_s3_uri")
        
        if preferred_uri:
            result_item = item.copy()
            result_item["audio_s3_uri"] = preferred_uri
            return result_item
        
        return None
        
    except Exception as e:
        _log("Cache lookup failed", error=str(e))
        return None
        
def cache_response_audio_under_both_hashes(
    user_input: str,
    chatbot_response: str, 
    response_audio_uri: str,
    locale: str,
    contact_id: str,
    inbox_id: str
):
    """
    Cache the response audio under both the user input hash AND the chatbot response hash.
    This way, looking up either the user input or chatbot response will return the response audio.
    """
    current_time = int(time.time())
    
    # Cache under user input hash -> response audio
    user_hash_item = {
        "utterance_hash": utt_hash(user_input, short=True),      # User input hash as primary key
        "locale": locale,
        "audio_s3_uri": response_audio_uri,                      # But response audio as the value!
        "status": "approved",
        "cached_text": chatbot_response,                         # Store response text
        "original_utterance": user_input,
        "chatbot_response": chatbot_response,
        "approval_type": "chatbot_response",                     # Mark as response type
        "inbox_id": inbox_id,
        "contact_id": contact_id,
        "created_at": current_time,
        "approved_by": "human_reviewer_response_priority"
    }
    
    # Cache under chatbot response hash -> response audio  
    response_hash_item = {
        "utterance_hash": utt_hash(chatbot_response, short=True), # Response hash as primary key
        "locale": locale,
        "audio_s3_uri": response_audio_uri,                       # Response audio
        "status": "approved",
        "cached_text": chatbot_response,                          # Response text
        "original_utterance": user_input,
        "chatbot_response": chatbot_response,
        "approval_type": "chatbot_response",
        "inbox_id": inbox_id,
        "contact_id": contact_id,
        "created_at": current_time,
        "approved_by": "human_reviewer_response_priority"
    }
    
    try:
        # Store both items - now any lookup will return response audio
        UTT_CACHE.put_item(Item=user_hash_item)
        UTT_CACHE.put_item(Item=response_hash_item)
        
        _log("Cached response audio under both hashes",
             user_hash=user_hash_item["utterance_hash"],
             response_hash=response_hash_item["utterance_hash"],
             audio_uri=_redact(response_audio_uri, 40))
        
        return {
            "success": True,
            "user_hash": user_hash_item["utterance_hash"],
            "response_hash": response_hash_item["utterance_hash"],
            "audio_uri": response_audio_uri
        }
        
    except Exception as e:
        _log("Failed to cache under both hashes", error=str(e))
        return {"success": False, "error": str(e)}

def handle_async_inbox_log(event):
    """
    Handler for async DynamoDB logging
    Runs in background, doesn't block the main call flow
    """
    inbox_id = event.get("inbox_id")
    user_input = event.get("user_input")
    chatbot_response = event.get("chatbot_response")
    locale = event.get("locale", "ko-KR")
    contact_id = event.get("contact_id", "unknown")
    
    _log("Starting async inbox logging", inbox_id=inbox_id[:8])
    
    try:
        # Write to SttInboxNew table (no blocking)
        inbox_item = {
            "inbox_id": inbox_id,
            "contact_id": contact_id,
            "turn_ts": int(time.time()),
            "locale": locale,
            "utterance_text": user_input,
            "utterance_norm": normalize_utt(user_input),
            "utterance_hash": utt_hash(user_input, short=True),
            "proposed_response_text": chatbot_response,
            "response_norm": normalize_utt(chatbot_response),
            "response_hash": utt_hash(chatbot_response, short=True),
            "review_status": "open",
            "needs_review": True,
            "cache_both": True,
            "logged_at": int(time.time())
        }
        
        INBOX.put_item(Item=inbox_item)
        
        _log("Inbox logged successfully", 
             inbox_id=inbox_id[:8],
             user_hash=inbox_item["utterance_hash"][:8],
             response_hash=inbox_item["response_hash"][:8])
        
        return {
            "success": True,
            "inbox_id": inbox_id,
            "logged_at": inbox_item["logged_at"]
        }
        
    except Exception as e:
        _log("Async inbox logging failed", error=str(e), inbox_id=inbox_id)
        return {
            "success": False,
            "error": str(e),
            "inbox_id": inbox_id
        }

def debug_inbox_schema(inbox_id: str):
    """Debug what the actual inbox table schema expects"""
    try:
        # Try the current approach
        resp1 = INBOX.get_item(Key={"inbox_id": inbox_id})
        return {"method1_success": True, "item": resp1.get("Item")}
    except Exception as e1:
        try:
            # Maybe it's contact_id + turn_ts as composite key?
            # This won't work without knowing the sort key value, but let's see the error
            resp2 = INBOX.get_item(Key={"contact_id": "unknown", "turn_ts": 0})
            return {"method2_success": True}
        except Exception as e2:
            return {
                "method1_error": str(e1),
                "method2_error": str(e2),
                "suggestion": "Check your INBOX table key schema in DynamoDB console"
            }

# Fixed approval function that works with your current schema
def approve_both_fixed_schema(inbox_id: str):
    """
    Fixed approval that handles schema correctly and uses dual-hash caching
    """
    try:
        _log("Starting approval with inbox_id", inbox_id=inbox_id[:8])
        
        # Try to get inbox item - handle potential schema issues
        try:
            inbox_resp = INBOX.get_item(Key={"inbox_id": inbox_id})
            inbox_item = inbox_resp.get("Item")
        except Exception as schema_error:
            _log("Schema error getting inbox item", error=str(schema_error))
            
            # If inbox_id doesn't work, maybe scan for it
            try:
                scan_resp = INBOX.scan(
                    FilterExpression="inbox_id = :id",
                    ExpressionAttributeValues={":id": inbox_id},
                    Limit=1
                )
                items = scan_resp.get("Items", [])
                inbox_item = items[0] if items else None
                _log("Found inbox item via scan", found=bool(inbox_item))
            except Exception as scan_error:
                _log("Scan also failed", error=str(scan_error))
                return {"success": False, "error": f"Cannot find inbox item: {str(schema_error)}"}
        
        if not inbox_item:
            return {"success": False, "error": f"Inbox item not found: {inbox_id}"}
        
        # Extract data from inbox item
        user_input = inbox_item.get("utterance_text") or inbox_item.get("user_input", "")
        chatbot_response = inbox_item.get("proposed_response_text") or inbox_item.get("chatbot_response", "")
        locale = inbox_item.get("locale", "ko-KR")
        contact_id = inbox_item.get("contact_id", "unknown")
        
        if not user_input or not chatbot_response:
            return {"success": False, "error": "Missing user_input or chatbot_response in inbox item"}
        
        _log("Inbox item data", 
             user_input=user_input[:50],
             chatbot_response=chatbot_response[:50])
        
        # Generate response audio
        try:
            response_audio_uri = generate_and_upload_tts_audio(
                text=chatbot_response,
                key_prefix=f"approved/responses/{inbox_id}",
                locale=locale,
                voice_style="agent"
            )
            _log("Generated response audio", uri=_redact(response_audio_uri, 40))
        except Exception as tts_error:
            _log("TTS generation failed", error=str(tts_error))
            return {"success": False, "error": f"TTS generation failed: {str(tts_error)}"}
        
        # Use dual-hash caching (response audio under both hashes)
        try:
            cache_result = cache_response_audio_under_both_hashes(
                user_input=user_input,
                chatbot_response=chatbot_response,
                response_audio_uri=response_audio_uri,
                locale=locale,
                contact_id=contact_id,
                inbox_id=inbox_id
            )
            
            if not cache_result["success"]:
                return {"success": False, "error": f"Caching failed: {cache_result.get('error')}"}
                
            _log("Dual-hash caching successful", 
                 user_hash=cache_result["user_hash"],
                 response_hash=cache_result["response_hash"])
                 
        except Exception as cache_error:
            _log("Caching failed", error=str(cache_error))
            return {"success": False, "error": f"Caching failed: {str(cache_error)}"}
        
        # Update inbox status (handle schema carefully)
        try:
            INBOX.update_item(
                Key={"inbox_id": inbox_id},
                UpdateExpression="SET review_status = :status, approved_at = :ts, response_audio_uri = :uri, dual_hash_cached = :dual",
                ExpressionAttributeValues={
                    ":status": "approved_dual_hash_fixed",
                    ":ts": int(time.time()),
                    ":uri": response_audio_uri,
                    ":dual": True
                }
            )
            _log("Updated inbox status", inbox_id=inbox_id[:8])
        except Exception as update_error:
            _log("Inbox update failed but approval succeeded", error=str(update_error))
            # Don't fail the whole operation just because inbox update failed
        
        return {
            "success": True,
            "inbox_id": inbox_id,
            "response_audio_uri": response_audio_uri,
            "user_hash": cache_result["user_hash"],
            "response_hash": cache_result["response_hash"],
            "dual_hash_cached": True,
            "caching_strategy": "response_audio_under_both_hashes"
        }
        
    except Exception as e:
        _log("Approval failed", error=str(e), inbox_id=inbox_id)
        return {"success": False, "error": str(e), "inbox_id": inbox_id}

# Updated handler for approve_both
def handle_approve_both_fixed(event):
    """Handle approve_both with fixed schema handling"""
    params = event.get("Details", {}).get("Parameters", {})
    inbox_id = params.get("inbox_id")
    
    if not inbox_id:
        return {"success": False, "error": "Missing inbox_id parameter"}
    
    # Debug schema first if needed
    debug_result = debug_inbox_schema(inbox_id)
    _log("Schema debug result", debug=debug_result)
    
    # Run the fixed approval
    try:
        result = approve_both_fixed_schema(inbox_id)
        return result
    except Exception as e:
        _log("Handle approve both failed", error=str(e))
        return {"success": False, "error": str(e)}


    
def resolve_audio(text: str, locale: str, chatbot_response_or_none: Optional[str]):
    # 1) Try UTTERANCE hit
    audio_uri = ddb_get_cached_audio(text, locale)
    if audio_uri:
        return {"action": "play_cached", "audio_s3_uri": audio_uri}

    # 2) Try RESPONSE hit (if you have one)
    if chatbot_response_or_none:
        audio_uri = ddb_get_cached_audio(chatbot_response_or_none, locale)
        if audio_uri:
            return {"action": "play_cached", "audio_s3_uri": audio_uri}

    # 3) Miss → log to SttInbox and return neutral+SMS
    try:
        INBOX.put_item(Item={
            "inbox_id": uuid4().hex,        # or use uuid.uuid4().hex if you prefer not to import uuid4
            "ts": int(time.time()),
            "locale": locale,
            "utterance_text": text,
            "utterance_norm": normalize_utt(text),
            "utterance_hash": utt_hash(text, short=True),
            "proposed_response_text": chatbot_response_or_none or "",
            "response_norm": normalize_utt(chatbot_response_or_none) if chatbot_response_or_none else "",
            "response_hash": utt_hash(chatbot_response_or_none, short=True) if chatbot_response_or_none else "",
            "review_status": "open"
        })
    except Exception as e:
        _log("Inbox log failed", error=str(e))

    return {
        "action": "play_neutral_and_sms",
        "neutral_audio_s3_uri": "s3://promo-audio-prefill/ko-KR/neutral_line.wav",
        "send_sms": True
    }

def lookup_cache(h, locale):
    resp = UTT_CACHE.get_item(Key={"utterance_hash": h, "locale": locale})
    item = resp.get("Item")
    return item if item and item.get("status") == "approved" else None

def log_inbox(contact_id, norm_text, raw_text, locale):
    INBOX.put_item(Item={
        "contact_id": contact_id,
        "turn_ts": int(time.time()),
        "raw_text": raw_text,
        "normalized_utterance": norm_text,
        "locale": locale,
        "needs_review": True,
        "review_status": "open"
    })

# CACHE_TABLE = os.getenv("CACHE_TABLE", "ConnectPromptCache")
ddb = boto3.resource("dynamodb", region_name=CONNECT_REGION).Table(CACHE_TABLE)
print(f"[INIT] DynamoDB ENABLED - using table {CACHE_TABLE}")

ASSUME_ROLE_ARN = os.getenv("ASSUME_ROLE_ARN", "")
ASSUME_ROLE_EXTERNAL_ID = os.getenv("ASSUME_ROLE_EXTERNAL_ID", "")
prefetch = int(os.getenv("BATCH_LIMIT", "1"))

TTS_BASE = os.environ.get("TTS_BASE_URL")  # e.g., http://tts.internal:8000

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
    return f"https://{bucket}.s3.{CONNECT_REGION}.amazonaws.com/{key}"

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

def lookup_cache_by_utterance(h, locale):
    resp = UTT_CACHE.get_item(Key={"utterance_hash": h, "locale": locale})
    return resp.get("Item")

def lookup_cache_by_response(resp_h):
    resp = UTT_CACHE.query(
        IndexName="ResponseIndex",
        KeyConditionExpression=Key("response_hash").eq(resp_h)
    )
    items = resp.get("Items", [])
    return items[0] if items else None


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


def _canon(s: str) -> str:
    import unicodedata
    s = (s or "").strip()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s

def h_short(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()

def parse_s3_uri(uri: str):
    # "s3://bucket/key/..." -> ("bucket","key/...")
    if not uri or not uri.startswith("s3://"): 
        return None, None
    _, rest = uri.split("s3://", 1)
    parts = rest.split("/", 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]

def _presign_s3_https(s3_uri: str, expires=3600) -> str:
    bucket, key = parse_s3_uri(s3_uri)
    if not bucket or not key:
        return ""
    # use the already-initialized client
    return _s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )

def _http_post_json_chat(url: str, payload: dict, token: str = None, timeout: float = 4.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context()) as resp:
            body = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            # try JSON; fall back to text
            if "application/json" in ctype or (body[:1] in (b"{", b"[")):
                return json.loads(body.decode("utf-8", "ignore"))
            return {"raw": body.decode("utf-8", "ignore")}
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", "ignore")
        raise RuntimeError(f"HTTP {e.code}: {err_body[:200]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"URL error: {e.reason}")

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

def analyze_customer_sentiment(user_speech: str) -> str:
    """
    Analyze customer speech to determine appropriate neutral message type
    Returns: "general", "busy_response", "consideration", "not_interested"
    """
    user_lower = user_speech.lower()
    
    # Busy/time-pressed indicators
    busy_keywords = ["바빠", "바쁘", "시간", "급해", "나중에", "지금은", "못해"]
    if any(keyword in user_lower for keyword in busy_keywords):
        return "busy_response"
    
    # Not interested indicators  
    not_interested_keywords = ["관심 없", "필요 없", "안 할래", "그만", "끊", "싫어", "아니"]
    if any(keyword in user_lower for keyword in not_interested_keywords):
        return "not_interested"
    
    # Consideration/hesitation indicators
    consideration_keywords = ["생각해", "고민", "검토", "비교", "확인", "알아보", "좀 더"]
    if any(keyword in user_lower for keyword in consideration_keywords):
        return "consideration"
    
    # Default to general
    return "general"

def get_neutral_audio_url(customer_response_type="general", index=1):
    """
    Return neutral audio URL directly - no S3 head call
    Let S3 return 404 if file doesn't exist (handled by Connect)
    """
    key = f"neutral/{customer_response_type}/{index:02d}.wav"
    
    # If neutrals are public, use direct URL (no presigning)
    if os.getenv("NEUTRAL_PUBLIC", "1") == "1":
        return f"https://{TTS_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"
    
    # Otherwise presign (but still no head check)
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": TTS_BUCKET, "Key": key},
        ExpiresIn=300
    )

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
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logger = logging.getLogger("connect-tts")

# at top-level
import json, time, os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LEVELS = {"NONE": 0, "ERROR": 1, "WARN": 2, "INFO": 3, "DEBUG": 4, "TRACE": 5}

def _should(level: str) -> bool:
    return _LEVELS.get(level, 3) <= _LEVELS.get(LOG_LEVEL, 3)

def _log(msg: str, level: str = "INFO", **fields):
    # Always safe print to stdout (CloudWatch captures it)
    if not _should(level):
        return
    try:
        rec = {"ts": int(time.time()), "level": level, "msg": msg}
        if fields:
            rec.update(fields)
        print(json.dumps(rec, ensure_ascii=False, default=str), flush=True)
    except Exception as e:
        # never fail the lambda due to logging
        print(f'{{"level":"ERROR","msg":"LOGFAIL","why":"{e}","orig":"{msg}"}}', flush=True)

def _dbg(msg: str, **fields):
    _log(msg, level="DEBUG", **fields)


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


def _redact(s: Optional[str], keep: int = 8) -> str:
    if not s:
        return ""
    return (s[:keep] + "...") if len(s) > keep else "***"

# ========= AWS clients =========
def _s3_client():
    if ASSUME_ROLE_ARN:
        sts = boto3.client("sts", region_name=CONNECT_REGION)
        params = {"RoleArn": ASSUME_ROLE_ARN, "RoleSessionName": "connect-tts"}
        if ASSUME_ROLE_EXTERNAL_ID:
            params["ExternalId"] = ASSUME_ROLE_EXTERNAL_ID
        creds = sts.assume_role(**params)["Credentials"]
        _log("Assumed role", role=_redact(ASSUME_ROLE_ARN, 12))
        return boto3.client(
            "s3",
            region_name=CONNECT_REGION,
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
        )
    return boto3.client("s3", region_name=CONNECT_REGION)



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

def chatbot_request(user_text: str, intent_hint: str | None = None, session_id: str | None = None) -> str:
    """Stable contract for both sync and async paths. Returns RAW reply text."""
    if not CHAT_URL:
        return ""
    payload = {"text": user_text}
    if intent_hint:
        payload["intent"] = intent_hint
    if session_id:
        payload["session_id"] = session_id

    try:
        data = _http_post_json_chat(CHAT_URL, payload, token=CHAT_TOKEN, timeout=4.0)
        return (data.get("response") or data.get("message") or data.get("answer") or "").strip("\n")
    except RuntimeError as e:
        # If the server rejects unexpected keys (422), retry with minimal {"text":...}
        if "422" in str(e):
            data = _http_post_json_chat(CHAT_URL, {"text": user_text}, token=CHAT_TOKEN, timeout=4.0)
            return (data.get("response") or data.get("message") or data.get("answer") or "").strip("\n")
        raise


def _upload_bytes_to_s3(data: bytes, content_type: str = "audio/wav", key: Optional[str] = None) -> Tuple[str, str]:
    put_key = key or f"{KEY_PREFIX.rstrip('/')}/{uuid.uuid4().hex}.wav"
    extra = {"ContentType": content_type}
    if os.getenv("CROSS_ACCOUNT_ACL", "0") == "1":
        extra["ACL"] = "bucket-owner-full-control"
    _s3.put_object(Bucket=TTS_BUCKET, Key=put_key, Body=data, **extra)
    url = _s3.generate_presigned_url(
        "get_object", Params={"Bucket": TTS_BUCKET, "Key": put_key}, ExpiresIn=300
    )
    _log("Uploaded audio to S3", bucket=TTS_BUCKET, key=put_key, content_type=content_type)

    # ✅ Add preview generation here
    try:
        if os.getenv("PREVIEW_COPY", "0") == "1":
            preview_bytes = make_pcm16_preview_from_ulaw_wav(data)
            preview_key = put_key.replace("/full/", "/preview/").replace("/full_cache/", "/preview/")
            _s3.put_object(
                Bucket=TTS_BUCKET,
                Key=preview_key,
                Body=preview_bytes,
                ContentType="audio/wav"
            )
            _log("Preview written", preview_key=preview_key, size=len(preview_bytes))
    except Exception as e:
        _log("Preview generation failed", error=str(e))

    return url, put_key

import json, re, unicodedata

def _canon(text: str) -> str:
    """NFKC, collapse whitespace, strip trailing full-stops/ellipses (KR/EN)."""
    t = unicodedata.normalize("NFKC", str(text))
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"[\.。ㆍ·…‥]+$", "", t)
    return t

def _extract_params(evt: dict):
    d = evt.get("Details") or {}
    p = d.get("Parameters") or {}
    if isinstance(p, str):
        try:
            p = json.loads(p)
        except Exception:
            p = {}
    return d, p

def get_user_input(evt: dict) -> str:
    """
    Preferred order:
      1) Lex slot: $.Lex.Slots.UserInput
      2) Parameters: user_input / text
      3) ContactData: CustomerInput / Attributes.UserInput / Attributes.CustomerInput
      4) Lex transcript & other fallbacks
    Always return canon-normalized text.
    """
    d = evt.get("Details") or {}
    p = d.get("Parameters") or {}
    if isinstance(p, str):
        try:
            p = json.loads(p)
        except Exception:
            p = {}
    lex = d.get("Lex") or {}
    contact = d.get("ContactData") or {}
    attrs = contact.get("Attributes") or {}

    # 1) Lex slot
    slots = lex.get("Slots") or {}
    slot_val = slots.get("UserInput")
    if isinstance(slot_val, str) and slot_val.strip():
        return _canon(slot_val)

    # 2) Parameters
    for key in ("user_input", "UserInput", "CaptureUtterance", "text", "utterance", "speech"):
        v = p.get(key)
        if isinstance(v, str) and v.strip():
            return _canon(v)

    # 3) ContactData
    v = contact.get("CustomerInput")
    if isinstance(v, str) and v.strip():
        return _canon(v)

    for key in ("UserInput", "CustomerInput"):
        v = attrs.get(key)
        if isinstance(v, str) and v.strip():
            return _canon(v)

    # 4) Lex transcript & other fallbacks
    for v in (
        lex.get("InputTranscript"),
        evt.get("InputTranscript"),
        evt.get("inputTranscript"),
    ):
        if isinstance(v, str) and v.strip():
            return _canon(v)

    try:
        from_this = _extract_user_speech_input(evt)
        if from_this:
            return _canon(from_this)
    except Exception:
        pass

    return ""


# def get_user_input(evt: dict) -> str:
#     """
#     Preferred order:
#       1) Lex slot: $.Lex.Slots.UserInput (Connect block maps this)
#       2) Parameters: user_input / text
#       3) Lex transcript fields as fallback
#     Always return canon-normalized text.
#     """
#     d, p = _extract_params(evt)
#     lex = d.get("Lex") or {}

#     # 1) Lex slot (what your Contact Flow should map now)
#     slots = lex.get("Slots") or {}
#     slot_val = slots.get("UserInput")
#     if isinstance(slot_val, str) and slot_val.strip():
#         return _canon(slot_val)

#     # 2) Parameters that your flow may set
#     for key in ("user_input", "UserInput", "CaptureUtterance", "text", "utterance", "speech"):
#         v = p.get(key)
#         if isinstance(v, str) and v.strip():
#             return _canon(v)

#     # 3) As a final fallback, look at Lex input transcript-esque fields
#     for v in (
#         lex.get("InputTranscript"),
#         evt.get("InputTranscript"),
#         evt.get("inputTranscript"),
#     ):
#         if isinstance(v, str) and v.strip():
#             return _canon(v)

#     # Last-ditch: your old extractor (if still present)
#     try:
#         from_this = _extract_user_speech_input(evt)  # existing function in your file
#         if from_this:
#             return _canon(from_this)
#     except Exception:
#         pass

#     return ""


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
        # params.get("Details.Parameters.user_input"),
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

# Make sure your call_chatbot function is exactly as it was working before:
# def call_chatbot(user_input: str, session_id: str = None) -> str:
#     """
#     Chatbot integration function - EXACTLY as it was working before
#     """
#     if not CHAT_URL:
#         _log("No CHAT_URL configured, returning original input", input=user_input[:50])
#         return user_input
    
#     if BYPASS_CHAT:
#         _log("BYPASS_CHAT is enabled, returning original input")
#         return user_input
    
#     # Generate session ID if not provided
#     if not session_id:
#         session_id = str(uuid.uuid4())
    
#     # Prepare payload matching your chatbot_connect.py format
#     payload = {
#         "session_id": session_id,
#         "question": user_input.strip()
#     }
    
#     headers = {
#         'Content-Type': 'application/json'
#     }
    
#     # Add authorization if token is provided
#     if CHAT_TOKEN:
#         headers['Authorization'] = f'Bearer {CHAT_TOKEN}'
    
#     try:
#         _log("Calling chatbot", 
#              url=_redact(CHAT_URL, 30), 
#              session_id=session_id[:8], 
#              input_length=len(user_input),
#              payload_preview=str(payload)[:200])
        
#         # Make the HTTP request
#         data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
#         req = urllib.request.Request(CHAT_URL, data=data, headers=headers, method="POST")
        
#         with urllib.request.urlopen(req, timeout=30) as response:
#             response_data = response.read()
#             chat_response = json.loads(response_data.decode("utf-8"))
        
#         _log("Raw chatbot response", raw_response=str(chat_response)[:500])
        
#         # Extract the answer from response
#         chatbot_answer = extract_chatbot_answer(chat_response)
        
#         if chatbot_answer and chatbot_answer.strip():
#             _log("Chatbot response received", 
#                  input_preview=user_input[:50], 
#                  output_preview=chatbot_answer[:50],
#                  changed=(chatbot_answer.strip() != user_input.strip()))
#             return chatbot_answer.strip()
#         else:
#             _log("Empty chatbot response, returning original input")
#             return user_input
            
#     except Exception as e:
#         _log("Chatbot call failed, returning original input", 
#              error=str(e), 
#              error_type=type(e).__name__)
#         return user_input

def call_chatbot(user_input: str, session_id: str | None = None, intent_hint: str | None = None) -> str:
    """
    Deterministic chatbot call.
    Uses {"text": "...", "intent": "...?"} to match your /respond endpoint.
    Returns RAW reply (no normalization) so hashing matches ResponseAudio seeding.
    """
    if not CHAT_URL:
        _log("No CHAT_URL configured, returning original input", input=user_input[:50])
        return user_input

    if BYPASS_CHAT:
        _log("BYPASS_CHAT is enabled, returning original input")
        return user_input

    # Minimal payload per your working test
    payload = {"text": user_input}
    if intent_hint:
        payload["intent"] = intent_hint
    if session_id:  # optional; include only if you want the server to use it
        payload["session_id"] = session_id

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if CHAT_TOKEN:
        headers["Authorization"] = f"Bearer {CHAT_TOKEN}"

    try:
        _log("Calling chatbot",
             url=_redact(CHAT_URL, 30),
             session_id=(session_id[:8] if session_id else None),
             input_length=len(user_input),
             payload_preview=str({k: (payload[k] if k != "text" else (payload[k][:30] + "…"))
                                  for k in payload})[:200])

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(CHAT_URL, data=data, headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=6) as response:
            body = response.read()
            chat_response = json.loads(body.decode("utf-8", "ignore"))

        _log("Raw chatbot response", raw_keys=list(chat_response.keys()))

        # Accept common fields without further normalization
        for k in ("response", "message", "answer", "agent_response", "text"):
            v = chat_response.get(k)
            if isinstance(v, str) and v.strip():
                out = v
                _log("Chatbot response received",
                    input_preview=user_input[:50],
                    output_preview=out[:50],
                    changed=(out.strip() != user_input.strip()))
                return out  # RAW

        _log("Empty chatbot response, returning original input")
        return user_input

    except urllib.error.HTTPError as e:
        # If server rejected unexpected keys, retry with minimal {"text": ...}
        err = e.read().decode("utf-8", "ignore")
        _log("Chatbot HTTPError", code=e.code, err_preview=err[:200])
        if e.code == 422 and ("session_id" in payload or "intent" in payload):
            try:
                minimal = {"text": user_input}
                data = json.dumps(minimal, ensure_ascii=False).encode("utf-8")
                req = urllib.request.Request(CHAT_URL, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=4) as response2:
                    body2 = response2.read()
                    chat_response2 = json.loads(body2.decode("utf-8", "ignore"))
                for k in ("response", "message", "answer"):
                    v = chat_response2.get(k)
                    if isinstance(v, str) and v.strip():
                        _log("Chatbot retry succeeded", output_preview=v[:50])
                        return v  # RAW
            except Exception as e2:
                _log("Chatbot retry failed", error=str(e2))

        _log("Chatbot call failed, returning original input",
             error=str(e), error_type=type(e).__name__)
        return user_input

    except Exception as e:
        _log("Chatbot call failed, returning original input",
             error=str(e), error_type=type(e).__name__)
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

# def _unified_voice_response_handler_ultra_fast(event):
#     """
#     GUARANTEED <100ms response on cache miss
#     Zero S3 heads, zero scans, zero blocking calls
#     """
#     user_speech = get_user_input(event) or "안녕하세요"
    
#     locale = "ko-KR"
#     contact_id = event.get("Details", {}).get("ContactData", {}).get("ContactId", "unknown")
    
#     # Single cache lookup (GetItem only - O(1))
#     cache_result = ddb_get_cached_audio(user_speech, locale)
#     if cache_result:
#         _log("Cache hit", hash=utt_hash(user_speech, short=True)[:8])
#         return {
#             "ready": "true",
#             "AudioS3Url0": cache_result["audio_s3_uri"],
#             "AudioS3UrlCount": "1",
#             "BatchCount": "1",
#             "HasMore": "false",
#             "cache_hit": "true",
#             "UserInput": user_speech,
#             "send_sms": "false",
#             "neutral_message": "false"
#         }
    
#     # Cache MISS - return instantly
#     inbox_id = uuid4().hex
    
#     # Fire async (non-blocking)
#     trigger_async_chatbot_and_logging(
#         inbox_id=inbox_id,
#         user_input=user_speech,
#         locale=locale,
#         contact_id=contact_id
#     )
    
#     # Get sentiment (local operation, <1ms)
#     customer_sentiment = analyze_customer_sentiment(user_speech)
    
#     # Get neutral URL (NO S3 head call!)
#     random_index = random.randint(1, 3)
#     neutral_audio_url = get_neutral_audio_url(customer_sentiment, index=random_index)
    
#     _log("Cache miss - instant return", inbox_id=inbox_id[:8])
    
#     # Return immediately
#     return {
#         "ready": "true",
#         "AudioS3Url0": neutral_audio_url,
#         "AudioS3UrlCount": "1",
#         "BatchCount": "1",
#         "HasMore": "false",
#         "cache_hit": "false",
#         "cache_miss": "true",
#         "send_sms": "true",
#         "neutral_message": "true",
#         "UserInput": user_speech,
#         "inbox_id": inbox_id,
#         "processing": "async"
#     }

def _unified_voice_response_handler_ultra_fast(event):
    """
    Ultra-fast handler that still tries to return the chatbot-generated audio
    on cache miss.
    """
    # intents we want to end right away (kept from your version)
    END_INTENTS = {"busy", "contact_namecard", "homepage", "phone", "contact_number"}

    # 0) user speech
    user_speech = get_user_input(event)
    if not user_speech:
        _log("No input detected, returning neutral", level="INFO")
        neutral_url = get_neutral_audio_url("general", index=1)
        return {
            "ready": "true",
            "AudioS3Url0": neutral_url,
            "AudioS3UrlCount": "1",
            "BatchCount": "1",
            "HasMore": "false",
            "chatbot_response": "",
            "user_input": "",
            "neutral_message": "true",
        }

    # locale/contact
    locale = (
        event.get("locale")
        or event.get("Details", {}).get("Parameters", {}).get("locale")
        or "ko-KR"
    )
    contact_id = event.get("Details", {}).get("ContactData", {}).get("ContactId", "unknown")
    inbox_id = uuid4().hex  # we’ll use this if we log or cache

    # 1) KR hard overrides (homepage, namecard, polite closing, …)
    override_intent, override_post_action = detect_special_intent_kr(user_speech)
    if override_intent:
        audio_url, default_post_action = SPECIAL_KR_OVERRIDES[override_intent]
        return {
            "ready": "true",
            "AudioS3Url0": audio_url,
            "AudioS3UrlCount": "1",
            "BatchCount": "1",
            "HasMore": "false",
            "intent": override_intent,
            "post_action": override_post_action or default_post_action or "end",
            "UserInput": user_speech,
        }

    # 2) quick busy detector — FIXED: actually define audio_url here
    sentiment = analyze_customer_sentiment(user_speech)
    if sentiment == "busy_response":
        # reuse your neutral bucket layout
        audio_url = get_neutral_audio_url("busy_response", index=random.randint(1, 3))
        return {
            "ready": "true",
            "AudioS3Url0": audio_url,
            "AudioS3UrlCount": "1",
            "BatchCount": "1",
            "HasMore": "false",
            "intent": "busy_response",
            "post_action": "end",
            "UserInput": user_speech,
        }

    # 3) check DynamoDB cache first (single GetItem, fast)
    cache_hit = ddb_get_cached_audio(user_speech, locale)
    if cache_hit:
        _log("Cache hit (utterance)", hash=utt_hash(user_speech, short=True)[:8])
        return {
            "ready": "true",
            "AudioS3Url0": cache_hit["audio_s3_uri"],
            "AudioS3UrlCount": "1",
            "BatchCount": "1",
            "HasMore": "false",
            "cache_hit": "true",
            "UserInput": user_speech,
            "chatbot_response": cache_hit.get("chatbot_response", ""),
        }

    # 4) no cache → call chatbot
    try:
        chat_payload = {"text": user_speech}
        chat_resp = _http_post_json_chat(CHAT_URL, chat_payload, CHAT_TOKEN, timeout=4.0)
        # your FastAPI chatbot returns "agent_response"
        chatbot_response = (
            chat_resp.get("agent_response")
            or chat_resp.get("reply")
            or chat_resp.get("text")
            or "네, 확인했습니다."
        )
    except Exception as e:
        _log("Chatbot call failed, falling back to neutral", error=str(e))
        neutral_url = get_neutral_audio_url("general", index=random.randint(1, 3))
        return {
            "ready": "true",
            "AudioS3Url0": neutral_url,
            "AudioS3UrlCount": "1",
            "BatchCount": "1",
            "HasMore": "false",
            "cache_hit": "false",
            "UserInput": user_speech,
            "chatbot_response": "",
            "neutral_message": "true",
        }

    # 5) turn chatbot text into TTS (your existing helper)
    try:
        # key_prefix can be anything structured
        key_prefix = f"connect/sessions/{contact_id}/{inbox_id}"
        tts_s3_uri = generate_and_upload_tts_audio(
            text=chatbot_response,
            key_prefix=key_prefix,
            locale=locale,
            voice_style="agent",
        )
        # presign so Connect can play it
        audio_url = _presign_s3_https(tts_s3_uri)
    except Exception as e:
        _log("TTS generation failed, falling back to neutral", error=str(e))
        neutral_url = get_neutral_audio_url("general", index=random.randint(1, 3))
        return {
            "ready": "true",
            "AudioS3Url0": neutral_url,
            "AudioS3UrlCount": "1",
            "BatchCount": "1",
            "HasMore": "false",
            "cache_hit": "false",
            "UserInput": user_speech,
            "chatbot_response": chatbot_response,
            "neutral_message": "true",
        }

    # 6) best-effort cache under BOTH hashes (user input + chatbot response)
    try:
        cache_response_audio_under_both_hashes(
            user_input=user_speech,
            chatbot_response=chatbot_response,
            response_audio_uri=tts_s3_uri,
            locale=locale,
            contact_id=contact_id,
            inbox_id=inbox_id,
        )
    except Exception as e:
        _log("Dual-hash cache failed (non-fatal)", error=str(e))

    # 7) final return to Connect
    return {
        "ready": "true",
        "AudioS3Url0": audio_url,
        "AudioS3UrlCount": "1",
        "BatchCount": "1",
        "HasMore": "false",
        "cache_hit": "false",
        "UserInput": user_speech,
        "chatbot_response": chatbot_response,
        "inbox_id": inbox_id,
        "processing": "sync",
    }




def trigger_async_chatbot_and_logging(inbox_id: str, user_input: str, 
                                       locale: str, contact_id: str):
    """
    Fire-and-forget async Lambda for BOTH chatbot call AND DDB logging
    This runs in the background after response is already sent
    """
    try:
        payload = {
            "action": "async_chatbot_and_log",
            "inbox_id": inbox_id,
            "user_input": user_input,
            "locale": locale,
            "contact_id": contact_id,
            "timestamp": int(time.time())
        }
        
        # InvocationType='Event' = fire-and-forget
        response = lambda_client.invoke(
            FunctionName=ASYNC_FUNCTION_NAME,
            InvocationType='Event',
            Payload=json.dumps(payload, ensure_ascii=False).encode('utf-8')
        )
        
        status = response.get('StatusCode', 0)
        if status == 202:
            _log("Async chatbot+logging triggered", inbox_id=inbox_id[:8])
        else:
            _log("Async trigger unexpected status", status=status)
        
    except Exception as e:
        _log("Async trigger failed (non-critical)", error=str(e))


def handle_async_chatbot_and_log(event):
    """
    Background handler - respects BYPASS_CHAT flag
    """
    inbox_id = event.get("inbox_id")
    user_input = event.get("user_input")
    locale = event.get("locale", "ko-KR")
    contact_id = event.get("contact_id", "unknown")
    
    _log("Async processing", inbox_id=inbox_id[:8])
    
    # Respect BYPASS_CHAT flag
    if BYPASS_CHAT:
        chatbot_response = user_input
        _log("Chatbot bypassed", inbox_id=inbox_id[:8])
    else:
        try:
            session_id = contact_id
            chatbot_response = call_chatbot(user_input, session_id)
            _log("Chatbot response received", response_preview=chatbot_response[:50])
        except Exception as e:
            _log("Chatbot failed", error=str(e))
            chatbot_response = user_input
    
    # Write to DDB (GetItem only, no scans)
    try:
        inbox_item = {
            "inbox_id": inbox_id,
            "contact_id": contact_id,
            "turn_ts": int(time.time()),
            "locale": locale,
            "utterance_text": user_input,
            "utterance_norm": normalize_utt(user_input),
            "utterance_hash": utt_hash(user_input, short=True),
            "proposed_response_text": chatbot_response,
            "response_norm": normalize_utt(chatbot_response),
            "response_hash": utt_hash(chatbot_response, short=True),
            "review_status": "open",
            "needs_review": True,
            "cache_both": True,
            "logged_at": int(time.time())
        }
        
        INBOX.put_item(Item=inbox_item)  # PutItem only (O(1))
        _log("Logged successfully", inbox_id=inbox_id[:8])
        
        return {"success": True, "inbox_id": inbox_id}
        
    except Exception as e:
        _log("Logging failed", error=str(e))
        return {"success": False, "error": str(e)}


def _unified_voice_response_handler_hybrid(event):
    """
    HYBRID approach: Try quick chatbot (with timeout), fallback to async
    Best of both worlds - fast when chatbot is fast, instant when slow
    """
    user_speech = get_user_input(event) or "안녕하세요"
    
    locale = "ko-KR"
    contact_id = event.get("Details", {}).get("ContactData", {}).get("ContactId", "unknown")
    
    # Step 1: Check cache (same as before)
    try:
        utterance_cache = ddb_get_cached_audio(user_speech, locale)
        if utterance_cache:
            _log("Cache hit", hash=utt_hash(user_speech, short=True)[:8])
            return {
                "ready": "true",
                "AudioS3Url0": utterance_cache["audio_s3_uri"],
                "AudioS3UrlCount": "1",
                "BatchCount": "1",
                "HasMore": "false",
                "cache_hit": "true",
                "UserInput": user_speech,
                "send_sms": "false",
                "neutral_message": "false"
            }
    except Exception as e:
        _log("Cache check failed", error=str(e))
    
    # Step 2: Try QUICK chatbot call with timeout
    session_id = _session_id_from_event(event, str(uuid.uuid4()))
    chatbot_response = None
    chatbot_success = False
    
    try:
        # Try chatbot with SHORT timeout (200ms max)
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Chatbot timeout")
        
        # Set 200ms timeout for chatbot
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, 0.2)  # 200ms timeout
        
        try:
            chatbot_response = call_chatbot(user_speech, session_id)
            signal.alarm(0)  # Cancel alarm
            chatbot_success = True
            _log("Quick chatbot success", response_preview=chatbot_response[:50])
        except TimeoutError:
            signal.alarm(0)
            _log("Chatbot timeout - will use async")
            chatbot_response = None
        
    except Exception as e:
        _log("Chatbot quick attempt failed", error=str(e))
        chatbot_response = None
    
    # Step 3: If we got chatbot response quickly, check its cache
    if chatbot_success and chatbot_response:
        try:
            response_cache = ddb_get_cached_audio(chatbot_response, locale)
            if response_cache:
                _log("Cache hit via quick chatbot response")
                return {
                    "ready": "true",
                    "AudioS3Url0": response_cache["audio_s3_uri"],
                    "AudioS3UrlCount": "1",
                    "BatchCount": "1",
                    "HasMore": "false",
                    "cache_hit": "true",
                    "UserInput": user_speech,
                    "ChatAnswer": chatbot_response,
                    "send_sms": "false",
                    "neutral_message": "false"
                }
        except Exception as e:
            _log("Response cache check failed", error=str(e))
    
    # Step 4: Cache miss - return instant neutral + async everything
    inbox_id = uuid4().hex
    
    if chatbot_success and chatbot_response:
        # We have chatbot response, just log async
        trigger_async_inbox_logging_with_response(
            inbox_id=inbox_id,
            user_input=user_speech,
            chatbot_response=chatbot_response,
            locale=locale,
            contact_id=contact_id
        )
    else:
        # Need to get chatbot response async
        trigger_async_chatbot_and_logging(
            inbox_id=inbox_id,
            user_input=user_speech,
            locale=locale,
            contact_id=contact_id
        )
    
    customer_sentiment = analyze_customer_sentiment(user_speech)
    neutral_audio_url = get_neutral_audio_url(customer_sentiment, index=1)
    
    _log("Returning instant neutral", inbox_id=inbox_id[:8])
    
    return {
        "ready": "true",
        "AudioS3Url0": neutral_audio_url,
        "AudioS3UrlCount": "1",
        "BatchCount": "1",
        "HasMore": "false",
        "cache_hit": "false",
        "cache_miss": "true",
        "send_sms": "true",
        "neutral_message": "true",
        "UserInput": user_speech,
        "ChatAnswer": chatbot_response or "",
        "inbox_id": inbox_id,
        "processing": "hybrid_async"
    }


def trigger_async_inbox_logging_with_response(inbox_id: str, user_input: str,
                                                chatbot_response: str, locale: str, 
                                                contact_id: str):
    """
    Async logging when we already have the chatbot response
    """
    try:
        payload = {
            "action": "async_inbox_log_with_response",
            "inbox_id": inbox_id,
            "user_input": user_input,
            "chatbot_response": chatbot_response,
            "locale": locale,
            "contact_id": contact_id,
            "timestamp": int(time.time())
        }
        
        lambda_client.invoke(
            FunctionName=ASYNC_FUNCTION_NAME,
            InvocationType='Event',
            Payload=json.dumps(payload, ensure_ascii=False).encode('utf-8')
        )
        _log("Async logging triggered (with response)", inbox_id=inbox_id[:8])
    except Exception as e:
        _log("Async trigger failed", error=str(e))


def handle_async_inbox_log_with_response(event):
    """
    Simple async logging when chatbot response is already known
    """
    inbox_id = event.get("inbox_id")
    user_input = event.get("user_input")
    chatbot_response = event.get("chatbot_response")
    locale = event.get("locale", "ko-KR")
    contact_id = event.get("contact_id", "unknown")
    
    try:
        inbox_item = {
            "inbox_id": inbox_id,
            "contact_id": contact_id,
            "turn_ts": int(time.time()),
            "locale": locale,
            "utterance_text": user_input,
            "utterance_norm": normalize_utt(user_input),
            "utterance_hash": utt_hash(user_input, short=True),
            "proposed_response_text": chatbot_response,
            "response_norm": normalize_utt(chatbot_response),
            "response_hash": utt_hash(chatbot_response, short=True),
            "review_status": "open",
            "needs_review": True,
            "cache_both": True,
            "logged_at": int(time.time())
        }
        
        INBOX.put_item(Item=inbox_item)
        _log("Logged with known response", inbox_id=inbox_id[:8])
        
        return {"success": True, "inbox_id": inbox_id}
    except Exception as e:
        _log("Logging failed", error=str(e))
        return {"success": False, "error": str(e)}

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
        # params.get("Details.Parameters.user_input"),
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

def to_regional_url(key: str) -> str:
    # Properly encode the key for URLs
    encoded_key = quote(key, safe='/')
    return f"https://{TTS_BUCKET}.s3.{CONNECT_REGION}.amazonaws.com/{encoded_key}"

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

def approve_and_cache_utterance(inbox_id: str, approved_audio_s3_uri: str):
    """
    Cache the customer utterance (updated for SttInboxNew schema)
    """
    # Get inbox item using inbox_id
    inbox_item = INBOX.get_item(Key={"inbox_id": inbox_id})["Item"]
    
    # Cache the customer utterance
    UTT_CACHE.put_item(Item={
        "utterance_hash": inbox_item["utterance_hash"],
        "locale": inbox_item["locale"],
        "audio_s3_uri": approved_audio_s3_uri,
        "status": "approved",
        "cached_text": inbox_item["utterance_text"],
        "original_utterance": inbox_item["utterance_text"],
        "chatbot_response": inbox_item["proposed_response_text"],
        "approval_type": "customer_utterance",
        "inbox_id": inbox_id,  # Reference back to inbox
        "contact_id": inbox_item["contact_id"],
        "turn_ts": inbox_item["turn_ts"],
        "created_at": int(time.time()),
        "approved_by": "human_reviewer"
    })

def approve_and_cache_response(inbox_id: str, approved_audio_s3_uri: str):
    """
    Cache the chatbot response (updated for SttInboxNew schema)
    """
    inbox_item = INBOX.get_item(Key={"inbox_id": inbox_id})["Item"]
    
    UTT_CACHE.put_item(Item={
        "utterance_hash": inbox_item["response_hash"],
        "locale": inbox_item["locale"],
        "audio_s3_uri": approved_audio_s3_uri,
        "status": "approved",
        "cached_text": inbox_item["proposed_response_text"],
        "original_utterance": inbox_item["utterance_text"],
        "chatbot_response": inbox_item["proposed_response_text"],
        "approval_type": "chatbot_response",
        "inbox_id": inbox_id,
        "contact_id": inbox_item["contact_id"],
        "turn_ts": inbox_item["turn_ts"],
        "created_at": int(time.time()),
        "approved_by": "human_reviewer"
    })

def approve_and_cache_both(inbox_id: str, utterance_audio_uri: str, response_audio_uri: str):
    """
    Cache both (updated for SttInboxNew schema)
    """
    approve_and_cache_utterance(inbox_id, utterance_audio_uri)
    approve_and_cache_response(inbox_id, response_audio_uri)
    
    # Mark inbox item as processed
    INBOX.update_item(
        Key={"inbox_id": inbox_id},
        UpdateExpression="SET review_status = :status, approved_at = :ts, both_cached = :both",
        ExpressionAttributeValues={
            ":status": "both_approved",
            ":ts": int(time.time()),
            ":both": True
        }
    )

def approve_utterance_with_tts_generation(inbox_id: str, generate_response_audio: bool = True):
    """
    Enhanced approval that generates TTS audio for the response and properly uploads it
    """
    try:
        # Get inbox item
        inbox_item = INBOX.get_item(Key={"inbox_id": inbox_id})["Item"]
        
        utterance_text = inbox_item["utterance_text"]
        proposed_response_text = inbox_item["proposed_response_text"]
        locale = inbox_item["locale"]
        
        _log("Starting approval with TTS generation", 
             inbox_id=inbox_id[:8],
             utterance_preview=utterance_text[:50],
             response_preview=proposed_response_text[:50])
        
        # Generate response audio using TTS
        if generate_response_audio and proposed_response_text:
            response_audio_uri = generate_and_upload_tts_audio(
                text=proposed_response_text,
                key_prefix=f"approved/responses/{inbox_id}",
                locale=locale
            )
            _log("Generated response audio", uri=_redact(response_audio_uri, 40))
        else:
            response_audio_uri = None
        
        # Cache the customer utterance (using a placeholder for now since we don't have the actual utterance audio)
        # In a full implementation, you might want to generate this too or get it from somewhere else
        utterance_audio_uri = f"s3://{TTS_BUCKET}/approved/utterances/{inbox_id}/utterance.wav"
        
        # Generate utterance audio if requested (optional)
        if utterance_text and utterance_text != proposed_response_text:
            try:
                utterance_audio_uri = generate_and_upload_tts_audio(
                    text=utterance_text,
                    key_prefix=f"approved/utterances/{inbox_id}",
                    locale=locale,
                    voice_style="customer"  # Different style for customer voice simulation
                )
                _log("Generated utterance audio", uri=_redact(utterance_audio_uri, 40))
            except Exception as e:
                _log("Failed to generate utterance audio, using placeholder", error=str(e))
        
        # Cache the response (this is the important one for your workflow)
        if response_audio_uri:
            UTT_CACHE.put_item(Item={
                "utterance_hash": inbox_item["response_hash"],
                "locale": locale,
                "audio_s3_uri": response_audio_uri,
                "status": "approved",
                "cached_text": proposed_response_text,
                "original_utterance": utterance_text,
                "chatbot_response": proposed_response_text,
                "approval_type": "chatbot_response",
                "inbox_id": inbox_id,
                "contact_id": inbox_item.get("contact_id", "unknown"),
                "turn_ts": inbox_item.get("turn_ts", int(time.time())),
                "created_at": int(time.time()),
                "approved_by": "human_reviewer_with_tts"
            })
        
        # Optionally cache the utterance too
        if utterance_audio_uri and utterance_text != proposed_response_text:
            UTT_CACHE.put_item(Item={
                "utterance_hash": inbox_item["utterance_hash"],
                "locale": locale,
                "audio_s3_uri": utterance_audio_uri,
                "status": "approved",
                "cached_text": utterance_text,
                "original_utterance": utterance_text,
                "chatbot_response": proposed_response_text,
                "approval_type": "customer_utterance",
                "inbox_id": inbox_id,
                "contact_id": inbox_item.get("contact_id", "unknown"),
                "turn_ts": inbox_item.get("turn_ts", int(time.time())),
                "created_at": int(time.time()),
                "approved_by": "human_reviewer_with_tts"
            })
        
        # Mark inbox item as processed
        INBOX.update_item(
            Key={"inbox_id": inbox_id},
            UpdateExpression="SET review_status = :status, approved_at = :ts, response_audio_uri = :response_uri, utterance_audio_uri = :utterance_uri",
            ExpressionAttributeValues={
                ":status": "approved_with_tts",
                ":ts": int(time.time()),
                ":response_uri": response_audio_uri or "",
                ":utterance_uri": utterance_audio_uri or ""
            }
        )
        
        return {
            "success": True,
            "response_audio_uri": response_audio_uri,
            "utterance_audio_uri": utterance_audio_uri,
            "inbox_id": inbox_id,
            "cached_response": bool(response_audio_uri),
            "cached_utterance": bool(utterance_audio_uri and utterance_text != proposed_response_text)
        }
        
    except Exception as e:
        _log("Approval with TTS generation failed", error=str(e), inbox_id=inbox_id)
        return {
            "success": False,
            "error": str(e),
            "inbox_id": inbox_id
        }

def generate_and_upload_tts_audio(text: str, key_prefix: str, locale: str = "ko-KR", voice_style: str = "agent"):
    """
    Generate TTS audio and upload to S3, returning the S3 URI
    """
    try:
        # Prepare text for TTS
        tts_text = prepare_korean_text_for_tts(text)
        
        # Use different voice settings based on style
        if voice_style == "customer":
            # Use slightly different settings for customer voice simulation
            payload = {
                "text": tts_text,
                "sample_rate": 8000,
                "key_prefix": key_prefix,
                "temperature": 0.8,  # Slightly more variation for customer
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "max_new_tokens": 512,
                "chunk_length": 64,
                "use_memory_cache": False
            }
        else:
            # Standard agent voice settings
            payload = {
                "text": tts_text,
                "sample_rate": 8000,
                "key_prefix": key_prefix,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.0,
                "max_new_tokens": 512,
                "chunk_length": 64,
                "use_memory_cache": False
            }
        
        _log("Calling TTS for approval audio generation", 
             text_preview=tts_text[:100],
             voice_style=voice_style)
        
        # Call TTS service
        tts_resp = _http_post_json(
            TTS_URL,
            payload,
            token=TTS_TOKEN,
            timeout=int(os.getenv("SYNTH_HTTP_TIMEOUT_SEC", "60"))
        )
        
        # Get the generated audio URL/key
        server_bucket = tts_resp.get("bucket", TTS_BUCKET)
        server_key = tts_resp.get("key")
        server_url = tts_resp.get("url") or tts_resp.get("s3_url")
        
        if not server_key:
            raise RuntimeError(f"TTS response missing key: {list(tts_resp.keys())}")
        
        # Define our target key structure for approved audio
        timestamp = int(time.time())
        final_key = f"{key_prefix.rstrip('/')}/{timestamp}.wav"
        
        _log("TTS generation successful", 
             server_key=server_key,
             target_key=final_key)
        
        # If the server used a different key, copy to our desired location
        if server_key != final_key:
            try:
                # Copy to our organized structure
                s3.copy_object(
                    Bucket=TTS_BUCKET,
                    CopySource={"Bucket": server_bucket, "Key": server_key},
                    Key=final_key,
                    MetadataDirective='COPY'
                )
                
                # Verify the copy
                copy_head = s3.head_object(Bucket=TTS_BUCKET, Key=final_key)
                copy_size = copy_head.get("ContentLength", 0)
                
                if copy_size == 0:
                    raise RuntimeError("Copied audio file is empty")
                
                _log("Audio copied to final location", 
                     final_key=final_key,
                     size=copy_size)
                
                # Clean up server's temporary file
                try:
                    s3.delete_object(Bucket=server_bucket, Key=server_key)
                except Exception as del_error:
                    _log("Failed to delete server temp file", error=str(del_error))
                
            except Exception as copy_error:
                _log("Failed to copy audio, using server location", error=str(copy_error))
                final_key = server_key
        
        # Return the S3 URI
        final_uri = f"s3://{TTS_BUCKET}/{final_key}"
        
        _log("TTS audio generation complete", 
             final_uri=final_uri,
             text_length=len(text))
        
        return final_uri
        
    except Exception as e:
        _log("TTS audio generation failed", error=str(e), text=text[:100])
        raise RuntimeError(f"TTS generation failed: {str(e)}")

def prepare_korean_text_for_tts(text: str) -> str:
    """
    Prepare Korean text for TTS to avoid common issues
    """
    # Apply your existing preprocessing
    text = text.strip()
    
    # Fix common Korean TTS issues
    text = text.replace("겠", "것")  # Avoid the problematic "겠" character
    text = text.replace("켇", "켜")  # Fix other problematic combinations
    
    # Ensure proper sentence ending for Korean prosody
    if not text.endswith((".", "!", "?", "다", "요", "니다", "습니다")):
        if "습니다" in text or "니다" in text:
            text += "."
        else:
            text += "요."
    
    return text

def review_and_approve(inbox_id: str, approved_audio_s3_uri: str, approval_type: str = "utterance"):
    """
    Approve an item from SttInbox and add to UtteranceCache
    approval_type: "utterance" or "response"
    """
    # Get inbox item
    inbox_item = INBOX.get_item(Key={"inbox_id": inbox_id})["Item"]
    
    # Determine which text to cache
    if approval_type == "utterance":
        cache_text = inbox_item["utterance_text"]
        cache_hash = inbox_item["utterance_hash"]
    else:  # response
        cache_text = inbox_item["proposed_response_text"] 
        cache_hash = inbox_item["response_hash"]
    
    # Add to cache
    UTT_CACHE.put_item(Item={
        "utterance_hash": cache_hash,
        "locale": inbox_item["locale"],
        "audio_s3_uri": approved_audio_s3_uri,
        "status": "approved",
        "original_text": cache_text,
        "approval_type": approval_type,
        "created_at": int(time.time())
    })
    
    # Mark inbox item as processed
    INBOX.update_item(
        Key={"inbox_id": inbox_id},
        UpdateExpression="SET review_status = :status, approved_at = :ts, approval_type = :type",
        ExpressionAttributeValues={
            ":status": "approved",
            ":ts": int(time.time()),
            ":type": approval_type
        }
    )

def approve_both_with_tts_generation(inbox_id: str):
    """
    Generate TTS audio for both customer utterance and chatbot response, then cache both
    """
    try:
        # Get inbox item
        inbox_item = INBOX.get_item(Key={"inbox_id": inbox_id})["Item"]
        
        utterance_text = inbox_item["utterance_text"]
        proposed_response_text = inbox_item["proposed_response_text"]
        locale = inbox_item["locale"]
        contact_id = inbox_item.get("contact_id", "unknown")
        turn_ts = inbox_item.get("turn_ts", int(time.time()))
        
        _log("Starting approve_both with TTS generation", 
             inbox_id=inbox_id[:8],
             utterance_preview=utterance_text[:50],
             response_preview=proposed_response_text[:50])
        
        # Generate both audio files
        results = {
            "utterance_audio_uri": None,
            "response_audio_uri": None,
            "utterance_cached": False,
            "response_cached": False,
            "errors": []
        }
        
        # Generate customer utterance audio (simulated agent reading the customer's words)
        try:
            utterance_audio_uri = generate_and_upload_tts_audio(
                text=utterance_text,
                key_prefix=f"approved/utterances/{inbox_id}",
                locale=locale,
                voice_style="customer"
            )
            results["utterance_audio_uri"] = utterance_audio_uri
            _log("Generated utterance audio", uri=_redact(utterance_audio_uri, 40))
        except Exception as e:
            error_msg = f"Failed to generate utterance audio: {str(e)}"
            results["errors"].append(error_msg)
            _log("Utterance audio generation failed", error=str(e))
        
        # Generate chatbot response audio (main agent voice)
        try:
            response_audio_uri = generate_and_upload_tts_audio(
                text=proposed_response_text,
                key_prefix=f"approved/responses/{inbox_id}",
                locale=locale,
                voice_style="agent"
            )
            results["response_audio_uri"] = response_audio_uri
            _log("Generated response audio", uri=_redact(response_audio_uri, 40))
        except Exception as e:
            error_msg = f"Failed to generate response audio: {str(e)}"
            results["errors"].append(error_msg)
            _log("Response audio generation failed", error=str(e))
        
        # Cache the customer utterance if audio was generated successfully
        if results["utterance_audio_uri"]:
            try:
                UTT_CACHE.put_item(Item={
                    "utterance_hash": inbox_item["utterance_hash"],
                    "locale": locale,
                    "audio_s3_uri": results["utterance_audio_uri"],
                    "status": "approved",
                    "cached_text": utterance_text,
                    "original_utterance": utterance_text,
                    "chatbot_response": proposed_response_text,
                    "approval_type": "customer_utterance",
                    "inbox_id": inbox_id,
                    "contact_id": contact_id,
                    "turn_ts": turn_ts,
                    "created_at": int(time.time()),
                    "approved_by": "human_reviewer_tts_both"
                })
                results["utterance_cached"] = True
                _log("Cached customer utterance", hash=inbox_item["utterance_hash"][:8])
            except Exception as e:
                error_msg = f"Failed to cache utterance: {str(e)}"
                results["errors"].append(error_msg)
                _log("Utterance caching failed", error=str(e))
        
        # Cache the chatbot response if audio was generated successfully
        if results["response_audio_uri"]:
            try:
                UTT_CACHE.put_item(Item={
                    "utterance_hash": inbox_item["response_hash"],
                    "locale": locale,
                    "audio_s3_uri": results["response_audio_uri"],
                    "status": "approved",
                    "cached_text": proposed_response_text,
                    "original_utterance": utterance_text,
                    "chatbot_response": proposed_response_text,
                    "approval_type": "chatbot_response",
                    "inbox_id": inbox_id,
                    "contact_id": contact_id,
                    "turn_ts": turn_ts,
                    "created_at": int(time.time()),
                    "approved_by": "human_reviewer_tts_both"
                })
                results["response_cached"] = True
                _log("Cached chatbot response", hash=inbox_item["response_hash"][:8])
            except Exception as e:
                error_msg = f"Failed to cache response: {str(e)}"
                results["errors"].append(error_msg)
                _log("Response caching failed", error=str(e))
        
        # Update inbox item status
        try:
            update_expression = "SET review_status = :status, approved_at = :ts, both_cached = :both"
            expression_values = {
                ":status": "both_approved_with_tts",
                ":ts": int(time.time()),
                ":both": results["utterance_cached"] and results["response_cached"]
            }
            
            # Add audio URIs if they exist
            if results["utterance_audio_uri"]:
                update_expression += ", utterance_audio_uri = :utterance_uri"
                expression_values[":utterance_uri"] = results["utterance_audio_uri"]
            
            if results["response_audio_uri"]:
                update_expression += ", response_audio_uri = :response_uri"
                expression_values[":response_uri"] = results["response_audio_uri"]
            
            INBOX.update_item(
                Key={"inbox_id": inbox_id},
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_values
            )
            _log("Updated inbox item status", inbox_id=inbox_id[:8])
        except Exception as e:
            error_msg = f"Failed to update inbox: {str(e)}"
            results["errors"].append(error_msg)
            _log("Inbox update failed", error=str(e))
        
        # Determine overall success
        success = (results["utterance_cached"] or results["response_cached"]) and len(results["errors"]) == 0
        
        response = {
            "success": success,
            "inbox_id": inbox_id,
            "utterance_audio_uri": results["utterance_audio_uri"],
            "response_audio_uri": results["response_audio_uri"],
            "utterance_cached": results["utterance_cached"],
            "response_cached": results["response_cached"],
            "both_successful": results["utterance_cached"] and results["response_cached"]
        }
        
        if results["errors"]:
            response["errors"] = results["errors"]
            response["partial_success"] = success and len(results["errors"]) > 0
        
        return response
        
    except Exception as e:
        _log("Approve both with TTS generation failed", error=str(e), inbox_id=inbox_id)
        return {
            "success": False,
            "error": str(e),
            "inbox_id": inbox_id
        }

def handle_approve_both(event):
    """Handle approve_both action with different input modes"""
    params = event.get("Details", {}).get("Parameters", {})
    inbox_id = params.get("inbox_id")
    
    if not inbox_id:
        return {"success": False, "error": "Missing inbox_id"}
    
    # Check if they provided both audio URIs (original workflow)
    utterance_audio_uri = params.get("utterance_audio_uri")
    response_audio_uri = params.get("response_audio_uri")
    
    if utterance_audio_uri and response_audio_uri:
        # Use provided audio files (original workflow)
        try:
            approve_and_cache_both(inbox_id, utterance_audio_uri, response_audio_uri)
            return {
                "success": True, 
                "message": "Both utterance and response cached with provided audio",
                "utterance_audio_uri": utterance_audio_uri,
                "response_audio_uri": response_audio_uri,
                "audio_generated": False
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    else:
        # Generate TTS audio for both (new workflow)
        try:
            result = approve_both_with_tts_generation(inbox_id)
            return result
        except Exception as e:
            _log("Approve both with TTS generation failed", error=str(e))
            return {"success": False, "error": str(e)}

def _extract_user_speech_input(event: dict) -> str:
    """
    Enhanced function to extract user speech input from Connect/Lex events
    Fixed to handle your specific event structure
    """
    d = event.get("Details", {}) or {}
    params = d.get("Parameters", {}) or {}
    contact = d.get("ContactData", {}) or {}
    attrs = contact.get("Attributes", {}) or {}
    
    # Handle Lex data properly
    lex_data = d.get("Lex", {})
    
    # Debug logging to see what we're getting
    _log("DEBUG: Input extraction", 
         event_keys=list(event.keys()),
         details_keys=list(d.keys()) if d else [],
         params_keys=list(params.keys()) if params else [],
         contact_keys=list(contact.keys()) if contact else [])
    
    candidates = [
        # Your specific event structure - check ContactData first
        contact.get("user_input"),           # From your event: ContactData.user_input
        contact.get("CustomerInput"),        # From your event: ContactData.CustomerInput
        
        # Top-level user_input (your event has this)
        event.get("user_input"),
        
        # Parameters
        params.get("user_input"),
        params.get("text"),
        # params.get("Details.Parameters.user_input"),
        params.get("Details.Parameters.text"),
        
        # Lex integration
        lex_data.get("InputTranscript"),
        
        # Contact attributes
        attrs.get("user_input"),
        attrs.get("UserInput"),
        
        # Direct event properties
        event.get("InputTranscript"),
        event.get("text"),
        
        # Lex slots if present
        lex_data.get("Slots", {}).get("UserInput") if isinstance(lex_data.get("Slots"), dict) else None,
    ]
    
    # Debug: Show all candidates
    for i, candidate in enumerate(candidates):
        if candidate is not None:
            _log(f"DEBUG: Candidate {i}", value=str(candidate)[:100])
    
    for candidate in candidates:
        if candidate is None:
            continue
        
        text = str(candidate).strip()
        text = re.sub(r'^\.+', '', text)  # Remove leading dots
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Skip empty strings, system messages, and obvious non-speech
        if (text and 
            len(text) > 0 and 
            not text.startswith("{") and 
            text != "null" and 
            text != "undefined" and
            text != "안녕하세요"):  # Skip default greeting
            
            _log("Found user speech input", 
                 source="extracted", 
                 length=len(text), 
                 preview=text[:100])
            return text
    
    _log("No valid user speech input found in event")
    return ""

def test_current_cache_lookup(user_input: str = "제가 바빠요"):
    """Test what your current cache lookup function returns"""
    locale = "ko-KR"
    
    _log("Testing current cache lookup", user_input=user_input)
    
    # Test your actual cache lookup function
    cache_result = ddb_get_cached_audio(user_input, locale)
    
    _log("Current lookup result",
         found=bool(cache_result),
         audio_uri=_redact(cache_result.get("audio_s3_uri", ""), 40) if cache_result else None,
         approval_type=cache_result.get("approval_type") if cache_result else None)
    
    return bool(cache_result)


def get_inbox_item_details(event):
    """Get details of an inbox item for preview before approval"""
    params = event.get("Details", {}).get("Parameters", {})
    inbox_id = params.get("inbox_id")
    
    if not inbox_id:
        return {"success": False, "error": "Missing inbox_id"}
    
    try:
        inbox_item = INBOX.get_item(Key={"inbox_id": inbox_id})["Item"]
        
        return {
            "success": True,
            "inbox_id": inbox_id,
            "utterance_text": inbox_item["utterance_text"],
            "proposed_response_text": inbox_item["proposed_response_text"],
            "locale": inbox_item["locale"],
            "contact_id": inbox_item.get("contact_id", "unknown"),
            "turn_ts": inbox_item.get("turn_ts"),
            "review_status": inbox_item.get("review_status", "open"),
            "customer_sentiment": inbox_item.get("customer_sentiment", "general")
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# Also add a function to manually verify one of your SttInbox entries was cached:
def verify_inbox_caching(inbox_id: str):
    """Check if an inbox entry that claims to be cached actually exists in cache"""
    try:
        # Get the inbox entry
        inbox_resp = INBOX.get_item(Key={"inbox_id": inbox_id})
        inbox_item = inbox_resp.get("Item")
        
        if not inbox_item:
            return {"error": "Inbox item not found"}
        
        # Extract the hashes that should have been cached
        utterance_hash = inbox_item.get("utterance_hash", "")
        response_hash = inbox_item.get("response_hash", "")
        locale = inbox_item.get("locale", "ko-KR")
        
        _log("Verifying cache for inbox item", 
             inbox_id=inbox_id[:8],
             utterance_hash=utterance_hash,
             response_hash=response_hash)
        
        results = []
        
        # Check if utterance hash exists in cache
        try:
            resp = UTT_CACHE.get_item(Key={"utterance_hash": utterance_hash, "locale": locale})
            utt_item = resp.get("Item")
            results.append({
                "hash_type": "utterance_hash",
                "hash": utterance_hash,
                "found": bool(utt_item),
                "status": utt_item.get("status") if utt_item else None,
                "approval_type": utt_item.get("approval_type") if utt_item else None
            })
        except Exception as e:
            results.append({"hash_type": "utterance_hash", "hash": utterance_hash, "error": str(e)})
        
        # Check if response hash exists in cache
        if response_hash and response_hash != utterance_hash:
            try:
                resp = UTT_CACHE.get_item(Key={"utterance_hash": response_hash, "locale": locale})
                resp_item = resp.get("Item")
                results.append({
                    "hash_type": "response_hash", 
                    "hash": response_hash,
                    "found": bool(resp_item),
                    "status": resp_item.get("status") if resp_item else None,
                    "approval_type": resp_item.get("approval_type") if resp_item else None
                })
            except Exception as e:
                results.append({"hash_type": "response_hash", "hash": response_hash, "error": str(e)})
        
        return {
            "inbox_id": inbox_id,
            "inbox_claims_cached": inbox_item.get("dual_hash_cached") == "True",
            "cache_verification": results
        }
        
    except Exception as e:
        return {"error": str(e), "inbox_id": inbox_id}

# ========= Main handler (router) =========
def lambda_handler(event, context):
    """
    Optimized main handler - guaranteed fast response
    """
    if _is_lex_code_hook(event):
        return _lex_delegate_with_interrupt_attrs(event)

    _req_filter.set_request_id(getattr(context, "aws_request_id", "-"))
    
    # Extract action
    try:
        if "Details" in event and "Parameters" in event["Details"]:
            action = event["Details"]["Parameters"].get("action", "voice_response")
        else:
            action = event.get("action", "warmup")
    except Exception as e:
        _log("Action extraction failed", error=str(e))
        action = "warmup"
    
    _log("Handler routing", action=action)

    try:
        if action == "voice_response":
            # ULTRA-FAST path only
            result = _unified_voice_response_handler_ultra_fast(event)
            return {"setAttributes": result}
        
        elif action == "async_chatbot_and_log":
            # Background processing
            return handle_async_chatbot_and_log(event)
        
        elif action == "get_inbox_items":
            # Admin function (scan is OK here)
            response = INBOX.scan(
                FilterExpression='review_status = :status',
                ExpressionAttributeValues={':status': 'open'},
                Limit=50
            )
            return {
                "success": True,
                "count": len(response.get('Items', [])),
                "items": response.get('Items', [])
            }
        
        elif action == "approve_both":
            return handle_approve_both_fixed(event)
        
        elif action == "get_inbox_details":
            return get_inbox_item_details(event)
        
        else:
            if "Details" in event:
                result = _unified_voice_response_handler_ultra_fast(event)
                return {"setAttributes": result}
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
            
    except Exception as e:
        logger.exception("Lambda handler failed")
        return {
            "error": "exception",
            "message": str(e),
            "setAttributes": {
                "error": "true",
                "error_message": str(e),
                "ready": "false"
            }
        }