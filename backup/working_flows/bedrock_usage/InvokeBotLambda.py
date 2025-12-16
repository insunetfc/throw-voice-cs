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

# DynamoDB Configuration
CACHE_TABLE = os.getenv("CACHE_TABLE", "ConnectPromptCache")
DISABLE_DDB = "0"
# Cache/table flags
USE_DDB = os.getenv("USE_DDB", "1").lower() in ("1","true","yes","y")
UTTERANCE_CACHE_TABLE = os.getenv("UTTERANCE_CACHE_TABLE", "UtteranceCache")
UTT_CACHE = boto3.resource("dynamodb", region_name=AWS_REGION).Table(UTTERANCE_CACHE_TABLE)
INBOX = boto3.resource("dynamodb").Table("SttInboxNew")

# External Services
CHAT_URL = os.getenv("CHAT_URL", "http://15.165.60.45:5000/chat")
CHAT_TOKEN = os.getenv("CHAT_TOKEN", "")
TTS_URL = os.getenv("TTS_URL", "https://honest-trivially-buffalo.ngrok-free.app/synthesize")
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
BYPASS_CHAT = int(os.getenv("BYPASS_CHAT", "0")) == "0"
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

# Korean regex patterns
KYES = r"(네|예|맞(아|습니다)|그렇(습)?니다|좋습니다)"
KASK = r"(왜|무엇|뭐(야|예요)|어떻게|언제|어디|얼마|가능|되는지|설명|자세히)"
KCONFQ = r"(맞(나요|습니까)|괜찮(나요|습니까)|되(나요|겠습니까)|이거(로)? (할까요|진행할까요)|확인(해)?주시겠어요)"
KTHANK = r"(감사|고맙)"
KFRUS = r"(느리|답답|짜증|문제|안라|안 돼|에러|오류|힘들|헷갈|복잡)"

# Warm-up guards
_IS_WARM = False
_WARM_TS = 0.0
WARMED = False
_PART_RE = re.compile(r"/?part(\d+)\.wav$")

filler_text = "NIPA 클라우드는 일정 시간 미사용 시 세션이 재시작되므로, 전체 과정을 2분 이내에 재실행할 수 있는 스크립트를 작성하여 재사용성을 확보했습니다. 또한 전화 통화에서는 FishSpeech 시스템이 사용자 발화를 성공적으로 인식하고 챗봇 응답을 받은 뒤 오디오를 생성하여 S3에 업로드하는 데 성공했습니다. 다만, 오디오 재생에서 권한 및 형식 문제로 인해 일부 문제가 발생하였으며, 현재 이를 해결하기 위해 디버깅 중입니다. 흐름이 정상적으로 동작하도록 마무리하면 기본 시스템 프로토타입이 완성될 예정입니다."

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
connect = boto3.client("connect", region_name=CONNECT_REGION)
lambda_client = boto3.client("lambda", region_name=AWS_REGION)

# def normalize_utt(text: str) -> str:
#     if not text:
#         return ""
#     # Ensure consistent Unicode normalization
#     t = unicodedata.normalize("NFKC", text)
#     t = t.lower().replace("%", " percent ")
#     t = re.sub(r"[^0-9a-z가-힣\s]", " ", t)
#     return re.sub(r"\s+", " ", t).strip()

def normalize_utt(text: str) -> str:
    s = unicodedata.normalize("NFKC", text).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\.!\?]+$", "", s)  # strip trailing punctuation often unstable in ASR
    return s

def utt_hash(text: str, short=True) -> str:
    norm = normalize_utt(text)
    h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    return h[:16] if short else h

def _utt_hashes_both(text: str) -> tuple[str, str]:
    """
    Returns (normalized-hash, raw-hash) for backward compatibility.
    Assumes `utt_hash(text, short=True)` uses your normalization.
    """
    try:
        norm_h = utt_hash(text, short=True)  # current normalized behavior
    except Exception:
        # as a fallback, still compute raw in case utt_hash fails
        norm_h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    raw_h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return (norm_h, raw_h)

def ddb_get_cached_audio_with_context_fixed(utterance_or_response: str, locale: str) -> Optional[dict]:
    """
    Backward-compatible cache lookup:
    1) Try normalized hash (current runtime behavior)
    2) Try raw-text hash (legacy rows)
    Prefer `response_audio_uri` if present.
    """
    if not USE_DDB:
        return None
    if not utterance_or_response:
        return None

    for uh in _utt_hashes_both(utterance_or_response):
        try:
            resp = UTT_CACHE.get_item(
                Key={"utterance_hash": uh, "locale": locale},
                ConsistentRead=True
            )
            item = resp.get("Item")
            if item and item.get("status") == "approved":
                # Prefer response_audio_uri over audio_s3_uri when available
                if item.get("response_audio_uri"):
                    item["audio_s3_uri"] = item["response_audio_uri"]
                    _log("Using response_audio_uri instead of audio_s3_uri", hash=uh, locale=locale)
                _log("Cache hit", hash=uh, locale=locale)
                return item
        except Exception as e:
            _log("Cache lookup failed", error=str(e), hash=uh, locale=locale)
            # continue to next hash
            continue

    _log("Cache miss (both hashes)", locale=locale, text_preview=utterance_or_response[:40])
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

# Debug the actual table schema first
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

# Then your existing simple lookup will work:
# Update your main handler to use the fixed lookup:
def _unified_voice_response_handler_simple(event):
    """
    Your existing handler but using the fixed cache lookup
    """
    user_speech = _extract_user_speech_input(event)
    if not user_speech:
        user_speech = "안녕하세요"
    
    locale = "ko-KR"
    contact_id = event.get("Details", {}).get("ContactData", {}).get("ContactId", "unknown")
    
    # Step 1: Check utterance cache (will return response audio if available)
    try:
        utterance_cache = ddb_get_cached_audio_response_preferred(user_speech, locale)
        if utterance_cache:
            _log("Cache hit - returning response audio", 
                 hash=utt_hash(user_speech, short=True)[:8])
            return {
                "ready": "true",
                "AudioS3Url0": utterance_cache["audio_s3_uri"],  # This will be response audio now
                "AudioS3UrlCount": "1",
                "BatchCount": "1",
                "HasMore": "false",
                "cache_hit": "true",
                "cache_type": "response_audio_preferred",
                "UserInput": user_speech,
                "send_sms": "false",
                "neutral_message": "false"
            }
    except Exception as e:
        _log("Utterance cache check failed", error=str(e))
    
    # Step 2: Get chatbot response
    session_id = _session_id_from_event(event, str(uuid.uuid4()))
    try:
        chatbot_response = call_chatbot(user_speech, session_id)
    except Exception as e:
        _log("Chatbot failed", error=str(e))
        chatbot_response = user_speech
    
    # Step 3: Check response cache (will return response audio if available)
    try:
        response_cache = ddb_get_cached_audio_response_preferred(chatbot_response, locale)
        if response_cache:
            _log("Cache hit via chatbot response", 
                 hash=utt_hash(chatbot_response, short=True)[:8])
            return {
                "ready": "true",
                "AudioS3Url0": response_cache["audio_s3_uri"],  # This will be response audio now
                "AudioS3UrlCount": "1",
                "BatchCount": "1",
                "HasMore": "false",
                "cache_hit": "true",
                "cache_type": "response_audio_preferred",
                "UserInput": user_speech,
                "ChatAnswer": chatbot_response,
                "send_sms": "false",
                "neutral_message": "false"
            }
    except Exception as e:
        _log("Response cache check failed", error=str(e))
    
    # Step 4: Cache miss - rest of your existing logic
    try:
        inbox_id = uuid4().hex
        inbox_item = {
            "inbox_id": inbox_id,
            "contact_id": contact_id,
            "turn_ts": int(time.time()),
            "locale": locale,
            "utterance_text": user_speech,
            "utterance_norm": normalize_utt(user_speech),
            "utterance_hash": utt_hash(user_speech, short=True),
            "proposed_response_text": chatbot_response,
            "response_norm": normalize_utt(chatbot_response),
            "response_hash": utt_hash(chatbot_response, short=True),
            "review_status": "open",
            "needs_review": True,
            "cache_both": True
        }
        
        INBOX.put_item(Item=inbox_item)
        _log("Logged to inbox", inbox_id=inbox_id[:8])
        
    except Exception as e:
        _log("Inbox logging failed", error=str(e))
    
    return {
        "ready": "true",
        "AudioS3Url0": "https://tts-bucket-250810.s3.ap-northeast-2.amazonaws.com/neutral/general/01.wav",
        "AudioS3UrlCount": "1",
        "BatchCount": "1",
        "HasMore": "false",
        "cache_hit": "false",
        "cache_miss": "true",
        "send_sms": "true",
        "neutral_message": "true",
        "UserInput": user_speech,
        "ChatAnswer": chatbot_response,
        "inbox_logged": "true"
    }


def _state_get(event):
    """Extract per-call filler state (JSON) from Connect attributes."""
    params = (event.get("Details", {}) or {}).get("Parameters", {}) or {}
    st_raw = params.get("FillerState") or params.get("filler_state")
    try:
        return json.loads(st_raw) if st_raw else {}
    except Exception:
        return {}
    
def resolve_audio(text: str, locale: str, chatbot_response_or_none: Optional[str]):
    # 1) Try UTTERANCE hit
    audio_uri = ddb_get_cached_audio_response_preferred(text, locale)
    if audio_uri:
        return {"action": "play_cached", "audio_s3_uri": audio_uri}

    # 2) Try RESPONSE hit (if you have one)
    if chatbot_response_or_none:
        audio_uri = ddb_get_cached_audio_response_preferred(chatbot_response_or_none, locale)
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
    ddb = boto3.resource("dynamodb", region_name=CONNECT_REGION).Table(CACHE_TABLE)
    print(f"[INIT] DynamoDB ENABLED - using table {CACHE_TABLE}")

ASSUME_ROLE_ARN = os.getenv("ASSUME_ROLE_ARN", "")
ASSUME_ROLE_EXTERNAL_ID = os.getenv("ASSUME_ROLE_EXTERNAL_ID", "")
prefetch = int(os.getenv("BATCH_LIMIT", "1"))



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


def _filler_key_url(category: str | None = None, index: int | None = None):
    cats = list(FILLER_CATS.keys())
    cat = category if category in FILLER_CATS else random.choice(cats)
    n = FILLER_CATS[cat]
    idx = index if (isinstance(index, int) and 1 <= index <= n) else random.randint(1, n)
    
    key = f"{(FILLER_PREFIX + '/') if FILLER_PREFIX and not FILLER_PREFIX.endswith('/') else FILLER_PREFIX}{cat}/{idx:02d}.wav"
    
    # Use presigned URL instead of direct S3 URL
    url = _s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": TTS_BUCKET, "Key": key},
        ExpiresIn=3600
    )
    
    return cat, idx, key, url

def _check_full(event):
    params = (event.get("Details", {}) or {}).get("Parameters", {}) or {}

    # ---------- S3-ONLY MODE ----------
    if DISABLE_DDB:
        bucket = params.get("job_bucket") or TTS_BUCKET
        key    = params.get("job_key") or ""
        if not key:
            return {"setAttributes": {"ready":"false", "Error":"missing_job_key"}}

        try:
            _s3.head_object(Bucket=bucket, Key=key)
            url = f"https://{bucket}.s3.{CONNECT_REGION}.amazonaws.com/{key}"
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

def _head_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=TTS_BUCKET, Key=key)
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
    bucket = bucket or (globals().get("TTS_BUCKET") or globals().get("TTS_BUCKET"))
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
                    TTS_BUCKET, key, f"{job_id}-part{i}"
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
                TTS_BUCKET, final_key, f"{job_id}-final"
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
    resp = s3.list_objects_v2(Bucket=TTS_BUCKET, Prefix=prefix, MaxKeys=1000) 
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
    return f"https://{TTS_BUCKET}.s3.{CONNECT_REGION}.amazonaws.com/{key}"


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
    Get the appropriate neutral audio URL based on customer response analysis
    """
    # Map response types to S3 keys
    neutral_key = f"neutral/{customer_response_type}/{index:02d}.wav"
    
    # Check if the specific neutral message exists
    try:
        s3.head_object(Bucket=TTS_BUCKET, Key=neutral_key)
        return f"https://{TTS_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{neutral_key}"
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "404":
            # Fallback to general neutral message
            fallback_key = "neutral/general/01.wav"
            return f"https://{TTS_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{fallback_key}"
        raise
    

def get_sms_message_content(customer_response_type="general"):
    """
    Get appropriate SMS content based on customer sentiment
    """
    sms_messages = {
        "general": "차집사 다이렉트 안내 관련하여 정확한 정보를 검토 후 다시 연락드리겠습니다. 문의사항이 있으시면 언제든 연락주세요.",
        
        "busy_response": "바쁘신 중에 시간 내주셔서 감사했습니다. 간단한 보험료 비교 자료를 문자로 보내드렸습니다. 편하실 때 검토해보시고 문의사항 있으시면 연락주세요.",
        
        "consideration": "신중하게 검토하시는 자세가 좋습니다. 상세한 상품 비교 자료와 혜택 정보를 문자로 보내드렸습니다. 궁금한 점 있으시면 언제든 연락주세요.",
        
        "not_interested": "현재는 관심이 없으시더라도 향후 보험료 절약이 필요하실 때를 위해 연락처를 남겨드립니다. 차집사 다이렉트 차은하 팀장 010-XXXX-XXXX"
    }
    
    return sms_messages.get(customer_response_type, sms_messages["general"])

def trigger_sms_notification(contact_id: str, user_speech: str, customer_response_type="general"):
    """
    Trigger contextual SMS notification based on customer sentiment
    """
    try:
        # Get customer phone number from Connect contact attributes or customer data
        # This would need to be implemented based on your Connect setup
        
        # Get appropriate SMS content
        sms_content = get_sms_message_content(customer_response_type)
        
        # Log the SMS trigger (actual SMS sending would depend on your service)
        _log("SMS notification triggered", 
             contact_id=contact_id[:8],
             sentiment_type=customer_response_type,
             message_preview=sms_content[:50])
        
        # Here you would integrate with your SMS service (SNS, Twilio, etc.)
        # Example:
        # sns = boto3.client('sns')
        # sns.publish(
        #     PhoneNumber=customer_phone,
        #     Message=sms_content
        # )
        
        return {"success": True, "message_type": customer_response_type}
        
    except Exception as e:
        _log("SMS notification failed", error=str(e))
        return {"success": False, "error": str(e)}

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

s3 = _s3_client()
lambda_client = boto3.client("lambda", region_name=os.getenv("AWS_REGION", CONNECT_REGION))

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


def _is_chat_on(params: dict) -> bool:
    if params and "chat_on" in params:
        return str(params["chat_on"]).lower() in ("1","true","y","yes")
    bypass = os.getenv("BYPASS_CHAT", "0")
    return not (str(bypass).lower() in ("1","true","y","yes"))

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
    bucket     = item.get("bucket", TTS_BUCKET)
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

    created = ddb_put_pending_if_absent(pk, TTS_BUCKET, key, text)
    if DISABLE_DDB:
        created = True  # force worker start in S3-only mode
    if not created:
        # Race condition: someone else just created it
        item = ddb_get(pk) or {}
        return {
            "job_id": "",
            "bucket": item.get("bucket", TTS_BUCKET),
            "key": item.get("key", key),
            "pk": pk,
            "prompt_arn": "",
            "ready": "false",
            "cache_hit": "true",  # ← Race condition is still a cache scenario
            "setAttributes": {
                "ready": "false",
                "cache_hit": "true",
                "job_bucket": item.get("bucket", TTS_BUCKET),
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
            "bucket":TTS_BUCKET,"key":key,"pk":pk,
            "chat_on": chat_on
        }).encode("utf-8")
    )
    
    return {
        "job_id": job_id,
        "bucket": TTS_BUCKET,
        "key": key,
        "pk": pk,
        "prompt_arn": "", 
        "ready": "false",
        "cache_hit": "false",  # ← New job, not a cache hit
        "setAttributes": {
            "job_bucket": TTS_BUCKET, 
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
        bucket = params.get("job_bucket") or TTS_BUCKET
        key    = params.get("job_key")    or params.get("key") or ""
        if not key:
            return {"setAttributes": {"ready":"false","Error":"missing_job_key"}}
        try:
            s3.head_object(Bucket=bucket, Key=key)
            url = f"https://{bucket}.s3.{os.getenv('AWS_REGION', CONNECT_REGION)}.amazonaws.com/{key}"
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
        bucket = item.get("bucket", TTS_BUCKET)
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
        bucket = item.get("bucket", TTS_BUCKET)
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
    if os.getenv("CREATE_PROMPTS","0") == "1":
        prompt_id, prompt_arn = _ensure_connect_prompt_for_key(TTS_BUCKET, key, job_id)
    else:
        prompt_arn = ""

    # Persist artifacts for the checker (and auditing)
    s3.put_object(Bucket=TTS_BUCKET, Key=key + ".txt",
                  Body=final_text.encode("utf-8"), ContentType="text/plain")
    s3.put_object(Bucket=TTS_BUCKET, Key=key + ".prompt",
                  Body=prompt_arn.encode("utf-8"), ContentType="text/plain")

    # Flip DynamoDB to ready
    if pk:
        ddb_mark_ready(pk, prompt_arn=prompt_arn, final_text=final_text, bucket=TTS_BUCKET, key=key)

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
        bucket = item.get("bucket", TTS_BUCKET)
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
    cache_bucket = TTS_BUCKET
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
        cache_bucket = item.get("bucket", TTS_BUCKET)
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

            if not idx:
                idx = DEFAULT_INDEX_CATEGORY
            
            _log("S3-only mode: checking cache", cache_key=cache_key)
            
            # Check if this audio already exists in S3
            try:
                head_response = _s3.head_object(Bucket=TTS_BUCKET, Key=cache_key)
                
                if head_response.get("ContentLength", 0) > 0:
                    # Cache hit - return the cached audio immediately
                    s3_https = f"https://{TTS_BUCKET}.s3.{CONNECT_REGION}.amazonaws.com/{cache_key}"
                    idx_safe = idx if isinstance(idx, int) else DEFAULT_INDEX_CATEGORY  # idx came back from _filler_key_url
                    filler_https = f"https://{TTS_BUCKET}.s3.{CONNECT_REGION}.amazonaws.com/{FILLER_PREFIX}/{raw_cat}/{idx_safe:02d}.wav" if FILLER_PREFIX else f"https://{TTS_BUCKET}.s3.{CONNECT_REGION}.amazonaws.com/{raw_cat}/{idx_safe:02d}.wav"
                    
                    _log("S3 cache hit for full synthesis - returning immediately", cache_key=cache_key)
                    
                    attrs = {
                        "ready": "true",
                        "AudioS3Url0": filler_https, #s3_url(TTS_BUCKET, f"{raw_cat}/{raw_idx:02d}.wav"),
                        "AudioS3UrlCount": "1",
                        "BatchCount": "1",
                        "HasMore": "true",                 # we will poll for final
                        "NextIndexOut": "1",
                        "mode": "full_with_filler",
                        "ChatAnswer": chatbot_response,
                        "filler_category": raw_cat,
                        "filler_index": str(raw_idx),
                        "job_bucket": TTS_BUCKET,
                        "job_key": cache_key,             # e.g. "connect/sessions/full_cache/<id>.wav"
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
                bucket = item.get("bucket", TTS_BUCKET)
                key = item["key"]
                s3_url = f"https://{bucket}.s3.{CONNECT_REGION}.amazonaws.com/{key}"
                
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
                    "job_key": cache_key
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

        if not idx:
            idx = DEFAULT_INDEX_CATEGORY
        
        _log("Selected filler audio", category=cat, index=idx, key=filler_key)
        
        # 2. Set up cache entry for background generation
        if not DISABLE_DDB:
            # Only create DDB entry if DDB is enabled
            created = ddb_put_pending_if_absent(pk, TTS_BUCKET, cache_key, chatbot_response)
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
                "bucket": TTS_BUCKET,
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
            "job_bucket": TTS_BUCKET,
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



def debug_cache_lookup_detailed(user_input: str, chatbot_response: str):
    """
    Detailed debugging to see why cache lookup is failing
    """
    locale = "ko-KR"
    
    # Step 1: Compute hashes
    user_hash = utt_hash(user_input, short=True)
    response_hash = utt_hash(chatbot_response, short=True)
    
    _log("=== CACHE LOOKUP DEBUG ===")
    _log("Input analysis", 
         user_input=user_input,
         chatbot_response=chatbot_response,
         user_hash=user_hash,
         response_hash=response_hash,
         locale=locale)
    
    results = {
        "user_input": user_input,
        "chatbot_response": chatbot_response,
        "user_hash": user_hash,
        "response_hash": response_hash,
        "lookups": []
    }
    
    # Step 2: Try lookup by user input hash
    try:
        resp = UTT_CACHE.get_item(
            Key={"utterance_hash": user_hash, "locale": locale},
            ConsistentRead=True
        )
        item = resp.get("Item")
        
        lookup_result = {
            "method": "user_hash_lookup",
            "hash_used": user_hash,
            "found": bool(item),
            "status": item.get("status") if item else None,
            "approval_type": item.get("approval_type") if item else None,
            "audio_uri": _redact(item.get("audio_s3_uri", ""), 40) if item else None,
            "cached_text": item.get("cached_text", "")[:50] if item else None
        }
        results["lookups"].append(lookup_result)
        
        _log("User hash lookup", **lookup_result)
        
    except Exception as e:
        _log("User hash lookup failed", error=str(e))
        results["lookups"].append({
            "method": "user_hash_lookup", 
            "hash_used": user_hash,
            "error": str(e)
        })
    
    # Step 3: Try lookup by response hash
    try:
        resp = UTT_CACHE.get_item(
            Key={"utterance_hash": response_hash, "locale": locale},
            ConsistentRead=True
        )
        item = resp.get("Item")
        
        lookup_result = {
            "method": "response_hash_lookup",
            "hash_used": response_hash,
            "found": bool(item),
            "status": item.get("status") if item else None,
            "approval_type": item.get("approval_type") if item else None,
            "audio_uri": _redact(item.get("audio_s3_uri", ""), 40) if item else None,
            "cached_text": item.get("cached_text", "")[:50] if item else None
        }
        results["lookups"].append(lookup_result)
        
        _log("Response hash lookup", **lookup_result)
        
    except Exception as e:
        _log("Response hash lookup failed", error=str(e))
        results["lookups"].append({
            "method": "response_hash_lookup",
            "hash_used": response_hash, 
            "error": str(e)
        })
    
    # Step 4: Scan for any entries with matching text
    try:
        scan_resp = UTT_CACHE.scan(
            FilterExpression=(
                "#status = :approved AND #locale = :locale AND "
                "(#original_utterance = :user_input OR #cached_text = :user_input OR "
                "#chatbot_response = :user_input OR #original_utterance = :response OR "
                "#cached_text = :response OR #chatbot_response = :response)"
            ),
            ExpressionAttributeNames={
                "#status": "status",
                "#locale": "locale",
                "#original_utterance": "original_utterance",
                "#cached_text": "cached_text",
                "#chatbot_response": "chatbot_response"
            },
            ExpressionAttributeValues={
                ":approved": "approved",
                ":locale": locale,
                ":user_input": user_input,
                ":response": chatbot_response
            },
            Limit=5
        )
        
        scan_items = scan_resp.get("Items", [])
        _log("Text scan found items", count=len(scan_items))
        
        for i, item in enumerate(scan_items):
            scan_result = {
                "method": f"text_scan_{i+1}",
                "item_hash": item.get("utterance_hash"),
                "approval_type": item.get("approval_type"),
                "audio_uri": _redact(item.get("audio_s3_uri", ""), 40),
                "cached_text": item.get("cached_text", "")[:50],
                "original_utterance": item.get("original_utterance", "")[:50],
                "chatbot_response": item.get("chatbot_response", "")[:50]
            }
            results["lookups"].append(scan_result)
            _log(f"Scan result {i+1}", **scan_result)
            
    except Exception as e:
        _log("Text scan failed", error=str(e))
        results["lookups"].append({
            "method": "text_scan",
            "error": str(e)
        })
    
    # Step 5: Summary
    found_any = any(lookup.get("found") for lookup in results["lookups"])
    _log("Cache debug summary", 
         found_any_entries=found_any,
         total_lookups=len(results["lookups"]))
    
    return results

def _synthesize_full_worker(event):
    """
    Background worker that generates the full audio using the non-streaming /synthesize endpoint.
    Now supports both DDB and S3-only modes with enhanced debugging.
    """
    try:
        text = (event.get("text") or "").strip()
        bucket = event.get("bucket") or TTS_BUCKET
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

        # # Create Connect prompt if needed
        prompt_arn = ""
        # if os.getenv("USE_CONNECT_PROMPT", "0") == "1":
        #     try:
        #         prompt_id, prompt_arn = _ensure_connect_prompt_for_key(
        #             bucket, key, key.split('/')[-1].replace('.wav', '')
        #         )
        #         _log("synthesize_full_worker: created Connect prompt", prompt_arn=prompt_arn[:60] + "...")
        #     except Exception as prompt_error:
        #         _log("synthesize_full_worker: Connect prompt creation failed", error=str(prompt_error))

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

def _check_full_ready(event):
    """
    Check if the full audio is ready. Now supports both DDB and S3-only modes.
    Fixed to use the correct S3 client with proper permissions.
    """
    params = event.get("Details", {}).get("Parameters", {}) or {}
    
    if DISABLE_DDB:
        # S3-only mode: check by job_bucket and job_key
        bucket = params.get("job_bucket") or TTS_BUCKET
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
                s3_url = f"https://{bucket}.s3.{CONNECT_REGION}.amazonaws.com/{key}"
                
                # Try to create Connect prompt if needed
                prompt_arn = ""
                # try:
                #     prompt_id, prompt_arn = _ensure_connect_prompt_for_key(bucket, key, key.split('/')[-1].replace('.wav', ''))
                #     _log("Created Connect prompt for full audio", prompt_arn=prompt_arn[:60] + "...")
                # except Exception as prompt_error:
                #     _log("Failed to create Connect prompt", error=str(prompt_error))
                
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
            bucket = item.get("bucket", TTS_BUCKET)
            key = item.get("key", "")
            prompt_arn = item.get("prompt_arn", "")
            final_text = item.get("final_text", "")
            
            s3_url = f"https://{bucket}.s3.{CONNECT_REGION}.amazonaws.com/{key}"
            
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

def _enhanced_chat_and_stream_with_cache(event, user_speech: str):
    """
    Enhanced chat and stream with caching logic per your flowchart
    """
    if not user_speech:
        user_speech = "안녕하세요"
    
    locale = "ko-KR"  # or extract from event
    
    # Step 1: Check utterance cache
    audio_uri = ddb_get_cached_audio_response_preferred(user_speech, locale)
    if audio_uri:
        _log("Cache hit - playing cached audio", uri=_redact(audio_uri, 40))
        return {"setAttributes": {
            "ready": "true",
            "AudioS3Url0": audio_uri,
            "AudioS3UrlCount": "1",
            "BatchCount": "1",
            "HasMore": "false",
            "cache_hit": "true",
            "UserInput": user_speech
        }}
    
    # Step 2: Get chatbot response
    session_id = _session_id_from_event(event, str(uuid.uuid4()))
    try:
        chatbot_response = call_chatbot(user_speech, session_id)
    except Exception as e:
        _log("Chatbot failed, using original input", error=str(e))
        chatbot_response = user_speech
    
    # Step 3: Check if we have cached audio for the response
    response_audio_uri = ddb_get_cached_audio_response_preferred(chatbot_response, locale)
    if response_audio_uri:
        _log("Response cache hit", uri=_redact(response_audio_uri, 40))
        return {"setAttributes": {
            "ready": "true",
            "AudioS3Url0": response_audio_uri,
            "AudioS3UrlCount": "1",
            "BatchCount": "1",
            "HasMore": "false",
            "cache_hit": "true",
            "UserInput": user_speech,
            "ChatAnswer": chatbot_response
        }}
    
    # Step 4: Cache miss - log to SttInbox for review
    try:
        inbox_id = uuid4().hex  # Generate unique inbox_id
        inbox_item = {
            "inbox_id": inbox_id,  # Use inbox_id as hash key (matches SttInboxNew)
            "contact_id": contact_id,  # Keep as regular attribute
            "turn_ts": int(time.time()),  # Keep as regular attribute
            "locale": locale,
            "utterance_text": user_speech,
            "utterance_norm": normalize_utt(user_speech),
            "utterance_hash": utt_hash(user_speech, short=True),
            "proposed_response_text": chatbot_response,
            "response_norm": normalize_utt(chatbot_response),
            "response_hash": utt_hash(chatbot_response, short=True),
            "review_status": "open",
            "needs_review": True,
            "cache_both": True
        }
        
        _log("About to write to SttInboxNew", inbox_id=inbox_id)
        
        INBOX.put_item(Item=inbox_item)
        
        _log("Successfully logged to SttInboxNew", inbox_id=inbox_id)
        
    except Exception as e:
        _log("ERROR: Failed to log to SttInboxNew", error=str(e))
    
    # Step 5: Generate audio anyway (for immediate user experience)
    return _enhanced_chat_and_stream(event)

def _synthesize_full_with_filler_cached(event, user_speech: str):
    """
    Full synthesis with filler, cache-aware version
    """
    if not user_speech:
        user_speech = "안녕하세요"
    
    locale = "ko-KR"
    
    # Check cache first
    audio_uri = ddb_get_cached_audio_response_preferred(user_speech, locale)
    if audio_uri:
        return {
            "ready": "true",
            "AudioS3Url0": audio_uri,
            "AudioS3UrlCount": "1",
            "BatchCount": "1",
            "HasMore": "false",
            "cache_hit": "true",
            "UserInput": user_speech
        }
    
    # Get chatbot response and check its cache
    session_id = _session_id_from_event(event, str(uuid.uuid4()))
    try:
        chatbot_response = call_chatbot(user_speech, session_id)
        response_audio_uri = ddb_get_cached_audio_response_preferred(chatbot_response, locale)
        if response_audio_uri:
            return {
                "ready": "true",
                "AudioS3Url0": response_audio_uri,
                "AudioS3UrlCount": "1",
                "BatchCount": "1",
                "HasMore": "false",
                "cache_hit": "true",
                "UserInput": user_speech,
                "ChatAnswer": chatbot_response
            }
    except Exception:
        chatbot_response = user_speech
    
    # Log to inbox and proceed with filler synthesis
    _log_to_stt_inbox(user_speech, chatbot_response, locale, event)
    return _synthesize_full_with_filler(event)

def _log_to_stt_inbox(user_speech: str, chatbot_response: str, locale: str, event: dict):
    """
    Log utterance and response to SttInbox for review
    """
    try:
        contact_id = event.get("Details", {}).get("ContactData", {}).get("ContactId", "unknown")
        
        INBOX.put_item(Item={
            "inbox_id": uuid4().hex,
            "ts": int(time.time()),
            "contact_id": contact_id,
            "locale": locale,
            "utterance_text": user_speech,
            "utterance_norm": normalize_utt(user_speech),
            "utterance_hash": utt_hash(user_speech, short=True),
            "proposed_response_text": chatbot_response,
            "response_norm": normalize_utt(chatbot_response),
            "response_hash": utt_hash(chatbot_response, short=True),
            "review_status": "open",
            "needs_review": True
        })
        _log("Logged to SttInbox for review", 
             utterance_preview=user_speech[:50],
             response_preview=chatbot_response[:50])
    except Exception as e:
        _log("Failed to log to SttInbox", error=str(e))

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
    url = f"https://{TTS_BUCKET}.s3.{FILLER_REGION}.amazonaws.com/{key}"
    
    return cat, idx, key, url


def ddb_get_cached_audio_response_preferred(utterance_or_response: str, locale: str) -> Optional[dict]:
    """
    Simple fix: Always prefer response_audio_uri over audio_s3_uri when available.
    No complex logic, just return response audio if it exists in the cache item.
    """
    if not USE_DDB:
        return None
    
    uh = utt_hash(utterance_or_response, short=True)
    try:
        resp = UTT_CACHE.get_item(
            Key={"utterance_hash": uh, "locale": locale},
            ConsistentRead=True
        )
        item = resp.get("Item")
        if not item or item.get("status") != "approved":
            return None
        
        # Simple preference: use response_audio_uri if it exists, otherwise use audio_s3_uri
        preferred_audio_uri = item.get("response_audio_uri") or item.get("audio_s3_uri")
        
        if preferred_audio_uri:
            # Create a new item dict with the preferred audio URI
            result_item = item.copy()
            result_item["audio_s3_uri"] = preferred_audio_uri
            
            _log("Cache hit with response preference", 
                 hash=uh[:8],
                 has_response_uri=bool(item.get("response_audio_uri")),
                 has_audio_uri=bool(item.get("audio_s3_uri")),
                 using_uri=_redact(preferred_audio_uri, 40))
            
            return result_item
        
        return None
        
    except Exception as e:
        _log("Response-preferred cache lookup failed", error=str(e))
        return None

def _synthesize_with_cache_check(event):
    """
    Production version based on working debug code
    """
    user_speech = _extract_user_speech_input(event)
    if not user_speech:
        user_speech = "안녕하세요"
    
    locale = "ko-KR"
    contact_id = event.get("Details", {}).get("ContactData", {}).get("ContactId", "unknown")
    
    # Step 1: Check utterance cache
    try:
        utterance_cache = ddb_get_cached_audio_response_preferred(user_speech, locale)
        if utterance_cache:
            _log("Cache hit - utterance", hash=utt_hash(user_speech, short=True)[:8])
            url = utterance_cache["audio_s3_uri"]
            base = {
                "ready": "true",
                "AudioS3Url0": url,
                "AudioS3UrlCount": "1",
                "BatchCount": "1",
                "HasMore": "false",
                "cache_hit": "true",
                "cache_type": "customer_utterance",
                "UserInput": user_speech,
            }
            # 👇 Connect’s UpdateContactAttributes reads from $.External.setAttributes.*
            base["setAttributes"] = {
                "ready": "true",
                "AudioS3Url0": url,
                "AudioS3UrlCount": "1",
                "BatchCount": "1",
                "HasMore": "false",
                "cache_hit": "true",
                "cache_type": "customer_utterance",
                "UserInput": user_speech,
            }
            return base

    except Exception as e:
        _log("Utterance cache check failed", error=str(e))
    
    # Step 2: Get chatbot response
    session_id = _session_id_from_event(event, str(uuid.uuid4()))
    try:
        chatbot_response = call_chatbot(user_speech, session_id)
    except Exception as e:
        _log("Chatbot failed", error=str(e))
        chatbot_response = user_speech
    
    # Step 3: Check response cache
    try:
        response_cache = ddb_get_cached_audio_response_preferred(chatbot_response, locale)
        if response_cache:
            _log("Cache hit - response", hash=utt_hash(chatbot_response, short=True)[:8])
            url = response_cache["audio_s3_uri"]
            base = {
                "ready": "true",
                "AudioS3Url0": url,
                "AudioS3UrlCount": "1",
                "BatchCount": "1",
                "HasMore": "false",
                "cache_hit": "true",
                "cache_type": "chatbot_response",
                "UserInput": user_speech,
            }
            base["setAttributes"] = {
                "ready": "true",
                "AudioS3Url0": url,
                "AudioS3UrlCount": "1",
                "BatchCount": "1",
                "HasMore": "false",
                "cache_hit": "true",
                "cache_type": "chatbot_response",
                "UserInput": user_speech,
            }
            return base

    except Exception as e:
        _log("Response cache check failed", error=str(e))
    
    # Step 4: Cache miss - log to inbox
    try:
        inbox_id = uuid4().hex
        inbox_item = {
            "inbox_id": inbox_id,
            "contact_id": contact_id,
            "turn_ts": int(time.time()),
            "locale": locale,
            "utterance_text": user_speech,
            "utterance_norm": normalize_utt(user_speech),
            "utterance_hash": utt_hash(user_speech, short=True),
            "proposed_response_text": chatbot_response,
            "response_norm": normalize_utt(chatbot_response),
            "response_hash": utt_hash(chatbot_response, short=True),
            "review_status": "open",
            "needs_review": True,
            "cache_both": True
        }
        
        INBOX.put_item(Item=inbox_item)
        _log("Logged to inbox", inbox_id=inbox_id[:8])
        
    except Exception as e:
        _log("Inbox logging failed", error=str(e))
    
    # Return neutral message
    neutral_audio_url = "https://tts-bucket-250810.s3.ap-northeast-2.amazonaws.com/neutral/we-will-get-back-to-you.wav"
    
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
        "ChatAnswer": chatbot_response,
        "inbox_logged": "true"
    }

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
        params.get("Details.Parameters.user_input"),
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

def _debug_event_structure(event: dict):
    """
    Debug function to understand your event structure
    """
    try:
        _log("DEBUG: Full event structure", 
             top_level_keys=list(event.keys()))
        
        if "Details" in event:
            details = event["Details"]
            _log("DEBUG: Details structure", 
                 details_keys=list(details.keys()))
            
            if "ContactData" in details:
                contact_data = details["ContactData"]
                _log("DEBUG: ContactData structure", 
                     contact_keys=list(contact_data.keys()),
                     contact_values={k: str(v)[:50] for k, v in contact_data.items()})
            
            if "Parameters" in details:
                params = details["Parameters"]
                _log("DEBUG: Parameters structure", 
                     param_keys=list(params.keys()),
                     param_values={k: str(v)[:50] for k, v in params.items()})
        
        # Check for top-level user input fields
        user_input_fields = ["user_input", "CustomerInput", "InputTranscript", "text"]
        for field in user_input_fields:
            if field in event:
                _log(f"DEBUG: Found top-level {field}", value=str(event[field])[:100])
                
    except Exception as e:
        _log("DEBUG: Event structure analysis failed", error=str(e))

def test_current_cache_lookup(user_input: str = "제가 바빠요"):
    """Test what your current cache lookup function returns"""
    locale = "ko-KR"
    
    _log("Testing current cache lookup", user_input=user_input)
    
    # Test your actual cache lookup function
    cache_result = ddb_get_cached_audio_with_context_fixed(user_input, locale)
    
    _log("Current lookup result",
         found=bool(cache_result),
         audio_uri=_redact(cache_result.get("audio_s3_uri", ""), 40) if cache_result else None,
         approval_type=cache_result.get("approval_type") if cache_result else None)
    
    return bool(cache_result)

def debug_cache_write_test():
    """Test if we can actually write to the UtteranceCache table"""
    test_item = {
        "utterance_hash": "test_hash_12345",
        "locale": "ko-KR",
        "audio_s3_uri": "s3://test-bucket/test.wav",
        "status": "approved",
        "cached_text": "테스트 텍스트",
        "original_utterance": "테스트",
        "chatbot_response": "테스트 응답",
        "approval_type": "test",
        "created_at": int(time.time())
    }
    
    try:
        # Try to write a test item
        UTT_CACHE.put_item(Item=test_item)
        _log("Test write successful")
        
        # Try to read it back
        resp = UTT_CACHE.get_item(
            Key={"utterance_hash": "test_hash_12345", "locale": "ko-KR"}
        )
        found_item = resp.get("Item")
        
        if found_item:
            _log("Test read successful", status=found_item.get("status"))
            
            # Clean up - delete the test item
            UTT_CACHE.delete_item(
                Key={"utterance_hash": "test_hash_12345", "locale": "ko-KR"}
            )
            _log("Test cleanup successful")
            
            return {"success": True, "can_write": True, "can_read": True}
        else:
            _log("Test read failed - item not found")
            return {"success": False, "can_write": True, "can_read": False}
            
    except Exception as e:
        _log("Cache write test failed", error=str(e))
        return {"success": False, "error": str(e)}

def debug_existing_cache_items():
    """Debug what's actually in the cache table"""
    try:
        # Scan without any filters to see everything
        scan_resp = UTT_CACHE.scan(Limit=10)
        items = scan_resp.get("Items", [])
        
        _log("Cache table scan", total_items=len(items))
        
        detailed_items = []
        for item in items:
            detailed_item = {
                "utterance_hash": item.get("utterance_hash", ""),
                "locale": item.get("locale", ""),
                "status": item.get("status", ""),
                "approval_type": item.get("approval_type", ""),
                "cached_text": item.get("cached_text", "")[:50],
                "has_audio_uri": bool(item.get("audio_s3_uri")),
                "audio_path_type": "responses" if "responses" in item.get("audio_s3_uri", "") else "utterances" if "utterances" in item.get("audio_s3_uri", "") else "other"
            }
            detailed_items.append(detailed_item)
            _log("Cache item details", **detailed_item)
        
        return {
            "total_items": len(items),
            "items": detailed_items,
            "table_name": UTTERANCE_CACHE_TABLE
        }
        
    except Exception as e:
        _log("Cache scan failed", error=str(e))
        return {"error": str(e)}

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
    Fixed main handler that properly handles different event types
    """
    if _is_lex_code_hook(event):
        return _lex_delegate_with_interrupt_attrs(event)

    # Set up logging
    _req_filter.set_request_id(getattr(context, "aws_request_id", "-"))
    
    try:
        preview = json.dumps(event, ensure_ascii=False, default=str)[:2000]
    except Exception:
        preview = (str(event) if event is not None else "null")[:2000]
    _log("EVENT_IN", preview=preview)

    # Extract action - handle both Connect and direct Lambda invocations
    try:
        # For Connect invocations
        if "Details" in event and "Parameters" in event["Details"]:
            action = event["Details"]["Parameters"].get("action", "voice_response")
            # Handle missing ContactData gracefully
            contact_data = event["Details"].get("ContactData", {})
            contact_id = contact_data.get("ContactId", "test-contact")
        # For direct Lambda invocations (like warmup, testing)
        else:
            action = event.get("action", "warmup")
            contact_id = event.get("contact_id", "direct-invocation")
    except Exception as e:
        _log("Action extraction failed", error=str(e))
        action = "warmup"
        contact_id = "unknown"
    
    _log("Lambda handler routing", action=action, contact_id=contact_id[:8])

    # Warmup check
    global WARMED
    if not WARMED:
        try:
            _http_post_json(_warmup_url(TTS_URL), {})
        except Exception as e:
            logging.warning(f"Warmup call failed: {e}")
        WARMED = True

    try:
        # Route to appropriate handler
        if action == "voice_response":
            # Main voice response workflow (implements your flowchart)
            result = _unified_voice_response_handler_simple(event)
            return {"setAttributes": result}
            
        elif action == "warmup":
            _warm_once(force=False)
            return {"ok": True, "warmed": True, "message": "Lambda warmed successfully"}     
        elif action == "trigger_sms":
            # Enhanced SMS trigger with sentiment-based content
            params = event.get("Details", {}).get("Parameters", {})
            user_speech = params.get("user_speech") or params.get("UserInput", "")
            customer_response_type = params.get("neutral_type", "general")  # Get sentiment type from flow
            
            try:
                result = trigger_sms_notification(contact_id, user_speech, customer_response_type)
                return {"success": True, "sms_result": result}
            except Exception as e:
                _log("SMS trigger failed", error=str(e))
                return {"success": False, "error": str(e)}
                
        elif action == "get_inbox_items":
            # Get pending items for human review
            try:
                response = INBOX.scan(
                    FilterExpression='review_status = :status',
                    ExpressionAttributeValues={':status': 'open'},
                    Limit=50
                )
                
                items = response.get('Items', [])
                return {
                    "success": True,
                    "count": len(items),
                    "items": items,
                    "setAttributes": {"inbox_count": str(len(items))}
                }
            except Exception as e:
                _log("Get inbox items failed", error=str(e))
                return {"success": False, "error": str(e)}
                
        elif action == "debug_cache_miss":
            params = event.get("Details", {}).get("Parameters", {})
            user_input = params.get("user_input", "제가 바빠요")
            chatbot_response = params.get("chatbot_response") 
            
            # If no chatbot_response provided, generate it
            if not chatbot_response:
                session_id = str(uuid.uuid4())
                chatbot_response = call_chatbot(user_input, session_id)
            
            debug_results = debug_cache_lookup_detailed(user_input, chatbot_response)
            
            return {
                "debug_complete": True,
                "found_cache_entries": any(lookup.get("found") for lookup in debug_results["lookups"]),
                "lookup_attempts": len(debug_results["lookups"]),
                "user_hash": debug_results["user_hash"],
                "response_hash": debug_results["response_hash"]
            }
        elif action == "debug_cache_writes":
            write_test = debug_cache_write_test()
            scan_result = debug_existing_cache_items()
            
            return {
                "write_test": write_test,
                "existing_items": scan_result,
                "diagnosis": "cache_write_test_complete"
            }
        elif action == "test_direct":
            # Test direct invocation
            return {
                "success": True,
                "message": "Direct invocation working",
                "event_keys": list(event.keys()),
                "contact_id": contact_id
            }
                
        # Legacy actions for backwards compatibility
        # Add this action to your lambda_handler for testing:
        elif action == "debug_cache_miss":
            params = event.get("Details", {}).get("Parameters", {})
            user_input = params.get("user_input", "제가 바빠요")
            chatbot_response = params.get("chatbot_response") 
            
            # If no chatbot_response provided, generate it
            if not chatbot_response:
                session_id = str(uuid.uuid4())
                chatbot_response = call_chatbot(user_input, session_id)
            
            debug_results = debug_cache_lookup_detailed(user_input, chatbot_response)
            
            return {
                "debug_complete": True,
                "found_cache_entries": any(lookup.get("found") for lookup in debug_results["lookups"]),
                "lookup_attempts": len(debug_results["lookups"]),
                "user_hash": debug_results["user_hash"],
                "response_hash": debug_results["response_hash"],
                "setAttributes": {
                    "debug_found": str(any(lookup.get("found") for lookup in debug_results["lookups"])),
                    "debug_user_hash": debug_results["user_hash"],
                    "debug_response_hash": debug_results["response_hash"]
                }
            }
        elif action == "verify_caching":
            params = event.get("Details", {}).get("Parameters", {})
            inbox_id = params.get("inbox_id", "472ffebed60e4fcb82c4497edd56b0b3")  # Use one from your CSV
            
            result = verify_inbox_caching(inbox_id)
            return result
        elif action == "start":
            return _start(event)
        elif action == "check":
            return _check_ready_by_pk(event)
        elif action == "worker":
            return _worker(event)
        elif action == "synthesize_full_worker":
            return _synthesize_full_worker(event)
        elif action == "check_full_ready":
            return _check_full_ready(event)
        elif action == "get_next_batch":
            return _get_next_batch(event)
        elif action == "approve_both":
            return handle_approve_both_fixed(event)
        elif action == "get_inbox_details":
            return get_inbox_item_details(event)
        else:
            _log("Unknown action, defaulting to voice_response", action=action)
            if "Details" in event:
                result = _unified_voice_response_handler_simple(event)
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