# lambda_connect_tts.py
# Single-Lambda router for Amazon Connect:
#   action=start  -> kick off async TTS, return job_bucket/job_key
#   action=check  -> poll S3 for object; when ready, return presigned URL
#   action=worker -> (self-invoked) do slow TTS + upload to the exact key
# Uses urllib.request (no 'requests'), structured logs, and your existing helpers.

import os
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

CACHE_TABLE = os.getenv("CACHE_TABLE", "ConnectPromptCache")
# ========= Env vars =========
CONNECT_REGION = os.getenv("CONNECT_REGION", "ap-northeast-2")
CONNECT_INSTANCE_ID = os.getenv("CONNECT_INSTANCE_ID", "eefed165-54dc-428e-a0f1-02c2ec35a22e")
PROMPT_NAME_PREFIX = os.getenv("PROMPT_NAME_PREFIX", "dyn-tts-")

connect = boto3.client("connect", region_name=CONNECT_REGION)

CHAT_URL   = os.getenv("CHAT_URL", "http://15.165.60.45:5000/chat")  # optional
CHAT_TOKEN = os.getenv("CHAT_TOKEN", "")  # optional bearer
TTS_URL    = os.getenv("TTS_URL", "https://honest-trivially-buffalo.ngrok-free.app/synthesize")
TTS_TOKEN  = os.getenv("TTS_TOKEN", "")  # optional bearer

COMPANY_BUCKET  = os.getenv("COMPANY_BUCKET", "seoul-bucket-65432")
COMPANY_REGION  = os.getenv("COMPANY_BUCKET_REGION", "ap-northeast-2")
KEY_PREFIX      = os.getenv("KEY_PREFIX", "connect/sessions")

FORCE_REUPLOAD  = os.getenv("FORCE_REUPLOAD", "0") == "1"
BATCH = int(os.getenv("STREAM_BATCH", "3"))

ASSUME_ROLE_ARN = os.getenv("ASSUME_ROLE_ARN", "")
ASSUME_ROLE_EXTERNAL_ID = os.getenv("ASSUME_ROLE_EXTERNAL_ID", "")

# Use ONE env var for self-invoke
ASYNC_FUNCTION_NAME = os.getenv("ASYNC_FUNCTION_NAME", "InvokeBotLambda")  # set to THIS Lambda's function name
BYPASS_CHAT = 0
ddb = boto3.resource("dynamodb", region_name=COMPANY_REGION).Table(CACHE_TABLE)
WARMED = False

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

def _build_batch(job_id: str, next_index: int):
    urls = []
    i = next_index
    while len(urls) < BATCH:
        key = f"{job_id}/part{i}.wav"
        if _head_exists(key):
            urls.append(to_regional_url(key))
            i += 1
        else:
            break

    if not urls and _head_exists(f"{job_id}/final.wav"):
        urls = [to_regional_url(f"{job_id}/final.wav")]
        return {
            # For TTS mode, return the URL as text (not recommended)
            "AudioS3Url0": urls[0],  # This will be read as text by TTS
            "BatchCount": "1",
            "AudioS3UrlCount": "1", 
            "HasMore": "false",
            "NextIndexOut": str(next_index),
        }

    batch_count = str(len(urls))
    next_index_out = str(i if urls else next_index)
    
    # Option 2A: Return URL as text (Amazon Connect will try to read the URL as speech)
    response = {
        "BatchCount": batch_count,
        "AudioS3UrlCount": batch_count,
        "HasMore": "true",
        "NextIndexOut": next_index_out,
    }
    
    # For TTS mode, return the first URL as text
    if urls:
        response["AudioS3Url0"] = urls[0]  # This becomes the TTS text
        
    # Option 2B: Better - return a message for TTS
    # response["AudioS3Url0"] = "오디오가 준비되었습니다."  # Korean TTS message
    
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

# ========= HTTP helpers (urllib) =========
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

# ========= Audio/S3 helpers =========
def _upload_bytes_to_s3(data: bytes, content_type: str = "audio/wav", key: Optional[str] = None) -> Tuple[str, str]:
    put_key = key or _make_key()
    extra = {"ContentType": content_type}
    if os.getenv("CROSS_ACCOUNT_ACL", "0") == "1":
        extra["ACL"] = "bucket-owner-full-control"
    s3.put_object(Bucket=COMPANY_BUCKET, Key=put_key, Body=data, **extra)
    url = s3.generate_presigned_url(
        "get_object", Params={"Bucket": COMPANY_BUCKET, "Key": put_key}, ExpiresIn=300
    )
    _log("Uploaded audio to S3", bucket=COMPANY_BUCKET, key=put_key, content_type=content_type)
    return url, put_key

def _ensure_connect_playable(url_or_data: str, target_key: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Accepts:
      - data: URL ('data:audio/wav;base64,...') -> upload to our S3 -> presign
      - http(s) URL -> return as-is or re-upload based on FORCE_REUPLOAD or when target_key is given
    Returns (presigned_url_or_http_url, s3_key_or_None)
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
        _log("Handling http(s) audio URL", src=_redact(url_or_data, 40), reupload=FORCE_REUPLOAD or bool(target_key))
        if FORCE_REUPLOAD or target_key:
            content, ctype = _http_get_bytes(url_or_data)
            return _upload_bytes_to_s3(content, ctype or "audio/wav", key=target_key)
        return url_or_data, None

    raise ValueError("Unsupported audio format (expected data: or http(s) URL)")

# ========= Input extraction =========
def _extract_text(event: dict) -> str:
    d = event.get("Details", {}) or {}
    params = d.get("Parameters", {}) or {}
    contact = d.get("ContactData", {}) or {}
    attrs = contact.get("Attributes", {}) or {}

    candidates = [
        params.get("input_text"),
        params.get("user_input"),
        # Add these lines to handle dot notation keys from Connect
        params.get("Details.Parameters.user_input"),
        params.get("Details.Parameters.input_text"),
        attrs.get("user_input"),
        attrs.get("CustomerInput"),      # some people name it this
        # Lex (commonly used paths)
        params.get("lex_input_transcript"),
        d.get("Lex", {}).get("InputTranscript"),
        d.get("Lex", {}).get("Slots", {}),   # if you pass a slot value through params instead, handle above
        # Connect chat / misc
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
    
    # Check DynamoDB state first
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

from urllib.parse import urlparse, quote

def _warmup_url(base: str) -> str:
    # if base endswith '/synthesize', post to '/synthesize/warmup', else '/warmup'
    parsed = urlparse(base)
    if parsed.path.rstrip("/").endswith("/synthesize"):
        return base.rstrip("/") + "/warmup"
    return base.rstrip("/") + "/synthesize/warmup" 

def to_regional_url(key: str) -> str:
    return f"https://{COMPANY_BUCKET}.s3.{COMPANY_REGION}.amazonaws.com/{quote(key)}"

# ========= Main handler (router) =========
def lambda_handler(event, context):
    _log("RAW_EVENT", raw=json.dumps(event, indent=2, default=str))
    global WARMED
    if not WARMED:
        try:
            _http_post_json(_warmup_url(TTS_URL), {})
        except Exception as e:
            logging.warning(f"Warmup call failed: {e}")
        WARMED = True

    # Tie logs to this invocation
    _req_filter.set_request_id(getattr(context, "aws_request_id", "-"))

    # Safe, truncated event preview
    try:
        preview = json.dumps(event, ensure_ascii=False, default=str)[:2000]
    except Exception:
        preview = (str(event) if event is not None else "null")[:2000]
    _log("EVENT_IN", preview=preview)

    # Determine action
    try:
        action = (
            event.get("Details", {}).get("Parameters", {}).get("action")
            or event.get("action")
            or "start"
        ).lower()
    except Exception:
        action = "start"
    _log("Invocation router", action=action)

    # Log extracted text length for visibility
    try:
        txt = _extract_text(event)
        _log("EXTRACTED_TEXT", length=len(txt), sample=txt[:200])
    except Exception as e:
        _log("EXTRACTED_TEXT_ERROR", error=str(e))

    try:
        if action == "start":
            return _start(event)
        elif action == "start_stream":
            text = event.get("text", "안녕하세요, 테스트 중입니다.")
            start_url = f"{TTS_URL.rstrip('/').replace('synthesize', 'synthesize_stream_start')}"
            res = _http_post_json(start_url, {"text": text, "sample_rate": 8000}, token=TTS_TOKEN, timeout=5)

            job_id = res.get("job_id")
            if not job_id:
                return {"setAttributes": {"Error": "no_job_id_from_server", "ready": "false"}}

            next_index = 0
            batch_attrs = _build_batch(job_id, next_index)
            # quick backoff to avoid empty first batch
            if batch_attrs.get("BatchCount") == "0":
                time.sleep(0.5)
                batch_attrs = _build_batch(job_id, next_index)
                if batch_attrs.get("BatchCount") == "0":
                    time.sleep(0.6)
                    batch_attrs = _build_batch(job_id, next_index)

            batch_attrs["JobId"] = job_id
            batch_attrs["ready"] = "true"
            return {"setAttributes": batch_attrs}
        
        elif action == "get_next_batch":
            # Debug: Log the entire event structure
            _log("Full event structure", event=json.dumps(event, indent=2, default=str)[:1000])
            
            # Extract parameters from Connect format
            details = event.get("Details", {})
            params = details.get("Parameters", {})
            
            job_id = params.get("JobId") or event.get("JobId")
            next_idx_str = params.get("NextIndex") or event.get("NextIndex") or "0"
            
            try:
                next_idx = int(next_idx_str)
            except (ValueError, TypeError):
                next_idx = 0
            
            _log("Extracted parameters", job_id=job_id, next_idx=next_idx)

            if not job_id:
                error_msg = f"Missing JobId. Available keys: event={list(event.keys())}, params={list(params.keys())}"
                _log("Parameter error", error=error_msg)
                return {"setAttributes": {"Error": error_msg, "ready": "false"}}

            # Use the simpler original function first
            batch_attrs = _build_batch(job_id, next_idx)
            batch_attrs["JobId"] = job_id
            
            # Log what we're returning
            _log("Batch response", response=json.dumps(batch_attrs, indent=2))
            
            # Convert everything to strings for Connect
            batch_attrs = {k: str(v) for k, v in batch_attrs.items()}
            
            return {"setAttributes": batch_attrs}
        elif action == "check":
            return _check_ready_by_pk(event)
        elif action == "diag_url":
            return _diag_url(event)
        elif action == "worker":
            return _worker(event)
        elif action == "check_ready":
            return _check_ready_by_pk(event)
        elif action == "get_prompt":
            return _get_prompt(event)
        else:
            return {"error": "bad_action", "message": f"Unknown action: {action}"}
    except Exception as e:
        logger.exception("Invocation failed (router)")
        return {"error": "exception", "message": str(e)}