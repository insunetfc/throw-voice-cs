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
TTS_BUCKET = COMPANY_BUCKET
S3_REGION = COMPANY_REGION
_s3 = boto3.client("s3", region_name=S3_REGION)
USE_PRESIGN = False
FORCE_REUPLOAD  = os.getenv("FORCE_REUPLOAD", "0") == "1"
BATCH = int(os.getenv("STREAM_BATCH", "3"))
CHAT_MODE = "echo"

ASSUME_ROLE_ARN = os.getenv("ASSUME_ROLE_ARN", "")
ASSUME_ROLE_EXTERNAL_ID = os.getenv("ASSUME_ROLE_EXTERNAL_ID", "")
prefetch = int(os.getenv("BATCH_LIMIT", "1"))

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
    return f"https://{COMPANY_BUCKET}.s3.{COMPANY_REGION}.amazonaws.com/{quote(key)}"

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

# ========= Main handler (router) =========
def lambda_handler(event, context):
    if _is_lex_code_hook(event):
        return _lex_delegate_with_interrupt_attrs(event)

    # 1) Otherwise, this is your **Amazon Connect** invocation: keep your existing logic.
    session_state = event.get("sessionState", {})  # harmless if Connect calls you
    session_attributes = session_state.get("sessionAttributes", {}) or {}

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
            return _warm_once(force=False)
        elif action == "chat_and_stream":
            return _enhanced_chat_and_stream(event)
        elif action == "start_stream":
            params = (event.get("Details", {}) or {}).get("Parameters", {}) or {}
            text = (params.get("text") or event.get("text") or "").strip() or filler_text
            attrs = _start_stream(text, filler_text, include_prompts=False)  # S3 direct
            return {"setAttributes": attrs}
        elif action == "get_next_batch":
            return _get_next_batch(event)
        elif action == "chat_and_stream_cached":
            return _unified_chat_and_stream(event) 
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


