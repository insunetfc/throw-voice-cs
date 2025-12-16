import os, sys, json, time, uuid, base64, re, pathlib, requests, boto3
from botocore.exceptions import ClientError, BotoCoreError
import hashlib, unicodedata

# =====================
# Remote endpoints
# =====================
CHAT_URL   = os.getenv("CHAT_URL", "http://15.165.60.45:5000/chat")
CHAT_TOKEN = os.getenv("CHAT_TOKEN", "")  # optional bearer
TTS_URL    = os.getenv("TTS_URL", "https://honest-trivially-buffalo.ngrok-free.app/synthesize")
TTS_TOKEN  = os.getenv("TTS_TOKEN", "")  # optional bearer

# =====================
# Cross-account S3 (company)
# =====================
MY_BUCKET  = os.getenv("COMPANY_BUCKET", "seoul-bucket-65432")
MY_REGION  = os.getenv("COMPANY_BUCKET_REGION", "ap-northeast-2")
KEY_PREFIX = os.getenv("KEY_PREFIX", "sessions/local")
FORCE_REUPLOAD = os.getenv("FORCE_REUPLOAD", "0") == "1"

# =====================
# DynamoDB (prompt cache)
# =====================
AWS_REGION   = os.getenv("AWS_REGION", "ap-northeast-2")
CACHE_TABLE  = os.getenv("CACHE_TABLE", "ConnectPromptCache")
WRITE_TO_DDB = os.getenv("WRITE_TO_DDB", "1") == "1"

# optional: identify the channel/origin of this job (e.g., "local" vs "connect")
JOB_SOURCE   = os.getenv("JOB_SOURCE", "local")

# AWS clients
s3 = boto3.client("s3", region_name="ap-northeast-2")
dynamodb = boto3.resource("dynamodb", region_name="ap-northeast-2")

def _table():
    return dynamodb.Table(CACHE_TABLE)

OUT_DIR = pathlib.Path("./out"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def _canon(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text))
    t = re.sub(r'^\.+', '', t).strip()
    t = re.sub(r'\s+', ' ', t)
    return t

def make_pk(text: str, *, voice="Jihye", sr=8000, lang="ko-KR", chat=False) -> str:
    payload = {"t": _canon(text), "v": voice, "sr": sr, "lang": lang, "chat": bool(chat)}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()

def _make_key(job_id: str | None = None) -> str:
    return f"{KEY_PREFIX.rstrip('/')}/{(job_id or uuid.uuid4().hex)}.wav"

def upload_bytes_to_my_s3(data: bytes, content_type="audio/wav", *, key: str | None = None):
    put_key = key or _make_key()
    extra = {"ContentType": content_type}
    if os.getenv("CROSS_ACCOUNT_ACL", "0") == "1":
        extra["ACL"] = "bucket-owner-full-control"
    s3.put_object(Bucket=MY_BUCKET, Key=put_key, Body=data, **extra)
    url = s3.generate_presigned_url(
        "get_object", Params={"Bucket": MY_BUCKET, "Key": put_key}, ExpiresIn=300
    )
    return MY_BUCKET, put_key, url


# ------------- HTTP helpers -------------
def post_json(url, obj, token="", timeout=30):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.post(url, json=obj, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ------------- S3 helpers -------------
def upload_bytes_to_my_s3(data: bytes, content_type="audio/wav"):
    key = f"{KEY_PREFIX.rstrip('/')}/{uuid.uuid4().hex}.wav"
    s3.put_object(Bucket=MY_BUCKET, Key=key, Body=data, ContentType=content_type)
    url = s3.generate_presigned_url(
        "get_object", Params={"Bucket": MY_BUCKET, "Key": key}, ExpiresIn=300
    )
    return MY_BUCKET, key, url

def handle_audio_to_s3(url_or_data: str, *, target_key: str | None = None):
    if url_or_data.startswith("data:"):
        m = re.match(r"data:(.*?);base64,(.*)", url_or_data)
        if not m:
            raise ValueError("bad data URL")
        content_type = m.group(1) or "audio/wav"
        body = base64.b64decode(m.group(2))
        bucket, key, presigned = upload_bytes_to_my_s3(body, content_type, key=target_key)
        return {"audio_url": presigned, "s3_bucket": bucket, "s3_key": key}

    # http(s) URL → reupload if FORCE_REUPLOAD or a target_key is given
    if FORCE_REUPLOAD or target_key:
        resp = requests.get(url_or_data, timeout=60)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "audio/wav")
        bucket, key, presigned = upload_bytes_to_my_s3(resp.content, content_type, key=target_key)
        return {"audio_url": presigned, "s3_bucket": bucket, "s3_key": key}

    return {"audio_url": url_or_data, "s3_bucket": None, "s3_key": None}


# ------------- Local save helper -------------
def save_local_copy(src_url_or_data_url: str, dest: pathlib.Path):
    if src_url_or_data_url.startswith("data:"):
        m = re.match(r"data:.*?;base64,(.*)", src_url_or_data_url)
        dest.write_bytes(base64.b64decode(m.group(1)))
        return str(dest)
    r = requests.get(src_url_or_data_url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return str(dest)

# ------------- DynamoDB write helpers -------------
def ddb_put_pending_if_absent(pk: str, bucket: str, key: str, input_text: str, ttl_days=14) -> bool:
    if not WRITE_TO_DDB:
        return False
    now = int(time.time())
    try:
        _table().put_item(
            Item={
                "pk": pk,
                "state": "pending",
                "bucket": bucket,
                "key": key,
                "input_text": _canon(input_text),
                "source": JOB_SOURCE,
                "updated_at": now,
                "expires_at": now + ttl_days*24*3600,
            },
            ConditionExpression="attribute_not_exists(pk)",
        )
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            return False
        raise


def ddb_mark_ready(pk: str, bucket: str | None, key: str | None, reply_text: str, audio_url: str):
    if not WRITE_TO_DDB:
        return
    try:
        _table().update_item(
            Key={"pk": pk},
            UpdateExpression="SET #s=:s, #b=:b, #k=:k, #r=:r, #u=:u, #a=:a",
            ExpressionAttributeNames={
                "#s": "state", "#b": "bucket", "#k": "key",
                "#r": "final_text", "#u": "updated_at", "#a": "audio_url",
            },
            ExpressionAttributeValues={
                ":s": "ready",
                ":b": bucket, ":k": key,
                ":r": reply_text[:2000],
                ":u": int(time.time()), ":a": audio_url,
            },
        )
    except (ClientError, BotoCoreError) as e:
        print(f"[ddb] update_item (READY) failed: {e}")


# ------------- Orchestrator -------------

def run_once(user_text: str) -> dict:
    if not CHAT_URL or not TTS_URL:
        raise SystemExit("Set CHAT_URL and TTS_URL env vars first.")

    # (A) derive a pk compatible with Lambda (so Connect can cache-hit)
    pk = make_pk(user_text, sr=8000, lang="ko-KR", voice="Jihye", chat=True)

    # (B) pre-allocate job/key; write pending with bucket/key
    job_id = uuid.uuid4().hex
    target_key = _make_key(job_id)
    ddb_put_pending_if_absent(pk, MY_BUCKET, target_key, user_text)

    # (C) chat -> tts
    chat = post_json(CHAT_URL, {"session_id": job_id, "question": user_text}, token=CHAT_TOKEN, timeout=20)
    reply = chat.get("answer") or chat.get("text") or user_text

    tts = post_json(TTS_URL, {"text": reply, "key_prefix": KEY_PREFIX, "sample_rate": 8000}, token=TTS_TOKEN, timeout=60)
    audio_url = tts.get("url") or tts.get("s3_url") or tts.get("audio_url")
    if not audio_url:
        raise RuntimeError(f"TTS returned no audio url: {tts}")

    # (D) force the audio into s3://MY_BUCKET/target_key (exact key)
    out = handle_audio_to_s3(audio_url, target_key=target_key)

    # (E) local copy + mark ready with SAME bucket/key
    local_path = save_local_copy(out["audio_url"], OUT_DIR / f"reply_{int(time.time())}.wav")
    ddb_mark_ready(pk, bucket=MY_BUCKET, key=target_key, reply_text=reply, audio_url=out["audio_url"])

    result = {
        "pk": pk,
        "input": user_text,
        "reply": reply,
        "audio_url": out["audio_url"],
        "s3_bucket": MY_BUCKET,
        "s3_key": target_key,
        "local_file": local_path,
        "table": CACHE_TABLE if WRITE_TO_DDB else None,
        "region": AWS_REGION if WRITE_TO_DDB else None,
        "state": "ready" if WRITE_TO_DDB else None,
    }
    print(json.dumps(result, ensure_ascii=False))
    return result



if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "아스파라거스를 좋아합니다"
    run_once(text)
