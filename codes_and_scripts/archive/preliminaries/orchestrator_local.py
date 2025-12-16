import os, sys, json, time, uuid, base64, re, pathlib, requests, boto3

# === Remote endpoints ===
CHAT_URL   = os.getenv("CHAT_URL", "http://15.165.60.45:5000/chat")
CHAT_TOKEN = os.getenv("CHAT_TOKEN", "")  # optional bearer
TTS_URL    = os.getenv("TTS_URL", "https://honest-trivially-buffalo.ngrok-free.app/synthesize")
TTS_TOKEN  = os.getenv("TTS_TOKEN", "")  # optional bearer

# === Cross-account S3 (company) ===
ASSUME_ROLE_ARN  = os.getenv("ASSUME_ROLE_ARN")  # arn:aws:iam::<COMPANY_ID>:role/S3UploadFromPartner (or S3UploadPartner)
EXT_ID           = os.getenv("ASSUME_ROLE_EXTERNAL_ID", "")  # set only if trust policy requires
MY_BUCKET   = os.getenv("COMPANY_BUCKET", "seoul-bucket-65432")               # e.g. tts-bucket-250810
MY_REGION   = os.getenv("COMPANY_BUCKET_REGION", "ap-northeast-2")
KEY_PREFIX       = os.getenv("KEY_PREFIX", "sessions/local")

# Reupload http(s) audio into your bucket?
FORCE_REUPLOAD = os.getenv("FORCE_REUPLOAD", "0") == "1"

s3 = boto3.client("s3", region_name=MY_REGION)
OUT_DIR = pathlib.Path("./out"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def post_json(url, obj, token="", timeout=30):
    headers = {"Content-Type": "application/json"}
    if token: headers["Authorization"] = f"Bearer {token}"
    r = requests.post(url, json=obj, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()

def upload_bytes_to_my_s3(data: bytes, content_type="audio/wav"):
    key = f"{KEY_PREFIX.rstrip('/')}/{uuid.uuid4().hex}.wav"
    s3.put_object(Bucket=MY_BUCKET, Key=key, Body=data, ContentType=content_type)
    url = s3.generate_presigned_url("get_object",
        Params={"Bucket": MY_BUCKET, "Key": key}, ExpiresIn=300)
    return MY_BUCKET, key, url

def handle_audio_to_s3(url_or_data: str):
    """Accept data: URL or http(s) URL. Upload data: to S3; optionally reupload http(s)."""
    if url_or_data.startswith("data:"):
        m = re.match(r"data:(.*?);base64,(.*)", url_or_data)
        if not m: raise ValueError("bad data URL")
        content_type = m.group(1) or "audio/wav"
        body = base64.b64decode(m.group(2))
        bucket, key, presigned = upload_bytes_to_my_s3(body, content_type)
        return {"audio_url": presigned, "s3_bucket": bucket, "s3_key": key}

    # http(s)
    if FORCE_REUPLOAD:
        resp = requests.get(url_or_data, timeout=60)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "audio/wav")
        bucket, key, presigned = upload_bytes_to_my_s3(resp.content, content_type)
        return {"audio_url": presigned, "s3_bucket": bucket, "s3_key": key}
    return {"audio_url": url_or_data, "s3_bucket": None, "s3_key": None}

def save_local_copy(src_url_or_data_url: str, dest: pathlib.Path):
    if src_url_or_data_url.startswith("data:"):
        m = re.match(r"data:.*?;base64,(.*)", src_url_or_data_url)
        dest.write_bytes(base64.b64decode(m.group(1)))
        return str(dest)
    r = requests.get(src_url_or_data_url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return str(dest)

def main(text: str):
    if not CHAT_URL or not TTS_URL:
        raise SystemExit("Set CHAT_URL and TTS_URL env vars first.")

    # 1) Chat (your Cloud9 bot expects session_id+question)
    chat = post_json(CHAT_URL, {"session_id": "local-1", "question": text},
                     token=CHAT_TOKEN, timeout=20)
    reply = chat.get("answer") or chat.get("text") or ""
    if not reply:
        raise RuntimeError(f"chat returned no answer: {chat}")

    # 2) TTS
    tts = post_json(TTS_URL, {"text": reply, "key_prefix": KEY_PREFIX, "sample_rate": 16000},
                    token=TTS_TOKEN, timeout=60)
    audio_url = tts.get("url") or tts.get("s3_url") or tts.get("audio_url")
    if not audio_url:
        raise RuntimeError(f"tts returned no audio url: {tts}")

    # 3) Ensure we have a Connect-playable URL; store to your S3 if needed
    out = handle_audio_to_s3(audio_url)

    # 4) Save a local copy to listen
    saved = save_local_copy(out["audio_url"], OUT_DIR / f"reply_{int(time.time())}.wav")

    print(json.dumps({
        "input": text,
        "reply": reply,
        "audio_url": out["audio_url"],
        "s3_bucket": out["s3_bucket"],
        "s3_key": out["s3_key"],
        "local_file": saved
    }, ensure_ascii=False))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "아스파라거스를 좋아합니다")