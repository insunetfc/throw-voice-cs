import os, json, logging, urllib.request, urllib.error, time
import boto3

log = logging.getLogger()
log.setLevel(logging.INFO)

# === Env ===
CHAT_URL   = os.getenv("CHAT_URL", "http://15.165.60.45:5000/chat")
CHAT_TOKEN = os.getenv("CHAT_TOKEN", "")  # optional bearer

TTS_URL    = os.getenv("TTS_URL", "https://al-consensus-habitat-bachelor.trycloudflare.com/synthesize")
TTS_TOKEN  = os.getenv("TTS_TOKEN", "")  # optional bearer

TTS_BUCKET = os.getenv("TTS_BUCKET", "tts-bucket-250810")  # only used if your TTS returns bucket/key
KEY_PREFIX = os.getenv("KEY_PREFIX", "sessions/demo")

# If you ever want to use Connect Prompts instead of external URL playback:
USE_PROMPT  = os.getenv("USE_PROMPT", "0") == "1"
CONNECT_INSTANCE_ID   = os.getenv("CONNECT_INSTANCE_ID", "")
CONNECT_PROMPT_BUCKET = os.getenv("CONNECT_PROMPT_BUCKET", "")
PROMPT_NAME_PREFIX    = os.getenv("PROMPT_NAME_PREFIX", "fishspeech-tts")

s3 = boto3.client("s3")
connect = boto3.client("connect")

def _post_json(url: str, payload: dict, token: str = "", timeout=20) -> dict:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def _get_user_text(event: dict) -> tuple[str, str]:
    session_id = (
        event.get("session_id")
        or event.get("Details", {}).get("Parameters", {}).get("session_id")
        or "session-default"
    )
    user_text = (
        event.get("user_text")
        or event.get("Details", {}).get("Parameters", {}).get("user_text")
        or event.get("Details", {}).get("ContactData", {}).get("Attributes", {}).get("user_text")
        or event.get("inputTranscript")
        or event.get("Lex", {}).get("InputTranscript")
    )
    return user_text, session_id

def _maybe_copy_to_prompt_bucket(src_bucket: str, src_key: str) -> tuple[str, str]:
    if CONNECT_PROMPT_BUCKET and CONNECT_PROMPT_BUCKET != src_bucket:
        dst_key = f"prompts/{int(time.time())}-{os.path.basename(src_key)}"
        s3.copy_object(
            Bucket=CONNECT_PROMPT_BUCKET,
            Key=dst_key,
            CopySource={"Bucket": src_bucket, "Key": src_key},
            ContentType="audio/wav",
            MetadataDirective="REPLACE",
        )
        return CONNECT_PROMPT_BUCKET, dst_key
    return src_bucket, src_key

def _create_prompt_from_s3(s3_bucket: str, s3_key: str) -> str:
    name = f"{PROMPT_NAME_PREFIX}-{int(time.time())}"
    s3_uri = f"s3://{s3_bucket}/{s3_key}"
    resp = connect.create_prompt(
        InstanceId=CONNECT_INSTANCE_ID,
        Name=name,
        Description="FishSpeech TTS prompt",
        S3Uri=s3_uri,
    )
    return resp["PromptId"]

def lambda_handler(event, context):
    log.info("Event: %s", json.dumps(event, ensure_ascii=False))
    user_text, session_id = _get_user_text(event)
    if not user_text:
        return {"error": "no user_text"}

    # 1) Chat -> get reply text
    # If your chatbot expects {"text": "..."} instead of {"question": "..."}, swap the payload.
    try:
        chat_payload = {"session_id": session_id, "question": user_text}
        chat = _post_json(CHAT_URL, chat_payload, token=CHAT_TOKEN, timeout=20)
        answer = chat.get("answer") or chat.get("text") or ""
        if not answer:
            return {"error": "chat returned no answer", "chat": chat}
    except urllib.error.URLError as e:
        return {"error": f"chat error: {e}"}

    # 2) TTS -> accept either url/s3_url or bucket/key
    try:
        tts_payload = {"text": answer, "key_prefix": f"sessions/{session_id}", "sample_rate": 16000}
        tts = _post_json(TTS_URL, tts_payload, token=TTS_TOKEN, timeout=60)
        # normalize
        audio_url = tts.get("url") or tts.get("s3_url") or tts.get("audio_url")
        src_bucket = tts.get("bucket") or tts.get("s3_bucket")
        src_key = tts.get("key") or tts.get("s3_key")
        if not (audio_url or (src_bucket and src_key)):
            return {"error": "tts returned neither url nor bucket/key", "tts": tts}
    except urllib.error.URLError as e:
        return {"error": f"tts error: {e}"}

    # 3A) External playback (recommended & simplest)
    if not USE_PROMPT and audio_url:
        return {
            "answer": answer,
            "audio_url": audio_url,      # Connect → Play prompt → External → JSONPath: $.audio_url
            "s3_bucket": src_bucket,     # (optional info)
            "s3_key": src_key
        }

    # 3B) Prompt-based playback (needs bucket/key and Connect prompt permissions)
    if USE_PROMPT and (src_bucket and src_key):
        try:
            pbucket, pkey = _maybe_copy_to_prompt_bucket(src_bucket, src_key)
            prompt_id = _create_prompt_from_s3(pbucket, pkey)
            return {
                "answer": answer,
                "prompt_id": prompt_id,
                "s3_bucket": pbucket,
                "s3_key": pkey
            }
        except Exception as e:
            return {"error": f"create prompt failed: {e}"}

    # Fallback if USE_PROMPT=1 but no bucket/key available
    if audio_url:
        return {"answer": answer, "audio_url": audio_url}

    return {"error": "unreachable state"}
