import os, json, uuid, random

REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
BUCKET = os.environ.get("AUDIO_BUCKET", "tts-bucket-250810")
PREFIX_DIR = os.environ.get("PREFIX_DIR", "filler_audio/neutral/fillers/")
COUNT = int(os.environ.get("PREFIX_COUNT", "20"))

PREFIX_KEYS = [f"{PREFIX_DIR}{i:02d}.wav" for i in range(1, COUNT + 1)]
_last_key = None  # persists across warm invocations

def _https(bucket: str, key: str) -> str:
    return f"https://{bucket}.s3.{REGION}.amazonaws.com/{key}"

def lambda_handler(event, context):
    global _last_key
    response_hash = str(uuid.uuid4())

    if len(PREFIX_KEYS) <= 1:
        prefix_key = PREFIX_KEYS[0]
    else:
        prefix_key = random.choice(PREFIX_KEYS)
        if _last_key == prefix_key:
            prefix_key = random.choice([k for k in PREFIX_KEYS if k != _last_key])
    _last_key = prefix_key

    prefix_url = _https(BUCKET, prefix_key)
    return {
        "setAttributes": {
            "AudioS3Url0Filler": prefix_url,
            "response_hash": response_hash,
            "prefetch_started": "true"
        }
    }
