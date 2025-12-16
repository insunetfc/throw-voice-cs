import os, sys, json, boto3
from botocore.exceptions import ClientError

BUCKET = os.environ.get("TTS_BUCKET", "tts-bucket-250810")
REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
PROFILE = os.environ.get("AWS_PROFILE", "")

if not BUCKET:
    sys.exit("Set TTS_BUCKET, e.g. export TTS_BUCKET=connect-prompt-cache")

print("Using:", {"bucket": BUCKET, "region": REGION, "profile": PROFILE})

sess = boto3.session.Session(region_name=REGION, profile_name=PROFILE or None)
s3 = sess.client("s3", region_name=REGION)

tests = [
    ("plain", {"ContentType":"text/plain"}),
    ("AES256", {"ContentType":"text/plain","ServerSideEncryption":"AES256"}),
    ("KMS-noKey", {"ContentType":"text/plain","ServerSideEncryption":"aws:kms"}),
]

KMS_KEY = os.getenv("TTS_KMS_KEY_ID","").strip()
if KMS_KEY:
    tests.append(("KMS-withKey", {"ContentType":"text/plain","ServerSideEncryption":"aws:kms","SSEKMSKeyId":KMS_KEY}))

for name, extra in tests:
    key = f"ping/doctor-{name}.txt"
    try:
        s3.put_object(Bucket=BUCKET, Key=key, Body=b"hello", **extra)
        print(f"[OK] {name}: s3://{BUCKET}/{key}")
    except ClientError as e:
        print(f"[DENY] {name}: {e.response.get('Error',{})}")
