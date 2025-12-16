import boto3, os, io
from botocore.client import Config

s3 = boto3.client(
    "s3",
    region_name=os.getenv("AWS_REGION", "ap-northeast-2"),
    config=Config(signature_version="s3v4"),
)
BUCKET = os.getenv("TTS_BUCKET", "tts-bucket-250810")

def upload_wav_to_s3(wav_bytes: bytes, key: str) -> str:
    s3.put_object(Bucket=BUCKET, Key=key, Body=wav_bytes, ContentType="audio/wav")
    return s3.generate_presigned_url(
        "get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=86400
    )
