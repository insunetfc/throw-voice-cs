import boto3
import time
import json
from datetime import datetime

# === Configuration ===
bucket_name = "transcription-seoul-bucket"
audio_file = "promotional_calls.wav"
region = "ap-northeast-2"
language_code = "ko-KR"
media_format = "wav"

# === Generate timestamped job name ===
timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")  # up to microseconds
job_name = f"transcribe-{audio_file.rsplit('.', 1)[0]}-{timestamp}"

# === Initialize clients ===
transcribe = boto3.client("transcribe", region_name=region)
s3 = boto3.client("s3", region_name=region)

# === Start transcription job ===
media_uri = f"s3://{bucket_name}/{audio_file}"
print(f"📤 Starting transcription for {media_uri}...")

transcribe.start_transcription_job(
    TranscriptionJobName=job_name,
    LanguageCode=language_code,
    MediaFormat=media_format,
    Media={"MediaFileUri": media_uri},
    OutputBucketName=bucket_name
)

# === Poll for job completion ===
print("⏳ Waiting for transcription to complete...")
while True:
    status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
    job_status = status["TranscriptionJob"]["TranscriptionJobStatus"]
    if job_status in ["COMPLETED", "FAILED"]:
        break
    time.sleep(2)

# === Fetch transcript ===
if job_status == "COMPLETED":
    key = f"{job_name}.json"
    print(f"✅ Transcription complete. Fetching result from s3://{bucket_name}/{key}")

    response = s3.get_object(Bucket=bucket_name, Key=key)
    result_json = json.loads(response["Body"].read().decode("utf-8"))

    transcript_text = result_json["results"]["transcripts"][0]["transcript"]
    print("\n🔊 Transcribed Text:\n" + transcript_text)
else:
    print("❌ Transcription failed.")
