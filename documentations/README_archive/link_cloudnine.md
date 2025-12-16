# AWS Lambda → Cloud9 FastAPI Bridge for TTS (FishSpeech)

This README describes how to set up an **AWS Cloud9** environment running a **FastAPI** service (such as FishSpeech TTS) that can be called from an **AWS Lambda** function.  
This architecture allows Amazon Connect (or any AWS service) to send a request through Lambda to Cloud9, process the text using TTS, save the audio to S3, and return the URL for playback.

---

## 1. Architecture Overview

```
Amazon Connect → Lambda → Cloud9 FastAPI (TTS) → S3 (audio) → Lambda returns URL → Connect plays audio
```

---

## 2. Prerequisites

- AWS Cloud9 environment (EC2-backed)
- Python 3.10+
- AWS CLI configured with correct permissions
- IAM Role for Lambda with:
  - `AmazonS3FullAccess` (or scoped-down permissions)
  - `AWSLambdaVPCAccessExecutionRole`
- Security group access configured for Cloud9 and Lambda
- (Optional) GPU instance (`g5.xlarge`) for heavy TTS workloads

---

## 3. Set Up Python Environment in Cloud9

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

---

## 4. Install Dependencies

### Base packages
```bash
pip install fastapi uvicorn pydantic boto3
```

### FishSpeech (optional)
```bash
pip install torch torchaudio  # For GPU, match CUDA version
pip install -U huggingface_hub
huggingface-cli login  # Ensure model access
```

---

## 5. Create FastAPI Service

Create a file called `test_text.py`:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TextIn(BaseModel):
    text: str

@app.get("/healthz")
def health_check():
    return {"ok": True}

@app.post("/process")
def process_text(data: TextIn):
    # Placeholder — replace with FishSpeech TTS call
    return {"result": f"Received: {data.text}"}
```

Run the API:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Test locally:
```bash
curl -s http://127.0.0.1:8000/healthz
curl -s -X POST http://127.0.0.1:8000/process      -H 'Content-Type: application/json'      -d '{"text":"hi"}'
```

---

## 6. Make the API Reachable

### Option A — Public (Testing)
1. In AWS Console, find your Cloud9 EC2 instance.
2. Edit **Inbound Rules** of its Security Group:
   - Custom TCP Rule: Port `8000`, Source `0.0.0.0/0` (remove later for security).
3. Get public IP:
   ```bash
   curl -s http://checkip.amazonaws.com
   ```
4. Test from outside:
   ```bash
   curl http://<PUBLIC_IP>:8000/healthz
   ```

### Option B — Private VPC (Recommended)
1. Place Lambda in the same VPC and subnet as Cloud9.
2. Create two security groups:
   - Cloud9 SG: allow TCP 8000 inbound from Lambda SG.
   - Lambda SG: allow outbound TCP 8000 to Cloud9 SG.
3. Use Cloud9’s private IP in Lambda API URL.

---

## 7. Create Lambda Client

Create `bridge_client.py`:

```python
import os, sys, json, urllib.request

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/process")

def lambda_handler(event, context):
    text = (event or {}).get("text", "Hello from Lambda")
    payload = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(API_URL, data=payload, headers={"Content-Type": "application/json"})

    with urllib.request.urlopen(req, timeout=30) as resp:
        resp_data = resp.read().decode("utf-8")
    return json.loads(resp_data)

if __name__ == "__main__":
    txt = "Hello from local" if len(sys.argv) < 2 else " ".join(sys.argv[1:])
    print(lambda_handler({"text": txt}, None))
```

---

## 8. Deploy to Lambda

1. In AWS Lambda console, create a Python 3.x function.
2. Upload `bridge_client.py` (zip if needed).
3. Set handler:
   ```
   bridge_client.lambda_handler
   ```
4. Add environment variable:
   ```
   API_URL=http://<PUBLIC_OR_PRIVATE_IP>:8000/process
   ```
5. If using VPC, configure Lambda to match Cloud9’s VPC/subnet/security group.

---

## 9. Test Lambda

Test event:
```json
{
  "text": "Hello from Lambda"
}
```

Expected output:
```json
{
  "result": "Received: Hello from Lambda"
}
```

---

## 10. Next Steps

### Step 1 — Integrate FishSpeech
Replace placeholder in `/process`:
```python
from fish_speech.inference_engine import TTSInferenceEngine
# engine = TTSInferenceEngine.load(...)
# audio_path = engine.infer(data.text)
```

### Step 2 — Save Audio to S3
```python
import boto3, uuid
s3 = boto3.client("s3")
key = f"tts-output/{uuid.uuid4()}.wav"
s3.upload_file(local_audio_path, "your-bucket", key)
s3_url = f"https://{your-bucket}.s3.amazonaws.com/{key}"
return {"result": s3_url}
```

### Step 3 — Return S3 URL to Lambda
Lambda simply forwards the URL to Amazon Connect.

### Step 4 — Play in Amazon Connect
Use **Play Prompt** with **External audio** set to the returned S3 URL.

### Step 5 — Secure the API
Restrict inbound rules to Lambda SG only.

---

## 11. Notes

- Keep FastAPI persistent using `tmux` or `screen`.
- For production, consider ECS/Fargate or EC2 with `systemd`.
- GPU workloads require switching to `g5.xlarge` and installing CUDA.

---
