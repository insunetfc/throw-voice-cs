# Architecture & Implementation Details

# 🇰🇷 Korean Voice Bot — Amazon Connect + Lex V2 + Lambda + Streaming TTS (ko_KR)

Low-latency **Korean** voice bot using **Amazon Connect + Lex V2 (ko_KR) + Lambda**, with a custom TTS server that streams **μ-law WAV (8 kHz)** parts to S3 for **progressive playback**. Designed for **first-utterance capture** (no double speaking) and smooth, chunked audio in Connect.

> **Region/Locale:** All resources (Connect, Lex, Lambda, S3) are intended for **ap-northeast-2 (Seoul)**. Lex locale is **`ko_KR`**.

---

## ✅ Status (as of 2025-08-27 00:49:01)
- First-utterance capture **working** via **Dialog code hook** (no double speaking).
- End-to-end **Connect playback** of chunked audio **working**.
- BYPASS path available for quick echo if chatbot is unavailable.

---

## ✨ What’s new in this README
- Incorporated your latest files: `ChatStreamFlow.json`, `InvokeBotLambda.py`, `TranscribeCustomerSpeech.py`, `app.py`.
- Added a **Live Config Snapshot** (detected defaults) and **Recommended Overrides** (for a smoother demo).
- Clarified **Fulfillment patterns** (Lex speaks vs. Connect streaming) and **no-double-audio** rules.

---

## 🗺️ Architecture (recap)

```
Caller → Amazon Connect (ko_KR)
   └─► Get customer input (Lex V2, intent = CaptureUtterance, slot = UserInput: AMAZON.FreeFormInput [Required])
        └─► Initial response → Invoke Dialog code hook (Lambda)
              └─ copies inputTranscript → {UserInput}, then Delegate
                └─► Fulfillment code hook (Lambda)
                     ├─ Pattern A: return PlainText (Lex/Polly speaks)
                     └─ Pattern B: start TTS stream, return messages: [], set attributes for Connect playback

Loop in Connect (Pattern B):
  ┌─► Lambda: action=get_next_batch(JobId, NextIndex)
  │     └─ Returns { PromptARN or AudioS3Url, NextIndexOut, HasMore }
  ├─► Play the returned part NOW
  └─► If HasMore=true → repeat get_next_batch(NextIndexOut)
```

**S3 layout**
```
s3://<TTS_BUCKET>/<KEY_PREFIX>/<JOB_ID>/
  part0.wav
  part1.wav
  ...
  final.wav      # sentinel marks completion
```

---

## 📦 Files in this repo (latest)
- `ChatStreamFlow.json` — Amazon Connect contact flow implementing streaming playback (dynamic PromptARN/URL).
- `InvokeBotLambda.py` — Lambda router for `chat_and_stream` and `get_next_batch`, prompt management.
- `TranscribeCustomerSpeech.py` — (optional) transcription utility; not required for the core demo.
- `app.py` — TTS chunker/uploader (μ-law mono 8 kHz) that writes `partN.wav` to S3.
- `config.yaml` — service configuration for the TTS app (not parsed in this README).
- `FreeKoreanSpeech-*.zip` — Lex V2 export (ap-northeast-2 / ko_KR).

---

## 🧩 Lex configuration (final)

- **Intent**: `CaptureUtterance`
- **Slot**: `UserInput` → **`AMAZON.FreeFormInput`**, **Required = ON**
- **Initial response**: **Next step = Invoke Dialog code hook**, **Skip elicitation prompt = ON**
- **Slot prompts**: allow barge-in; **Invoke Lambda after each elicitation = OFF**; **Retries = 0/1**
- **Confirmation prompt**: **OFF**
- **Code hooks** (after attaching Lambda at the **Alias → ko_KR** level):
  - **Initialization and validation (Dialog code hook)**: **ON**
  - **Fulfillment code hook**: **ON** (pick pattern A or B below)
- **Build the alias** after any Lambda version change.

### Dialog code hook (first-utterance capture)
```python
def lambda_handler(event, context):
    intent = event["sessionState"]["intent"]
    slots = intent.get("slots") or {}
    if event.get("invocationSource") == "DialogCodeHook":
        first_text = event.get("inputTranscript") or ""
        if first_text and not (slots.get("UserInput") and slots["UserInput"].get("value")):
            slots["UserInput"] = {"value": {"originalValue": first_text, "interpretedValue": first_text}}
        return {
            "sessionState": {
                "dialogAction": {"type": "Delegate"},
                "intent": {**intent, "state": "InProgress", "slots": slots},
            }
        }
```

### Fulfillment patterns

**A) Lex speaks (PlainText message)**
```python
if event.get("invocationSource") == "FulfillmentCodeHook":
    text = get_user_text(event)
    reply = your_chatbot_call(text)
    return {
        "sessionState": {
            "dialogAction": {"type": "Close"},
            "intent": {"name": event["sessionState"]["intent"]["name"], "state": "Fulfilled"}
        },
        "messages": [{"contentType": "PlainText", "content": reply}]
    }
```

**B) Connect streams audio (no double speak)**
```python
if event.get("invocationSource") == "FulfillmentCodeHook":
    text = get_user_text(event)
    job_id, first_part = start_tts_stream(text)  # your orchestrator
    out = {
        "sessionState": {
            "dialogAction": {"type": "Close"},
            "intent": {"name": event["sessionState"]["intent"]["name"], "state": "Fulfilled"},
            "sessionAttributes": {
                "JobId": job_id,
                "HasMore": "true",
                "NextIndexOut": str(0 if first_part is None else 1)
            }
        },
        "messages": []   # IMPORTANT: no Lex TTS; Connect will play audio
    }
    if first_part and "PromptARN" in first_part:
        out["sessionState"]["sessionAttributes"]["PromptARN"] = first_part["PromptARN"]
    elif first_part and "AudioS3Url" in first_part:
        out["sessionState"]["sessionAttributes"]["AudioS3Url"] = first_part["AudioS3Url"]
    return out
```

**Helper to read text robustly**
```python
def get_user_text(event):
    slots = (event.get("sessionState", {}).get("intent", {}).get("slots") or {})
    v = slots.get("UserInput")
    if v and v.get("value"):
        return v["value"].get("originalValue") or v["value"].get("interpretedValue")
    return event.get("inputTranscript") or ""
```

---

## ☁️ Connect contact flow (Pattern B)

**Ordering to avoid skipping `part0`:**
1. **Invoke Lambda (`chat_and_stream`)** → receive `JobId`, `HasMore`, `NextIndexOut`, and optionally `PromptARN`/**`AudioS3Url`**.
2. **Set contact attributes** with those values.
3. **If PromptARN/AudioS3Url present** → **Play now**.
4. **If `HasMore` is true** → loop `get_next_batch(NextIndexOut)` and play until done.

---

## 🔎 Live config snapshot (detected from your code)

### Lambda env defaults detected

| Key | Default/Detected |
|---|---|
| `ASSUME_ROLE_ARN` | `` |
| `ASSUME_ROLE_EXTERNAL_ID` | `` |
| `ASYNC_FUNCTION_NAME` | `InvokeBotLambda` |
| `BATCH` | `3` |
| `CACHE_TABLE` | `ConnectPromptCache` |
| `CHAT_TOKEN` | `` |
| `CHAT_URL` | `http://15.165.60.45:5000/chat` |
| `COMPANY_BUCKET` | `tts-bucket-250810` |
| `COMPANY_REGION` | `ap-northeast-2` |
| `CONNECT_INSTANCE_ID` | `eefed165-54dc-428e-a0f1-02c2ec35a22e` |
| `CONNECT_REGION` | `ap-northeast-2` |
| `FORCE_REUPLOAD` | `0` |
| `KEY_PREFIX` | `connect/sessions` |
| `LOG_LEVEL` | `INFO` |
| `PROMPT_NAME_PREFIX` | `dyn-tts-` |
| `TTS_TOKEN` | `` |
| `TTS_URL` | `https://honest-trivially-buffalo.ngrok-free.app/synthesize` |


### App env defaults detected

| Key | Default/Detected |
|---|---|
| `API_TOKEN` | `` |
| `AWS_REGION` | `ap-northeast-2` |
| `CFG_PATH` | `/home/work/VALL-E/fish-speech/fishspeech_infer/config.yaml` |
| `CHUNK_MAX_CHARS` | `1` |
| `KEY_PREFIX_DEF` | `sessions/demo` |
| `PRESIGN_EXPIRES` | `600` |
| `TTS_BUCKET` | `seoul-bucket-65432` |
| `TTS_REP` | `1.0` |
| `TTS_TEMP` | `0.7` |
| `TTS_TOP_P` | `0.9` |


**Lex alias regions referenced in flow**: ap-northeast-1

---

## ✅ Recommended overrides for demo smoothness

**Lambda**  
- `STREAM_BATCH`: set to **1** (current: `None`)

**TTS App**  
- `CHUNK_MAX_CHARS`: set to **140** (current: `1`)

> Rationale: `STREAM_BATCH=1` ensures sequential playback; `CHUNK_MAX_CHARS≈140` reduces micro-chunks and prompt churn.

---

## 🔧 IAM & env

**Lambda needs:** `s3:ListBucket`, `s3:GetObject`, (optional) `connect:ListPrompts`/`CreatePrompt`/`DescribePrompt` if you use Prompt ARNs, and a resource policy allowing **Lex** to invoke the function.  
**Env examples (Lambda):**
```
STREAM_BATCH=1
PROMPT_NAME_PREFIX=dyn-tts-
USE_PRESIGN=1
```

**Env examples (TTS app):**
```
CHUNK_MAX_CHARS=140
STREAM_UPLOAD_SECONDS=1.0
```

---

## 🧪 Testing

**Lex console**: one sentence → `Slots.UserInput` should populate on the **first turn**.  
**Connect call**: hear `part0` quickly, then seamless parts.  
**Logs Insights**:
```
fields @timestamp,
sessionState.invocationSource as src,
sessionState.intent.slots.UserInput.value.originalValue as user
| sort @timestamp desc
| limit 20
```

---

## 🛟 Troubleshooting (quick)

- **Always says “안녕하세요”** → Slot empty; ensure Dialog hook copies `inputTranscript` into `{UserInput}` before `Delegate`.
- **Double audio** → Using Pattern B? Return **`messages: []`** from fulfillment and play only via Connect.
- **Part0 missing** → Always play the part returned by `chat_and_stream` before polling.
- **Prompt quota** → Keep `PROMPT_NAME_PREFIX` stable; update existing prompts instead of creating new ones.
- **Region mismatch** → Connect instance and Lex alias should be in the **same region** (ideally `ap-northeast-2`).

---

## 📜 Changelog
- **2025-08-27 00:49:01**: Updated README with live-config snapshot, overrides, and validation from your latest files.
- **2025-08-27**: First-utterance capture, Fulfillment patterns A/B, BYPASS flag, flow ordering, troubleshooting.


---

# 🔄 Enhancements (2025 Updates)

## 🎙️ Category-based Fillers
Lambda now supports **categorised fillers**:  
- `확인` (confirm)  
- `설명` (explain)  
- `공감` (empathy)  
- `시간벌기형` (stall)

Filler selection is **automatic** in `synthesize_with_filler` if no category is provided.  
The router inspects `user_text`, `asr_confidence`, `waiting_ms`, and dialog state to pick a category【83†source】.

---

## ⚡ TTS Warmup
The OpenVoice TTS server (`app_openvoice.py`) now **warms up on startup**:  
- Loads engine + CUDA context once.  
- Runs a micro-synthesis to JIT kernels.  
- Exposes `/healthz` and `/synthesize/warmup` for monitoring【82†source】.

Lambda also pings warmup once on cold start.

---

## ⚙️ Updated Environment Variables

### Lambda
- `STREAM_BATCH` — batch size for parts (recommended: 1).  
- `DISABLE_DDB=1` — enable S3-only mode.  

### TTS App
- `STREAM_MAX_CHARS=100` — first chunk small for fast playback.  
- `FIRST_URL_WAIT_MS=1400` — allow time for part0.  
- `OV_SPEED=1.5` — faster but natural rate.  
- `OV_TGT_SE_PTH` — precomputed speaker embedding.  

---

## 🛠️ Deployment Runbook

### 1. TTS Server
```bash
uvicorn app_openvoice:app --host 0.0.0.0 --port 8000 --workers 1 --loop uvloop --http httptools
```

Check warmup:
```bash
curl http://localhost:8000/healthz
```

### 2. Upload Fillers
```bash
aws s3 cp fillers/ s3://$COMPANY_BUCKET/fillers/ --recursive --acl bucket-owner-full-control
```

### 3. Lambda
- Deploy `lambda.py`.  
- Env vars: `COMPANY_BUCKET`, `TTS_URL`, `TTS_BASE_URL`.  
- Attach `s3:GetObject`, `s3:PutObject`, `connect:CreatePrompt`.  

### 4. Lex
```bash
aws lexv2-models import-bot --region ap-northeast-2 --file Placa_CallBot-1-DSJBTAHSHN-LexJson.zip
```

### 5. Connect
- Import `Promotional Call With Interrupt.json`.  
- Connect Lambda + Lex alias.  

### 6. Test
- Call Connect number.  
- Verify: filler → part0.wav → streaming parts → final.wav.

---

## 🚀 Production Notes
- Enable **Provisioned Concurrency** (1–2) for Lambda.  
- Run TTS server in same VPC/region.  
- Use cached embeddings (`OV_TGT_SE_PTH`).  
- Monitor `/healthz` and CloudWatch.  
