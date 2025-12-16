# 🇰🇷 Korean Voice Bot System (Handover Repository)

This repository contains a **production-grade Korean voice bot system** built around **Amazon Connect + Lex V2 (ko_KR) + AWS Lambda**, with multiple **Text-to-Speech (TTS)** backends and a **local web interface** for testing and orchestration.

This README is written for **handover purposes**. Some experimental or deprecated components (e.g. OpenVoice) have been **moved into deeper folders or archives** and are **not required for day-to-day operation**.

---

## 1. System Purpose

This repository contains a **production-grade Korean voice bot system** built around **Amazon Connect + Lex V2 (ko_KR) + AWS Lambda**, with optional **SIP-based calling via FreeSWITCH** for advanced telephony integration.

The system has evolved over time; this README **keeps the original structure and intent**, while removing redundancy and clearly separating **active components** from **archived or experimental ones**.

The system supports:

- Automated **outbound phone calls** via Amazon Connect
- Korean **speech recognition and intent handling** via Lex V2 (ko_KR)
- Multiple **TTS engines** (FishSpeech, ElevenLabs, GPT Voice)
- **Caching and batching** of synthesized speech for low-latency playback
- A **local web UI** for testing calls, TTS, and chatbots

All AWS resources are deployed in **ap-northeast-2 (Seoul)** unless stated otherwise.

---

## 2. High-Level Architecture

### 2.1 Amazon Connect / AWS Path (Primary)

```
Caller → Amazon Connect
   → Lex V2 (ko_KR)
      → Lambda (dialog / fulfillment)
         → TTS (FishSpeech / ElevenLabs / GPT)
         → S3 (audio storage)
         → DynamoDB (utterance cache)
```

Playback is handled in **batched streaming mode**, allowing long responses to be delivered smoothly inside Connect contact flows.

---

### 2.2 SIP / FreeSWITCH Path (Optional)

```
SIP Client / PSTN
   → FreeSWITCH
      → SIP App / HTTP bridge
         → TTS backend
         → Audio streaming / playback
```

The FreeSWITCH path was introduced later to support **SIP-based calling**, testing outside Amazon Connect, and future extensibility toward non-AWS telephony environments.

```
Caller → Amazon Connect
   → Lex V2 (ko_KR)
      → Lambda (dialog / fulfillment)
         → TTS (FishSpeech / ElevenLabs / GPT)
         → S3 (audio storage)
         → DynamoDB (utterance cache)
```

Playback is handled in **batched streaming mode**, allowing long responses to be delivered smoothly inside Connect contact flows.

---

## 3. Recommended Entry Points (Important)

To stay consistent with the original README intent, the system can be approached from **three main entry points**, depending on use case:

For handover and maintenance, **only the following entry points are critical**:

### 🔴 Local Web Interface (Primary)
- **File:** `local_app.py`
- **Purpose:** Unified UI for
  - Outbound phone calls (Connect / SIP)
  - TTS generation
  - Chatbot testing

This remains the **recommended starting point** for understanding and testing the system end-to-end.
- **File:** `local_app.py`
- **Purpose:** Unified UI for
  - Outbound phone calls
  - TTS generation
  - Chatbot testing

This is the fastest way to understand and test the system.

### 🟠 Server Components
- **Directory:** `server_components/`
- Contains:
  - TTS servers
  - Chatbot logic
  - Phone call orchestration
  - Provider adapters (ElevenLabs, GPT, etc.)

### 🟢 AWS Runtime Logic
- **Directory:** `backup/`
- Includes:
  - Lambda handlers
  - Amazon Connect flow backups
  - Transcription logic

---

## 4. Repository Structure (Maintained)

The structure below follows the **original README layout**, with redundant or deprecated elements clearly grouped instead of removed.

```
.
├── local_app.py                 # Main local testing UI (IMPORTANT)
│
├── server_components/           # Core backend logic
│   ├── app.py                  # Main server entry
│   ├── phone_call/             # Call orchestration
│   ├── chatbot/                # Chatbot engine & models
│   ├── tts/                    # TTS logic, caching, batching
│   ├── bridge_api/             # Multi-provider TTS abstraction
│   ├── environments/           # Run scripts and env files
│   └── requirements.txt
│
├── backup/                      # AWS-related backups (IMPORTANT)
│   ├── aws_connect_backup/      # Contact flows, IAM
│   ├── InvokeBotLambda.py       # Main Lambda logic
│   └── TranscribeCustomerSpeech.py
│
├── lex_bots/                    # Lex V2 bot exports & build scripts
│
├── reports/                     # Daily progress reports & diagrams
│
├── amazon-connect-realtime-transcription-master/  # Real-time transcription reference
├── freeswitch/                   # SIP / FreeSWITCH integration
│   ├── conf/                     # FreeSWITCH configuration
│   ├── scripts/                  # Dialplan / control scripts
│   └── README.md                 # SIP setup notes  # Reference integration
│
├── README_archive/              # Old / deprecated documentation
├── README.md                    # This file
├── README_ko.md                 # Korean documentation
└── requirements.txt
```

---

## 5. TTS Engines (Operational Status)

This section preserves the original multi-TTS philosophy while clarifying current usage.

### ✅ FishSpeech (Primary / Production)

- Used for **Korean TTS**
- Supports **utterance caching** via DynamoDB
- Optimized for **low latency** and repeated prompts

**Status:** Actively used and recommended

---

### ⚠️ ElevenLabs

- Used mainly for:
  - English or multilingual voices
  - Demonstration purposes

**Status:** Optional / external dependency

---

### 🗄️ OpenVoice (Archived / Reference)

- Early real-time TTS experiments
- Kept for reproducibility and comparison
- Not required for current production or SIP flows

**Status:** Archived, reference only

- Older real-time TTS experiments
- Code still exists for reference
- **Not required** for current production flow

**Status:** Archived / reference only

---

## 6. Local Development (Minimal Setup)

### Requirements

```bash
pip install -r requirements.txt
```

### Environment Variables (Minimal)

```bash
export NIPA_BASE="https://<backend-endpoint>"
export NIPA_AUTH="Bearer <TOKEN>"
```

### Run Local UI

```bash
python local_app.py
```

Access:
```
http://localhost:5051
```

---

## 7. AWS Deployment Notes (Handover)

This section consolidates previously duplicated AWS notes into a single reference.

- **Amazon Connect** handles call control and playback
- **Lex V2 (ko_KR)** captures customer utterances
- **Lambda** performs:
  - First-utterance capture
  - TTS selection
  - Cache lookup
- **S3** stores generated audio
- **DynamoDB** stores normalized utterance hashes

⚠️ IAM permissions and Connect flow configuration are critical. Refer to:

```
backup/aws_connect_backup/
```

---

## 8. What Can Be Safely Ignored (For Maintenance)

The following are intentionally preserved but **not required** for normal operation:

For day-to-day operation or handover:

- `README_archive/`
- Experimental notebooks or scripts
- Archived OpenVoice folders
- Old test scripts not referenced by `local_app.py`

These are kept **only for traceability**.

---

## 9. Recommended Handover Checklist

- [ ] Confirm `local_app.py` runs
- [ ] Verify outbound call works via FishSpeech
- [ ] Verify Lex V2 bot is deployed (ko_KR)
- [ ] Check Lambda environment variables
- [ ] Confirm S3 + DynamoDB access
- [ ] Review Connect contact flows (backup folder)

---

## 10. Maintainer Notes

This repository reflects a **real production system with historical layers**. Not all folders represent equal importance.

- This repository prioritizes **practical deployment** over cleanliness
- Some redundancy exists by design (AWS backup safety)
- When modifying TTS or call logic, start from:

```
server_components/phone_call/
server_components/tts/
```

---

## 11. Contact / Context

This system was developed as part of an **AWS-based outbound voice automation project** for Korean-language use cases.

If extending or refactoring, it is recommended to **keep FishSpeech + caching logic intact**, as this is the most stable and cost-efficient path.

