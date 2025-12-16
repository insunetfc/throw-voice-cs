# 🇰🇷 Korean Voice Bot System

A comprehensive Korean voice bot system featuring **Amazon Connect + Lex V2 (ko_KR) + Lambda** integration with multiple TTS backends and a unified web interface for testing and interaction.

> **Region/Locale:** All AWS resources (Connect, Lex, Lambda, S3) are deployed in **ap-northeast-2 (Seoul)**. Lex locale is **`ko_KR`**.

---

## ✅ System Overview

This project provides:

1. **AWS Voice Bot Infrastructure** - Production-ready voice bot using Amazon Connect and Lex V2
2. **Dual TTS Systems** - OpenVoice (real-time) and FishSpeech (cached) backends
3. **Local Web Interface** - Unified testing and interaction UI (`local_app.py`)
4. **Chatbot Integration** - Multiple chatbot backends (Chajipsa, GPT)
5. **Phone Call System** - Programmatic outbound calling with various TTS engines

---

## 🚀 Quick Start - Local Web Interface

The `local_app.py` provides a unified web interface for testing all system components.

### Prerequisites

```bash
pip install fastapi uvicorn httpx
```

### Configuration

Set these environment variables before running:

```bash
# Base endpoint for NIPA services
export NIPA_BASE="https://your-ngrok-url.ngrok-free.app"

# Authentication token
export NIPA_AUTH="Bearer YOUR_TOKEN"

# Optional: Override specific endpoints
export ELEVEN_BASE="${NIPA_BASE}/voice/eleven"
export ELEVEN_BRAIN_BASE="${NIPA_BASE}/voice/brain/gpt-voice-eleven"
export GPT_VOICE_BASE="${NIPA_BASE}/voice/brain/gpt-voice/start"
```

### Running the Server

```bash
python local_app.py
```

The server will start on `http://0.0.0.0:5051`

### Web Interface Features

#### 1. **Phone Call Tab**
Make outbound calls using different TTS engines:

- **Call with FishSpeech** - High-quality Korean voice
- **Call with ElevenLabs** - Natural English/multilingual voice
- **Call with GPT** - AI-powered conversational voice

**Features:**
- Enter phone number (international format)
- Optional display name
- Generate custom intro message
- Real-time call status updates

#### 2. **TTS Generation Tab**
Generate speech using multiple TTS engines:

- **FishSpeech** - High-quality Korean TTS with reference audio support
  - Upload reference audio (WAV format)
  - Adjust temperature parameter (0.0-1.0)
  - Download generated audio
  
- **ElevenLabs** - Real-time TTS with style control
  - Multiple voice options
  - Style presets (professional, casual, excited, etc.)
  - Cache support for faster generation
  
- **GPT Voice** - AI-powered voice synthesis
  - Processes text through GPT first
  - Natural conversational tone
  - Temperature control

#### 3. **Chatbot Tab**
Interactive chat interface with multiple backends:

- **Chajipsa Engine** - Specialized Korean chatbot
- **GPT Engine** - OpenAI GPT-based responses

**Features:**
- Real-time messaging
- Engine indicator badges
- Message history
- Dark/Light theme support

---

## 🏗️ AWS Infrastructure Architecture

### High-Level Flow

```
Caller → Amazon Connect (ko_KR)
   └─► Get customer input (Lex V2, intent = CaptureUtterance)
        └─► Dialog code hook (Lambda) - Captures first utterance
              └─► Fulfillment code hook (Lambda)
                   ├─ Cache lookup (DynamoDB)
                   ├─ TTS generation (if needed)
                   └─ Stream audio to caller

Loop in Connect:
  ┌─► Lambda: get_next_batch(JobId, NextIndex)
  │     └─ Returns { PromptARN/AudioS3Url, NextIndex, HasMore }
  ├─► Play audio chunk
  └─► If HasMore=true → repeat
```

### Key Components

1. **Amazon Connect** - Voice platform handling calls
2. **Lex V2** - Natural language understanding (ko_KR locale)
3. **Lambda Functions** - Business logic and orchestration
4. **S3** - Audio file storage
5. **DynamoDB** - TTS cache and metadata
6. **TTS Servers** - OpenVoice/FishSpeech backends

---

## 🎯 TTS System Selection

### FishSpeech (Cached)

**Best for:**
- ✅ Production deployments
- ✅ Repetitive FAQ responses
- ✅ Optimized latency
- ✅ Higher audio quality

**Configuration:**
```bash
# Lambda (FishSpeech mode)
CACHE_TABLE=UtteranceCache
BATCH_APPROVAL_MODE=true
NORMALIZE_KOREAN=true
```

**DynamoDB Tables:**
- `UtteranceCache` - Normalized utterances → S3 audio URLs
- `SttInboxNew` - Pending utterances awaiting approval

---

## 📦 Repository Structure

```
.
├── local_app.py                    # 🔴 Unified web interface
├── server_components/              # Main server components
│   ├── app.py                     # Core TTS server
│   ├── api_test/                  # API testing scripts
│   │   ├── call_phone.py         # Call testing
│   │   ├── chatbot_response.py   # Chatbot testing
│   │   └── tts_response.py       # TTS testing
│   ├── bridge_api/                # Bridge API for multiple TTS providers
│   │   ├── app.py                # Bridge API server
│   │   ├── adapters/             # TTS provider adapters
│   │   │   ├── elevenlabs.py    # ElevenLabs adapter
│   │   │   ├── gpt_realtime.py  # GPT Realtime adapter
│   │   │   └── minimax.py       # Minimax adapter
│   │   └── utils/                # Shared utilities
│   │       ├── audio.py         # Audio processing
│   │       ├── logger.py        # Logging utilities
│   │       └── s3.py            # S3 operations
│   ├── chatbot/                  # Chatbot system
│   │   ├── app.py               # Chatbot server
│   │   ├── data/                # Training data & datasets
│   │   │   ├── intent_dataset.csv
│   │   │   ├── response_bank_large.csv
│   │   │   └── UtteranceCacheDataset.csv
│   │   ├── intent_training/     # Intent classification training
│   │   │   ├── train.py        # Training script
│   │   │   └── test.py         # Testing script
│   │   └── models/              # Trained models
│   │       ├── intent_classification_model_promotion.pth
│   │       └── retriever_multitask_pairs.pt
│   ├── phone_call/              # Phone call orchestration
│   │   └── app.py              # Call management
│   ├── tts/                     # TTS utilities and scripts
│   │   ├── app.py              # Main TTS app
│   │   ├── config.yaml         # TTS configuration
│   │   ├── batch_approve_inbox.py  # Approval workflow
│   │   ├── coverage_tester.py  # Cache coverage testing
│   │   └── generate_fillers_tts.py  # Filler generation
│   ├── environments/            # Environment configs
│   │   ├── run_server.sh       # Server startup script
│   │   └── envrc               # Environment variables
│   ├── NIPA_cloud/             # NIPA cloud container configs
│   └── requirements.txt         # Python dependencies
├── backup/                      # AWS configuration backups
│   ├── aws_connect_backup/     # Connect flow backups
│   │   ├── connect-backup/    # Contact flows
│   │   └── iam-backup/        # IAM configurations
│   ├── InvokeBotLambda.py     # Lambda functions
│   └── TranscribeCustomerSpeech.py
├── lex_bots/                    # Lex V2 bot definitions
│   ├── bot_archive/            # Bot export files
│   └── build_script/           # Bot build scripts
├── amazon-connect-realtime-transcription-master/  # Transcription integration
├── reports/                     # Daily progress reports
│   ├── *.pdf                   # Daily reports
│   └── Flow_Diagram.png        # System flow diagrams
├── README_archive/              # Archived documentation
├── requirements.txt             # Top-level dependencies
├── README.md                   # This file
└── README_ko.md                # Korean documentation
```

---

## 🔌 API Endpoints Reference

### Local Web App (`local_app.py`)

#### TTS Endpoints

**FishSpeech TTS**
```http
POST /api/tts
Content-Type: multipart/form-data

text: string
temperature: float (default: 0.8)
ref_audio: file (optional)
```

**ElevenLabs TTS**
```http
POST /api/tts-eleven
Content-Type: multipart/form-data

text: string
style: string (optional)
voice_id: string (optional)
use_cache: boolean (default: true)
```

**GPT Voice TTS**
```http
POST /api/tts-gpt-voice
Content-Type: multipart/form-data

text: string
temperature: float (default: 0.6)
```

#### Phone Call Endpoints

```http
POST /api/call_fish
POST /api/call_eleven
POST /api/call_gpt
Content-Type: multipart/form-data

phone_number: string
```

```http
POST /api/generate-intro
Content-Type: multipart/form-data

phone_number: string
display_name: string (optional)
```

#### Chatbot Endpoints

```http
POST /api/chat-chajipsa
POST /api/chat-gpt
Content-Type: multipart/form-data

message: string
```

---

## 🛠️ Development Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit with your credentials
nano .env
```

### 4. Start Services

**Option A: Local Web Interface Only**
```bash
python local_app.py
```

**Option B: Full Stack (TTS Server + Web Interface)**
```bash
# Terminal 1: Start TTS server
cd server_components
source environments/run_server.sh

# Terminal 2: Start web interface
python local_app.py
```

**Option C: With Ngrok Tunneling**
```bash
# Terminal 1: TTS server
cd /home/work/VALL-E/
source uvicorn.sh

# Terminal 2: Ngrok
cd /home/work/VALL-E/
source ngrok.sh

# Terminal 3: Web interface
python local_app.py
```

---

## 🧪 Testing

### Test Phone Calls

```bash
# Call with FishSpeech
curl -X POST http://localhost:5051/api/call_fish \
  -F "phone_number=+821012345678"

# Call with GPT Voice
curl -X POST http://localhost:5051/api/call_gpt \
  -F "phone_number=+821012345678"

# Call with ElevenLabs
curl -X POST http://localhost:5051/api/call_eleven \
  -F "phone_number=+821012345678"
```

### Test TTS Generation

```bash
# Test FishSpeech
curl -X POST http://localhost:5051/api/tts \
  -F "text=안녕하세요" \
  -F "temperature=0.8"

# Test ElevenLabs
curl -X POST http://localhost:5051/api/tts-eleven \
  -F "text=Hello world" \
  -F "style=professional"

# Test GPT Voice
curl -X POST http://localhost:5051/api/tts-gpt-voice \
  -F "text=Tell me about AI"
```

### Test Chatbot

```bash
# Chajipsa chatbot
curl -X POST http://localhost:5051/api/chat-chajipsa \
  -F "message=안녕하세요"

# GPT chatbot
curl -X POST http://localhost:5051/api/chat-gpt \
  -F "message=Hello, how are you?"
```

---

## 🔧 Configuration Guide

### Required Environment Variables

```bash
# Core Configuration
NIPA_BASE="https://your-backend.ngrok-free.app"
NIPA_AUTH="Bearer YOUR_SECRET_TOKEN"

# Optional Overrides
ELEVEN_BASE="${NIPA_BASE}/voice/eleven"
ELEVEN_BRAIN_BASE="${NIPA_BASE}/voice/brain/gpt-voice-eleven"
GPT_VOICE_BASE="${NIPA_BASE}/voice/brain/gpt-voice/start"

# AWS Configuration (for production deployment)
COMPANY_BUCKET="tts-bucket-250810"
COMPANY_REGION="ap-northeast-2"
CONNECT_INSTANCE_ID="eefed165-54dc-428e-a0f1-02c2ec35a22e"
CACHE_TABLE="UtteranceCache"

# TTS Configuration
STREAM_BATCH=1
LOG_LEVEL="INFO"
TTS_MODE="openvoice"  # or "fishspeech"
USE_FILLERS=true
```

### Lambda Environment Variables

```bash
# Common Settings
COMPANY_BUCKET=tts-bucket-250810
COMPANY_REGION=ap-northeast-2
CONNECT_REGION=ap-northeast-2
STREAM_BATCH=1

# OpenVoice Mode
TTS_URL=http://openvoice-server:8000/synthesize
USE_FILLERS=true

# FishSpeech Mode
CACHE_TABLE=UtteranceCache
BATCH_APPROVAL_MODE=true
TTS_URL=https://fishspeech.ngrok-free.app/synthesize
```

---

## 🚨 Troubleshooting

### Local App Issues

**Problem: Server won't start**
```bash
# Check if port 5051 is in use
lsof -i :5051

# Kill existing process
kill -9 <PID>
```

**Problem: Connection refused errors**
```bash
# Verify NIPA_BASE is accessible
curl ${NIPA_BASE}/healthz

# Check authentication
curl -H "Authorization: ${NIPA_AUTH}" ${NIPA_BASE}/api/status
```

**Problem: TTS generation fails**
- Verify ngrok tunnel is active
- Check TTS server logs
- Ensure reference audio is in correct format (WAV, 16kHz)

### AWS Infrastructure Issues

**Problem: Double audio playback**
- Ensure Lambda fulfillment returns `messages: []`
- Verify Connect flow plays from cache correctly

**Problem: High latency**
- Check `STREAM_BATCH=1` in Lambda
- Verify S3 bucket region matches Connect region
- Enable provisioned concurrency for Lambda

**Problem: Cache misses**
- Run batch approval workflow
- Check DynamoDB `UtteranceCache` table
- Verify hash generation consistency

---

## 📊 Performance Metrics

### TTS Systems Comparison

| Metric | FishSpeech | ElevenLabs | GPT Voice |
|--------|-------------|------------|-----------|
| **First Response** | ~0.5s (cached) | ~3-4s | ~1-2s |
| **Audio Quality** | Excellent | Excellent | Good |
| **Korean Support** | EExcellent | Limited | Fair |
| **Cache Support** | Yes | Yes | No |
| **Cost** | Low | High | Medium |

### System Performance

- **Call Setup Time:** < 2 seconds
- **TTS Generation (cached):** < 500ms
- **TTS Generation (uncached):** 2-4 seconds
- **Cache Hit Rate:** ~85-95% (FAQ systems)

---

## 🔐 Security Considerations

### Local Development

1. **Never commit credentials** to version control
2. Use `.env` files for local configuration
3. Rotate `NIPA_AUTH` tokens regularly
4. Use HTTPS for all production endpoints

### Production Deployment

1. **Use AWS Secrets Manager** for sensitive data
2. **Enable CloudWatch logging** for audit trails
3. **Implement rate limiting** on API endpoints
4. **Use VPC endpoints** for S3/DynamoDB access
5. **Enable S3 bucket encryption**
6. **Restrict IAM permissions** to minimum required

---

## 📚 Additional Resources

### Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed system architecture
- [README_ko.md](README_ko.md) - Korean setup guide
- `/reports` - Daily development progress reports

### Testing Scripts

- `api_test/call_phone.py` - Phone call testing
- `api_test/chatbot_response.py` - Chatbot testing
- `api_test/tts_response.py` - TTS endpoint testing

### Utilities

- `tts/batch_approve_inbox.py` - Approve pending TTS requests
- `tts/coverage_tester.py` - Test cache coverage
- `tts/ddb_setup_and_seed.py` - Initialize DynamoDB tables

---

## 🤝 Contributing

### Reporting Issues

When reporting issues, please include:
1. Steps to reproduce
2. Expected vs actual behavior
3. Relevant logs (sanitize credentials!)
4. Environment details (OS, Python version, etc.)

### Development Workflow

1. Create feature branch from `main`
2. Make changes and test locally
3. Update documentation if needed
4. Submit pull request with description

---

## 📝 Changelog

- **2025-11-17**: Added comprehensive documentation for `local_app.py`
- **2025-08-27**: Integrated OpenVoice and FishSpeech systems
- **2025-08-27**: Added first-utterance capture and streaming playback
- **2025-08-27**: Category-based fillers and TTS warmup