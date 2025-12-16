# 🇰🇷 Korean Voice Bot System (Handover Repository)

> **Status:** Production / Handover-ready
> **Languages:** English & Korean (below)
> **Scope:** Code & configuration only (no datasets, no model weights, no secrets)

---

## ENGLISH VERSION

### 1. Overview

This repository contains a **production-grade Korean voice bot system** built around **Amazon Connect + Lex V2 (ko_KR) + AWS Lambda**, with support for **multiple TTS backends** and an **optional SIP-based calling path via FreeSWITCH**.

The repository is prepared specifically for **handover and long-term maintenance**. Historical experiments, backups, and reference implementations are preserved, but clearly separated from active runtime components.

---

### 2. System Capabilities

* Automated **outbound phone calls** via Amazon Connect
* Korean **speech recognition and intent handling** using Lex V2 (ko_KR)
* Multiple **Text-to-Speech (TTS)** engines

  * FishSpeech (primary)
  * ElevenLabs (optional)
  * GPT Voice (experimental)
* **Low-latency audio playback** using batching and caching
* **Local web UI** for testing and orchestration
* **SIP / FreeSWITCH integration** for non-AWS telephony environments

All AWS resources are deployed in **ap-northeast-2 (Seoul)** unless otherwise noted.

---

### 3. High-Level Architecture

#### 3.1 Amazon Connect Path (Primary)

```
Caller → Amazon Connect
   → Lex V2 (ko_KR)
      → AWS Lambda
         → TTS Engine
         → S3 (Audio Storage)
         → DynamoDB (Utterance Cache)
```

This is the main production path. Audio responses are generated in batches and streamed for smooth playback.

#### 3.2 SIP / FreeSWITCH Path (Optional)

```
SIP Client / PSTN
   → FreeSWITCH
      → sip_app.py / HTTP Bridge
         → TTS Engine
         → Audio Streaming
```

This path enables SIP-based calling for testing, on-premise setups, or non-AWS deployments.

---

### 4. Primary Entry Points

#### 🔴 Local Web Interface (Recommended)

* **File:** `local_app.py`
* **Purpose:**

  * Test outbound calls
  * Test TTS engines
  * Test chatbot behavior

#### 🔴 SIP Application

* **File:** `sip_app.py`
* **Purpose:**

  * Entry point for FreeSWITCH-based SIP calls

#### 🟠 Core Backend Services

* **Directory:** `server_components/`
* **Purpose:**

  * Chatbot logic
  * Call orchestration
  * TTS abstraction and caching

#### 🟢 AWS Runtime Logic

* **Directory:** `lambda_functions/`, `flows/`, `backup/`
* **Purpose:**

  * Amazon Connect flows
  * AWS Lambda handlers
  * Backup and recovery artifacts

---

### 5. Repository Structure

```
.
├── local_app.py                 # Local UI (IMPORTANT)
├── sip_app.py                   # SIP entrypoint (IMPORTANT)
│
├── server_components/           # Core backend services
│   ├── app.py
│   ├── phone_call/
│   ├── chatbot/
│   ├── tts/
│   ├── bridge_api/
│   └── environments/
│
├── lambda_functions/            # Active AWS Lambda code
│   ├── InvokeBotLambda.py
│   └── kvs_Trigger/
│
├── flows/                       # Amazon Connect flows
│
├── backup/                      # AWS backups & snapshots
│
├── documentations/              # Human-readable docs
│   ├── FreeSWITCH/
│   └── AI_PhoneCallSystem_Guide.*
│
├── codes_and_scripts/           # Utilities & experiments
│   ├── sip_app.py
│   ├── backup_scripts/
│   └── archive/
│
├── amazon-connect-realtime-transcription-master/
│                               # AWS reference implementation
│
├── files/                       # Audio prompts & reference files
├── ddb/                         # DynamoDB exports (reference)
├── reports/                     # Progress reports & diagrams
│
├── README.md
├── README_ko.md
└── requirements.txt
```

---

### 6. TTS Engines

#### FishSpeech (Primary)

* Optimized for Korean
* Supports caching via DynamoDB
* Recommended for production use

#### ElevenLabs (Optional)

* Used for multilingual or demo purposes

#### OpenVoice (Archived)

* Kept for reference only
* Not required for production

---

### 7. Local Development

```bash
pip install -r requirements.txt
python local_app.py
```

Default UI:

```
http://localhost:5051
```

---

### 8. Security & Data Policy

* **No datasets** are included (e.g., KSponSpeech)
* **No trained model weights** are included
* **No secrets or credentials** are committed
* Secrets must be provided via local `.envrc` or external secret storage

---

### 9. Handover Checklist

* [ ] `local_app.py` runs successfully
* [ ] SIP calls work via FreeSWITCH (if used)
* [ ] Amazon Connect flows deployed
* [ ] Lambda environment variables configured
* [ ] S3 and DynamoDB permissions verified

---

### 10. Maintainer Notes

This repository prioritizes **operational clarity** over minimalism. Some redundancy exists intentionally for AWS recovery and traceability.

---

## 한국어 버전 (KOREAN VERSION)

### 1. 개요

이 저장소는 **Amazon Connect + Lex V2 (한국어) + AWS Lambda** 기반의 **한국어 음성 봇 시스템**입니다. 여러 TTS 엔진을 지원하며, **FreeSWITCH 기반 SIP 통화**도 선택적으로 사용할 수 있습니다.

본 문서는 **인수인계(Handover)** 를 목적으로 작성되었으며, 실험적 코드 및 과거 백업은 유지하되 운영 코드와 명확히 구분되어 있습니다.

---

### 2. 주요 기능

* Amazon Connect 기반 **아웃바운드 전화 발신**
* Lex V2(ko_KR)를 이용한 **한국어 음성 인식 및 의도 처리**
* 다중 TTS 엔진 지원

  * FishSpeech (주력)
  * ElevenLabs (선택)
  * GPT Voice (실험)
* **저지연 오디오 재생** (배치 및 캐싱)
* **로컬 웹 UI** 제공
* **FreeSWITCH 기반 SIP 통화 지원**

---

### 3. 아키텍처 개요

#### 3.1 Amazon Connect 경로 (주요)

```
발신자 → Amazon Connect
   → Lex V2 (ko_KR)
      → AWS Lambda
         → TTS 엔진
         → S3
         → DynamoDB
```

#### 3.2 SIP / FreeSWITCH 경로 (선택)

```
SIP / PSTN
   → FreeSWITCH
      → sip_app.py
         → TTS 엔진
```

---

### 4. 주요 실행 지점

* `local_app.py` : 로컬 테스트 UI (권장)
* `sip_app.py` : SIP 통화 엔트리포인트
* `server_components/` : 핵심 백엔드 로직

---

### 5. 디렉터리 구조

(영문 구조와 동일하며, 중요도 기준으로 분류됨)

* 🔴 필수: `local_app.py`, `sip_app.py`, `server_components/`, `lambda_functions/`, `flows/`
* 🟠 참고: `documentations/`, `files/`, `reports/`
* ⚪ 보관용: `codes_and_scripts/archive/`, OpenVoice 관련 코드

---

### 6. 보안 및 데이터 정책

* 데이터셋은 포함되지 않음
* 학습된 모델 가중치는 포함되지 않음
* API 키 및 인증 정보는 Git에 포함되지 않음

---

### 7. 인수인계 체크리스트

* [ ] 로컬 UI 실행 확인
* [ ] SIP 통화 동작 확인 (사용 시)
* [ ] Amazon Connect 설정 확인
* [ ] Lambda 환경 변수 설정

---

### 8. 유지보수 참고사항

본 저장소는 실제 운영 시스템을 기반으로 하며, 일부 중복 또는 기록용 디렉터리가 의도적으로 포함되어 있습니다.

---

**문의:** 본 시스템은 AWS 기반 한국어 음성 자동화 프로젝트의 일부로 개발되었습니다.
