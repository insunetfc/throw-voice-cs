# 🇰🇷 Korean Voice Bot System (Handover Repository)

---

## 한국어 버전

### 1. 개요

본 저장소는 **Amazon Connect + Lex V2 (ko_KR) + AWS Lambda**를 중심으로 구축된 **프로덕션 수준의 한국어 음성 봇 시스템**을 포함하고 있으며, **다중 TTS 백엔드**를 지원하고 **FreeSWITCH 기반 SIP 통화 경로(선택 사항)**를 제공합니다.

이 저장소는 **인수인계 및 장기 유지보수**를 목적으로 특별히 정리되었습니다. 과거 실험, 백업, 참조용 구현물은 보존되어 있으나, 실제 운영에 사용되는 구성요소와는 명확히 분리되어 있습니다.

---

### 2. 시스템 기능

* Amazon Connect를 통한 **자동 아웃바운드 전화 발신**
* Lex V2 (ko_KR)를 활용한 한국어 **음성 인식 및 인텐트 처리**
* 다중 **텍스트-투-스피치(TTS)** 엔진 지원
  * FishSpeech (기본)
  * ElevenLabs (선택)
  * GPT Voice (실험적)
* 배치 처리 및 캐싱을 활용한 **저지연 오디오 재생**
* 테스트 및 오케스트레이션을 위한 **로컬 웹 UI**
* **SIP / FreeSWITCH 연동**을 통한 비(非) AWS 텔레포니 환경 지원

별도 명시가 없는 한, 모든 AWS 리소스는 **ap-northeast-2 (서울)** 리전에 배포됩니다.

---

### 3. 상위 수준 아키텍처

#### 3.1 Amazon Connect 경로

```
Caller → Amazon Connect
   → Lex V2 (ko_KR)
      → AWS Lambda
         → TTS 엔진
         → S3 (오디오 저장소)
         → DynamoDB (발화 캐시)
```

본 경로는 메인 프로덕션 경로입니다. 음성 응답은 배치 단위로 생성되며, 부드러운 재생을 위해 스트리밍 방식으로 전달됩니다.

#### 3.2 SIP / FreeSWITCH 경로

```
SIP Client / PSTN
   → FreeSWITCH
      → sip_app.py / HTTP 브리지
         → TTS 엔진
         → 오디오 스트리밍
```

이 경로는 테스트, 온프레미스 환경, 또는 AWS 외부 환경에서의 SIP 기반 통화를 가능하게 합니다.

---

### 4. 주요 진입 지점

#### 🔴 로컬 웹 인터페이스

* **파일:** `local_app.py`
* **목적:**
  * 아웃바운드 전화 테스트
  * TTS 엔진 테스트
  * 챗봇 동작 테스트

#### 🔴 SIP 애플리케이션

* **파일:** `sip_app.py`
* **목적:**
  * FreeSWITCH 기반 SIP 통화의 진입 지점

#### 🟠 핵심 백엔드 서비스

* **디렉토리:** `server_components/`
* **목적:**
  * 챗봇 로직
  * 콜 오케스트레이션
  * TTS 추상화 및 캐싱

#### 🟢 AWS 런타임 로직

* **디렉토리:**
  * `flows/` - Amazon Connect 플로우
  * `lambda_functions/` - AWS Lambda 핸들러
  * `lex_bots/` - Lex Bot JSON 설정
  * `backup/` - 백업 및 복구 아티팩트

#### 🟢 NIPA 서버 브리지

* **디렉토리: (`flows/`)**
  * `api_test/` - API 테스트 스크립트
  * `bridge_api/` - ElevenLabs, GPTVoice, MiniMax 연동 브리지
  * `chatbot/` - `ddb/`의 발화 및 응답을 기반으로 한 자체 개발 챗봇 (학습 데이터 및 스크립트 포함)
  * `fishspeech_tts/` - FishSpeech TTS 앱 호스팅을 위한 백업 코드
  * `phone_call/` - Amazon Connect용 전화 통화 앱
  * `run_server.sh` - NIPA 클라우드에서 서버 실행 스크립트

---

### 5. 저장소 구조

```
.
├── local_app.py                 # 로컬 UI (중요)
├── sip_app.py                   # SIP 진입점 (중요)
│
├── server_components/           # 핵심 백엔드 서비스
│   ├── app.py
│   ├── phone_call/
│   ├── chatbot/
│   ├── tts/
│   └── bridge_api/
│
├── lambda_functions/            # 실제 사용 중인 AWS Lambda 코드
│   ├── InvokeBotLambda.py
│   └── kvs_Trigger/
│
├── flows/                       # Amazon Connect 플로우
│
├── backup/                      # AWS 백업 및 스냅샷
│
├── documentations/              # 사람이 읽을 수 있는 문서
│   ├── FreeSWITCH/              # FreeSWITCH 설치 문서 및 스크립트
│   └── AI_PhoneCallSystem_Guide.*
│
├── codes_and_scripts/           # 유틸리티 및 실험 코드
│   ├── sip_app.py
│   ├── backup_scripts/
│   └── archive/
│
├── amazon-connect-realtime-transcription-master/
│   # Amazon Kinesis Video Stream (KVS)
│   # Java 및 Node.js 런타임 초기화 코드 업데이트됨
│   # AWS 공식 참조 구현:
│   # https://github.com/amazon-connect/amazon-connect-realtime-transcription
│
├── files/                       # 오디오 프롬프트, 참조 파일, CSV 파일 등
├── ddb/                         # DynamoDB exports (발화 및 응답 테이블 참조)
├── reports/                     # 진행 보고서 및 다이어그램
│
├── README.md
└── requirements.txt
```

---

### 6. TTS 엔진

#### FishSpeech (기본)

* 한국어에 최적화됨
* DynamoDB 기반 캐싱 지원

#### ElevenLabs (선택)

* 다국어 또는 데모 목적에 사용

#### OpenVoice (아카이브)

* 참조용으로만 유지
* 프로덕션 환경에서는 필요하지 않음

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

#### 3.1 Amazon Connect Path

```
Caller → Amazon Connect
   → Lex V2 (ko_KR)
      → AWS Lambda
         → TTS Engine
         → S3 (Audio Storage)
         → DynamoDB (Utterance Cache)
```

This is the main production path. Audio responses are generated in batches and streamed for smooth playback.

#### 3.2 SIP / FreeSWITCH Path

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

#### 🔴 Local Web Interface

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

* **Directory:**
  * `flows/` - Amazon Connect flows
  * `lambda_functions/` - AWS Lambda handlers
  * `lex_bots/` - Lex Bots JSON configuration
  * `backup/` - Backup and recovery artifacts

#### 🟢 NIPA Server Bridge

* **Directory: (`flows/`)**
  * `api_test/` - API testing scripts
  * `bridge_api/` - Bridge for ElevenLabs, GPTVoice, MiniMax
  * `chatbot/` - In-house built chatbot from `ddb/` utterances and response. Contains training data and scripts
  * `fishspeech_tts/` - Backup codes for hosting FishSpeech TTS app
  * `phone_call/` - Phone call app for Amazon Connect
  * `run_server.sh` - Script to run server on NIPA cloud

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
│   └── bridge_api/
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
│   ├── FreeSWITCH/              # FreeSWITCH installation documents and scripts
│   └── AI_PhoneCallSystem_Guide.*
│
├── codes_and_scripts/           # Utilities & experiments
│   ├── sip_app.py
│   ├── backup_scripts/
│   └── archive/
│
├── amazon-connect-realtime-transcription-master/
│   # Amazon Kinesis Video Stream (KVS). Updated Java and Node.js runtime initialization. AWS reference implementation:
│   # https://github.com/amazon-connect/amazon-connect-realtime-transcription
│
├── files/                       # Audio prompts & reference files, CSV files, etc.
├── ddb/                         # DynamoDB exports (reference utterance and response table)
├── reports/                     # Progress reports & diagrams
│
├── README.md
└── requirements.txt
```

---

### 6. TTS Engines

#### FishSpeech (Primary)

* Optimized for Korean
* Supports caching via DynamoDB

#### ElevenLabs (Optional)

* Used for multilingual or demo purposes

#### OpenVoice (Archived)

* Kept for reference only
* Not required for production