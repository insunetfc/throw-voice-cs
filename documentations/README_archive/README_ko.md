# 🇰🇷 한국어 음성 봇 시스템

**Amazon Connect + Lex V2 (ko_KR) + Lambda** 통합과 다중 TTS 백엔드, 그리고 테스트 및 상호작용을 위한 통합 웹 인터페이스를 갖춘 종합 한국어 음성 봇 시스템입니다.

> **리전/로케일:** 모든 AWS 리소스(Connect, Lex, Lambda, S3)는 **ap-northeast-2 (서울)**에 배포됩니다. Lex 로케일은 **`ko_KR`**입니다.

---

## ✅ 시스템 개요

이 프로젝트는 다음을 제공합니다:

1. **AWS 음성 봇 인프라** - Amazon Connect와 Lex V2를 사용한 프로덕션급 음성 봇
2. **이중 TTS 시스템** - OpenVoice(실시간) 및 FishSpeech(캐시) 백엔드
3. **로컬 웹 인터페이스** - 통합 테스트 및 상호작용 UI (`local_app.py`)
4. **챗봇 통합** - 다중 챗봇 백엔드 (차집사, GPT)
5. **전화 발신 시스템** - 다양한 TTS 엔진을 사용한 프로그래밍 방식의 아웃바운드 통화

---

## 🚀 빠른 시작 - 로컬 웹 인터페이스

`local_app.py`는 모든 시스템 구성 요소를 테스트할 수 있는 통합 웹 인터페이스를 제공합니다.

### 사전 요구사항

```bash
pip install fastapi uvicorn httpx
```

### 환경 설정

실행 전에 다음 환경 변수를 설정하세요:

```bash
# NIPA 서비스를 위한 기본 엔드포인트
export NIPA_BASE="https://your-ngrok-url.ngrok-free.app"

# 인증 토큰
export NIPA_AUTH="Bearer YOUR_TOKEN"

# 선택사항: 특정 엔드포인트 재정의
export ELEVEN_BASE="${NIPA_BASE}/voice/eleven"
export ELEVEN_BRAIN_BASE="${NIPA_BASE}/voice/brain/gpt-voice-eleven"
export GPT_VOICE_BASE="${NIPA_BASE}/voice/brain/gpt-voice/start"
```

### 서버 실행

```bash
python local_app.py
```

서버가 `http://0.0.0.0:5051`에서 시작됩니다.

### 웹 인터페이스 기능

#### 1. **전화 걸기 탭**
다양한 TTS 엔진을 사용하여 아웃바운드 통화를 합니다:

- **FishSpeech로 통화** - 고품질 한국어 음성
- **ElevenLabs로 통화** - 자연스러운 영어/다국어 음성
- **GPT로 통화** - AI 기반 대화형 음성

**기능:**
- 전화번호 입력 (국제 형식)
- 선택적 표시 이름
- 사용자 정의 인트로 메시지 생성
- 실시간 통화 상태 업데이트

#### 2. **TTS 생성 탭**
다양한 TTS 엔진을 사용하여 음성을 생성합니다:

- **FishSpeech** - 참조 오디오 지원이 있는 고품질 한국어 TTS
  - 참조 오디오 업로드 (WAV 형식)
  - 온도 매개변수 조정 (0.0-1.0)
  - 생성된 오디오 다운로드
  
- **ElevenLabs** - 스타일 제어가 가능한 실시간 TTS
  - 다양한 음성 옵션
  - 스타일 프리셋 (전문적, 캐주얼, 흥분 등)
  - 빠른 생성을 위한 캐시 지원
  
- **GPT Voice** - AI 기반 음성 합성
  - 먼저 GPT를 통해 텍스트 처리
  - 자연스러운 대화 톤
  - 온도 제어

#### 3. **챗봇 탭**
다양한 백엔드를 사용한 대화형 채팅 인터페이스:

- **차집사 엔진** - 전문 한국어 챗봇
- **GPT 엔진** - OpenAI GPT 기반 응답

**기능:**
- 실시간 메시징
- 엔진 표시 배지
- 메시지 기록
- 다크/라이트 테마 지원

---

## 🏗️ AWS 인프라 아키텍처

### 고수준 흐름

```
발신자 → Amazon Connect (ko_KR)
   └─► 고객 입력 받기 (Lex V2, intent = CaptureUtterance)
        └─► 대화 코드 훅 (Lambda) - 첫 발화 캡처
              └─► 이행 코드 훅 (Lambda)
                   ├─ 캐시 조회 (DynamoDB)
                   ├─ TTS 생성 (필요시)
                   └─ 발신자에게 오디오 스트리밍

Connect 루프:
  ┌─► Lambda: get_next_batch(JobId, NextIndex)
  │     └─ 반환 { PromptARN/AudioS3Url, NextIndex, HasMore }
  ├─► 오디오 청크 재생
  └─► HasMore=true이면 → 반복
```

### 주요 구성 요소

1. **Amazon Connect** - 통화를 처리하는 음성 플랫폼
2. **Lex V2** - 자연어 이해 (ko_KR 로케일)
3. **Lambda 함수** - 비즈니스 로직 및 오케스트레이션
4. **S3** - 오디오 파일 저장소
5. **DynamoDB** - TTS 캐시 및 메타데이터
6. **TTS 서버** - OpenVoice/FishSpeech 백엔드

---

## 🎯 TTS 시스템 선택

### OpenVoice (실시간)

**최적 사용 사례:**
- ✅ 동적 콘텐츠 생성
- ✅ 빠른 프로토타이핑
- ✅ 낮은 인프라 복잡성
- ✅ 즉각적인 합성 필요

**설정:**
```bash
# TTS 앱 (OpenVoice)
STREAM_MAX_CHARS=100
FIRST_URL_WAIT_MS=1400
OV_SPEED=1.5
OV_TGT_SE_PTH=/path/to/speaker.pth
CHUNK_MAX_CHARS=140
```

**시작:**
```bash
uvicorn app_openvoice:app --host 0.0.0.0 --port 8000
```

### FishSpeech (캐시)

**최적 사용 사례:**
- ✅ 프로덕션 배포
- ✅ 반복적인 FAQ 응답
- ✅ 최적화된 지연 시간
- ✅ 더 높은 오디오 품질

**설정:**
```bash
# Lambda (FishSpeech 모드)
CACHE_TABLE=UtteranceCache
BATCH_APPROVAL_MODE=true
NORMALIZE_KOREAN=true
```

**DynamoDB 테이블:**
- `UtteranceCache` - 정규화된 발화 → S3 오디오 URL
- `SttInboxNew` - 승인 대기 중인 발화

---

## 📦 리포지토리 구조

```
├── local_app.py                    # 🔴 통합 웹 인터페이스
├── server_components/              # 주요 서버 구성 요소
│   ├── app.py                     # 핵심 TTS 서버
│   ├── api_test/                  # API 테스트 스크립트
│   │   ├── call_phone.py         # 통화 테스트
│   │   ├── chatbot_response.py   # 챗봇 테스트
│   │   └── tts_response.py       # TTS 테스트
│   ├── bridge_api/                # 다중 TTS 제공자를 위한 브리지 API
│   │   ├── app.py                # 브리지 API 서버
│   │   ├── adapters/             # TTS 제공자 어댑터
│   │   │   ├── elevenlabs.py    # ElevenLabs 어댑터
│   │   │   ├── gpt_realtime.py  # GPT Realtime 어댑터
│   │   │   └── minimax.py       # Minimax 어댑터
│   │   └── utils/                # 공유 유틸리티
│   │       ├── audio.py         # 오디오 처리
│   │       ├── logger.py        # 로깅 유틸리티
│   │       └── s3.py            # S3 작업
│   ├── chatbot/                  # 챗봇 시스템
│   │   ├── app.py               # 챗봇 서버
│   │   ├── data/                # 학습 데이터 및 데이터셋
│   │   │   ├── intent_dataset.csv
│   │   │   ├── response_bank_large.csv
│   │   │   └── UtteranceCacheDataset.csv
│   │   ├── intent_training/     # 의도 분류 학습
│   │   │   ├── train.py        # 학습 스크립트
│   │   │   └── test.py         # 테스트 스크립트
│   │   └── models/              # 학습된 모델
│   │       ├── intent_classification_model_promotion.pth
│   │       └── retriever_multitask_pairs.pt
│   ├── phone_call/              # 전화 통화 오케스트레이션
│   │   └── app.py              # 통화 관리
│   ├── tts/                     # TTS 유틸리티 및 스크립트
│   │   ├── app.py              # 메인 TTS 앱
│   │   ├── config.yaml         # TTS 구성
│   │   ├── batch_approve_inbox.py  # 승인 워크플로우
│   │   ├── coverage_tester.py  # 캐시 커버리지 테스트
│   │   └── generate_fillers_tts.py  # 필러 생성
│   ├── environments/            # 환경 설정
│   │   ├── run_server.sh       # 서버 시작 스크립트
│   │   └── envrc               # 환경 변수
│   ├── NIPA_cloud/             # NIPA 클라우드 컨테이너 설정
│   └── requirements.txt         # Python 의존성
├── backup/                      # AWS 구성 백업
│   ├── aws_connect_backup/     # Connect 플로우 백업
│   │   ├── connect-backup/    # 연락 플로우
│   │   └── iam-backup/        # IAM 구성
│   ├── InvokeBotLambda.py     # Lambda 함수
│   └── TranscribeCustomerSpeech.py
├── lex_bots/                    # Lex V2 봇 정의
│   ├── bot_archive/            # 봇 내보내기 파일
│   └── build_script/           # 봇 빌드 스크립트
├── amazon-connect-realtime-transcription-master/  # 전사 통합
├── reports/                     # 일일 진행 보고서
│   ├── *.pdf                   # 일일 보고서
│   └── Flow_Diagram.png        # 시스템 흐름 다이어그램
├── README_archive/              # 보관된 문서
├── requirements.txt             # 최상위 의존성
├── README.md                   # 영문 문서
└── README_ko.md                # 이 파일
```

---

## 🔌 API 엔드포인트 참조

### 로컬 웹 앱 (`local_app.py`)

#### TTS 엔드포인트

**FishSpeech TTS**
```http
POST /api/tts
Content-Type: multipart/form-data

text: string
temperature: float (기본값: 0.8)
ref_audio: file (선택사항)
```

**ElevenLabs TTS**
```http
POST /api/tts-eleven
Content-Type: multipart/form-data

text: string
style: string (선택사항)
voice_id: string (선택사항)
use_cache: boolean (기본값: true)
```

**GPT Voice TTS**
```http
POST /api/tts-gpt-voice
Content-Type: multipart/form-data

text: string
temperature: float (기본값: 0.6)
```

#### 전화 통화 엔드포인트

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
display_name: string (선택사항)
```

#### 챗봇 엔드포인트

```http
POST /api/chat-chajipsa
POST /api/chat-gpt
Content-Type: multipart/form-data

message: string
```

---

## 🛠️ 개발 환경 설정

### 1. 리포지토리 클론

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 환경 구성

```bash
# 예제 환경 파일 복사
cp .env.example .env

# 자격 증명으로 편집
nano .env
```

### 4. 서비스 시작

**옵션 A: 로컬 웹 인터페이스만**
```bash
python local_app.py
```

**옵션 B: 전체 스택 (TTS 서버 + 웹 인터페이스)**
```bash
# 터미널 1: TTS 서버 시작
cd server_components
source environments/run_server.sh

# 터미널 2: 웹 인터페이스 시작
python local_app.py
```

**옵션 C: Ngrok 터널링 포함**
```bash
# 터미널 1: TTS 서버
cd /home/work/VALL-E/
source uvicorn.sh

# 터미널 2: Ngrok
cd /home/work/VALL-E/
source ngrok.sh

# 터미널 3: 웹 인터페이스
python local_app.py
```

---

## 🧪 테스트

### 전화 통화 테스트

```bash
# FishSpeech로 통화
curl -X POST http://localhost:5051/api/call_fish \
  -F "phone_number=+821012345678"

# GPT Voice로 통화
curl -X POST http://localhost:5051/api/call_gpt \
  -F "phone_number=+821012345678"

# ElevenLabs로 통화
curl -X POST http://localhost:5051/api/call_eleven \
  -F "phone_number=+821012345678"
```

### TTS 생성 테스트

```bash
# FishSpeech 테스트
curl -X POST http://localhost:5051/api/tts \
  -F "text=안녕하세요" \
  -F "temperature=0.8"

# ElevenLabs 테스트
curl -X POST http://localhost:5051/api/tts-eleven \
  -F "text=Hello world" \
  -F "style=professional"

# GPT Voice 테스트
curl -X POST http://localhost:5051/api/tts-gpt-voice \
  -F "text=Tell me about AI"
```

### 챗봇 테스트

```bash
# 차집사 챗봇
curl -X POST http://localhost:5051/api/chat-chajipsa \
  -F "message=안녕하세요"

# GPT 챗봇
curl -X POST http://localhost:5051/api/chat-gpt \
  -F "message=Hello, how are you?"
```

---

## 🔧 구성 가이드

### 필수 환경 변수

```bash
# 핵심 구성
NIPA_BASE="https://your-backend.ngrok-free.app"
NIPA_AUTH="Bearer YOUR_SECRET_TOKEN"

# 선택적 재정의
ELEVEN_BASE="${NIPA_BASE}/voice/eleven"
ELEVEN_BRAIN_BASE="${NIPA_BASE}/voice/brain/gpt-voice-eleven"
GPT_VOICE_BASE="${NIPA_BASE}/voice/brain/gpt-voice/start"

# AWS 구성 (프로덕션 배포용)
COMPANY_BUCKET="tts-bucket-250810"
COMPANY_REGION="ap-northeast-2"
CONNECT_INSTANCE_ID="eefed165-54dc-428e-a0f1-02c2ec35a22e"
CACHE_TABLE="UtteranceCache"

# TTS 구성
STREAM_BATCH=1
LOG_LEVEL="INFO"
TTS_MODE="openvoice"  # 또는 "fishspeech"
USE_FILLERS=true
```

### Lambda 환경 변수

```bash
# 공통 설정
COMPANY_BUCKET=tts-bucket-250810
COMPANY_REGION=ap-northeast-2
CONNECT_REGION=ap-northeast-2
STREAM_BATCH=1

# OpenVoice 모드
TTS_URL=http://openvoice-server:8000/synthesize
USE_FILLERS=true

# FishSpeech 모드
CACHE_TABLE=UtteranceCache
BATCH_APPROVAL_MODE=true
TTS_URL=https://fishspeech.ngrok-free.app/synthesize
```

---

## 🚨 문제 해결

### 로컬 앱 문제

**문제: 서버가 시작되지 않음**
```bash
# 포트 5051이 사용 중인지 확인
lsof -i :5051

# 기존 프로세스 종료
kill -9 <PID>
```

**문제: 연결 거부 오류**
```bash
# NIPA_BASE가 접근 가능한지 확인
curl ${NIPA_BASE}/healthz

# 인증 확인
curl -H "Authorization: ${NIPA_AUTH}" ${NIPA_BASE}/api/status
```

**문제: TTS 생성 실패**
- ngrok 터널이 활성화되어 있는지 확인
- TTS 서버 로그 확인
- 참조 오디오가 올바른 형식(WAV, 16kHz)인지 확인

### AWS 인프라 문제

**문제: 이중 오디오 재생**
- Lambda 이행이 `messages: []`를 반환하는지 확인
- Connect 플로우가 캐시에서 올바르게 재생되는지 확인

**문제: 높은 지연 시간**
- Lambda에서 `STREAM_BATCH=1` 확인
- S3 버킷 리전이 Connect 리전과 일치하는지 확인
- Lambda에 프로비저닝된 동시성 활성화

**문제: 캐시 미스**
- 배치 승인 워크플로우 실행
- DynamoDB `UtteranceCache` 테이블 확인
- 해시 생성 일관성 확인

---

## 📊 성능 지표

### TTS 시스템 비교

| 지표 | FishSpeech | ElevenLabs | GPT Voice |
|------|------------|------------|-----------|
| **첫 응답** | ~0.5초 (캐시) | ~3-4초 | ~1-2초 |
| **오디오 품질** | 우수 | 우수 | 좋음 |
| **한국어 지원** | 우수 | 제한적 | 보통 |
| **캐시 지원** | 있음 | 있음 | 없음 |
| **비용** | 낮음 | 높음 | 중간 |

### 시스템 성능

- **통화 설정 시간:** < 2초
- **TTS 생성 (캐시):** < 500ms
- **TTS 생성 (미캐시):** 2-4초
- **캐시 적중률:** ~85-95% (FAQ 시스템)

---

## 🔐 보안 고려사항

### 로컬 개발

1. **절대 자격 증명을 커밋하지 마세요** 버전 관리에
2. 로컬 구성에 `.env` 파일 사용
3. `NIPA_AUTH` 토큰을 정기적으로 교체
4. 모든 프로덕션 엔드포인트에 HTTPS 사용

### 프로덕션 배포

1. 민감한 데이터에 **AWS Secrets Manager 사용**
2. 감사 추적을 위한 **CloudWatch 로깅 활성화**
3. API 엔드포인트에 **속도 제한 구현**
4. S3/DynamoDB 액세스를 위한 **VPC 엔드포인트 사용**
5. **S3 버킷 암호화 활성화**
6. 필요한 최소한으로 **IAM 권한 제한**

---

## 📚 추가 리소스

### 문서

- [README.md](README.md) - 영문 설정 가이드
- `/reports` - 일일 개발 진행 보고서

### 테스트 스크립트

- `api_test/call_phone.py` - 전화 통화 테스트
- `api_test/chatbot_response.py` - 챗봇 테스트
- `api_test/tts_response.py` - TTS 엔드포인트 테스트

### 유틸리티

- `tts/batch_approve_inbox.py` - 대기 중인 TTS 요청 승인
- `tts/coverage_tester.py` - 캐시 커버리지 테스트
- `tts/ddb_setup_and_seed.py` - DynamoDB 테이블 초기화

---

## 🤝 기여하기

### 이슈 보고

이슈를 보고할 때 다음을 포함해주세요:
1. 재현 단계
2. 예상 동작 vs 실제 동작
3. 관련 로그 (자격 증명은 제거!)
4. 환경 세부사항 (OS, Python 버전 등)

### 개발 워크플로우

1. `main`에서 기능 브랜치 생성
2. 변경 사항 작성 및 로컬 테스트
3. 필요시 문서 업데이트
4. 설명과 함께 풀 리퀘스트 제출

---

## 📝 변경 로그

- **2025-11-17**: `local_app.py`에 대한 종합 문서 추가
- **2025-08-27**: OpenVoice 및 FishSpeech 시스템 통합
- **2025-08-27**: 첫 발화 캡처 및 스트리밍 재생 추가
- **2025-08-27**: 카테고리 기반 필러 및 TTS 워밍업

---

## 🎓 학습 리소스

### 웹 인터페이스 사용법

1. **TTS 생성 탭 사용하기**
   - 엔진 선택 (FishSpeech, ElevenLabs, GPT Voice)
   - 텍스트 입력
   - 필요시 참조 오디오 업로드
   - "생성" 버튼 클릭
   - 재생 또는 다운로드

2. **전화 걸기 탭 사용하기**
   - 전화번호 입력 (예: +821012345678)
   - TTS 엔진 선택
   - "📞 Call" 버튼 클릭
   - 상태 메시지 확인

3. **챗봇 탭 사용하기**
   - 엔진 선택 (차집사 또는 GPT)
   - 메시지 입력
   - Enter 키 또는 "전송" 버튼
   - 대화 기록 확인

### 일반적인 사용 사례

**사례 1: 한국어 TTS 테스트**
```
1. TTS 생성 탭 열기
2. FishSpeech 엔진 선택
3. 텍스트 입력: "안녕하세요, 반갑습니다"
4. 생성 클릭
5. 오디오 재생
```

**사례 2: 고객에게 전화하기**
```
1. 전화 걸기 탭 열기
2. 전화번호 입력
3. 적절한 TTS 엔진 선택
4. Call 버튼 클릭
5. 통화 상태 모니터링
```

**사례 3: 챗봇 응답 테스트**
```
1. 챗봇 탭 열기
2. 차집사 엔진 선택
3. 질문 입력
4. 응답 확인
5. 필요시 대화 계속
```

---

## 💡 팁과 트릭

### 성능 최적화

1. **캐시 활용**
   - 자주 사용하는 문구는 FishSpeech 캐시 사용
   - `use_cache=true` 옵션 활성화

2. **최적의 오디오 품질**
   - 16kHz 이상의 참조 오디오 사용
   - 배경 소음이 없는 깨끗한 녹음

3. **빠른 응답**
   - 짧은 텍스트에는 OpenVoice 사용
   - 긴 텍스트는 청크로 분할

### 문제 예방

1. **정기적인 토큰 교체**
   ```bash
   # 매달 새 토큰 생성
   export NIPA_AUTH="Bearer NEW_TOKEN"
   ```

2. **로그 모니터링**
   ```bash
   # 실시간 로그 확인
   tail -f logs/app.log
   ```

3. **백업 유지**
   ```bash
   # 구성 백업
   cp .env .env.backup
   ```

---

## 🌟 고급 기능

### 사용자 정의 음성 프로필

FishSpeech를 사용하여 사용자 정의 음성 프로필을 만들 수 있습니다:

```python
# 참조 오디오로 음성 프로필 생성
import requests

files = {'ref_audio': open('my_voice.wav', 'rb')}
data = {'text': '테스트 문장입니다'}

response = requests.post(
    'http://localhost:5051/api/tts',
    files=files,
    data=data
)
```

### 배치 TTS 생성

여러 문장을 한 번에 생성:

```python
import asyncio
import httpx

async def generate_batch(texts):
    async with httpx.AsyncClient() as client:
        tasks = [
            client.post(
                'http://localhost:5051/api/tts',
                data={'text': text}
            )
            for text in texts
        ]
        return await asyncio.gather(*tasks)

# 사용 예
texts = [
    "첫 번째 문장입니다",
    "두 번째 문장입니다",
    "세 번째 문장입니다"
]

results = asyncio.run(generate_batch(texts))
```

### 웹훅 통합

통화 이벤트에 대한 웹훅 설정:

```python
@app.post("/webhook/call-status")
async def call_status_webhook(data: dict):
    # 통화 상태 처리
    if data['status'] == 'completed':
        # 통화 완료 로직
        pass
    return {"status": "ok"}
```

---

## 🔍 디버깅 가이드

### 일반적인 오류 메시지

**"Connection refused"**
```bash
# 서비스가 실행 중인지 확인
ps aux | grep python

# 포트 확인
netstat -an | grep 5051
```

**"Authentication failed"**
```bash
# 토큰 확인
echo $NIPA_AUTH

# 토큰 유효성 테스트
curl -H "Authorization: $NIPA_AUTH" $NIPA_BASE/healthz
```

**"TTS generation timeout"**
```bash
# 서버 로그 확인
tail -f logs/tts_server.log

# 네트워크 지연 확인
ping your-server.com
```

### 로그 분석

```bash
# 오류만 필터링
grep ERROR logs/app.log

# 특정 시간대 로그
grep "2025-11-17" logs/app.log

# 통계 생성
grep "TTS generation" logs/app.log | wc -l
```