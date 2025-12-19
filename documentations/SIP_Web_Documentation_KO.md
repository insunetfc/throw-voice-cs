# SIP 웹 전화 & 내선 관리 시스템
*(FreeSWITCH + WebRTC)*

## 1. 개요

본 시스템은 다음 기능을 제공합니다:

- **브라우저 기반 SIP 소프트폰** (WebRTC, JsSIP)
- **FreeSWITCH 내선 관리용 웹 관리자 페이지**
- **정적 FreeSWITCH XML 디렉토리** 기반 내선 관리
- Python 백엔드는 SIP 처리 없음:
  - HTML 페이지 제공
  - XML 파일 생성/수정
  - `fs_cli -x reloadxml` 실행

---

## 2. 접속 주소

| 기능 | URL |
|----|-----|
| SIP 소프트폰 | `https://<domain>/sip` |
| 내선 관리자 | `https://<domain>/sip/admin` |

소프트폰 페이지 우측 상단에 ⚙️ **톱니바퀴 아이콘**이 있으며, 내선 관리자 페이지로 이동할 수 있습니다.

---

## 3. SIP 소프트폰 페이지

### 주요 기능
- **WSS(WebSocket Secure)** 기반 SIP 등록
- 내선 간 통화 송수신
- 다이얼 패드 및 **백스페이스(⌫)** 버튼
- 라이트 / 다크 모드 전환
- FreeSWITCH `mod_sofia` 연동

### 스크린샷 ① – SIP 소프트폰
![SIP Softphone](images/SIP_softphone.png)

---

## 4. 내선 관리자 페이지

### 목적
다음 경로에 위치한 **정적 XML 내선 파일**을 관리합니다:

```
/usr/local/freeswitch/conf/vanilla/directory/default/
```

각 내선은 아래 파일에 해당합니다:

```
<내선번호>.xml
```

### 제공 기능
- 내선 생성
- 비밀번호 설정 / 재설정
- 발신자 이름 / 번호 설정
- 내선 삭제
- 모든 변경 후 자동 `reloadxml` 실행

### 스크린샷 ② – 내선 관리자
![Extension Manager](images/Extension_manager.png)

---

## 5. 관리자 인증 (중요)

### 인증 방식

내선 관리자 페이지는 **HTTP Basic 인증**으로 보호됩니다.

아래 환경 변수가 설정된 경우에만 인증이 활성화됩니다:

```bash
SIP_ADMIN_PASS=<비밀번호>
```

선택 사항:
```bash
SIP_ADMIN_USER=admin   # 기본값: admin
```

### 동작 방식

| 조건 | 결과 |
|----|------|
| `SIP_ADMIN_PASS` 미설정 | 관리자 페이지 접근 제한 없음 (개발용) |
| `SIP_ADMIN_PASS` 설정 | 브라우저 로그인 창 표시 |

### 로그인 정보
- **사용자명:** `admin` (또는 `SIP_ADMIN_USER`)
- **비밀번호:** `SIP_ADMIN_PASS` 값 (!insunet-fc)

---

## 6. 비밀번호 처리 방식

- 비밀번호는 **FreeSWITCH XML 파일**에 저장됩니다.
- 내선 생성 시:
  - 비밀번호 미입력 → 자동 생성
- 비밀번호 재설정 시:
  - 직접 입력 가능
  - 비워두면 자동 생성
- 자동 생성 비밀번호는 **1회만 표시**됩니다.

---

## 7. 통화 흐름 요약

```
브라우저 (JsSIP)
   |
   |  SIP over WSS
   |
FreeSWITCH (mod_sofia)
   |
   |  RTP / SRTP
   |
브라우저 / SIP 단말
```

---

## 8. 운영 참고 사항

- 통화 중 **단방향 음성** 문제가 발생할 경우:
  - 페이지 새로고침
  - SIP 재등록
  - 브라우저 마이크 권한 확인
- 내선 변경 작업 시:
  - XML 디렉토리 쓰기 권한 필요
  - `reloadxml` 정상 수행 필수

---

## 9. 재시작 명령어

코드 수정 후:

```bash
python3 -m py_compile sip_app.py
sudo systemctl restart sipapp
```

FreeSWITCH 설정 반영:

```bash
fs_cli -x reloadxml
```
