#!/usr/bin/env python3
"""
Minimal SIP WebPhone server using FastAPI.

- Serves a single HTML page at "/"
- Page uses JsSIP in the browser to:
  - register a SIP extension (over WebSocket)
  - call another extension
  - hang up

Backend does NOT handle any SIP itself; it only serves the static HTML.
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

FAVICON_URL = "https://static.thenounproject.com/png/microphone-icon-1681031-512.png"
_cached_icon: bytes | None = None

app = FastAPI(title="Simple SIP WebPhone")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # relax as needed
    allow_methods=["*"],
    allow_headers=["*"],
)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>간단 SIP 웹폰 / Simple SIP Web Phone</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <!-- JsSIP from CDN -->
  <script type="text/javascript" src="https://jssip.net/download/releases/jssip-3.10.0.min.js"></script>
  <style>
    :root {
      color-scheme: light dark;

      /* default = dark */
      --bg: #0f172a;
      --border: #444444;
      --text: #e2e8f0;

      --input-bg: #020617;
      --input-border: #1f2937;
      --input-text: #e2e8f0;
      --input-placeholder: #64748b;

      --status-bg: rgba(15, 23, 42, 0.85);
      --status-border: #1f2937;

      --card-bg: #111827;
      --card-bg-hover: #1f2937;

      --note-text: #9ca3af;

      --btn-primary-bg: #38bdf8;
      --btn-primary-bg-hover: #0ea5e9;
      --btn-primary-text: #0f172a;

      --btn-call-bg: #22c55e;
      --btn-call-bg-hover: #16a34a;
      --btn-call-text: #022c22;

      --btn-hang-bg: #f97373;
      --btn-hang-bg-hover: #ef4444;
      --btn-hang-text: #450a0a;

      --divider: #1f2937;
      --topbar-bg: rgba(15,23,42,0.85);
    }

    body[data-theme="light"] {
      --bg: #f9fafb;
      --border: #e5e7eb;
      --text: #111827;

      --input-bg: #ffffff;
      --input-border: #d1d5db;
      --input-text: #111827;
      --input-placeholder: #9ca3af;

      --status-bg: #f3f4f6;
      --status-border: #e5e7eb;

      --card-bg: #f3f4f6;
      --card-bg-hover: #e5e7eb;

      --note-text: #6b7280;

      --btn-primary-bg: #0ea5e9;
      --btn-primary-bg-hover: #0284c7;
      --btn-primary-text: #f9fafb;

      --btn-call-bg: #16a34a;
      --btn-call-bg-hover: #15803d;
      --btn-call-text: #ecfdf3;

      --btn-hang-bg: #ef4444;
      --btn-hang-bg-hover: #dc2626;
      --btn-hang-text: #fef2f2;

      --divider: #e5e7eb;
      --topbar-bg: rgba(243,244,246,0.9);
    }

    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      max-width: 420px;
      margin: 20px auto;
      padding: 16px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: var(--bg);
      color: var(--text);
      transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
    }

    h2 {
      text-align: left;
      margin-top: 0;
      margin-bottom: 0.5rem;
      font-size: 1.3rem;
      display: flex;
      flex-direction: column;
      gap: 0.1rem;
    }

    h2 span.sub {
      font-size: 0.8rem;
      opacity: 0.8;
    }

    .top-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.8rem;
      padding: 0.4rem 0.6rem;
      border-radius: 999px;
      background: var(--topbar-bg);
      border: 1px solid var(--border);
      font-size: 0.8rem;
    }

    .top-bar-left {
      display: flex;
      flex-direction: column;
      gap: 0.1rem;
    }

    .top-bar-left strong {
      font-size: 0.85rem;
    }

    .top-bar-left span {
      opacity: 0.8;
    }

    .theme-toggle {
      cursor: pointer;
      border-radius: 999px;
      border: none;
      font-weight: 500;
      padding: 0.35rem 0.7rem;
      font-size: 0.8rem;
      background: var(--card-bg);
      color: var(--text);
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      transition: transform 0.08s ease, background 0.08s ease;
    }

    .theme-toggle:hover {
      transform: translateY(-1px);
      background: var(--card-bg-hover);
    }

    .row {
      margin-bottom: 0.6rem;
    }

    label {
      display: block;
      font-size: 0.8rem;
      margin-bottom: 0.1rem;
    }

    label span.en {
      opacity: 0.7;
      margin-left: 0.25rem;
    }

    input[type="text"],
    input[type="password"] {
      width: 100%;
      box-sizing: border-box;
      padding: 0.5rem 0.6rem;
      border-radius: 8px;
      border: 1px solid var(--input-border);
      background: var(--input-bg);
      color: var(--input-text);
      font-size: 0.9rem;
      outline: none;
      transition: border-color 0.12s ease, background 0.12s ease;
    }

    input::placeholder {
      color: var(--input-placeholder);
    }

    input:focus {
      border-color: #38bdf8;
    }

    button {
      cursor: pointer;
      border-radius: 999px;
      border: none;
      font-weight: 600;
      padding: 0.55rem 0.9rem;
      font-size: 0.9rem;
      transition: transform 0.08s, background 0.08s;
    }

    button:hover {
      transform: translateY(-1px);
    }

    #registerBtn {
      width: 100%;
      background: var(--btn-primary-bg);
      color: var(--btn-primary-text);
    }

    #registerBtn:hover {
      background: var(--btn-primary-bg-hover);
    }

    .status {
      margin: 0.6rem 0;
      padding: 0.55rem 0.6rem;
      border-radius: 8px;
      font-size: 0.8rem;
      background: var(--status-bg);
      border: 1px solid var(--status-border);
    }

    .status.connected {
      border-color: #22c55e;
      color: #bbf7d0;
    }

    /* Improve "Registered" visibility in LIGHT mode */
    body[data-theme="light"] .status.connected {
      color: #15803d;              /* darker green text */
      border-color: #22c55e;       /* keep the success border */
      background: #ecfdf5;         /* subtle mint background */
    }

    .status.error {
      border-color: #f97373;
      color: #fecaca;
    }

    /* Light mode: make error message readable */
    body[data-theme="light"] .status.error {
      color: #b91c1c;          /* darker red text */
      border-color: #f97373;   /* keep existing border tone */
      background: #fef2f2;     /* soft red background */
    }

    .dial-row {
      display: flex;
      gap: 0.4rem;
      align-items: center;
      margin-bottom: 0.4rem;
    }

    #dialNumber {
      flex: 1;
      padding: 0.55rem 0.6rem;
      border-radius: 8px;
      border: 1px solid var(--input-border);
      background: var(--input-bg);
      color: var(--input-text);
      font-size: 1rem;
    }

    .keypad {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0.4rem;
      margin: 0.6rem 0;
    }

    .keypad button {
      padding: 0.7rem 0;
      font-size: 1.1rem;
      background: var(--card-bg);
      color: var(--text);
    }

    .keypad button:hover {
      background: var(--card-bg-hover);
    }

    .controls {
      display: flex;
      gap: 0.6rem;
      margin-top: 0.4rem;
    }

    .controls button {
      flex: 1;
      text-align: center;
    }

    #callBtn {
      background: var(--btn-call-bg);
      color: var(--btn-call-text);
    }

    #callBtn:hover {
      background: var(--btn-call-bg-hover);
    }

    #hangupBtn {
      background: var(--btn-hang-bg);
      color: var(--btn-hang-text);
    }

    #hangupBtn:hover {
      background: var(--btn-hang-bg-hover);
    }

    audio {
      width: 100%;
      margin-top: 0.6rem;
    }

    small {
      color: var(--note-text);
      font-size: 0.75rem;
    }

    hr {
      border: none;
      border-top: 1px solid var(--divider);
      margin: 1rem 0;
    }
  </style>
</head>
<body data-theme="light">
  <div class="top-bar">
    <div class="top-bar-left">
      <strong>SIP 웹폰 데모</strong>
      <span>브라우저에서 바로 테스트용 / simple web test</span>
    </div>
    <button id="themeToggleBtn" class="theme-toggle">🌙 다크 모드</button>
  </div>

  <h2>
    간단 SIP 웹폰
    <span class="sub">Simple SIP Web Phone</span>
  </h2>

  <div class="row">
    <label>
      WebSocket 서버 (WS/WSS)
      <span class="en">Server URL</span>
    </label>
    <input id="wsServer" type="text" placeholder="ws://YOUR_IP_OR_DOMAIN:5066"
           value="wss://freeswitch.throw.im:7443">
  </div>

  <div class="row">
    <label>
      SIP 도메인
      <span class="en">SIP Domain</span>
    </label>
    <input id="sipDomain" type="text" placeholder="YOUR_SIP_DOMAIN_OR_IP" value="43.202.112.167">
  </div>

  <div class="row">
    <label>
      내 내선번호
      <span class="en">(Extension / SIP user)</span>
    </label>
    <input id="sipUser" type="text" placeholder="1001" value="1000">
  </div>

  <div class="row">
    <label>
      비밀번호
      <span class="en">Password</span>
    </label>
    <input id="sipPassword" type="password" placeholder="SIP password" value="p4ssw0rd-1000-EC2-xyz">
  </div>

  <div class="row">
    <button id="registerBtn">등록 / Register</button>
  </div>

  <div class="status" id="statusBox">미등록 상태 / Not registered</div>

  <hr>

  <div class="row">
    <label>
      발신할 내선번호
      <span class="en">(Number to dial)</span>
    </label>
  </div>
  <div class="dial-row">
    <input id="dialNumber" type="text" placeholder="예: 1002">
  </div>

  <div class="keypad">
    <button data-key="1">1</button>
    <button data-key="2">2</button>
    <button data-key="3">3</button>
    <button data-key="4">4</button>
    <button data-key="5">5</button>
    <button data-key="6">6</button>
    <button data-key="7">7</button>
    <button data-key="8">8</button>
    <button data-key="9">9</button>
    <button data-key="*">*</button>
    <button data-key="0">0</button>
    <button data-key="#">#</button>
  </div>

  <div class="controls">
    <button id="callBtn">통화 / Call</button>
    <button id="hangupBtn">끊기 / Hang Up</button>
  </div>

  <audio id="remoteAudio" autoplay></audio>

  <div class="row">
    <small>
      참고: 이 페이지는 브라우저 &rarr; PBX(예: FreeSWITCH)로
      WebSocket(SIP over WebSocket)만 사용합니다.
      Python 백엔드는 HTML만 서빙합니다. (simple static server only)
    </small>
  </div>

  <script>
    JsSIP.debug.enable('JsSIP:*');
    // --- Theme toggle (light / dark) ---------------------------------------
    const themeToggleBtn = document.getElementById('themeToggleBtn');

    function applyTheme(theme) {
      document.body.setAttribute('data-theme', theme);
      if (theme === 'dark') {
        themeToggleBtn.textContent = '🌙 다크 모드';
      } else {
        themeToggleBtn.textContent = '☀ 라이트 모드';
      }
    }

    (function initTheme() {
      const saved = localStorage.getItem('sipWebphoneTheme');
      let theme = saved;
      if (!theme) {
        const prefersDark = window.matchMedia &&
                            window.matchMedia('(prefers-color-scheme: dark)').matches;
        theme = prefersDark ? 'dark' : 'light';
      }
      applyTheme(theme);
    })();

    themeToggleBtn.addEventListener('click', () => {
      const current = document.body.getAttribute('data-theme') || 'dark';
      const next = current === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      localStorage.setItem('sipWebphoneTheme', next);
    });

    // --- Existing JsSIP connectivity logic (unchanged) ---------------------
    let ua = null;
    let currentSession = null;

    const statusBox     = document.getElementById('statusBox');
    const registerBtn   = document.getElementById('registerBtn');
    const callBtn       = document.getElementById('callBtn');
    const hangupBtn     = document.getElementById('hangupBtn');
    const dialNumberEl  = document.getElementById('dialNumber');
    const remoteAudio   = document.getElementById('remoteAudio');

    const wsInput       = document.getElementById('wsServer');
    const domainInput   = document.getElementById('sipDomain');

    function setStatus(text, type = '') {
      statusBox.textContent = text;
      statusBox.className = 'status' + (type ? ' ' + type : '');
    }

    // Auto-fill WS from domain if it’s empty
    domainInput.addEventListener('input', () => {
      const v = domainInput.value.trim();
      if (!wsInput.value && v) {
        wsInput.value = `wss://freeswitch.throw.im:7443`;
      }
    });

    // Dial pad
    document.querySelectorAll('.keypad button').forEach(btn => {
      btn.addEventListener('click', () => {
        const val = btn.getAttribute('data-key');
        if (!val) return;
        if (val === 'del') {
          dialNumberEl.value = dialNumberEl.value.slice(0, -1);
        } else {
          dialNumberEl.value += val;
        }
      });
    });

    // Register SIP extension
    registerBtn.addEventListener('click', () => {
      const sipDomain = domainInput.value.trim();
      const sipUser   = document.getElementById('sipUser').value.trim();
      const sipPass   = document.getElementById('sipPassword').value.trim();

      if (!sipDomain || !sipUser || !sipPass) {
        setStatus('도메인/내선/비밀번호를 입력하세요. (Fill in domain, extension, and password.)', 'error');
        return;
      }

      let wsServer = wsInput.value.trim();
      if (!wsServer) {
        wsServer = `wss://freeswitch.throw.im:7443`;
        wsInput.value = wsServer;
      }

      if (ua) {
        ua.stop();
        ua = null;
      }

      const socket = new JsSIP.WebSocketInterface(wsServer);
      const configuration = {
        sockets: [socket],
        uri: `sip:${sipUser}@${sipDomain}`,
        password: sipPass,
        session_timers: true
      };

      ua = new JsSIP.UA(configuration);

      ua.on('connected', () => setStatus('WebSocket 연결됨 / WebSocket connected'));
      ua.on('disconnected', () => setStatus('WebSocket 연결 끊김 / disconnected', 'error'));

      ua.on('registered', () => setStatus(`등록 완료 / Registered as ${sipUser}`, 'connected'));
      ua.on('unregistered', () => setStatus('등록 해제 / Unregistered'));

      ua.on('registrationFailed', (e) => {
        console.error('Registration failed', e);
        setStatus('등록 실패 / Registration failed: ' + (e.cause || ''), 'error');
      });

      function attachRemoteTrack(pc) {
        if (!pc) return;
        console.log('[WebRTC] attachRemoteTrack called with PC:', pc);

        pc.addEventListener('track', (evt) => {
          console.log('[WebRTC] Remote track received:', evt);
          if (evt.streams && evt.streams[0]) {
            remoteAudio.srcObject = evt.streams[0];
            remoteAudio.muted = false;
            const p = remoteAudio.play();
            if (p && p.catch) {
              p.catch(err => console.warn('remoteAudio.play() failed:', err));
            }
          }
        });
      }

      ua.on('newRTCSession', (e) => {
        const session = e.session;
        currentSession = session;
        window.currentSession = session;
        console.log('[SIP] newRTCSession, originator =', e.originator);

        // 1) If the peerconnection already exists, attach immediately
        if (session.connection) {
          console.log('[WebRTC] session.connection already present, attaching track handler');
          attachRemoteTrack(session.connection);
        }

        // 2) Also attach when the peerconnection event fires
        session.on('peerconnection', (ev2) => {
          console.log('[WebRTC] peerconnection event fired');
          attachRemoteTrack(ev2.peerconnection);
        });

        // Auto-answer incoming calls (phone -> browser)
        if (e.originator === 'remote') {
          setStatus('수신 전화 중... (Incoming call...)');

          const answerOptions = {
            mediaConstraints: { audio: true, video: false },
            pcConfig: {
              iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
            },
            sessionTimersExpires: 120
          };
          console.log('[SIP] Answering incoming call with options:', answerOptions);
          session.answer(answerOptions);
        }

        session.on('ended', () => {
          setStatus('통화 종료 / Call ended');
          currentSession = null;
        });

        session.on('failed', (ev) => {
          console.error('Call failed', ev);
          setStatus('통화 실패 / Call failed: ' + (ev.cause || ''), 'error');
          currentSession = null;
        });

        session.on('accepted', () => {
          setStatus('통화 중... / Call in progress...');
        });
      });

      ua.start();
      setStatus('SIP 서버에 연결 중... / Connecting to SIP server...');
    });

    // Call
    callBtn.addEventListener('click', () => {
      if (!ua || !ua.isRegistered()) {
        setStatus('먼저 등록하세요. (Not registered. Register first.)', 'error');
        return;
      }

      const sipDomain = domainInput.value.trim();
      const number    = dialNumberEl.value.trim();
      if (!number) {
        setStatus('발신할 내선번호를 입력하세요. (Enter extension to call.)', 'error');
        return;
      }

      const target = `sip:${number}@${sipDomain}`;
      const options = {
        mediaConstraints: { audio: true, video: false },
        pcConfig: {
          iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
        },
        sessionTimersExpires: 120  // avoids 422 Session Interval Too Small
      };

      console.log('[SIP] Calling', target, 'with options', options);
      currentSession = ua.call(target, options);
      setStatus(`발신 중... ${target} / Calling ...`);
    });

    // Hang up
    hangupBtn.addEventListener('click', () => {
      if (currentSession) {
        currentSession.terminate();
        currentSession = null;
        setStatus('통화 종료 / Call terminated');
      }
    });
  </script>

</body>
</html>
"""

@app.get("/favicon.ico")
async def favicon():
    global _cached_icon
    if not _cached_icon:
        # Either return 204 or redirect to the original URL
        # return Response(status_code=204)
        return RedirectResponse(FAVICON_URL)
    return Response(_cached_icon, media_type="image/png")


@app.get("/", response_class=HTMLResponse)
async def index():
  return HTML_PAGE

@app.get("/health", response_class=HTMLResponse)
async def health():
  return "ok"

if __name__ == "__main__":
  # Change port if you like (e.g., 8080)
  uvicorn.run("sip_app:app", host="0.0.0.0", port=8080, reload=True)
