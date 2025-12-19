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

from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials

import os
import re
import secrets
import string
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

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

    .admin-link{
      position: fixed;
      top: 10px;
      right: 12px;
      z-index: 9999;

      width: 30px;
      height: 30px;
      display: inline-flex;
      align-items: center;
      justify-content: center;

      border-radius: 999px;
      text-decoration: none;
      font-size: 16px;
      line-height: 1;

      opacity: 0.25;
      transition: opacity 120ms ease, transform 120ms ease;
      backdrop-filter: blur(6px);
    }

    .admin-link:hover{
      opacity: 0.9;
      transform: scale(1.05);
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
  
    /* Top nav */
    .topnav { position: sticky; top: 0; z-index: 50; backdrop-filter: blur(10px);
      background: rgba(2,6,23,0.65); border-bottom: 1px solid var(--border); }
    .topnav-inner { max-width: 1100px; margin: 0 auto; padding: 10px 16px;
      display: flex; align-items: center; gap: 12px; }
    .brand { font-weight: 700; letter-spacing: 0.2px; margin-right: 10px; }
    .tab { display: inline-block; padding: 8px 12px; border-radius: 10px; border: 1px solid transparent;
      text-decoration: none; color: var(--text); }
    .tab:hover { background: rgba(148,163,184,0.12); border-color: rgba(148,163,184,0.25); }
    .tab.active { background: rgba(59,130,246,0.18); border-color: rgba(59,130,246,0.35); }

  </style>
</head>
<body data-theme="light">
  <a class="admin-link" href="/sip/admin" title="내선 관리 / Extensions">⚙︎</a>

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
    <button data-key="backspace">⌫</button>
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
        if (val === 'backspace') {
          backspace();
        } else {
          dialNumberEl.value += val;
        }
      });
    });


    function backspace(){
      dialNumberEl.value = dialNumberEl.value.slice(0, -1);
      dialNumberEl.focus();
    }


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

# ----------------------------
# FreeSWITCH Directory (static XML) Extension Manager
# ----------------------------
FS_CONF_DIR = os.environ.get("FS_CONF_DIR", "/usr/local/freeswitch/conf/vanilla")
USERS_DIR = Path(os.environ.get("FS_USERS_DIR", str(Path(FS_CONF_DIR) / "directory" / "default")))
FS_CLI = os.environ.get("FS_CLI", "/usr/local/freeswitch/bin/fs_cli")

ADMIN_USERNAME = os.environ.get("SIP_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("SIP_ADMIN_PASS")  # if unset/empty -> admin endpoints are open (dev mode)

basic_security = HTTPBasic()


def _require_admin(credentials: HTTPBasicCredentials = Depends(basic_security)):
  """
  Minimal protection for admin endpoints.
  Set SIP_ADMIN_PASS to enable auth. Username defaults to 'admin' (SIP_ADMIN_USER).
  """
  if not ADMIN_PASSWORD:
    return True

  ok_user = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
  ok_pass = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
  if not (ok_user and ok_pass):
    raise HTTPException(
      status_code=status.HTTP_401_UNAUTHORIZED,
      detail="Unauthorized",
      headers={"WWW-Authenticate": "Basic"},
    )
  return True


def _rand_password(n: int = 14) -> str:
  alphabet = string.ascii_letters + string.digits
  return "".join(secrets.choice(alphabet) for _ in range(n))


def _reloadxml() -> None:
  # Keep it simple: rely on FS_CLI in PATH (or set FS_CLI to full path).
  # Typical config reload for static directory users.
  proc = subprocess.run([FS_CLI, "-x", "reloadxml"], capture_output=True, text=True)
  if proc.returncode != 0:
    raise RuntimeError(f"reloadxml failed: {proc.stderr.strip() or proc.stdout.strip()}")


def _safe_ext(ext: str) -> str:
  ext = (ext or "").strip()
  if not re.fullmatch(r"\d{2,8}", ext):
    raise HTTPException(status_code=400, detail="Extension must be 2-8 digits.")
  return ext


def _user_xml(ext: str, password: str, display_name: str | None = None, caller_id: str | None = None) -> str:
  dn = (display_name or ext).strip()
  cid = (caller_id or ext).strip()
  # Minimal XML template consistent with FreeSWITCH directory/default/*.xml usage.
  return f"""<include>
  <user id="{ext}">
    <params>
      <param name="password" value="{password}"/>
    </params>
    <variables>
      <variable name="user_context" value="default"/>
      <variable name="effective_caller_id_name" value="{dn}"/>
      <variable name="effective_caller_id_number" value="{cid}"/>
    </variables>
  </user>
</include>
"""


def _parse_user_file(path: Path) -> dict:
  # Best-effort parser; tolerates extra sections.
  tree = ET.parse(path)
  root = tree.getroot()

  # Find <user id="...">
  user = root.find(".//user")
  user_id = user.attrib.get("id") if user is not None else path.stem

  def _find_param(name: str) -> str | None:
    el = root.find(f".//param[@name='{name}']")
    return el.attrib.get("value") if el is not None else None

  def _find_var(name: str) -> str | None:
    el = root.find(f".//variable[@name='{name}']")
    return el.attrib.get("value") if el is not None else None

  mtime = path.stat().st_mtime
  return {
    "ext": str(user_id),
    "has_password": _find_param("password") is not None or _find_param("a1-hash") is not None,
    "caller_id_name": _find_var("effective_caller_id_name"),
    "caller_id_number": _find_var("effective_caller_id_number"),
    "user_context": _find_var("user_context"),
    "path": str(path),
    "mtime_epoch": mtime,
  }


ADMIN_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Extension Manager</title>
  <link rel="icon" href="/favicon.ico" />
  <style>
    :root { color-scheme: light dark; }
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0;
      background: #0b1020; color: #e2e8f0; }
    .topnav { position: sticky; top: 0; z-index: 50; backdrop-filter: blur(10px);
      background: rgba(2,6,23,0.65); border-bottom: 1px solid rgba(148,163,184,0.25); }
    .topnav-inner { max-width: 1100px; margin: 0 auto; padding: 10px 16px;
      display: flex; align-items: center; gap: 12px; }
    .brand { font-weight: 700; letter-spacing: 0.2px; margin-right: 10px; }
    .tab { display: inline-block; padding: 8px 12px; border-radius: 10px; border: 1px solid transparent;
      text-decoration: none; color: #e2e8f0; }
    .tab:hover { background: rgba(148,163,184,0.12); border-color: rgba(148,163,184,0.25); }
    .tab.active { background: rgba(59,130,246,0.18); border-color: rgba(59,130,246,0.35); }

    .wrap { max-width: 1100px; margin: 16px auto 40px; padding: 0 16px; }
    .card { background: rgba(17,24,39,0.85); border: 1px solid rgba(148,163,184,0.20);
      border-radius: 16px; padding: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.35); }
    h1 { margin: 0 0 10px; font-size: 20px; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: end; }
    label { font-size: 12px; color: rgba(226,232,240,0.8); display: block; margin-bottom: 6px; }
    input { width: 220px; padding: 10px 12px; border-radius: 12px; border: 1px solid rgba(148,163,184,0.25);
      background: rgba(2,6,23,0.6); color: #e2e8f0; outline: none; }
    input::placeholder { color: rgba(226,232,240,0.45); }
    button { padding: 10px 12px; border-radius: 12px; border: 1px solid rgba(148,163,184,0.25);
      background: rgba(59,130,246,0.18); color: #e2e8f0; cursor: pointer; }
    button:hover { filter: brightness(1.08); }
    .btn-danger { background: rgba(239,68,68,0.18); }
    .btn-ghost { background: rgba(148,163,184,0.10); }

    table { width: 100%; border-collapse: collapse; margin-top: 12px; }
    th, td { text-align: left; padding: 10px 8px; border-bottom: 1px solid rgba(148,163,184,0.15); font-size: 14px; }
    th { color: rgba(226,232,240,0.8); font-weight: 600; }
    .muted { color: rgba(226,232,240,0.65); font-size: 12px; }
    .right { text-align: right; }
    .pill { font-size: 12px; padding: 3px 8px; border-radius: 999px; border: 1px solid rgba(148,163,184,0.20);
      background: rgba(148,163,184,0.10); display: inline-block; }
    .note { margin-top: 10px; color: rgba(226,232,240,0.70); font-size: 12px; line-height: 1.4; }
    code { background: rgba(148,163,184,0.12); padding: 2px 6px; border-radius: 8px; }
  </style>
</head>
<body>
  <div class="topnav">
    <div class="topnav-inner">
      <div class="brand">SIP WebPhone</div>
      <a class="tab" href="/sip">소프트폰 / Softphone</a>
      <a class="tab active" href="/admin">내선 관리 / Extensions</a>
    </div>
  </div>

  <div class="wrap">
    <div class="card">
      <h1>FreeSWITCH 내선 관리 (정적 XML) / FreeSWITCH Extensions (static XML)</h1>

      <div class="row">
        <div>
          <label>내선번호 / Extension</label>
          <input id="ext" placeholder="e.g., 1102" />
        </div>
        <div>
          <label>표시 이름 (선택) / Display name (optional)</label>
          <input id="dn" placeholder="e.g., Calling as 1102" />
        </div>
        <div>
          <label>발신자 번호 (선택) / Caller ID number (optional)</label>
          <input id="cid" placeholder="e.g., 1102" />
        </div>
        <div>
          <label>비밀번호 (선택) / Password (optional)</label>
          <input id="pw" placeholder="auto-generate if empty" />
        </div>

        <div>
          <button onclick="createExt()">생성 / Create</button>
          <button class="btn-ghost" onclick="refresh()">새로고침 / Refresh</button>
        </div>
      </div>

      <div class="note">
        Source directory: <code id="srcDir"></code><br/>
        After every change, the server runs <code>fs_cli -x reloadxml</code>.
      </div>

      <table>
        <thead>
          <tr>
            <th>Ext</th>
            <th>Caller ID Name</th>
            <th>Caller ID #</th>
            <th>Has Auth</th>
            <th class="right">동작 / Actions</th>
          </tr>
        </thead>
        <tbody id="tbody">
          <tr><td colspan="5" class="muted">Loading…</td></tr>
        </tbody>
      </table>
    </div>
  </div>

<script>
async function api(path, opts={}) {
  const res = await fetch(path, Object.assign({headers: {"Content-Type":"application/json"}}, opts));
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch(e) {}
  if (!res.ok) {
    const msg = (data && (data.detail || data.error)) ? (data.detail || data.error) : text || ("HTTP " + res.status);
    throw new Error(msg);
  }
  return data;
}

function esc(s){ return (s ?? "").toString().replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;"); }

async function refresh() {
  const tbody = document.getElementById("tbody");
  tbody.innerHTML = `<tr><td colspan="5" class="muted">Loading…</td></tr>`;
  const data = await api("/admin/api/extensions");
  document.getElementById("srcDir").textContent = data.source_dir;

  if (!data.items.length) {
    tbody.innerHTML = `<tr><td colspan="5" class="muted">No extensions found.</td></tr>`;
    return;
  }

  tbody.innerHTML = data.items.map(u => {
    const hasAuth = u.has_password ? `<span class="pill">yes</span>` : `<span class="pill">no</span>`;
    return `
      <tr>
        <td><b>${esc(u.ext)}</b><div class="muted">${new Date(u.mtime_epoch*1000).toLocaleString()}</div></td>
        <td>${esc(u.caller_id_name || "")}</td>
        <td>${esc(u.caller_id_number || "")}</td>
        <td>${hasAuth}</td>
        <td class="right">
          <button class="btn-ghost" onclick="resetPw('${esc(u.ext)}')">비밀번호 재설정 / Reset PW</button>
          <button class="btn-danger" onclick="delExt('${esc(u.ext)}')">삭제 / Delete</button>
        </td>
      </tr>
    `;
  }).join("");
}

async function createExt() {
  const ext = document.getElementById("ext").value.trim();
  const display_name = document.getElementById("dn").value.trim();
  const caller_id = document.getElementById("cid").value.trim();
  const password = document.getElementById("pw").value;

  const payload = {ext, display_name, caller_id, password};
  const data = await api("/admin/api/extensions", {method:"POST", body: JSON.stringify(payload)});
  alert(`Created extension ${data.ext}\nPassword: ${data.password}\n\n(Password is only shown once.)`);
  document.getElementById("pw").value = "";
  await refresh();
}

async function resetPw(ext) {
  const pw = prompt(`새 비밀번호 설정 (${ext}) — 비워두면 랜덤 생성`, "");
  const payload = pw && pw.trim() ? { password: pw.trim() } : {};

  const data = await api(
    `/sip/admin/api/extensions/${encodeURIComponent(ext)}/reset_password`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }
  );

  alert(`New password for ${ext}: ${data.password}`);
  await refresh();
}



async function delExt(ext) {
  if (!confirm(`내선 ${ext}을(를) 삭제할까요? XML 파일이 삭제됩니다.`)) return;
  await api(`/admin/api/extensions/${encodeURIComponent(ext)}`, {method:"DELETE"});
  await refresh();
}

refresh().catch(e => {
  document.getElementById("tbody").innerHTML = `<tr><td colspan="5" class="muted">${esc(e.message)}</td></tr>`;
});
</script>
</body>
</html>
"""


@app.get("/favicon.ico")
async def favicon():
  # Keep favicon requests quiet (either redirect to hosted icon or serve generated png)
  if _cached_icon is None:
    return RedirectResponse(FAVICON_URL)
  return Response(_cached_icon, media_type="image/png")


@app.get("/", include_in_schema=False)
async def root():
  return RedirectResponse("/sip")


@app.get("/sip", response_class=HTMLResponse)
async def sip_page():
  return HTML_PAGE


@app.get("/admin", response_class=HTMLResponse)
@app.get("/sip/admin", response_class=HTMLResponse)
async def admin_page(_ok: bool = Depends(_require_admin)):
  return ADMIN_PAGE


@app.get("/admin/api/extensions")
@app.get("/sip/admin/api/extensions")
async def list_extensions(_ok: bool = Depends(_require_admin)):
  try:
    USERS_DIR.mkdir(parents=True, exist_ok=True)
    items = []
    for p in sorted(USERS_DIR.glob("*.xml")):
      ext = p.stem
      if not ext.isdigit():
          continue
      # Skip obvious non-user files if any
      if p.name.lower().startswith("default"):
        continue      
      items.append(_parse_user_file(p))
    # Sort numerically if possible
    def _k(x):
      try:
        return int(x["ext"])
      except Exception:
        return 10**18
    items.sort(key=_k)
    return {"source_dir": str(USERS_DIR), "items": items}
  except Exception as e:
    return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/admin/api/extensions")
@app.post("/sip/admin/api/extensions")
async def create_extension(req: Request, _ok: bool = Depends(_require_admin)):
  try:
    data = await req.json()
  except Exception:
    data = {}

  ext = _safe_ext(str(data.get("ext", "")))
  password = (data.get("password") or "").strip() or _rand_password()
  display_name = (data.get("display_name") or "").strip() or None
  caller_id = (data.get("caller_id") or "").strip() or None

  user_file = USERS_DIR / f"{ext}.xml"
  if user_file.exists():
    raise HTTPException(status_code=409, detail=f"Extension {ext} already exists.")

  USERS_DIR.mkdir(parents=True, exist_ok=True)
  user_file.write_text(_user_xml(ext, password, display_name, caller_id), encoding="utf-8")

  _reloadxml()
  return {"ext": ext, "password": password}


@app.post("/admin/api/extensions/{ext}/reset_password")
@app.post("/sip/admin/api/extensions/{ext}/reset_password")
async def reset_password(ext: str, req: Request, _ok: bool = Depends(_require_admin)):
  ext = _safe_ext(ext)
  if not ext.isdigit():
    raise HTTPException(status_code=400, detail="Invalid extension id")

  user_file = USERS_DIR / f"{ext}.xml"
  if not user_file.exists():
    raise HTTPException(status_code=404, detail=f"Extension {ext} not found.")

  # Try to read optional JSON body: {"password": "..."}
  body = {}
  try:
    body = await req.json()
  except Exception:
    body = {}

  password = (body.get("password") or "").strip()
  if not password:
    password = _rand_password()

  # (Optional) basic validation
  if len(password) < 4:
    raise HTTPException(status_code=400, detail="Password too short")
  if any(ch.isspace() for ch in password):
    raise HTTPException(status_code=400, detail="Password must not contain spaces")

  # Keep existing display/caller-id if present
  info = _parse_user_file(user_file)
  display_name = info.get("caller_id_name") or ext
  caller_id = info.get("caller_id_number") or ext

  user_file.write_text(_user_xml(ext, password, display_name, caller_id), encoding="utf-8")
  _reloadxml()
  return {"ext": ext, "password": password}


@app.delete("/admin/api/extensions/{ext}")
@app.delete("/sip/admin/api/extensions/{ext}")
async def delete_extension(ext: str, _ok: bool = Depends(_require_admin)):
  ext = _safe_ext(ext)
  if not ext.isdigit():
    raise HTTPException(status_code=400, detail="Invalid extension id")
  user_file = USERS_DIR / f"{ext}.xml"
  if not user_file.exists():
    raise HTTPException(status_code=404, detail=f"Extension {ext} not found.")
  user_file.unlink()
  _reloadxml()
  return {"ok": True, "ext": ext}


@app.get("/health", response_class=HTMLResponse)
async def health():
  return "ok"


if __name__ == "__main__":
  # Change port if you like (e.g., 8080)
  uvicorn.run("sip_app:app", host="0.0.0.0", port=8080, reload=True)
