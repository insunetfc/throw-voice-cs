from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import time, uuid, asyncio
import os, io, boto3
import json, base64, threading
from pydantic import BaseModel
from typing import Optional
from queue import Queue, Empty
from contextlib import contextmanager
import audioop, struct
import openai
from websocket import create_connection, WebSocketTimeoutException
from dataclasses import dataclass, field
import threading
from pydub import AudioSegment
from fastapi.params import Query as QueryParam
from fastapi import FastAPI, WebSocket, APIRouter, HTTPException, Body, Query, File, UploadFile, Form, HTTPException
import json
import os
from pathlib import Path
import hashlib
import time, requests
import websockets
from botocore.exceptions import ClientError  # NEW

from .utils.s3 import upload_wav_to_s3
from .utils.audio import ensure_mulaw_8k
from .utils.logger import log_metric
from .adapters.elevenlabs import fetch_elevenlabs_audio
from .adapters.minimax import generate_minimax_audio
from .adapters.gpt_realtime import gpt_ws_relay

# ============================================================================
# CONFIGURATION
# ============================================================================

AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET = os.getenv("TTS_BUCKET", "tts-bucket-250810")
S3_PREFIX = "bridge/audio/gpt"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
REALTIME_MODEL = "gpt-4o-mini-realtime-preview-2024-12-17"
WS_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"

ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
MODEL_ID = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
BASE_URL = "https://api.elevenlabs.io/v1/text-to-speech"
DEFAULT_ELEVEN_VOICE_ID = os.getenv(
    "ELEVENLABS_VOICE",
    "AW5wrnG1jVizOYY7R1Oo",   # This should be a valid voice_id, not "JiYoung"
)
AUDIO_CACHE: dict[str, str] = {} 
S3_PREFIX_ELEVEN = "bridge/audio/eleven"


AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
s3 = boto3.client("s3", region_name=AWS_REGION)
dynamodb = boto3.client("dynamodb", region_name=AWS_REGION)         # NEW
PROMPT_KR = """당신은 친절한 한국어 고객 상담원입니다.

규칙:
1. 고객의 말에 직접적으로 답변하세요
2. "안녕", "hello" 같은 인사에는 인사로 답하세요
3. "바빠요" 같은 상태 표현에는 공감하고 도움을 제안하세요
4. 질문에는 구체적으로 답변하세요
5. 짧고 명확하게 답하되, 맥락에 맞게 다르게 응답하세요
6. 고객 발화를 그대로 반복(에코)하지 마세요

예시:
- "안녕하세요" → "안녕하세요! 무엇을 도와드릴까요?"
- "바빠요" → "바쁘신 것 같네요. 빠르게 도와드리겠습니다. 무엇이 필요하신가요?"
- "보험 문의" → "네, 보험 상담을 도와드리겠습니다. 어떤 보험이 궁금하신가요?"
"""

# 1) Korean base rules for counselor
PROMPT_RULES_KR = """당신은 차집사 다이렉트 자동차보험 아웃바운드 상담원입니다.

규칙:
1. 고객 발화에 직접적으로 답변하세요.
2. "안녕하세요", "hello" 같은 인사에는 정중하게 인사로 답하세요.
3. 고객이 "바빠요", "시간 없어요" 같은 표현을 사용하면 공감 후 간단히 안내하세요.
4. 질문에는 간단하고 명확하게, 그러나 실제 콜센터 톤으로 답하세요.
5. 고객 발화를 그대로 반복(에코)하지 마세요.
6. 존댓말을 사용하고, 한두 문장 내로 짧게 답하세요.
"""

# 2) Scenario file
try:
    scenario_path = Path(__file__).parent / "scenario_prompt.txt"
    SCENARIO_TEXT = scenario_path.read_text(encoding="utf-8").strip()
except:
    SCENARIO_TEXT = ""

# 3) Final combined system prompt
FINAL_SYSTEM_PROMPT = PROMPT_RULES_KR + "\n\n[시나리오 참고]\n" + SCENARIO_TEXT

# ============================================================================
# TIMING & METRICS
# ============================================================================

class T:
    def __enter__(self): self.t = time.time(); return self
    def __exit__(self, *a): self.ms = int((time.time() - self.t)*1000)

class EWMA:
    def __init__(self, alpha=0.2): 
        self.v = None
        self.a = alpha
        self.lock = threading.Lock()
    
    def add(self, x):
        with self.lock:
            self.v = x if self.v is None else (self.a*x + (1-self.a)*self.v)
    
    def get(self):
        with self.lock:
            return None if self.v is None else round(self.v, 1)

METRICS = {
    "total_ms": EWMA(),
    "gpt_ms": EWMA(),
    "s3_ms": EWMA(),
    "connect_ms": EWMA(),
}

def _now_ms():
    return int(time.perf_counter() * 1000)

# ============================================================================
# WEBSOCKET CONNECTION POOL
# ============================================================================

def ensure_s3_bucket(bucket_name: str):
    """Ensure S3 bucket exists; create it if necessary."""
    if not bucket_name:
        return
    try:
        s3.head_bucket(Bucket=bucket_name)
        return
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code not in ("404", "NoSuchBucket", "400", "403"):
            raise
    
    create_kwargs = {"Bucket": bucket_name}
    if AWS_REGION and AWS_REGION != "us-east-1":
        create_kwargs["CreateBucketConfiguration"] = {
            "LocationConstraint": AWS_REGION
        }
    s3.create_bucket(**create_kwargs)


def ensure_ddb_table(table_name: str):
    """Ensure DynamoDB table with PK: scenario_id, SK: short_key exists."""
    if not table_name:
        return

    try:
        dynamodb.describe_table(TableName=table_name)
        return
    except dynamodb.exceptions.ResourceNotFoundException:
        pass

    dynamodb.create_table(
        TableName=table_name,
        KeySchema=[
            {"AttributeName": "scenario_id", "KeyType": "HASH"},
            {"AttributeName": "short_key", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "scenario_id", "AttributeType": "S"},
            {"AttributeName": "short_key", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    waiter = dynamodb.get_waiter("table_exists")
    waiter.wait(TableName=table_name)

@dataclass
class PooledWS:
    ws: any
    idle_event: threading.Event = field(default_factory=lambda: threading.Event())
    alive: bool = True
    lock: threading.Lock = field(default_factory=threading.Lock)

    def mark_in_use(self):
        # pause keepalive reader
        self.idle_event.clear()

    def mark_idle(self):
        # resume keepalive reader
        self.idle_event.set()

class WebSocketPool:
    """
    Connection pool for WebSocket connections to avoid cold starts.
    Maintains warm connections ready for immediate use.
    """
    def __init__(self, url: str, pool_size: int = 3, timeout: int = 10):
        self.url = url
        self.pool_size = pool_size
        self.timeout = timeout
        self.pool: Queue = Queue(maxsize=pool_size)
        self.lock = threading.Lock()
        self.active = True
        self.stats = {"created": 0, "reused": 0, "errors": 0}
        
        # Pre-warm the pool
        self._prewarm()
        
        # Start maintenance thread
        self.maintenance_thread = threading.Thread(
            target=self._maintain_pool, 
            daemon=True
        )
        self.maintenance_thread.start()
        
    def _prewarm(self):
        """Create initial connections"""
        print(f"🔥 Pre-warming WebSocket pool with {self.pool_size} connections...")
        for i in range(self.pool_size):
            try:
                ws = self._create_connection()
                if ws:
                    self.pool.put(ws, block=False)
                    print(f"  ✓ Connection {i+1}/{self.pool_size} ready")
            except Exception as e:
                print(f"  ✗ Failed to pre-warm connection {i+1}: {e}")
    
    def _create_connection(self):
        """Create a new WebSocket connection"""
        try:
            headers = [
                f"Authorization: Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta: realtime=v1"
            ]
            ws = create_connection(
                self.url,
                timeout=self.timeout,
                header=headers,
                enable_multithread=True
            )
            pws = PooledWS(ws=ws)
            self._start_keepalive(pws)
            self.stats["created"] += 1
            return pws
        except Exception as e:
            self.stats["errors"] += 1
            print(f"❌ Failed to create WebSocket: {e}")
            return None
    
    def _is_connection_alive(self, pws: PooledWS) -> bool:
        try:
            if not pws or not pws.alive:
                return False
            if not getattr(pws.ws, "connected", False):
                return False
            # optional probe; don't call recv() here
            pws.ws.ping()
            return True
        except Exception:
            return False
    
    def _maintain_pool(self):
        while self.active:
            time.sleep(30)
            try:
                current_size = self.pool.qsize()
                if current_size < self.pool_size:
                    needed = self.pool_size - current_size
                    print(f"🔧 Pool maintenance: adding {needed} connections")
                    for _ in range(needed):
                        pws = self._create_connection()
                        if pws:
                            try: self.pool.put(pws, block=False)
                            except: pass
            except Exception as e:
                print(f"⚠️ Pool maintenance error: {e}")
    
    @contextmanager
    def get_connection(self):
        pws = None
        try:
            try:
                pws = self.pool.get(timeout=0.5)
                if not self._is_connection_alive(pws):
                    print("⚠️ Stale connection, creating new one")
                    try: pws.ws.close()
                    except: pass
                    pws = self._create_connection()
                else:
                    self.stats["reused"] += 1
            except Empty:
                print("📡 Creating new connection (pool empty)")
                pws = self._create_connection()
    
            if not pws:
                raise Exception("Failed to get WebSocket connection")
    
            # pause keepalive while in use
            pws.mark_in_use()
            try:
                yield pws.ws    # <--- IMPORTANT: expose the raw websocket
            finally:
                # resume keepalive for idle period
                pws.mark_idle()
    
        except Exception as e:
            self.stats["errors"] += 1
            raise e
        finally:
            if pws:
                try:
                    if self._is_connection_alive(pws) and self.pool.qsize() < self.pool_size:
                        self.pool.put(pws, block=False)   # put the PooledWS back
                    else:
                        try: pws.ws.close()
                        except: pass
                except:
                    try: pws.ws.close()
                    except: pass
                        
    def _start_keepalive(self, pws: PooledWS):
        def _loop():
            pws.ws.settimeout(5)
            # socket starts idle in the pool
            pws.idle_event.set()
            while pws.alive:
                # Wait until the socket is idle; while in-use, this blocks here.
                pws.idle_event.wait()
                try:
                    # A short recv lets websocket-client process ping/pong frames.
                    # If there’s no data for 5s, it just times out and loops.
                    _ = pws.ws.recv()
                    # We ignore app data while idle; the app never uses the socket
                    # concurrently, so anything we "read" here is only control frames.
                except WebSocketTimeoutException:
                    continue
                except Exception:
                    # closed/errored; let the pool recycle it
                    pws.alive = False
                    break
        t = threading.Thread(target=_loop, daemon=True)
        t.start()

    def get_stats(self):
        return {
            **self.stats,
            "pool_size": self.pool.qsize(),
            "max_size": self.pool_size
        }
    
    def shutdown(self):
        self.active = False
        while not self.pool.empty():
            try:
                pws = self.pool.get_nowait()
                try: pws.ws.close()
                except: pass
            except: pass
                
# Global pool
ws_pool = WebSocketPool(WS_URL, pool_size=3, timeout=10)

# ============================================================================
# AUDIO UTILITIES
# ============================================================================

def wav_wrap_ulaw(raw_bytes: bytes, sample_rate=8000, channels=1) -> bytes:
    """Wrap μ-law raw bytes into a valid WAV container."""
    audio_format = 7  # μ-law format
    bits_per_sample = 8
    byte_rate = sample_rate * channels * 1
    block_align = channels * 1
    data_size = len(raw_bytes)
    riff_size = 36 + data_size
    
    header = b"RIFF" + struct.pack("<I", riff_size) + b"WAVE"
    fmt = (b"fmt " + struct.pack("<I", 16) + 
           struct.pack("<H", audio_format) +
           struct.pack("<H", channels) + 
           struct.pack("<I", sample_rate) +
           struct.pack("<I", byte_rate) + 
           struct.pack("<H", block_align) +
           struct.pack("<H", bits_per_sample))
    data = b"data" + struct.pack("<I", data_size)
    
    return header + fmt + data + raw_bytes

# ============================================================================
# GPT VOICE GENERATION (COMPLETE, NO STREAMING)
# ============================================================================

def generate_complete_gpt_voice(prompt: str, voice: str = "alloy") -> bytes:
    """
    Generate complete GPT voice audio synchronously.
    Returns complete WAV file in μ-law format for Amazon Connect.
    
    This function waits for the entire audio to be generated before returning.
    No streaming or partial audio files.
    """
    t_start = _now_ms()
    
    print(f"🎤 Generating complete audio for: '{prompt}'")
    
    try:
        # Get connection from pool
        with ws_pool.get_connection() as ws:
            t_connect = _now_ms()
            METRICS["connect_ms"].add(t_connect - t_start)
            print(f"  ✓ Connected in {t_connect - t_start}ms")
            
            # Configure session
            session_config = {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "voice": voice,
                    "output_audio_format": "pcm16",
                    "turn_detection": None,
                    "temperature": 0.6,
                    "max_response_output_tokens": 256,
                    "instructions": FINAL_SYSTEM_PROMPT,
                }
            }
            ws.send(json.dumps(session_config))
            
            # Send conversation item
            conversation_item = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}]
                }
            }
            ws.send(json.dumps(conversation_item))
            
            # Request response
            response_create = {
                "type": "response.create",
                "response": {
                    "modalities": ["text", "audio"],   # <--- ensure text deltas arrive
                    # (optional) reinforce non-parroting just for this turn too
                    # "instructions": "고객의 말을 반복하지 말고, 한두 문장으로 도와주세요."
                }
            }

            ws.send(json.dumps(response_create))
            
            # Collect ALL audio chunks before returning
            audio_chunks = []
            text_response = ""
            chunk_count = 0
            first_chunk_time = None
            
            print(f"  ⏳ Waiting for complete audio generation...")
            
            while True:
                try:
                    raw = ws.recv()
                    if not raw:
                        break
                    
                    msg = json.loads(raw)
                    msg_type = msg.get("type", "")
                    
                    # Collect audio deltas
                    if msg_type == "response.audio.delta":
                        b64_audio = msg.get("delta", "")
                        if b64_audio:
                            chunk_count += 1
                            if first_chunk_time is None:
                                first_chunk_time = _now_ms()
                                ttfb = first_chunk_time - t_start
                                print(f"  ⚡ First chunk in {ttfb}ms")
                            
                            pcm16_bytes = base64.b64decode(b64_audio)
                            audio_chunks.append(pcm16_bytes)
                    
                    # Collect text response
                    elif msg_type == "response.text.delta":
                        text_response += msg.get("delta", "")
                    
                    # Wait for completion
                    elif msg_type == "response.done":
                        print(f"  ✓ Audio generation complete ({chunk_count} chunks)")
                        break
                    
                    # Handle errors
                    elif msg_type == "error":
                        error_detail = msg.get("error", {})
                        print(f"  ❌ GPT error: {error_detail}")
                        raise Exception(f"GPT error: {error_detail}")
                
                except WebSocketTimeoutException:
                    print(f"  ⚠️ WebSocket timeout")
                    break
                except Exception as e:
                    print(f"  ❌ Error receiving: {e}")
                    raise
            
            # Process complete audio
            if not audio_chunks:
                print(f"  ⚠️ No audio chunks received")
                # Return silence as fallback (8kHz μ-law)
                silent_1s = bytes([0xFF]) * 8000
                return wav_wrap_ulaw(silent_1s, sample_rate=8000)
            
            print(f"  📦 Processing {len(audio_chunks)} chunks...")
            
            # Concatenate all PCM16 chunks
            full_pcm16 = b"".join(audio_chunks)
            
            # GPT returns 24kHz PCM16, we need to downsample to 8kHz for telephony
            # Step 1: Resample from 24kHz to 8kHz (reduce by factor of 3)
            pcm16_8k = audioop.ratecv(full_pcm16, 2, 1, 24000, 8000, None)[0]
            
            # Step 2: Convert PCM16 to μ-law (G.711)
            ulaw_bytes = audioop.lin2ulaw(pcm16_8k, 2)
            
            # Step 3: Wrap in WAV container (8kHz μ-law for telephony)
            final_wav = wav_wrap_ulaw(ulaw_bytes, sample_rate=8000, channels=1)
            
            t_total = _now_ms()
            total_duration = t_total - t_start
            METRICS["gpt_ms"].add(total_duration)
            
            print(f"  ✅ Complete in {total_duration}ms")
            print(f"     Text: {text_response[:100]}{'...' if len(text_response) > 100 else ''}")
            print(f"     Audio: {len(final_wav)} bytes (8kHz μ-law)")
            
            return final_wav
    
    except Exception as e:
        print(f"  ❌ Fatal error in generate_complete_gpt_voice: {e}")
        import traceback
        traceback.print_exc()
        
        # Return silence as fallback
        silent_1s = bytes([0xFF]) * 8000
        return wav_wrap_ulaw(silent_1s)

# ============================================================================
# FASTAPI APP & ENDPOINTS
# ============================================================================

app = FastAPI(title="Amazon Connect GPT Voice API")

@app.post("/api/scenario/store-audio")
async def api_scenario_store_audio(
    table_name: str = Form(...),
    scenario_id: str = Form(...),
    category: str = Form(""),
    short_key: str = Form(...),
    response_text: str = Form(...),
    audio_url: str = Form(...),
):
    """
    For a single scenario row:
    - ensure DynamoDB table + S3 bucket (same name as table)
    - download audio_url
    - upload to S3 bucket
    - store metadata in DynamoDB
    """
    table_name = table_name.strip()
    scenario_id = scenario_id.strip()
    short_key = short_key.strip()

    if not table_name:
        raise HTTPException(status_code=400, detail="Missing table_name")
    if not scenario_id or not short_key:
        raise HTTPException(status_code=400, detail="Missing scenario_id or short_key")
    if not audio_url:
        raise HTTPException(status_code=400, detail="Missing audio_url")

    bucket_name = table_name  # 1:1 mapping (table name == bucket name)

    # 1) Ensure AWS resources
    ensure_ddb_table(table_name)
    ensure_s3_bucket(bucket_name)

    # 2) Download audio from existing URL
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(audio_url)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail="audio_download_failed")
        audio_bytes = r.content

    # 3) Upload to our S3
    safe_sid = scenario_id.replace(" ", "_")
    safe_sk = short_key.replace(" ", "_")
    object_key = f"{safe_sid}/{safe_sk}-{uuid4().hex}.wav"

    s3.put_object(
        Bucket=bucket_name,
        Key=object_key,
        Body=audio_bytes,
        ContentType="audio/wav",
    )

    # Construct S3 URL
    s3_url = f"https://{bucket_name}.s3.{AWS_REGION}.amazonaws.com/{object_key}"

    # 4) Write to DynamoDB
    now_str = datetime.utcnow().isoformat() + "Z"
    dynamodb.put_item(
        TableName=table_name,
        Item={
            "scenario_id": {"S": scenario_id},
            "short_key": {"S": short_key},
            "category": {"S": category},
            "response": {"S": response_text},
            "audio_url": {"S": s3_url},
            "source_url": {"S": audio_url},
            "created_at": {"S": now_str},
        },
    )

    return {
        "status": "ok",
        "s3_url": s3_url,
        "s3_key": object_key,
        "table_name": table_name,
    }


@app.post("/scenario/save-text")
async def scenario_save_text(
    table_name: str = Form(...),
    rows_json: str = Form(...),
):
    """
    선택된 행들의 텍스트 정보를 DynamoDB에만 저장.
    - S3 업로드 / 오디오 생성은 하지 않음
    - audio_url 이 있으면 그대로 저장하고, 없으면 필드 생략
    """
    try:
        rows = json.loads(rows_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="rows_json must be valid JSON array")

    if not isinstance(rows, list) or not rows:
        raise HTTPException(status_code=400, detail="rows_json must be a non-empty list")

    # 테이블 없으면 자동 생성 (E 옵션)
    ensure_ddb_table(table_name)

    saved = 0
    for row in rows:
        if not isinstance(row, dict):
            continue

        scenario_id = (row.get("scenario_id") or "").strip()
        short_key   = (row.get("short_key") or "").strip()
        response    = (row.get("response") or "").strip()
        category    = (row.get("category") or "").strip()
        audio_url   = (row.get("audio_url") or "").strip()

        # 필수값 없으면 스킵 (B 옵션에서 "선택된 + 유효한 것만" 저장)
        if not scenario_id or not short_key or not response:
            continue

        item = {
            "scenario_id": {"S": scenario_id},
            "short_key":   {"S": short_key},
            "response":    {"S": response},
        }
        if category:
            item["category"] = {"S": category}
        # C 옵션: audio_url 이 있을 때만 저장
        if audio_url:
            item["audio_url"] = {"S": audio_url}

        dynamodb.put_item(TableName=table_name, Item=item)
        saved += 1

    return {"status": "ok", "saved": saved}


@app.post("/chat/gpt")
async def chat_gpt_endpoint(payload: dict):
    """
    Simple text chat with GPT - no audio.
    Used by the local UI for text-only conversations.
    """
    user_message = payload.get("message") or payload.get("text") or ""
    payload_system = payload.get("system_prompt")  # optional extra, if caller sends one
    temperature = payload.get("temperature", 0.6)

    # NEW: scenario override from caller
    scenario_prompt = (payload.get("scenario_prompt") or "").strip()
    scenario_id = (payload.get("scenario_id") or "").strip()  # not strictly needed yet

    # Decide system prompt:
    # 1) If caller sends scenario_prompt, use PROMPT_RULES_KR + that prompt
    # 2) Else fall back to FINAL_SYSTEM_PROMPT (rules + scenario_prompt.txt)
    # 3) Else (very unlikely) use generic helper
    if scenario_prompt:
        system_prompt = PROMPT_RULES_KR + "\n\n[시나리오 참고]\n" + scenario_prompt
    elif FINAL_SYSTEM_PROMPT:
        system_prompt = FINAL_SYSTEM_PROMPT
    elif payload_system:
        system_prompt = payload_system
    else:
        system_prompt = "You are a helpful assistant."
    
    if not user_message:
        raise HTTPException(400, "message is required")
    
    if not OPENAI_API_KEY:
        raise HTTPException(503, "OpenAI API key not configured")
    
    print(f"\n{'='*60}")
    print(f"[CHAT] User: {user_message}")
    
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4.1-nano",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": 500,
                "temperature": temperature,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        reply = data["choices"][0]["message"]["content"].strip()
        
        print(f"[CHAT] GPT: {reply[:100]}{'...' if len(reply) > 100 else ''}")
        print(f"{'='*60}\n")
        
        return {
            "ok": True,
            "reply": reply,
            "model": "gpt-4.1-nano",
        }
    except Exception as e:
        print(f"[CHAT] ❌ Error: {e}")
        raise HTTPException(500, f"Chat failed: {e}")

class GPTVoiceRequest(BaseModel):
    text: str
    voice: str = "alloy"
    owner: str = "default"

@app.post("/brain/gpt-voice/start")
async def start_gpt_voice(req: GPTVoiceRequest):
    """
    Generate complete GPT voice audio for Amazon Connect.
    
    This endpoint is SYNCHRONOUS - it waits for complete audio generation
    before returning. No streaming or partial files.
    
    Perfect for Amazon Connect contact flows that need a single, complete
    audio file URL.
    """
    t_request_start = _now_ms()
    
    try:
        prompt = req.text.strip()
        if not prompt:
            raise HTTPException(400, "text is required")
        
        print(f"\n{'='*60}")
        print(f"🎯 Amazon Connect Voice Request: '{prompt}'")
        
        # Generate COMPLETE audio (this blocks until done)
        with T() as t_gen:
            audio_wav = generate_complete_gpt_voice(prompt, voice=req.voice)
        
        print(f"  ✓ Audio generated in {t_gen.ms}ms")
        
        # Upload to S3
        job_id = str(uuid.uuid4())[:8]
        s3_key = f"{S3_PREFIX}/{time.strftime('%Y%m%d')}/{job_id}.wav"
        
        with T() as t_s3:
            s3.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Body=audio_wav,
                ContentType="audio/wav"
            )
        
        METRICS["s3_ms"].add(t_s3.ms)
        print(f"  ✓ Uploaded to S3 in {t_s3.ms}ms")
        
        # Generate presigned URL (Lambda will strip parameters for Amazon Connect)
        audio_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": s3_key},
            ExpiresIn=3600
        )
        
        t_request_end = _now_ms()
        total_ms = t_request_end - t_request_start
        METRICS["total_ms"].add(total_ms)
        
        pool_stats = ws_pool.get_stats()
        
        print(f"  ✅ Request complete in {total_ms}ms")
        print(f"  📊 Pool: {pool_stats['reused']} reused, {pool_stats['created']} created")
        print(f"{'='*60}\n")
        
        return JSONResponse({
            "ok": True,
            "status": "complete",  # Not "queued" - it's done!
            "job_id": job_id,
            "audio_url": audio_url,
            "prompt": prompt,
            "audio_size_bytes": len(audio_wav),
            "timings": {
                "total_ms": total_ms,
                "generation_ms": t_gen.ms,
                "s3_upload_ms": t_s3.ms,
                "connect_ms_ewma": METRICS["connect_ms"].get(),
                "gpt_ms_ewma": METRICS["gpt_ms"].get(),
                "s3_ms_ewma": METRICS["s3_ms"].get(),
            },
            "pool_stats": pool_stats
        })
    
    except Exception as e:
        print(f"❌ Error in start_gpt_voice: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Internal error: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint for Amazon Connect monitoring"""
    pool_stats = ws_pool.get_stats()
    return {
        "status": "healthy",
        "service": "amazon-connect-gpt-voice",
        "pool_available": pool_stats["pool_size"] > 0,
        "pool_stats": pool_stats,
        "timestamp": int(time.time())
    }

from fastapi.params import Query as QueryParam

def get_cache_key(
    owner: str,
    text: str,
    voice_id: str,
    ref_id: str | None = None,
    style: str | None = None,
) -> str:
    parts = [owner, text, voice_id]
    if ref_id:
        parts.append(f"ref={ref_id}")
    if style:
        parts.append(f"style={style}")
    content = "|".join(parts)
    return hashlib.md5(content.encode("utf-8")).hexdigest()


@app.post("/eleven")
async def eleven_tts_endpoint(
    text_q: Optional[str] = Query(None),
    body: Optional[dict] = Body(None),
    owner: Optional[str] = Query(None),
    voice_id: Optional[str] = Query(None),
    hq: bool = Query(False),
    extra: float = Query(3.0),
    use_cache: bool = Query(True),
):
    # normalize when called internally (not via HTTP)
    if isinstance(owner, QueryParam):
        owner = None
    if isinstance(voice_id, QueryParam):
        voice_id = None
    if isinstance(hq, QueryParam):
        hq = False
    if isinstance(extra, QueryParam):
        extra = 3.0
    if isinstance(use_cache, QueryParam):
        use_cache = True

    # 1) extract text + owner + voice_id from body/query
    if body is not None and "text" in body:
        text = body["text"]
        if not owner:
            owner = body.get("owner")
        if not voice_id:
            voice_id = body.get("voice_id")
        if "use_cache" in body:
            use_cache = body["use_cache"]
    elif text_q is not None:
        text = text_q
    else:
        raise HTTPException(400, "text is required")

    # 2) decide which voice to use
    if voice_id:
        voice_to_use = voice_id
        print(f"🔍 Using explicit voice_id={voice_id}")
    else:
        voice_to_use = get_active_voice_id(owner)
        print(f"🔍 owner={owner}, active voice={voice_to_use}")

    if not voice_to_use or len(voice_to_use) < 10:
        raise HTTPException(500, f"Invalid voice_id '{voice_to_use}'")

    # 3) optional extras from body (for future: reference/style)
    ref_id = None
    style = None
    if body is not None:
        ref_id = body.get("reference_id")
        style = body.get("style")

    # 4) now we can safely build the cache key
    cache_key = get_cache_key(owner or "default", text, voice_to_use, ref_id, style)

    # 5) cache check
    if use_cache and cache_key in AUDIO_CACHE:
        cached_url = AUDIO_CACHE[cache_key]
        print(f"⚡ Cache HIT! key={cache_key[:12]}...")
        return {
            "ok": True,
            "audio_url": cached_url,
            "mode": "hq" if hq else "tel",
            "owner": owner,
            "voice_id_used": voice_to_use,
            "cached": True,
        }

    print(f"🔄 Cache MISS. Generating: '{text[:50]}...'")

    # 4) Call ElevenLabs TTS
    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "Accept": "audio/wav",
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": MODEL_ID,
        "voice_settings": {
            "stability": 0.6,
            "similarity_boost": 0.8,
        },
    }
    el_url = f"{BASE_URL}/{voice_to_use}"
    print(f"🎤 ELEVEN TTS URL >>> {el_url}")
    
    el_resp = requests.post(el_url, headers=headers, data=json.dumps(payload))

    if el_resp.status_code == 404 and "voice_not_found" in el_resp.text:
        print(f"❌ Voice {voice_to_use} not found in ElevenLabs")
        raise HTTPException(404, f"Voice {voice_to_use} not found. Please select a valid voice.")
    
    # still bad? then fail
    if el_resp.status_code >= 300:
        print(f"❌ ElevenLabs API Error: {el_resp.status_code} - {el_resp.text}")
        raise HTTPException(el_resp.status_code, f"ElevenLabs API error: {el_resp.text}")

    
    if el_resp.status_code >= 300:
        print(f"❌ ElevenLabs API Error: {el_resp.status_code} - {el_resp.text}")
        raise HTTPException(el_resp.status_code, f"ElevenLabs API error: {el_resp.text}")
    
    raw_audio = el_resp.content
    ctype = el_resp.headers.get("Content-Type", "").lower()

    # 5) Process audio
    buf = io.BytesIO(raw_audio); buf.seek(0)
    looks_like_mp3 = raw_audio[:2] == b"\xff\xfb"
    if "audio/mpeg" in ctype or looks_like_mp3:
        seg = AudioSegment.from_file(buf, format="mp3")
    else:
        seg = AudioSegment.from_file(buf, format="wav")

    target_dbfs = -1.0
    change_needed = target_dbfs - seg.max_dBFS
    seg = seg.apply_gain(change_needed)
    if extra:
        seg = seg.apply_gain(extra)

    # HQ path
    if hq:
        out_buf = io.BytesIO()
        seg.export(out_buf, format="wav")
        out_buf.seek(0)
        key = f"{S3_PREFIX_ELEVEN}/{time.strftime('%Y%m%d')}_{uuid.uuid4().hex}_hq.wav"
        s3_url = upload_wav_to_s3(out_buf.getvalue(), key)
        
        if use_cache:
            AUDIO_CACHE[cache_key] = s3_url
            print(f"💾 Cached (HQ): {len(AUDIO_CACHE)} total")
        
        return {
            "ok": True,
            "audio_url": s3_url,
            "mode": "hq",
            "owner": owner,
            "voice_id_used": voice_to_use,
            "cached": False,
        }

    # Telephony path
    seg = seg.set_frame_rate(8000).set_channels(1)
    out_buf = io.BytesIO()
    seg.export(out_buf, format="wav", codec="pcm_mulaw")
    out_buf.seek(0)

    key = f"{S3_PREFIX_ELEVEN}/{time.strftime('%Y%m%d')}_{uuid.uuid4().hex}.wav"
    s3_url = upload_wav_to_s3(out_buf.getvalue(), key)

    if use_cache:
        AUDIO_CACHE[cache_key] = s3_url
        print(f"💾 Cached: {len(AUDIO_CACHE)} total")

    return {
        "ok": True,
        "audio_url": s3_url,
        "mode": "tel",
        "owner": owner,
        "voice_id_used": voice_to_use,
        "cached": False,
    }

@app.post("/eleven/register")
async def register_eleven_voice(
    owner: str = Form(...),
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    voice_name: Optional[str] = Form(None),  # Human-friendly name
    set_active: bool = Form(True),  # Whether to make this the active voice
):
    """
    Register a new voice for an owner.
    - Keeps all previous voices unless set_active=True and you want to delete old ones
    - set_active=True makes this voice the default for the owner
    """
    if not ELEVEN_API_KEY:
        raise HTTPException(500, "ELEVENLABS_API_KEY not set")

    audio_bytes = await file.read()

    # Initialize owner's voice list if needed
    if owner not in VOICE_REGISTRY:
        VOICE_REGISTRY[owner] = []

    # Upload to ElevenLabs
    files = {
        "files": (file.filename, audio_bytes, file.content_type or "audio/wav"),
    }
    eleven_name = name or voice_name or f"custom_{owner}_{uuid.uuid4().hex[:8]}"
    data = {
        "name": eleven_name,
        "model_id": MODEL_ID,
    }
    headers = {"xi-api-key": ELEVEN_API_KEY}

    print(f"📤 Registering new voice for owner={owner}, name={eleven_name}")
    resp = requests.post(
        "https://api.elevenlabs.io/v1/voices/add",
        headers=headers,
        data=data,
        files=files,
    )
    if resp.status_code >= 300:
        raise HTTPException(resp.status_code, resp.text)

    info = resp.json()
    voice_id = info.get("voice_id")
    if not voice_id:
        raise HTTPException(500, "No voice_id returned from ElevenLabs")

    # Add to registry
    voice_entry = {
        "voice_id": voice_id,
        "name": voice_name or eleven_name,
        "eleven_name": eleven_name,
        "timestamp": time.time(),
        "active": set_active,
    }

    # If set_active, mark all others as inactive
    if set_active:
        for v in VOICE_REGISTRY[owner]:
            v["active"] = False

    VOICE_REGISTRY[owner].append(voice_entry)
    
    # Save to disk
    save_registry(VOICE_REGISTRY)
    
    print(f"✅ Registered voice_id={voice_id} for owner={owner}")
    print(f"📋 Owner {owner} now has {len(VOICE_REGISTRY[owner])} voice(s)")

    return {
        "ok": True,
        "owner": owner,
        "voice_id": voice_id,
        "name": voice_name or eleven_name,
        "active": set_active,
        "total_voices": len(VOICE_REGISTRY[owner]),
    }

@app.get("/eleven/voices/{owner}")
async def list_owner_voices(owner: str):
    """List all voices for an owner."""
    if owner not in VOICE_REGISTRY:
        return {"ok": True, "owner": owner, "voices": [], "count": 0}
    
    voices = VOICE_REGISTRY[owner]
    active_voice = next((v for v in voices if v.get("active")), voices[-1] if voices else None)
    
    return {
        "ok": True,
        "owner": owner,
        "voices": voices,
        "count": len(voices),
        "active_voice": active_voice,
    }

DETERMINISTIC_TTS_OVERRIDES = {}

# At the top of app.py, replace VOICE_REGISTRY initialization:
REGISTRY_FILE = Path("voice_registry.json")

def load_registry() -> dict:
    """Load voice registry from disk."""
    if REGISTRY_FILE.exists():
        try:
            with open(REGISTRY_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load registry: {e}")
    return {}

def save_registry(registry: dict):
    """Save voice registry to disk."""
    try:
        with open(REGISTRY_FILE, "w") as f:
            json.dump(registry, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to save registry: {e}")

# Initialize registry from disk
VOICE_REGISTRY = load_registry()

def pick_tts_text(user_text: str, gpt_text: str) -> str:
    """Use GPT's response unless there's a manual override."""
    if not user_text:
        return gpt_text
    
    user_text_normalized = user_text.strip().lower()
    
    if user_text_normalized in DETERMINISTIC_TTS_OVERRIDES:
        mapped = DETERMINISTIC_TTS_OVERRIDES[user_text_normalized]
        print(f"[OVERRIDE] '{user_text}' -> '{mapped}'")
        return mapped
    
    return gpt_text

def get_active_voice_id(owner: str | None) -> str:
    """Always use default voice - no registry lookup."""
    return DEFAULT_ELEVEN_VOICE_ID
    
# def get_active_voice_id(owner: str | None) -> str:
#     """Get the active voice_id for an owner, or default."""
#     if not owner or owner not in VOICE_REGISTRY or not VOICE_REGISTRY[owner]:
#         return DEFAULT_ELEVEN_VOICE_ID
    
#     # Find active voice
#     active = next((v for v in VOICE_REGISTRY[owner] if v.get("active")), None)
    
#     # If no active voice, use the most recent one
#     if not active:
#         active = VOICE_REGISTRY[owner][-1]
    
#     return active["voice_id"]

def gpt_brain_reply(user_text: str, system_prompt: str | None = None) -> str:
    """
    Call GPT (text-only) and get the reply text.
    """
    if not OPENAI_API_KEY:
        return "현재 안내를 준비 중입니다. 어떤 점을 도와드릴까요?"
    
    default_prompt = "당신은 한국어 콜센터 상담원입니다. 짧고 공손하게 대답하세요."
    
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4.1-nano",
                "messages": [
                    {"role": "system", "content": FINAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                "max_tokens": 120,
                "temperature": 0.6,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[GPT Error] {e}")
        raise

async def gpt_voice_brain_reply(user_text: str, system_prompt: str | None = None, temperature: float = 0.6) -> tuple[str, bytes]:
    """
    Call GPT Voice realtime API for conversational responses.
    NOW USES CONNECTION POOL for much faster responses!
    
    Note: GPT Voice API requires temperature >= 0.6
    Use 0.6 for most deterministic responses (minimum allowed)
    
    Returns: (reply_text, audio_pcm_24k)
    """
    if not OPENAI_API_KEY:
        return "현재 안내를 준비 중입니다.", b""
    
    # Clamp temperature to valid range (0.6 - 1.2)
    temperature = max(0.6, min(1.2, temperature))
    
    default_prompt = FINAL_SYSTEM_PROMPT
    
    audio_chunks = []
    transcript_parts = []
    
    try:
        print(f"[GPT Voice] Using connection pool")
        
        # Use the connection pool (same as synchronous endpoint)
        # This reuses existing WebSocket connections = MUCH FASTER
        with ws_pool.get_connection() as ws:
            print(f"[GPT Voice] Got connection from pool")
            
            # Session config
            session_update = {
                "type": "session.update",
                "session": {
                    "instructions": FINAL_SYSTEM_PROMPT,
                    "voice": "alloy",
                    "modalities": ["text", "audio"],
                    "turn_detection": None,  # Manual turn control
                    "temperature": temperature,
                }
            }
            ws.send(json.dumps(session_update))
            print(f"[GPT Voice] Session configured")
            
            # Create conversation item with user's text
            ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": user_text
                    }]
                }
            }))
            print(f"[GPT Voice] User message sent: {user_text}")
            
            # Trigger response
            ws.send(json.dumps({
                "type": "response.create"
            }))
            print(f"[GPT Voice] Response requested")
            
            # Collect response
            response_complete = False
            event_count = 0
            
            while not response_complete:
                try:
                    # Use blocking recv with timeout
                    ws.settimeout(30.0)
                    message = ws.recv()
                    event = json.loads(message)
                    event_type = event.get("type", "")
                    event_count += 1
                    
                    # Debug: print event types
                    if event_count <= 20:
                        print(f"[GPT Voice Event {event_count}] {event_type}")
                    
                    # Handle errors
                    if event_type == "error":
                        error_msg = event.get("error", {})
                        print(f"[GPT Voice Error] {error_msg}")
                        break
                    
                    # Collect transcript (multiple possible event types)
                    if event_type == "response.text.delta":
                        delta = event.get("delta", "")
                        transcript_parts.append(delta)
                        print(f"[GPT Voice] Text delta: {delta}")
                    
                    elif event_type == "response.text.done":
                        final_text = event.get("text", "")
                        if final_text:
                            transcript_parts = [final_text]  # Replace with final
                        print(f"[GPT Voice] Text done: {final_text}")
                    
                    # AUDIO TRANSCRIPT events
                    elif event_type == "response.audio_transcript.delta":
                        delta = event.get("delta", "")
                        transcript_parts.append(delta)
                        if event_count <= 20:
                            print(f"[GPT Voice] Audio transcript delta: {delta}")
                    
                    elif event_type == "response.audio_transcript.done":
                        final_text = event.get("transcript", "")
                        if final_text:
                            transcript_parts = [final_text]  # Replace with final
                        print(f"[GPT Voice] Audio transcript done: {final_text}")
                    
                    elif event_type == "response.output_item.done":
                        # Alternative: check for text in output items
                        item = event.get("item", {})
                        content = item.get("content", [])
                        for c in content:
                            if c.get("type") == "text":
                                text = c.get("text", "")
                                if text:
                                    transcript_parts.append(text)
                                    print(f"[GPT Voice] Output item text: {text}")
                    
                    # Collect audio
                    elif event_type == "response.audio.delta":
                        audio_b64 = event.get("delta")
                        if audio_b64:
                            audio_chunks.append(base64.b64decode(audio_b64))
                    
                    elif event_type == "response.audio.done":
                        print(f"[GPT Voice] Audio complete: {len(audio_chunks)} chunks")
                    
                    # End conditions
                    elif event_type == "response.done":
                        print(f"[GPT Voice] Response complete")
                        response_complete = True
                        break
                        
                except WebSocketTimeoutException:
                    print("[GPT Voice] Timeout waiting for response")
                    break
                except Exception as e:
                    print(f"[GPT Voice] Error in recv loop: {e}")
                    break
        
        transcript = "".join(transcript_parts).strip()
        audio_bytes = b"".join(audio_chunks)
        
        print(f"[GPT Voice] Final transcript: '{transcript}' ({len(transcript)} chars)")
        print(f"[GPT Voice] Final audio: {len(audio_bytes)} bytes")
        
        if not transcript:
            # Fallback if no transcript
            print("[GPT Voice Warning] No transcript received, using fallback")
            transcript = "안녕하세요! 무엇을 도와드릴까요?"
        
        return transcript, audio_bytes
        
    except Exception as e:
        print(f"[GPT Voice Error] {e}")
        import traceback
        traceback.print_exc()
        return "죄송합니다. 다시 말씀해 주시겠어요?", b""


@app.post("/brain/gpt-eleven")
async def voice_brain_gpt_eleven(payload: dict):
    """
    Old endpoint using GPT text API (not voice).
    Kept for backwards compatibility.
    """
    user_text = payload.get("text") or payload.get("user_text") or ""
    owner = payload.get("owner") or "default"
    use_cache = payload.get("use_cache", True)
    voice_id = payload.get("voice_id")
    hq = payload.get("hq", False)
    extra = payload.get("extra", 3.0)

    # 1) call GPT text API and time it
    gpt_start = time.time()
    try:
        gpt_reply = gpt_brain_reply(user_text)
    except Exception as e:
        print(f"[BRAIN] ❌ GPT failed: {e}")
        gpt_reply = "죄송합니다. 다시 말씀해 주시겠어요?"
    gpt_ms = round((time.time() - gpt_start) * 1000)

    # 2) pick deterministic TTS text
    tts_text = pick_tts_text(user_text, gpt_reply)

    # 3) call our own /eleven
    eleven_body = {
        "text": tts_text,
        "owner": owner,
        "use_cache": use_cache,
    }
    if voice_id:
        eleven_body["voice_id"] = voice_id

    eleven_resp = await eleven_tts_endpoint(
        body=eleven_body,
        owner=owner,
        voice_id=voice_id,
        hq=hq,
        extra=extra,
        use_cache=use_cache,
    )

    audio_url = eleven_resp["audio_url"]
    vid_used = eleven_resp.get("voice_id_used")
    cached = eleven_resp.get("cached", False)

    tts_ms = 0
    total_ms = gpt_ms + tts_ms

    return {
        "ok": True,
        "reply_text": gpt_reply,
        "audio_url": audio_url,
        "voice_id_used": vid_used,
        "cached": cached,
        "mode": payload.get("mode", "tel"),
        "owner": owner,
        "timings": {
            "gpt_ms": gpt_ms,
            "tts_ms": tts_ms,
            "total_ms": total_ms,
        },
    }

@app.post("/brain/gpt-voice-eleven")
async def voice_brain_gpt_voice_eleven(payload: dict):
    start_time = time.time()

    user_text = payload.get("text") or payload.get("user_text") or ""
    owner = payload.get("owner") or "default"
    use_cache = payload.get("use_cache", True)
    voice_id = payload.get("voice_id")
    hq = payload.get("hq", False)
    extra = payload.get("extra", 3.0)

    if not user_text:
        raise HTTPException(400, "text or user_text is required")

    print(f"\n{'='*60}")
    print(f"[BRAIN] User: {user_text}")
    print(f"[BRAIN] Owner: {owner}")

    # 1) Use normal GPT text brain (no Realtime)
    gpt_start = time.time()
    try:
        gpt_reply = gpt_brain_reply(user_text)
        gpt_ms = round((time.time() - gpt_start) * 1000)
        print(f"[BRAIN] GPT text ({gpt_ms}ms): {gpt_reply}")
    except Exception as e:
        print(f"[BRAIN] ❌ GPT failed: {e}")
        gpt_reply = "죄송합니다. 다시 말씀해 주시겠어요?"
        gpt_ms = 0

    if not gpt_reply:
        gpt_reply = "죄송합니다. 응답을 생성할 수 없습니다."

    # 2) Pick TTS text
    tts_text = pick_tts_text(user_text, gpt_reply)

    # 3) Generate with ElevenLabs
    print(f"[BRAIN] 🎤 Generating TTS: {tts_text[:50]}...")
    tts_start = time.time()

    eleven_resp = await eleven_tts_endpoint(
        body={"text": tts_text, "owner": owner, "use_cache": use_cache},
        owner=owner,
        voice_id=voice_id,
        hq=hq,
        extra=extra,
        use_cache=use_cache,
    )

    tts_ms = round((time.time() - tts_start) * 1000)
    cached = eleven_resp.get("cached", False)
    status = "⚡ CACHED" if cached else f"🔄 GENERATED ({tts_ms}ms)"
    print(f"[BRAIN] TTS: {status}")

    total_ms = round((time.time() - start_time) * 1000)
    print(f"[BRAIN] ✅ Total: {total_ms}ms")
    print(f"{'='*60}\n")

    return {
        "ok": True,
        "reply_text": gpt_reply,
        "tts_text": tts_text,
        "audio_url": eleven_resp["audio_url"],
        "voice_id_used": eleven_resp.get("voice_id_used"),
        "cached": cached,
        "owner": owner,
        "timings": {
            "gpt_ms": gpt_ms,
            "tts_ms": tts_ms,
            "total_ms": total_ms,
        },
    }


@app.get("/metrics")
def voice_metrics():
    """Get performance metrics"""
    return {
        "model": REALTIME_MODEL,
        "total_ms_ewma": METRICS["total_ms"].get(),
        "gpt_ms_ewma": METRICS["gpt_ms"].get(),
        "s3_ms_ewma": METRICS["s3_ms"].get(),
        "connect_ms_ewma": METRICS["connect_ms"].get(),
        "pool_stats": ws_pool.get_stats(),
        "timestamp": int(time.time())
    }

@app.on_event("shutdown")
def shutdown_event():
    """Clean shutdown"""
    print("🛑 Shutting down WebSocket pool...")
    ws_pool.shutdown()
    print("✅ Shutdown complete")

# ============================================================================
# INFO
# ============================================================================

@app.get("/")
def root():
    return {
        "service": "Amazon Connect GPT Voice API",
        "version": "2.0",
        "mode": "synchronous",
        "model": REALTIME_MODEL,
        "endpoints": {
            "generate_voice": "/voice/brain/gpt-voice/start",
            "health": "/health",
            "metrics": "/voice/metrics"
        },
        "note": "This API generates COMPLETE audio files synchronously - no streaming. Perfect for Amazon Connect."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)