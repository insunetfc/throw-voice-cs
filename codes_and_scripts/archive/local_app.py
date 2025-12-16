#!/usr/bin/env python3
import os
from fastapi import FastAPI, Form, UploadFile, File, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uvicorn
from fastapi.responses import FileResponse
import tempfile
import shutil
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import json
from fastapi.responses import Response, RedirectResponse
from starlette.datastructures import FormData
import boto3
from uuid import uuid4
from datetime import datetime
from fastapi.responses import PlainTextResponse
import io


s3 = boto3.client("s3")
AUDIO_BUCKET = "your-s3-bucket-name"    # set your bucket
AUDIO_FOLDER = "chat_multi_tts"         # prefix

# your existing endpoints
NIPA_BASE = os.getenv("NIPA_BASE", "https://honest-trivially-buffalo.ngrok-free.app")
NIPA_PHONE_CALL = f"{NIPA_BASE}/phone/call"
NIPA_PHONE_INTRO = f"{NIPA_BASE}/phone/generate-intro"
NIPA_TTS_REF = f"{NIPA_BASE}/tts/reference"
NIPA_TTS_SYN = f"{NIPA_BASE}/tts/synthesize2"
NIPA_CHATBOT_RESPOND = f"{NIPA_BASE}/chatbot/respond"
NIPA_AUTH = os.getenv("NIPA_AUTH", "Bearer YOUR_TOKEN")
FAVICON_URL = "https://static.thenounproject.com/png/microphone-icon-1681031-512.png"
ELEVEN_BASE = os.getenv(
    "ELEVEN_BASE",
    "https://honest-trivially-buffalo.ngrok-free.app/voice/eleven"  # Direct TTS endpoint
)
ELEVEN_BRAIN_BASE = os.getenv(
    "ELEVEN_BRAIN_BASE", 
    "https://honest-trivially-buffalo.ngrok-free.app/voice/brain/gpt-voice-eleven"  # GPT + TTS
)
GPT_VOICE_BASE = os.getenv(
    "GPT_VOICE_BASE",
    "https://honest-trivially-buffalo.ngrok-free.app/voice/brain/gpt-voice/start"
)
_cached_icon: bytes | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cached_icon
    async with httpx.AsyncClient() as client:
        r = await client.get(FAVICON_URL)
        r.raise_for_status()
        _cached_icon = r.content
    # 👇 very important
    yield

app = FastAPI(title="Local TTS / Call UI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

import re

def normalize_scenario_id(raw: str, fallback="SCENARIO"):
    if not raw:
        return fallback

    # 1. Lowercase everything first
    s = raw.lower()

    # 2. Keep only letters, digits, and spaces/underscores
    s = re.sub(r"[^a-z0-9\s_]+", " ", s)

    # 3. Replace whitespace with a single underscore
    s = re.sub(r"[\s]+", "_", s)

    # 4. Remove leading/trailing underscores
    s = s.strip("_")

    # 5. Uppercase final form
    s = s.upper()

    # 6. Fallback if empty
    if not s:
        return fallback

    return s


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


@app.post("/api/scenario/export-txt")
async def api_scenario_export_txt(payload: dict = Body(...)):
    """
    Build a TXT export of all scenario rows.
    payload = {
        "table_name": "...",
        "rows": [
            {
              "scenario_id": "...",
              "category": "...",
              "short_key": "...",
              "response_text": "...",
              "audio_url": "..."
            },
            ...
        ]
    }
    """
    table_name = (payload.get("table_name") or "").strip()
    rows = payload.get("rows") or []

    lines: list[str] = []
    lines.append("# Scenario rows export")
    if table_name:
        lines.append(f"# Table: {table_name}")
    lines.append("")

    for i, row in enumerate(rows, start=1):
        scenario_id = row.get("scenario_id", "") or ""
        category    = row.get("category", "") or ""
        short_key   = row.get("short_key", "") or ""
        response    = row.get("response_text", "") or ""
        audio_url   = row.get("audio_url", "") or ""

        lines.append(f"# {i}")
        lines.append(f"시나리오 ID: {scenario_id}")
        lines.append(f"카테고리: {category}")
        lines.append(f"대답: {response}")
        lines.append(f"Short Key: {short_key}")        
        lines.append(f"Audio URL: {audio_url}")
        lines.append("")

    content = "\n".join(lines)
    filename = f"{table_name}_scenario_rows.txt" if table_name else "scenario_rows.txt"

    return PlainTextResponse(
        content,
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )

@app.post("/api/call_eleven")
async def api_call(phone_number: str = Form(...)):
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f'{NIPA_PHONE_CALL}_eleven',
            data={"phone_number": phone_number},
            headers={"Authorization": NIPA_AUTH},
        )
    return JSONResponse(status_code=resp.status_code, content=resp.json())

@app.post("/api/call_gpt")
async def api_call(phone_number: str = Form(...)):
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f'{NIPA_PHONE_CALL}_gpt',
            data={"phone_number": phone_number},
            headers={"Authorization": NIPA_AUTH},
        )
    return JSONResponse(status_code=resp.status_code, content=resp.json())

@app.post("/api/call_fish")
async def api_call(phone_number: str = Form(...)):
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f'{NIPA_PHONE_CALL}_fish',
            data={"phone_number": phone_number},
            headers={"Authorization": NIPA_AUTH},
        )
    return JSONResponse(status_code=resp.status_code, content=resp.json())


@app.post("/api/generate-intro")
async def api_generate_intro(
    phone_number: str = Form(...),
    display_name: str = Form(""),
):
    async with httpx.AsyncClient(timeout=12.0) as client:
        resp = await client.post(
            NIPA_PHONE_INTRO,
            data={"phone_number": phone_number, "display_name": display_name},
            headers={"Authorization": NIPA_AUTH},
        )
    return JSONResponse(status_code=resp.status_code, content=resp.json())


@app.post("/api/tts")
async def api_tts(
    text: str = Form(...),
    temperature: float = Form(0.8),
    ref_audio: UploadFile | None = File(None),
):
    async with httpx.AsyncClient(timeout=30.0) as client:
        ref_id = None
        if ref_audio is not None:
            files = {
                "ref_wav": (
                    ref_audio.filename,
                    await ref_audio.read(),
                    ref_audio.content_type or "audio/wav",
                )
            }
            ref_resp = await client.post(
                NIPA_TTS_REF,
                files=files,
                headers={"Authorization": NIPA_AUTH},
            )
            if ref_resp.status_code != 200:
                return JSONResponse(
                    status_code=503,
                    content={"error": "server_down"},
                )
            ref_id = ref_resp.json().get("ref_id")

        payload = {"text": text, "temperature": temperature}
        if ref_id:
            payload["ref_id"] = ref_id

        resp = await client.post(
            NIPA_TTS_SYN,
            json=payload,
            headers={"Authorization": NIPA_AUTH, "Content-Type": "application/json"},
        )

    if resp.status_code != 200:
        return JSONResponse(status_code=503, content={"error": "server_down"})

    return {"status": "ok", "result": resp.json()}

@app.get("/api/tts-download")
async def api_tts_download(url: str, filename: str = "tts_output.wav"):
    """
    Generic download proxy for any TTS audio URL (S3, NIPA, etc.).
    Forces download as an attachment from the same origin.
    """
    async with httpx.AsyncClient(timeout=25.0) as client:
        r = await client.get(url)
        r.raise_for_status()

    # Serve from our own origin with 'attachment'
    return StreamingResponse(
        io.BytesIO(r.content),
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )

@app.get("/api/tts-eleven/download")
async def api_tts_eleven_download(url: str, filename: str | None = None):
    # download the remote wav/mp3 and return it
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp_path = tmp.name
    tmp.close()

    async with httpx.AsyncClient(timeout=25.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            f.write(r.content)

    # if frontend gives a filename (e.g., "01.wav"), use that
    if not filename:
        filename = url.split("/")[-1].split("?")[0] or "tts_eleven.wav"

    return StreamingResponse(
        open(tmp_path, "rb"),
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )

from fastapi import Form

@app.post("/api/scenario/save-text")
async def api_scenario_save_text(payload: dict = Body(...)):
    """
    Proxy: save multiple scenario rows (text + optional audio_url) into remote DDB.
    Frontend sends JSON:
      { "table_name": "...", "rows_json": "[{...}, {...}]" }

    Backend expects form fields.
    """
    table_name = (payload.get("table_name") or "").trim() if hasattr(str, "trim") else (payload.get("table_name") or "").strip()
    rows_json  = payload.get("rows_json")

    if not table_name or not rows_json:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "error": "missing table_name or rows_json"},
        )

    async with httpx.AsyncClient(timeout=40.0) as client:
        resp = await client.post(
            f"{NIPA_BASE}/voice/scenario/save-text",  # ✅ no /api
            data={                                   # ✅ Form data, not JSON
                "table_name": table_name,
                "rows_json": rows_json,
            },
            headers={"Authorization": NIPA_AUTH},
        )

    try:
        content = resp.json()
    except Exception:
        content = {"status": "error", "error": resp.text}

    # ✅ Normalize so frontend always sees status: "ok" when HTTP 200
    if resp.status_code == 200 and content.get("status") in (None, "ok"):
        content.setdefault("status", "ok")

    return JSONResponse(status_code=resp.status_code, content=content)

@app.post("/api/scenario/store-audio")
async def api_scenario_store_audio(
    table_name: str = Form(...),
    row_index: int = Form(...),
    text: str = Form(...),
    audio_url: str = Form(...),
):
    """
    Proxy: store scenario TTS info (text + audio_url) into remote DDB via NIPA backend.
    """
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            f"{NIPA_BASE}/voice/api/scenario/store-audio",
            data={
                "table_name": table_name,
                "row_index": row_index,
                "text": text,
                "audio_url": audio_url,
            },
            headers={"Authorization": NIPA_AUTH},
        )

    # pass through status + json
    try:
        content = resp.json()
    except Exception:
        content = {"error": resp.text}

    return JSONResponse(status_code=resp.status_code, content=content)


@app.post("/api/tts-eleven")
async def api_tts_eleven(
    text: str = Form(...),
    style: str = Form(""),
    voice_id: str = Form(""),
    use_cache: bool = Form(True),
):
    # If no voice_id provided, don't send owner (backend will use default)
    # This avoids the "Active voice missing" warning and retry latency
    payload = {
        "text": text,
        "use_cache": use_cache,
    }
    
    # Only include owner if we're NOT providing an explicit voice_id
    # This makes the backend use default voice instead of looking up active voice
    if not voice_id:
        # Don't send owner - let backend use its default voice
        pass
    else:
        # User selected a specific voice
        payload["voice_id"] = voice_id
        payload["owner"] = "manager1"  # Include owner for tracking
    
    if style:
        payload["style"] = style

    async with httpx.AsyncClient(timeout=25.0) as client:
        resp = await client.post(
            ELEVEN_BASE,
            json=payload,
            headers={"Content-Type": "application/json"},
        )

    if resp.status_code != 200:
        return JSONResponse(status_code=503, content={"error": "eleven_failed"})

    data = resp.json()
    # unify key names a bit
    audio_url = data.get("audio_url") or data.get("s3_url") or data.get("url")
    return {
        "status": "ok",
        "audio_url": audio_url,
        "raw": data,
    }

@app.post("/api/tts-gpt-voice")
async def api_tts_gpt_voice(
    text: str = Form(...),
    temperature: float = Form(0.6),
):
    """GPT Voice TTS - generates audio using GPT's voice model"""
    payload = {
        "text": text,
        "user_text": text,
        "owner": "default",
        "temperature": temperature,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            GPT_VOICE_BASE,
            json=payload,
            headers={"Content-Type": "application/json"},
        )

    if resp.status_code != 200:
        return JSONResponse(status_code=503, content={"error": "gpt_voice_failed"})

    data = resp.json()
    audio_url = data.get("audio_url")
    reply_text = data.get("reply_text", "")
    
    return {
        "status": "ok",
        "audio_url": audio_url,
        "reply_text": reply_text,
        "raw": data,
    }

@app.post("/api/chat-gpt")
async def api_chat_gpt(
    message: str = Form(...),
    scenario_id: str = Form(""),
    scenario_prompt: str = Form(""),
):
    """Text chat with GPT (no audio) - proxies to remote server, with optional scenario override."""
    payload = {
        "message": message,
    }

    # Only send these if they exist – your backend can override its default scenario
    if scenario_id:
        payload["scenario_id"] = scenario_id
    if scenario_prompt:
        payload["scenario_prompt"] = scenario_prompt

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{NIPA_BASE}/voice/chat/gpt",
            json=payload,
            headers={"Content-Type": "application/json"},
        )

    if resp.status_code != 200:
        return JSONResponse(
            status_code=resp.status_code,
            content={"error": "GPT chat failed"},
        )

    data = resp.json()
    reply = data.get("reply", "")

    return {
        "status": "ok",
        "reply": reply,
    }

@app.post("/api/chat-chajipsa")
async def api_chat_chajipsa(
    message: str = Form(...),
):
    """
    Text chat with 차집사 챗봇 (no audio).
    Proxies to NIPA /chatbot/respond, which returns agent_response.
    """
    payload = {"text": message}

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            NIPA_CHATBOT_RESPOND,
            json=payload,
            headers={"Content-Type": "application/json"},
        )

    if resp.status_code != 200:
        return JSONResponse(
            status_code=resp.status_code,
            content={"error": "chajipsa chat failed"},
        )

    data = resp.json()
    reply = (
        data.get("agent_response")
        or data.get("reply")
        or data.get("text")
        or ""
    )

    return {
        "status": "ok",
        "reply": reply,
        "raw": data,
    }


@app.post("/api/chat-multi")
async def api_chat_multi(
    engine: str = Form(...),     # "retriever" or "gpt4o"
    text: str = Form(...),
    top_k: int = Form(10),
    scenario_id: str = Form(""),
    scenario_prompt: str = Form(""),
):
    """
    Unified top-K response generator for:
      - retriever (차집사 /chatbot/respond)
      - GPT (/voice/chat/gpt)
    """
    text = (text or "").strip()
    if not text:
        return {"status": "error", "error": "empty_text"}
    
    try:
        top_k = int(top_k)
    except ValueError:
        top_k = 3
    top_k = max(1, min(top_k, 10))

    # --------------------- RETRIEVER / 차집사 ---------------------
    if engine == "retriever":
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    NIPA_CHATBOT_RESPOND,
                    json={"text": text, "top_k": top_k},
                    headers={"Content-Type": "application/json"},
                )

            if resp.status_code != 200:
                return {
                    "status": "error",
                    "error": f"retriever_http_{resp.status_code}",
                }

            data = resp.json()
            candidates = data.get("top_k_candidates")

            items = []
            if isinstance(candidates, list) and candidates:
                for i, c in enumerate(candidates[:top_k]):
                    items.append({
                        "rank": c.get("rank", i + 1),
                        "response_text": (
                            c.get("response_text")
                            or c.get("agent_response")
                            or ""
                        ),
                    })
            else:
                # Fallback: agent_response / reply / text
                base = (
                    data.get("agent_response")
                    or data.get("reply")
                    or data.get("text")
                    or ""
                )

                items = []

                # Case 1: backend already returns a Python list
                if isinstance(base, list):
                    for i, s in enumerate(base[:top_k]):
                        if isinstance(s, dict):
                            txt = (
                                s.get("response_text")
                                or s.get("agent_response")
                                or s.get("text")
                                or str(s)
                            )
                        else:
                            txt = str(s)
                        items.append({"rank": i + 1, "response_text": txt})

                # Case 2: string that might be a JSON array: '["...", "..."]'
                elif isinstance(base, str):
                    try:
                        arr = json.loads(base)
                        if isinstance(arr, list) and arr:
                            for i, s in enumerate(arr[:top_k]):
                                txt = s if isinstance(s, str) else str(s)
                                items.append({"rank": i + 1, "response_text": txt})
                        else:
                            items.append({"rank": 1, "response_text": base})
                    except Exception:
                        items.append({"rank": 1, "response_text": base})

                # Case 3: anything else – just stringify
                else:
                    items.append({"rank": 1, "response_text": str(base)})

            return {"status": "ok", "engine": "retriever", "items": items}

        except Exception as e:
            return {"status": "error", "error": f"retriever_exc_{e}"}

    # --------------------- GPT / GPT-4o-nano style ----------------
    else:
        try:
            # If there's an explicit scenario_prompt, use that as description.
            # Otherwise, treat the user's text as the ad-hoc scenario description.            
            scenario_desc = scenario_prompt.strip()
            eff_scen_id = "SCENARIO" 
            user_message = (text or "").strip()

            try:
                tk = int(top_k)
            except ValueError:
                tk = 10
            tk = max(1, min(tk, 50))

            prompt = f"""
You are a call-center scenario script generator.

Read the following scenario description and propose {top_k} candidate responses
that an outbound call agent could speak.

Return ONLY valid JSON with this exact structure (no markdown, no explanation):

{{
  "scenario_id": "<INTENT_LABEL>",
  "items": [
    {{
      "category": "짧은 한국어 라벨, 예: 빠른 종료형, 예약 콜백, 인사, 안내",
      "short_key": "short snake_case identifier in English, e.g. fast_close, callback_offer, greet_default",
      "response": "natural Korean sentence suitable for spoken call-center dialog"
    }}
  ]
}}

Rules:
- Replace <INTENT_LABEL> with a simple 1-2 word UPPERCASE English intent classification of the message: "{user_message}".
- response must be in Korean, polite, and suitable for telephone speech.
- category MUST be a short Korean label (e.g. 빠른 종료형, 예약 콜백).
- short_key MUST be lowercase snake_case in English.
- Do NOT include comments, text outside JSON, or trailing commas.

Scenario description:
{scenario_prompt}
""".strip()

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{NIPA_BASE}/voice/chat/gpt",
                    json={"message": prompt},
                    headers={"Content-Type": "application/json"},
                )

            if resp.status_code != 200:
                return {
                    "status": "error",
                    "error": f"gpt_http_{resp.status_code}",
                }

            raw_reply = resp.json().get("reply", "").strip()

            # Try to parse GPT JSON
            try:
                parsed = json.loads(raw_reply)
            except Exception as e:
                return {
                    "status": "error",
                    "error": f"gpt_json_error: {e}",
                    "raw": raw_reply,
                }

            raw_id = parsed.get("scenario_id", "").strip()
            base_id = normalize_scenario_id(raw_id, fallback=eff_scen_id)
            if not base_id:
                base_id = eff_scen_id  # fallback, e.g. "ADHOC"

            base_id = base_id.upper()

            items: list[dict] = []
            for i, it in enumerate(parsed.get("items", [])):
                # per-row scenario ID: e.g. BUSY_01, BUSY_02, ...
                idx = i + 1
                row_id = f"{base_id}_{i+1:02d}"

                items.append(
                    {
                        "rank": idx,
                        "scenario_id": row_id,
                        "category": it.get("category", ""),
                        "short_key": it.get("short_key", ""),
                        "response_text": it.get("response", ""),
                    }
                )

            return {
                "status": "ok",
                "engine": "gpt4o",
                "items": items,
            }

        except Exception as e:
            return {"status": "error", "error": f"gpt_exc_{e}"}

@app.get("/api/eleven/voices")
async def api_eleven_voices(owner: str = "manager1"):
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(f"{NIPA_BASE}/voice/eleven/voices/{owner}")
    if resp.status_code != 200:
        return JSONResponse(status_code=resp.status_code, content={"error": resp.text})
    return resp.json()


# upload reference WAV to bridge → ElevenLabs
@app.post("/api/eleven/register")
async def api_eleven_register(
    owner: str = Form("manager1"),
    file: UploadFile = File(...),
    name: str = Form("web-upload"),
):
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Backend expects 'file' parameter
        files = {
            "file": (file.filename, await file.read(), file.content_type or "audio/wav")
        }
        data = {"owner": owner, "name": name, "set_active": "true"}
        resp = await client.post(f"{NIPA_BASE}/voice/eleven/register", data=data, files=files)
    if resp.status_code != 200:
        return JSONResponse(status_code=resp.status_code, content={"error": resp.text})
    return resp.json()

@app.post("/api/scenario_preview")
async def api_scenario_preview(
    scenario_id: str = Form(...),
    scenario_prompt: str = Form(...),
    top_k: int = Form(10),
):
    """
    Generate candidate responses for a scenario using the remote GPT endpoint.
    Returns JSON:
    {
      "status": "ok" | "error",
      "scenario_id": "...",
      "scenario_prompt": "...",
      "items": [
        {"category": "...", "short_key": "...", "response": "..."}
      ]
    }
    """
    # sanitize / clamp top_k
    try:
        top_k = int(top_k)
    except ValueError:
        top_k = 10
    top_k = max(10, min(top_k, 50))

    # Build a JSON-style prompt for the remote /voice/chat/gpt
    prompt = f"""
You are a call-center scenario script generator.

Read the following scenario description and propose {top_k} candidate responses
that an outbound call agent could speak.

Return ONLY valid JSON with this exact structure (no markdown, no explanation):

{{
  "scenario_id": "{scenario_id}",
  "items": [
    {{
      "category": "짧은 한국어 라벨, 예: 빠른 종료형, 예약 콜백, 인사, 안내",
      "short_key": "short snake_case identifier in English, e.g. fast_close, callback_offer, greet_default",
      "response": "natural Korean sentence suitable for spoken call-center dialog"
    }}
  ]
}}

Rules:
- response must be in Korean, polite, and suitable for telephone speech.
- category MUST be a short Korean label (e.g. 빠른 종료형, 예약 콜백).
- short_key MUST be lowercase snake_case in English.
- Do NOT include comments, text outside JSON, or trailing commas.

Scenario description:
{scenario_prompt}
""".strip()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
          gpt_payload = {"message": prompt}
          # optional: still forward scenario_id / scenario_prompt if you want backend to use them
          if scenario_id:
              gpt_payload["scenario_id"] = scenario_id
          if scenario_prompt:
              gpt_payload["scenario_prompt"] = scenario_prompt

          resp = await client.post(
              f"{NIPA_BASE}/voice/chat/gpt",
              json=gpt_payload,
              headers={"Content-Type": "application/json"},
          )

          if resp.status_code != 200:
              return {
                  "status": "error",
                  "error": f"gpt_http_{resp.status_code}",
              }

          raw = resp.json().get("reply", "").strip()

          try:
              parsed = json.loads(raw)
          except Exception as e:
              # If GPT returns something messy, just surface an error (optional:
              # you could fall back to a single row with raw text)
              return {
                  "status": "error",
                  "error": f"gpt_json_parse_error: {e}",
                  "raw": raw,
              }

          parsed_scenario_id = parsed.get("scenario_id") or scenario_id
          raw_items = parsed.get("items") or []

          items = []
          for i, it in enumerate(raw_items):
              items.append(
                  {
                      "rank": i + 1,
                      # what the table uses as main text
                      "response_text": it.get("response", ""),
                      # auto-filled by GPT
                      "category": it.get("category", ""),
                      "short_key": it.get("short_key", ""),
                      "scenario_id": parsed_scenario_id,
                  }
              )

          return {"status": "ok", "engine": "gpt4o", "items": items}

    except Exception as e:
        return {"status": "error", "error": f"gpt_exc_{e}"}



# -------------------- HTML with call + intro name + dark/light --------------------
HTML_PAGE = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <title>AI 음성 / 챗봇 서비스</title>
  <link rel="icon" href="/favicon.ico">
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {
      --bg: #0f172a;
      --card: #1f2937;
      --text: #e2e8f0;
      --subtle: rgba(255,255,255,.05);
    }
    body.light {
      --bg: #f3f4f6;
      --card: #ffffff;
      --text: #0f172a;
      --subtle: rgba(15,23,42,.06);
    }
    body {
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 1rem;
      padding-top: 2.5rem;
      font-family: system-ui, -apple-system, sans-serif;
      transition: background .2s, color .2s;
    }

    .container {
      max-width: 1400px;  /* was 900px */
      margin: 0 auto;
      padding: 0 0.25rem;  /* was 0 1rem */
    }
    /* Improve link color visibility in dark mode */
    body.dark a {
      color: #93c5fd !important;   /* bright, readable blue */
      text-decoration: underline;
    }

    /* Improve link color in light mode too */
    body.light a {
      color: #2563eb !important;   /* darker blue */
      text-decoration: underline;
    }
    .header {
      text-align: center;
      margin-bottom: 2rem;
    }
    .theme-toggle {
      position: fixed;
      top: 1rem;
      right: 1rem;
      z-index: 1000;
      padding: 0.5rem 0.8rem;
      border-radius: 8px;
      border: 1px solid var(--subtle);
      background: var(--card);
      color: var(--text);
      cursor: pointer;
      font-weight: 600;
      font-size: 0.9rem;
      transition: transform 0.1s;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      white-space: nowrap;
      min-width: auto;
      width: auto;
    }
    .theme-toggle:hover {
      transform: scale(1.05);
    }
    .icon-btn {
      width: auto;
      min-width: 0;
      padding: 0.25rem;
      border-radius: 999px;
      font-size: 0.85rem;
      display: flex;
      align-items: center;
      justify-content: center;
      background: var(--card);
      border: 1px solid var(--subtle);
      color: var(--text);
      cursor: pointer;
    }
    .icon-btn:hover {
      background: rgba(59, 130, 246, 0.25);
    }
    .card {
      background: var(--card);
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .row { display: flex; gap: 1rem; margin-bottom: 1rem; align-items: flex-start; }
    .row > div { flex: 1; min-width: 0; }
    input, select, button, textarea {
      width: 100%;
      max-width: 100%;
      box-sizing: border-box;
      padding: 0.8rem;
      border-radius: 8px;
      border: 1px solid var(--subtle);
      background: var(--subtle);
      color: var(--text);
      font-size: 1rem;
      font-family: inherit;
    }
    select option {
      background: var(--card);
      color: var(--text);
    }
    /* Fix for dark mode dropdowns in some browsers */
    select {
      color-scheme: light dark;
    }
    body.light select {
      color-scheme: light;
    }
    body.dark select {
      color-scheme: dark;
    }
    button {
      cursor: pointer;
      font-weight: 600;
      transition: transform 0.1s;
      background: #3b82f6;
      color: #fff;
      border: none;
    }
    button:hover {
      background: #2563eb;
      transform: translateY(-1px);
    }
    button:active { transform: translateY(0); }
    textarea { resize: vertical; min-height: 80px; }
    .tone-buttons {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
      gap: 0.5rem;
      margin-bottom: 1rem;
    }
    .tone-btn {
      padding: 0.6rem;
      font-size: 0.9rem;
      background: var(--subtle);
      color: var(--text);
      border: 2px solid transparent;
    }
    .tone-btn.active {
      border-color: #3b82f6;
      background: rgba(59, 130, 246, 0.2);
    }
    .output {
      padding: 1rem;
      border-radius: 8px;
      background: var(--subtle);
      margin-top: 1rem;
    }
    .scenario-play-btn {
      width: 26px;
      height: 26px;
      border-radius: 999px;
      padding: 0;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 0.75rem;
      background: #22c55e !important;   /* Tailwind green-500 */
      color: white !important;
    }
    .scenario-play-btn:hover {
      background: #16a34a !important;   /* green-600 */
    }
    #scenario-table td,
    #scenario-table th {
      vertical-align: middle !important;
    }
    .scenario-dl-btn {
      width: 26px;
      height: 26px;
      border-radius: 999px;
      padding: 0;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 0.75rem;
      background: #0ea5e9 !important;   /* sky-500 */
      color: white !important;
      text-decoration: none;
    }
    .scenario-dl-btn:hover {
      background: #0284c7 !important;   /* sky-600 */
    }
    /* Forest green TTS generation button */
    #scenario-generate-tts {
      background: #166534 !important;  /* forest green */
      color: white !important;
      border: none !important;
    }

    #scenario-generate-tts:hover {
      background: #14532d !important;  /* darker forest green */
    }
    audio { width: 100%; margin-top: 0.5rem; }
    #el-ref { padding: 0.4rem; }
    .slider-container {
      margin-bottom: 1rem;
    }
    .slider-label {
      display: flex;
      justify-content: space-between;
      margin-bottom: 0.5rem;
    }
    input[type="range"] {
      width: 100%;
      height: 6px;
      border-radius: 5px;
      background: var(--subtle);
      outline: none;
      padding: 0;
    }
    input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: #3b82f6;
      cursor: pointer;
    }
    input[type="range"]::-moz-range-thumb {
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: #3b82f6;
      cursor: pointer;
      border: none;
    }
    .chat-container {
      max-height: 400px;
      overflow-y: auto;
      overflow-x: hidden;   /* add this line */
      margin-bottom: 1rem;
      padding: 1rem;
      background: var(--subtle);
      border-radius: 8px;
    }
    .chat-meta {
      font-size: 0.7rem;
      opacity: 0.6;
      margin-top: 0px;
      margin-bottom: -6px;
    }

    .chat-meta.user {
      text-align: right;
      margin-right: 6px;
    }

    .chat-meta.assistant {
      text-align: left;
      margin-left: 6px;
    }

    .chat-row {
      display: flex;
      margin-bottom: 0.35rem;
    }
    .edit-btn {
      width: 1.4rem;
      height: 1.4rem;
      padding: 0;
      border-radius: 999px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.75rem;
      background: var(--card);
      border: 1px solid var(--subtle);
      cursor: pointer;
    }

    .edit-btn:hover {
      background: rgba(59, 130, 246, 0.2);
    }

    .chat-row.user {
      justify-content: flex-end;
    }
    .call-btn {
      flex: 1;             /* makes both buttons equal width */
      text-align: center;
      white-space: nowrap; /* prevents awkward wrapping */
    }
    .chat-row.assistant {
      justify-content: flex-start;
      align-items: center;
    }

    /* Tiny circular badges for engine type */
    .engine-badge {
      margin-left: 0.35rem;
      font-size: 0.7rem;
      width: 1.4rem;
      height: 1.4rem;
      border-radius: 999px;
      display: flex;
      align-items: center;
      justify-content: center;
      opacity: 0.85;
      flex-shrink: 0;
    }

    /* Different colors for each engine */
    /* tweak colors as you like */
    .engine-chajipsa {
      background: #10b981;  /* emerald-ish */
      color: #ffffff;
    }

    .engine-gpt {
      background: #3b82f6;  /* blue-ish */
      color: #ffffff;
    }

    .chat-message {
      margin-bottom: 0.15rem;   /* or 0.25rem / 0.3rem */
      padding: 0.8rem;
      border-radius: 8px;
      max-width: 80%;
      word-wrap: break-word;
      white-space: pre-wrap;
    }
    .chat-message.user {
      background: #3b82f6;
      color: white;
      /* no margin-left:auto here */
    }
    .chat-message.assistant {
      background: var(--card);
      border: 1px solid var(--subtle);
      word-wrap: break-word;
      white-space: pre-wrap;
    }

    .chat-input-row {
      display: flex;
      gap: 0.5rem;
      align-items: flex-start;
    }
    .chat-input-row textarea {
      flex: 1;
      min-height: 60px;
      min-width: 0;
    }
    .chat-input-row button {
      width: auto;
      min-width: 100px;
      flex-shrink: 0;
    }
    #chat-multi-results a,
    .chat-container a {
      color: #93c5fd;
      text-decoration: underline;
    }
    /* Light theme: darker blue */
    body.light #chat-multi-results a,
    body.light .chat-container a {
      color: #2563eb;   /* Tailwind blue-600 style */
    }

    /* Dark theme: light blue */
    body.dark #chat-multi-results a,
    body.dark .chat-container a {
      color: #93c5fd;   /* Tailwind blue-300 style */
    }
    #call,
    #intro {
      padding-left: 0.6rem;
      padding-right: 0.6rem;
    }
  </style>
</head>
<body id="body" class="light">
  <button id="theme" class="theme-toggle">☀️ Light</button>
  
  <div class="container">
    <div class="header">
      <h1>🎤 AI 음성 / 챗봇 서비스</h1>
    </div>

    <!-- CALL SECTION (MOVED TO TOP) -->
    <div class="card">
      <h2>📞 전화 발신 / Telephone Call</h2>
      <label><strong>전화번호</strong></label>
      <input id="phone" type="tel" placeholder="+821012345678" />

      <div style="margin-top: 1rem;">
        <label>표시 이름 (인트로용)</label>
        <input id="display_name" type="text" placeholder="차집사" />
      </div>

      <div style="margin-top: 1rem;">
        <label>전화 TTS 엔진</label>
        <select id="call_engine">          
          <option value="eleven">ElevenLabs</option>          
          <option value="fish">FishSpeech</option>
          <option value="gpt">GPT Voice</option>
        </select>
      </div>      

      <div class="row" style="margin-top: 1rem;">
        <button id="call" class="call-btn">📞 통화 실행 / Call</button>
        <button id="intro" class="call-btn">📝 인트로 생성 / Generate Intro</button>
      </div>

      <div id="call-out" class="output" style="display:none;">
        <div id="call-status">대기 중...</div>
      </div>
    </div>

    <!-- TTS SECTION -->
    <div class="card">
      <h2>🎙️ 음성 합성 / Text-to-Speech (TTS)</h2>
      <div id="tts-note" style="
        margin-top: 0.5rem;
        padding: 0.6rem;
        background: rgba(59,130,246,0.08);
        border-radius: 8px;
        margin-bottom: 0.8rem;
        line-height: 1.45;
        font-size: 0.9rem;
      "></div>

      <label><strong>엔진 선택</strong></label>
      <select id="engine">
        <option value="fishspeech">FishSpeech</option>
        <option value="elevenlabs">ElevenLabs</option>                
        <option value="gptvoice">GPT Voice</option>
      </select>

      <div style="margin-top: 1rem;">
        <label><strong>텍스트</strong></label>
        <textarea id="text" placeholder="여기에 텍스트를 입력하세요..."></textarea>
      </div>

      <!-- FishSpeech Panel -->      
      <div id="fish-panel">
        <label>톤 선택 (Tone Selection)</label>
        <div class="tone-buttons">
          <button class="tone-btn" data-tone="">[None]</button>
          <button class="tone-btn" data-tone="(happy)">(happy)</button>
          <button class="tone-btn" data-tone="(sad)">(sad)</button>
          <button class="tone-btn" data-tone="(angry)">(angry)</button>
          <button class="tone-btn" data-tone="(excited)">(excited)</button>
          <button class="tone-btn" data-tone="(friendly)">(friendly)</button>
          <button class="tone-btn" data-tone="(fearful)">(fearful)</button>
        </div>

        <div class="slider-container">
          <div class="slider-label">
            <label>Temperature</label>
            <span id="temp-value">0.90</span>
          </div>
          <input id="temp" type="range" min="0.1" max="1.0" step="0.05" value="0.70" />
        </div>

        <div>
          <label>참조 음성 (선택)</label>
          <input id="ref" type="file" accept="audio/*" />
        </div>
      </div>

      <!-- ElevenLabs Panel -->
      <div id="eleven-panel" style="display:none;">
        <div class="row">
          <div style="flex:1">
            <label>Owner</label>
            <input id="el-owner" type="text" value="manager1" />
          </div>
          <div style="flex:1">
            <label>Voice</label>
            <select id="el-voice"></select>
          </div>
        </div>
        <button id="el-load">Load Voices</button>
        <div style="margin-top: 0.5rem;">
          <label>Style (선택)</label>
          <input id="el-style" type="text" placeholder="예: friendly, professional" />
        </div>
        <div style="margin-top: 0.5rem;">
          <label style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer;">
            <input id="el-cache" type="checkbox" style="width: auto; cursor: pointer;" />
            <span>Use cache (faster but may return old audio)</span>
          </label>
        </div>
        <div style="margin-top: 1rem;">
          <label>Reference Voice Upload (선택)</label>
          <input id="el-ref" type="file" accept="audio/wav,audio/mp3" />
          <button id="el-upload" style="margin-top: 0.5rem;">Upload & Register</button>
        </div>
        <div id="el-status" style="margin-top: 0.5rem; font-size: 0.9rem; color: #888;"></div>
      </div>

      <!-- GPT Voice Panel -->
      <div id="gpt-panel" style="display:none;">
        <div class="slider-container">
          <div class="slider-label">
            <label>Temperature (GPT creativity)</label>
            <span id="gpt-temp-value">0.60</span>
          </div>
          <input id="gpt-temp" type="range" min="0.6" max="1.2" step="0.1" value="0.6" />
          <small style="color: #888;">0.6 = Most deterministic, 1.2 = Most creative</small>
        </div>
        <div style="margin-top: 0.5rem; padding: 0.8rem; background: rgba(59, 130, 246, 0.1); border-radius: 8px;">
          <strong>참고:</strong> GPT Voice는 텍스트를 먼저 GPT로 처리해 적절한 응답을 생성한 뒤, 그 내용을 음성으로 읽어줍니다. 직접적인 TTS만 원하신다면 FishSpeech 또는 ElevenLabs를 사용해 주세요.
        </div>
      </div>

      <button id="send" style="margin-top: 1rem;">🎵 음성 생성 / Generate Audio</button>

      <div id="output" class="output" style="display:none;">
        <div id="status">대기 중...</div>
        <div id="gpt-response" style="display:none; margin-top: 0.5rem; padding: 0.8rem; background: var(--card); border-radius: 8px;">
          <strong>GPT Response:</strong> <span id="gpt-response-text"></span>
        </div>
        <audio id="audio" controls></audio>
        <a id="download-link" style="display:none; margin-top: 1rem; display: inline-block; color: #3b82f6;">Download Audio</a>
      </div>
    </div>


    <div class="card">
    <h2>📦 시나리오 설정 / Scenario Config</h2>

      <label><strong>시나리오 프롬프트 (편집 가능)</strong></label>
      <textarea
        id="scenario-prompt"
        style="min-height: 120px;"
      >이 시나리오는 피자 배달 콜센터입니다. 자동차, 보험, 차집사라는 단어를 절대 쓰지 마세요.</textarea>

      <div style="margin-top:0.5rem; font-size:0.85rem; opacity:0.75;">
        이 시나리오는 GPT / Top-K 응답 생성 시에만 사용됩니다.<br>
        비워두면 서버의 기본 시나리오가 사용됩니다.
      </div>

    <h2>💬 챗봇 / Chat</h2>

    <!-- Engine + Top-K + TTS (for rows) -->
    <div style="
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.5rem;
      gap: 1rem;
    ">

  <!-- Chat engine -->
  <div style="display: flex; align-items: center; gap: 0.5rem;">
      <label for="chat-engine" style="font-size: 0.95rem;">
        <strong>챗봇 엔진:</strong>
      </label>
      <select
        id="chat-engine"
        style="
          padding: 2px 6px;
          font-size: 0.95rem;
          width: 140px;
          border-radius: 6px;
        "
      >
        <option value="gpt">GPT</option>
        <option value="chajipsa">차집사</option>        
      </select>
    </div>

    <!-- Top-K -->
    <div style="display: flex; align-items: center; gap: 0.5rem;">
      <label style="font-size: 0.95rem;"><strong>Top-K:</strong></label>
      <input
        id="chat-topk"
        type="number"
        min="1"
        max="50"
        value="10"
        style="
          width: 50px;
          padding: 4px;
          border-radius: 6px;
          font-size: 0.95rem;
        "
      />
    </div>

    <!-- TTS engine for rows -->
    <div style="display: flex; align-items: center; gap: 0.5rem;">
      <label style="font-size: 0.95rem;"><strong>TTS 엔진:</strong></label>
      <select
        id="chat-tts-engine"
        style="
          padding: 2px 6px;
          font-size: 0.95rem;
          width: 140px;
          border-radius: 6px;
        "
      >
        <option value="fishspeech" selected>FishSpeech</option>
        <option value="elevenlabs">ElevenLabs</option>        
      </select>
    </div>
  </div>

    <!-- Chat-specific reference voice upload (hidden by default, only for FishSpeech) -->
    <div id="chat-ref-wrapper" style="margin: 0.25rem 0 0.75rem; display: none;">
      <label style="font-size: 0.9rem; display:block; margin-bottom:0.25rem;">
        참조 음성 (채팅용, 선택)
      </label>
      <input
        id="chat-ref"
        type="file"
        accept="audio/*"
        style="padding: 0.3rem;"
      />
      <div style="font-size: 0.8rem; opacity: 0.7; margin-top: 0.25rem;">
        이 음성은 아래 Top-K 챗봇 응답을 FishSpeech로 재생할 때 사용됩니다.
      </div>
    </div>


    <div id="chat-container" class="chat-container" style="margin-top: 1rem;"></div>

    <div class="chat-input-row">
      <textarea id="chat-input" placeholder="여기에 메시지를 입력하세요..." style="margin-top: 1rem;"></textarea>
      <button id="chat-send" style="margin-top: 1rem;">전송 / Send</button>
      <button id="chat-clear" style="margin-top: 1rem; background: #6b7280;">
        대화 지우기 / Clear Chat
      </button>
    </div>

    <!-- NEW multi-response result panel -->
    <div id="chat-multi-results" style="margin-top: 1rem;"></div>


    <!-- Scenario candidates table -->
    <div style="margin-top: 1rem; overflow-x:auto;">
      <table id="scenario-table" style="width:100%; border-collapse: collapse; margin-top:0.5rem; display:none; max-height: 260px; overflow-y:auto;">
        <thead>
        <tr>
          <!-- select all -->
          <th style="width: 32px; text-align:center;">
            <input type="checkbox" id="scenario-check-all" />
          </th>

          <!-- new 번호 column -->
          <th style="width: 40px; text-align:center;">No.</th>

          <!-- 시나리오 ID column -->
          <th style="width: 120px;">시나리오 ID</th>

          <!-- 카테고리 -->
          <th style="width: 110px;">카테고리</th>

          <!-- Response -->
          <th style="width: 50%;">대답</th>

          <!-- Short Key -->
          <th style="width: 120px;">Short Key</th>          

          <!-- Play / URL / DL -->
          <th style="width: 120px;">Play</th>
          <th style="width: 220px;">Audio URL</th>
          <th style="width: 60px;">DL</th>
        </tr>
      </thead>

        <tbody id="scenario-body" style="max-height:260px; overflow-y:auto;"></tbody>
      </table>

    </div>

      <!-- DynamoDB + Generate Audio row -->
      <div
        class="dynamodb-response"
        style="
          margin-top: 0.75rem;
          display: flex;
          gap: 0.75rem;
          align-items: stretch;  /* let buttons be taller */
          flex-wrap: wrap;
        "
      >
        <input
          id="dynamodb-input"
          placeholder="DynamoDB 테이블 이름"
          style="
            flex: 0 0 300px;     /* fixed-ish width, slightly shorter */
            min-width: 220px;
            padding: 0.5rem;
          "
        />

        <button
          id="scenario-save-ddb"
          style="
            flex: 1;
            min-width: 190px;
            padding: 0.7rem 1rem;
            line-height: 1.3;
            text-align: center;
            white-space: normal;   /* allow wrap */
          "
        >
          💾 선택된 응답들 DynamoDB 업로드 <br> Upload Selected to DynamoDB
        </button>        

        <button
          id="scenario-export-txt"
          style="
            flex: 1;
            min-width: 190px;
            padding: 0.7rem 1rem;
            line-height: 1.3;
            text-align: center;
            white-space: normal;
            background: #0f766e;
            color: #ffffff;
            border: none;
          "
        >
          📄 모든 행<br/>TXT 내보내기
        </button>

        <button
          id="scenario-generate-tts"
          style="
            flex: 1;
            min-width: 190px;
            padding: 0.7rem 1rem;
            line-height: 1.3;
            text-align: center;
            white-space: normal;   /* allow wrap */
          "
        >
          🎙️ 선택된 응답들 음성 생성 <br> Generate TTS for Selected Responses
        </button>
      </div>

      <div id="scenario-tts-status"
           style="margin-top:0.5rem; font-size:0.9rem; color:#9ca3af;">
      </div>
    
  </div>


<script>
document.addEventListener("DOMContentLoaded", () => {
  // ======================= Helpers =======================
  const toastEl = document.getElementById("toast");

  function showToast(message = "Copied to clipboard!") {
    const toast = toastEl;
    if (!toast) return;
    toast.textContent = message;
    toast.style.visibility = "visible";
    toast.style.opacity = "1";
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.visibility = "hidden";
    }, 1500);
  }

  // Strip outer quotes from GPT-style strings: "안녕하세요" / “안녕하세요”
  function stripOuterQuotes(text) {
    if (!text) return text;
    let t = String(text).trim();
    if (
      (t.startsWith('"') && t.endsWith('"')) ||
      (t.startsWith("'") && t.endsWith("'")) ||
      (t.startsWith("“") && t.endsWith("”"))
    ) {
      t = t.slice(1, -1).trim();
    }
    return t;
  }

  function normalizeSpaces(text) {
    if (!text) return text;
    return String(text).replace(/\\s+/g, " ").trim();
  }

  // ======================= Theme toggle =======================
  const bodyEl = document.getElementById("body");
  const themeBtn = document.getElementById("theme");
  if (bodyEl && themeBtn) {
    function applyTheme(mode) {
      if (mode === "dark") {
        bodyEl.classList.remove("light");
        bodyEl.classList.add("dark");
        themeBtn.textContent = "🌙 Dark";
      } else {
        bodyEl.classList.remove("dark");
        bodyEl.classList.add("light");
        themeBtn.textContent = "☀️ Light";
      }
    }
    const stored = window.localStorage.getItem("theme") || "light";
    applyTheme(stored);
    themeBtn.addEventListener("click", () => {
      const next = bodyEl.classList.contains("dark") ? "light" : "dark";
      window.localStorage.setItem("theme", next);
      applyTheme(next);
    });
  }

  // ======================= Call section =======================
  const phoneInput        = document.getElementById("phone");
  const displayNameInput  = document.getElementById("display_name");
  const callEngineSel     = document.getElementById("call_engine");
  const callBtn           = document.getElementById("call");
  const introBtn          = document.getElementById("intro");
  const callOut           = document.getElementById("call-out");
  const callStatus        = document.getElementById("call-status");

  async function doCall(kind) {
    if (!phoneInput || !callOut || !callStatus) return;
    const phone = (phoneInput.value || "").trim();
    const displayName = (displayNameInput?.value || "").trim();
    const engine = callEngineSel?.value || "fishspeech";

    if (!phone) {
      alert("전화번호를 입력해 주세요.");
      return;
    }

    callStatus.textContent = kind === "intro"
      ? "📞 인사 콜 시작 중..."
      : "📞 통화 시작 중...";

    callOut.textContent = "";
    try {
      const fd = new FormData();
      fd.append("phone", phone);
      if (displayName) fd.append("display_name", displayName);
      fd.append("engine", engine);
      fd.append("kind", kind);

      const resp = await fetch("/api/call", {
        method: "POST",
        body: fd,
      });
      const data = await resp.json();

      if (!resp.ok || data.status !== "ok") {
        callStatus.textContent = "❌ 실패: " + (data.error || resp.status);
        return;
      }
      callStatus.textContent = "✅ 콜이 시작되었습니다.";
      callOut.textContent = data.message || "";
    } catch (err) {
      console.error(err);
      callStatus.textContent = "❌ 네트워크 오류";
    }
  }

  callBtn?.addEventListener("click", () => doCall("normal"));
  introBtn?.addEventListener("click", () => doCall("intro"));

  // ======================= TTS section =======================
  const ttsText        = document.getElementById("text");
  const ttsTemp        = document.getElementById("temperature");
  const ttsTempLabel   = document.getElementById("temperature-label");
  const ttsEngineSel   = document.getElementById("engine");
  const ttsToneBtns    = document.querySelectorAll(".tone-btn");
  const sendBtn        = document.getElementById("send");
  const outputDiv      = document.getElementById("output");
  const statusDiv      = document.getElementById("status");
  const gptRespDiv     = document.getElementById("gpt-response");
  const gptRespText    = document.getElementById("gpt-response-text");
  const audioEl        = document.getElementById("audio");
  const downloadLink   = document.getElementById("download-link");

  // ElevenLabs controls
  const elOwnerInput   = document.getElementById("el-owner");
  const elRefInput     = document.getElementById("el-ref");
  const elVoiceSelect  = document.getElementById("el-voice");
  const elLoadBtn      = document.getElementById("el-load-voices");
  const elUploadBtn    = document.getElementById("el-upload-ref");
  const elStatus       = document.getElementById("el-status");

  // ======================= TTS Panel Switching =======================
  const fishPanel   = document.getElementById("fish-panel");
  const elevenPanel = document.getElementById("eleven-panel");
  const gptPanel    = document.getElementById("gpt-panel");
  const noteEl      = document.getElementById("tts-note");
  const engineEl    = document.getElementById("engine");

  function updateTtsPanelAndNote() {
    // Hide all panels first
    fishPanel.style.display   = "none";
    elevenPanel.style.display = "none";
    gptPanel.style.display    = "none";

    if (engineEl.value === "elevenlabs") {
      noteEl.innerHTML = `
        🎧 <strong>TTS 엔진 안내</strong><br>
        ⚡ <strong>ElevenLabs</strong> — 매우 빠르고 실시간 스타일에 적합합니다.<br>
        🎤 음질은 좋지만 FishSpeech보다 약간 가볍습니다.
        <br><br>
        <em>ElevenLabs: faster, great for real-time use.</em>
      `;
      elevenPanel.style.display = "block";

    } else if (engineEl.value === "gptvoice") {
      noteEl.innerHTML = `
        🔊 <strong>GPT Voice 안내</strong><br>
        GPT Voice는 텍스트를 먼저 GPT로 이해하고 적절한 답변을 생성한 뒤, 그 내용을 음성으로 변환합니다.<br>
        직접적인 자연 음성 합성이 필요하다면 FishSpeech 또는 ElevenLabs가 더 적합합니다.
        <br><br>
        <em>GPT Voice: generates a GPT reply first, then speaks it.</em>
      `;
      gptPanel.style.display = "block";

    } else {
      // Default → FishSpeech
      noteEl.innerHTML = `
        🎧 <strong>TTS 엔진 안내</strong><br>
        ⏳ <strong>FishSpeech</strong> — 생성 속도는 느리지만 음질이 가장 자연스럽습니다.<br>
        ⚡ <strong>ElevenLabs</strong>에 비해 딜레이가 조금 더 있습니다.
        <br><br>
        <em>FishSpeech: slower, but highest voice quality.</em>
      `;
      fishPanel.style.display = "block";
    }
  }

  // Wire the change handler
  engineEl.addEventListener("change", updateTtsPanelAndNote);

  // Run once on page load
  updateTtsPanelAndNote();



  if (ttsTemp && ttsTempLabel) {
    ttsTemp.addEventListener("input", () => {
      ttsTempLabel.textContent = ttsTemp.value;
    });
  }

  function applyTonePreset(tone) {
    if (!ttsTemp || !ttsText) return;
    if (tone === "calm") {
      ttsTemp.value = "0.4";
    } else if (tone === "neutral") {
      ttsTemp.value = "0.7";
    } else if (tone === "excited") {
      ttsTemp.value = "0.9";
    }
    ttsTemp.dispatchEvent(new Event("input"));
  }

  ttsToneBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const tone = btn.dataset.tone;
      applyTonePreset(tone);
    });
  });

  function updateTtsEngineUi() {
    const eng = ttsEngineSel?.value || "fishspeech";
    const elevenPanel = document.getElementById("eleven-panel");
    if (elevenPanel) {
      elevenPanel.style.display = eng === "elevenlabs" ? "block" : "none";
    }
  }
  ttsEngineSel?.addEventListener("change", updateTtsEngineUi);
  updateTtsEngineUi();

  sendBtn?.addEventListener("click", async () => {
    if (!ttsText || !statusDiv || !audioEl || !outputDiv) return;

    const text = (ttsText.value || "").trim();
    if (!text) {
      alert("텍스트를 입력해 주세요.");
      return;
    }

    const engine = ttsEngineSel?.value || "fishspeech";
    outputDiv.style.display = "block";
    statusDiv.textContent = "🎵 음성 생성 중...";
    gptRespDiv.style.display = "none";
    gptRespText.textContent = "";
    audioEl.src = "";
    downloadLink.style.display = "none";

    try {
      let audioUrl = "";
      let usedGptText = "";

      if (engine === "elevenlabs") {
        const fd = new FormData();
        fd.append("text", text);
        fd.append("temperature", ttsTemp?.value || "0.7");
        if (elVoiceSelect && elVoiceSelect.value) {
          fd.append("voice_id", elVoiceSelect.value);
        }
        const resp = await fetch("/api/tts-eleven", {
          method: "POST",
          body: fd,
        });
        const data = await resp.json();
        if (!resp.ok || data.status !== "ok") {
          throw new Error(data.error || resp.statusText);
        }
        audioUrl = data.audio_url;
        usedGptText = data.gpt_text || "";
      } else {
        const fd = new FormData();
        fd.append("text", text);
        fd.append("temperature", ttsTemp?.value || "0.9");
        if (elRefInput && elRefInput.files && elRefInput.files[0]) {
          fd.append("ref_audio", elRefInput.files[0]);
        }
        const resp = await fetch("/api/tts", { method: "POST", body: fd });
        const data = await resp.json();
        if (!resp.ok || data.status !== "ok") {
          throw new Error(data.error || resp.statusText);
        }
        const result = data.result || data;
        audioUrl = result.audio_url || result.s3_url || result.url;
        usedGptText = result.gpt_text || "";
      }

      if (!audioUrl) {
        throw new Error("오디오 URL을 찾을 수 없습니다.");
      }

      statusDiv.textContent = "✅ 음성 생성 완료";
      audioEl.src = audioUrl;

      if (usedGptText) {
        gptRespDiv.style.display = "block";
        gptRespText.textContent = usedGptText;
      }

      downloadLink.style.display = "inline-block";
      downloadLink.textContent = "Download Audio";
      if (ttsEngineSel?.value === "elevenlabs") {
        const dlHref = `/api/tts-eleven/download?url=${encodeURIComponent(
          audioUrl
        )}&filename=${encodeURIComponent("tts_output.wav")}`;
        downloadLink.href = dlHref;
      } else {
        downloadLink.href = audioUrl;
        downloadLink.download = "tts_output.wav";
        downloadLink.removeAttribute("target");
      }
    } catch (err) {
      console.error(err);
      statusDiv.textContent = "❌ 오류: " + err.message;
    }
  });

  // ElevenLabs: load voices
  elLoadBtn?.addEventListener("click", async () => {
    const owner = (elOwnerInput?.value || "manager1").trim();
    if (!owner) return;
    if (elStatus) elStatus.textContent = "🔄 음성 목록 로드 중...";
    try {
      const resp = await fetch(
        `/api/eleven/voices?owner=${encodeURIComponent(owner)}`
      );
      const data = await resp.json();
      if (!resp.ok) {
        elStatus &&
          (elStatus.textContent = "❌ 실패: " + (data.error || resp.statusText));
        return;
      }
      const voices = data.voices || [];
      if (elVoiceSelect) {
        elVoiceSelect.innerHTML = "";
        voices.forEach((v) => {
          const opt = document.createElement("option");
          opt.value = v.voice_id || v.voiceId || "";
          opt.textContent =
            v.name || v.display_name || v.voice_id || "voice";
          elVoiceSelect.appendChild(opt);
        });
      }
      elStatus &&
        (elStatus.textContent = `✅ ${voices.length}개 음성 로드 완료`);
    } catch (err) {
      console.error(err);
      elStatus && (elStatus.textContent = "❌ 네트워크 오류");
    }
  });

  // ElevenLabs: upload ref audio
  elUploadBtn?.addEventListener("click", async () => {
    if (!elRefInput || !elRefInput.files || !elRefInput.files[0]) {
      alert("참조 음성 파일을 선택해주세요.");
      return;
    }
    const owner = (elOwnerInput?.value || "manager1").trim();
    const fd = new FormData();
    fd.append("owner", owner);
    fd.append("file", elRefInput.files[0]);
    fd.append("name", "web-upload");
    if (elStatus) elStatus.textContent = "🔄 음성 업로드 중...";
    try {
      const resp = await fetch("/api/eleven/register", {
        method: "POST",
        body: fd,
      });
      const data = await resp.json();
      if (!resp.ok) {
        elStatus &&
          (elStatus.textContent =
            "❌ 업로드 실패: " + (data.error || resp.statusText));
        return;
      }
      elStatus &&
        (elStatus.textContent =
          "✅ 업로드 완료. 'Load Voices'로 목록을 갱신하세요.");
    } catch (err) {
      console.error(err);
      elStatus && (elStatus.textContent = "❌ 네트워크 오류");
    }
  });

  // ======================= Scenario table =======================
  const scenarioIdEl           = document.getElementById("scenario-id");
  const scenarioPromptEl       = document.getElementById("scenario-prompt");
  const scenarioTable          = document.getElementById("scenario-table");
  const scenarioBody           = document.getElementById("scenario-body");
  const scenarioCheckAll       = document.getElementById("scenario-check-all");
  const dynamoInput            = document.getElementById("dynamodb-input");
  const scenarioSaveTextBtn    = document.getElementById("scenario-save-ddb");
  const scenarioGenerateTtsBtn = document.getElementById("scenario-generate-tts");
  const scenarioTtsStatus      = document.getElementById("scenario-tts-status");
  const scenarioExportTxtBtn   = document.getElementById("scenario-export-txt");

  function renderScenarioItems(items) {
    if (!scenarioBody) return;

    scenarioBody.innerHTML = "";

    // 비어있을 때 메시지
    if (!Array.isArray(items) || !items.length) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 9;
      td.textContent = "⚠ 표시할 응답이 없습니다.";
      td.style.textAlign = "center";
      td.style.padding = "0.6rem";
      tr.appendChild(td);
      scenarioBody.appendChild(tr);
      if (scenarioTable) scenarioTable.style.display = "table";
      return;
    }

    // 상단 시나리오 ID 입력값
    const scenIdFromTop = (scenarioIdEl?.value || "").trim();

    items.forEach((it, idx) => {
      const tr = document.createElement("tr");
      tr.dataset.rowIndex = String(idx);

      // 응답 텍스트
      const rawText =
        it.response_text || it.response || it.text || "";
      const textVal = normalizeSpaces(stripOuterQuotes(rawText));

      // 카테고리 / ShortKey 원본
      const catRaw = (it.category || "").toString();
      let skRaw    = (it.short_key || it.key || "").toString();

      // --- 시나리오 ID: it.scenario_id > 상단 입력 > "" ---
      const scenarioIdText =
        (it.scenario_id || it.scenarioId || scenIdFromTop || "").toString();

      // --- ShortKey 자동 부여: 없으면 resp_01, resp_02 ... ---
      if (!skRaw) {
        const n = idx + 1;
        const nStr = n < 10 ? "0" + n : String(n);
        skRaw = `resp_${nStr}`;
      }

      // --- 1) 체크박스 ---
      const tdCheck = document.createElement("td");
      tdCheck.style.textAlign = "center";
      const chk = document.createElement("input");
      chk.type = "checkbox";
      chk.checked = true;   
      chk.className = "scenario-row-check";
      tdCheck.appendChild(chk);
      tr.appendChild(tdCheck);

      // --- 2) 번호 (No.) ---
      const tdIdx = document.createElement("td");
      tdIdx.style.textAlign = "center";
      tdIdx.textContent = String(idx + 1);
      tr.appendChild(tdIdx);

      // --- 3) 시나리오 ID ---
      const tdId = document.createElement("td");
      tdId.style.verticalAlign = "top";
      tdId.innerHTML =
        `<input type="text" class="scenario-input-id" ` +
        `value="${scenarioIdText.replace(/"/g, "&quot;")}" ` +
        `style="width:100%; font-size:0.8rem; padding:2px 4px;" />`;
      tr.appendChild(tdId);

      // --- 4) 카테고리 ---
      const tdCat = document.createElement("td");
      tdCat.style.verticalAlign = "top";
      tdCat.innerHTML =
        `<input type="text" class="scenario-input-category" ` +
        `value="${catRaw.replace(/"/g, "&quot;")}" ` +
        `placeholder="ex) 인사, 안내" ` +
        `style="width:100%; font-size:0.8rem; padding:2px 4px;" />`;
      tr.appendChild(tdCat);      

      // --- 5) Response 텍스트 ---
      const tdResp = document.createElement("td");
      tdResp.style.verticalAlign = "top";
      tdResp.innerHTML =
        `<textarea class="scenario-input-response" ` +
        `style="width:100%; min-height:3.5rem; font-size:0.85rem; padding:4px;">` +
        (textVal || "") +
        `</textarea>`;
      tr.appendChild(tdResp);

      // --- 6) ShortKey ---
      const tdShort = document.createElement("td");
      tdShort.style.verticalAlign = "top";
      tdShort.innerHTML =
        `<input type="text" class="scenario-input-shortkey" ` +
        `value="${skRaw.replace(/"/g, "&quot;")}" ` +
        `style="width:100%; font-size:0.8rem; padding:2px 4px;" />`;
      tr.appendChild(tdShort);

      // --- 7) Play / URL / DL 셀: 처음엔 비워둠 ---
      const tdPlay = document.createElement("td");
      tdPlay.className = "scenario-play-cell";
      tdPlay.style.textAlign = "center";
      tr.appendChild(tdPlay);

      const tdUrl = document.createElement("td");
      tdUrl.className = "scenario-url-cell";
      tdUrl.style.fontSize = "0.8rem";
      tdUrl.style.wordBreak = "break-all";
      tr.appendChild(tdUrl);

      const tdDl = document.createElement("td");
      tdDl.className = "scenario-dl-cell";
      tdDl.style.textAlign = "center";
      tr.appendChild(tdDl);

      scenarioBody.appendChild(tr);
    });

    if (scenarioTable) scenarioTable.style.display = "table";
  }

  // 전체 선택 체크박스
  scenarioCheckAll?.addEventListener("change", () => {
    const checks = document.querySelectorAll(".scenario-row-check");
    checks.forEach((ch) => {
      ch.checked = scenarioCheckAll.checked;
    });
  });



  // 전체 선택 체크박스
  scenarioCheckAll?.addEventListener("change", () => {
    const checks = document.querySelectorAll(".scenario-row-check");
    checks.forEach((ch) => {
      ch.checked = scenarioCheckAll.checked;
    });
  });

  // ======================= Chat + Top-K + Scenario =======================
  const chatContainer     = document.getElementById("chat-container");
  const chatInput         = document.getElementById("chat-input");
  const chatSendBtn       = document.getElementById("chat-send");
  const chatClearBtn      = document.getElementById("chat-clear");
  const chatEngineSelect  = document.getElementById("chat-engine");
  const chatTopkInput     = document.getElementById("chat-topk");
  const chatTtsEngineSel  = document.getElementById("chat-tts-engine");
  const chatRefWrapper    = document.getElementById("chat-ref-wrapper");
  const chatRefInput      = document.getElementById("chat-ref");
  const chatMultiResults  = document.getElementById("chat-multi-results");

  // show/hide chat-specific ref for FishSpeech
  function updateChatTtsUi() {
    const eng = chatTtsEngineSel?.value || "fishspeech";
    if (chatRefWrapper) {
      chatRefWrapper.style.display =
        eng === "fishspeech" ? "block" : "none";
    }
  }
  chatTtsEngineSel?.addEventListener("change", updateChatTtsUi);
  updateChatTtsUi();

  function addChatMessage(role, content, engine) {
    if (!chatContainer) return;
    const row = document.createElement("div");
    row.className = `chat-row ${role}`;

    const msgDiv = document.createElement("div");
    msgDiv.className = `chat-message ${role}`;
    msgDiv.textContent = content;

    const metaDiv = document.createElement("div");
    metaDiv.className = `chat-meta ${role}`;
    const now = new Date();
    const timeStr = now.toLocaleTimeString("ko-KR", {
      hour: "2-digit",
      minute: "2-digit",
    });
    metaDiv.textContent = timeStr;
    metaDiv.title = now.toLocaleString("ko-KR");

    if (role === "user") {
      const wrapper = document.createElement("div");
      wrapper.style.display = "flex";
      wrapper.style.flexDirection = "column";
      wrapper.style.alignItems = "flex-end";
      wrapper.style.gap = "0.05rem";

      const topRow = document.createElement("div");
      topRow.style.display = "flex";
      topRow.style.flexDirection = "row";
      topRow.style.alignItems = "center";
      topRow.style.gap = "0.4rem";

      const editBtn = document.createElement("button");
      editBtn.className = "icon-btn edit-btn";
      editBtn.textContent = "✏️";
      editBtn.title = "메시지 수정 / Edit message";
      editBtn.style.flexShrink = "0";
      editBtn.onclick = () => {
        if (chatInput) {
          chatInput.value = content;
          chatInput.focus();
        }
      };

      topRow.appendChild(editBtn);
      topRow.appendChild(msgDiv);
      wrapper.appendChild(topRow);
      wrapper.appendChild(metaDiv);
      row.appendChild(wrapper);
    } else {
      const topRow = document.createElement("div");
      topRow.style.display = "flex";
      topRow.style.alignItems = "center";
      topRow.style.gap = "0.4rem";

      topRow.appendChild(msgDiv);

      if (engine) {
        const badge = document.createElement("div");
        badge.className = `engine-badge engine-${engine}`;
        badge.textContent = engine === "chajipsa" ? "C" : "G";
        topRow.appendChild(badge);
      }

      const copyBtn = document.createElement("button");
      copyBtn.className = "icon-btn copy-btn";
      copyBtn.title = "복사 / Copy";
      copyBtn.textContent = "📋";
      copyBtn.onclick = async () => {
        try {
          if (navigator.clipboard && navigator.clipboard.writeText) {
            // Modern API path
            await navigator.clipboard.writeText(content);
          } else {
            // Fallback for environments without navigator.clipboard
            const ta = document.createElement("textarea");
            ta.value = content;
            ta.style.position = "fixed";
            ta.style.left = "-9999px";
            document.body.appendChild(ta);
            ta.select();
            document.execCommand("copy");
            document.body.removeChild(ta);
          }
          showToast("복사되었습니다.");
        } catch (err) {
          console.error(err);
          showToast("복사 실패");
        }
      };

      topRow.appendChild(copyBtn);

      const wrapper = document.createElement("div");
      wrapper.style.display = "flex";
      wrapper.style.flexDirection = "column";
      wrapper.style.alignItems = "flex-start";
      wrapper.style.gap = "0.05rem";
      wrapper.appendChild(topRow);
      wrapper.appendChild(metaDiv);
      row.appendChild(wrapper);
    }

    chatContainer.appendChild(row);
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }

    // ---------------- Chat: Clear all messages ----------------
  chatClearBtn?.addEventListener("click", () => {
    if (chatContainer) chatContainer.innerHTML = "";
    if (chatMultiResults) chatMultiResults.innerHTML = "";
  });

  // ---------------- Chat: Send + Top-K → Scenario table ----------------
  async function handleChatSend() {
    if (!chatInput || !chatContainer) return;
    const userMsg = (chatInput.value || "").trim();
    if (!userMsg) return;

    const engineSel = chatEngineSelect?.value || "gpt";
    let topk = parseInt(chatTopkInput?.value || "10", 10);
    if (isNaN(topk)) topk = 10;
    topk = Math.max(1, Math.min(50, topk));

    // add user message bubble
    addChatMessage("user", userMsg, engineSel);
    chatInput.value = "";

    // clear multi-results view (we’re not listing all candidates here)
    if (chatMultiResults) chatMultiResults.innerHTML = "";

    // call /api/chat-multi -> scenario-style responses
    try {
      const fd = new FormData();
      fd.append("engine", engineSel === "chajipsa" ? "retriever" : "gpt4o");
      fd.append("text", userMsg);
      fd.append("top_k", String(topk));

      const scenId = (scenarioIdEl?.value || "").trim();
      const scenPrompt = (scenarioPromptEl?.value || "").trim();
      if (scenId) fd.append("scenario_id", scenId);
      if (scenPrompt) fd.append("scenario_prompt", scenPrompt);

      const resp = await fetch("/api/chat-multi", {
        method: "POST",
        body: fd,
      });
      const data = await resp.json();

      if (!resp.ok || data.status !== "ok" || !Array.isArray(data.items)) {
        addChatMessage(
          "assistant",
          "❌ multi 응답 생성 실패: " + (data.error || resp.status),
          engineSel
        );
        return;
      }

      const items = data.items;
      if (!items.length) {
        addChatMessage("assistant", "⚠ multi 응답이 없습니다.", engineSel);
        return;
      }

      // Top-1 → main chat reply
      const top1 = items[0];
      const top1Raw =
        top1.response_text || top1.response || top1.text || "";
      const top1Text = normalizeSpaces(stripOuterQuotes(top1Raw));
      addChatMessage("assistant", top1Text, engineSel);

      // Fill scenario table with same items (for caching / TTS)
      renderScenarioItems(items);

      // (Top-K .txt download: we’ll add this later once everything is stable)

    } catch (err) {
      console.error(err);
      addChatMessage(
        "assistant",
        "❌ /api/chat-multi 호출 중 오류",
        engineSel
      );
    }
  }

  chatSendBtn?.addEventListener("click", handleChatSend);
  chatInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleChatSend();
    }
  });

  // ======================= Scenario TTS generation (selected rows) =======================
  scenarioSaveTextBtn?.addEventListener("click", async () => {
    if (!scenarioBody) return;

    const tableName = (dynamoInput?.value || "").trim();
    if (!tableName) {
      alert("먼저 DynamoDB 테이블 이름을 입력해 주세요.");
      return;
    }

    const rows = Array.from(scenarioBody.querySelectorAll("tr"));
    const selectedRows = rows.filter((r) => {
      const chk = r.querySelector(".scenario-row-check");
      return chk && chk.checked;
    });

    if (!selectedRows.length) {
      alert("선택된 행이 없습니다.");
      return;
    }

    const payloadRows = [];
    for (const row of selectedRows) {
      const idInput = row.querySelector(".scenario-input-id");
      const catInput = row.querySelector(".scenario-input-category");
      const keyInput = row.querySelector(".scenario-input-shortkey");
      const respInput = row.querySelector(".scenario-input-response");
      const urlCell = row.querySelector(".scenario-url-cell");

      const scenarioId = (idInput?.value || "").trim();
      const category = (catInput?.value || "").trim();
      const shortKey = (keyInput?.value || "").trim();
      const response = (respInput?.value || "").trim();
      const audioUrl = (urlCell?.textContent || "").trim();

      if (!scenarioId || !shortKey || !response) continue;

      payloadRows.push({
        scenario_id: scenarioId,
        category,
        short_key: shortKey,
        response,
        audio_url: audioUrl,
      });
    }

    if (!payloadRows.length) {
      alert("저장할 유효한 행이 없습니다.");
      return;
    }

    if (scenarioTtsStatus) {
      scenarioTtsStatus.textContent = "💾 DynamoDB에 저장 중...";
    }

    try {
      const resp = await fetch("/api/scenario/save-text", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          table_name: tableName,
          rows_json: JSON.stringify(payloadRows)
        })
      });

      const data = await resp.json();
      if (!resp.ok || data.status !== "ok") {
        throw new Error(data.error || `HTTP ${resp.status}`);
      }

      if (scenarioTtsStatus) {
        const n = data.saved ?? payloadRows.length;
        scenarioTtsStatus.textContent = `✅ ${n}개 저장 완료`;
      }
    } catch (err) {
      console.error(err);
      if (scenarioTtsStatus) {
        scenarioTtsStatus.textContent = "❌ DynamoDB 저장 실패";
      }
      alert("DynamoDB 저장 중 오류가 발생했습니다.");
    }
  });

  
  // --- Scenario: Generate Audio for Selected rows (FishSpeech / ElevenLabs) ---
  scenarioGenerateTtsBtn?.addEventListener("click", async () => {
    if (!scenarioBody) return;

    const rows = Array.from(scenarioBody.querySelectorAll("tr"));
    const ttsEngineSelect = document.getElementById("chat-tts-engine");
    const engine = (ttsEngineSelect && ttsEngineSelect.value) || "fishspeech";

    // ✅ tableName is OPTIONAL for TTS; only needed if we want DDB + S3
    const tableName = (dynamoInput?.value || "").trim();

    const selectedRows = rows.filter((r) => {
      const chk = r.querySelector(".scenario-row-check");
      return chk && chk.checked;
    });

    if (!selectedRows.length) {
      alert("선택된 행이 없습니다.");
      return;
    }

    if (scenarioTtsStatus) {
      scenarioTtsStatus.textContent = tableName
        ? "⏳ 선택된 행들에 대해 TTS 생성 + DynamoDB 저장 중..."
        : "⏳ 선택된 행들에 대해 TTS 생성 중 (DynamoDB는 건너뜀)...";
    }

    let successCount = 0;

    for (const row of selectedRows) {
      const idInput   = row.querySelector(".scenario-input-id");
      const catInput  = row.querySelector(".scenario-input-category");
      const keyInput  = row.querySelector(".scenario-input-shortkey");
      const respInput = row.querySelector(".scenario-input-response");

      const scenarioId = (idInput?.value || "").trim();
      const category   = (catInput?.value || "").trim();
      const shortKey   = (keyInput?.value || "").trim();
      const response   = (respInput?.value || "").trim();

      if (!scenarioId || !shortKey || !response) {
        console.warn("Skipping row due to missing fields:", {
          scenarioId,
          shortKey,
          response,
        });
        continue;
      }

      const playCell =
        row.querySelector(".scenario-play-cell, .scenario-cell-play");
      const urlCell =
        row.querySelector(".scenario-url-cell, .scenario-cell-url");
      const dlCell =
        row.querySelector(".scenario-dl-cell, .scenario-cell-dl");

      if (playCell) playCell.textContent = "생성 중...";
      if (urlCell)  urlCell.textContent = "";
      if (dlCell)   dlCell.textContent = "";

      try {
        let audioUrl = "";

        // 1) Generate TTS
        if (engine === "elevenlabs") {
          const fdEleven = new FormData();
          fdEleven.append("text", response);

          const resp = await fetch("/api/tts-eleven", {
            method: "POST",
            body: fdEleven,
          });
          const data = await resp.json();
          if (!resp.ok || data.status !== "ok") {
            throw new Error(data.error || "ElevenLabs 실패");
          }
          audioUrl = data.audio_url;
        } else {
          const fd = new FormData();
          fd.append("text", response);
          fd.append("temperature", "0.9");

          // 채팅용 ref 우선, 없으면 TTS 패널 ref
          if (chatRefInput && chatRefInput.files && chatRefInput.files[0]) {
            fd.append("ref_audio", chatRefInput.files[0]);
          } else if (ref && ref.files && ref.files[0]) {
            fd.append("ref_audio", ref.files[0]);
          }

          const resp = await fetch("/api/tts", {
            method: "POST",
            body: fd,
          });
          const data = await resp.json();
          if (!resp.ok || data.status !== "ok") {
            throw new Error(data.error || "FishSpeech 실패");
          }
          const result = data.result || data;
          audioUrl = result.audio_url || result.s3_url || result.url;
        }

        if (!audioUrl) {
          throw new Error("오디오 URL 없음");
        }

        // 2) (Optional) Store to DDB + S3 only if tableName is provided
        let finalAudioUrl = audioUrl;
        if (tableName) {
          try {
            const fdStore = new FormData();
            fdStore.append("table_name", tableName);
            fdStore.append("scenario_id", scenarioId);
            fdStore.append("category", category);
            fdStore.append("short_key", shortKey);
            fdStore.append("response_text", response);
            fdStore.append("audio_url", audioUrl);

            const storeResp = await fetch("/voice/scenario/store-audio", {
              method: "POST",
              body: fdStore,
            });
            const storeData = await storeResp.json();
            if (storeResp.ok && storeData.status === "ok") {
              if (storeData.s3_url) {
                // ✅ if backend gives presigned S3, use it going forward
                finalAudioUrl = storeData.s3_url;
              }
            } else {
              console.warn("store-audio 실패:", storeData);
            }
          } catch (err) {
            console.error("store-audio 예외:", err);
          }
        }

        // 3) UI 업데이트: 작은 재생 버튼 + URL + 다운로드 버튼
        if (playCell) {
          playCell.innerHTML = "";
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "scenario-play-btn";  // small round button
          btn.title = "재생";
          btn.textContent = "▶";
          btn.addEventListener("click", () => {
            const audio = new Audio(finalAudioUrl);
            audio.play().catch((err) => console.error("Play error:", err));
          });
          playCell.appendChild(btn);
        }

        if (urlCell && finalAudioUrl) {
          urlCell.innerHTML = "";

          // Make a "clean" display URL (no query params)
          let displayUrl = finalAudioUrl;
          try {
            const u = new URL(finalAudioUrl);
            displayUrl = u.origin + u.pathname;   // strip ?query
          } catch (e) {
            // If URL constructor fails, just fall back to the full string
            displayUrl = finalAudioUrl;
          }

          const link = document.createElement("a");
          link.href = finalAudioUrl;      // still presigned for streaming
          link.target = "_blank";
          link.rel = "noopener";
          link.textContent = displayUrl;  // clean, non-presigned-looking text

          urlCell.appendChild(link);
        } else if (urlCell) {
          urlCell.textContent = "";
        }

        if (dlCell && finalAudioUrl) {
          const filename = (shortKey || "resp") + ".wav";

          // Always go through our same-origin proxy
          const dlHref =
            `/api/tts-download?url=${encodeURIComponent(
              finalAudioUrl
            )}&filename=${encodeURIComponent(filename)}`;

          dlCell.innerHTML =
            `<a class="scenario-dl-btn" ` +
            `href="${dlHref}" ` +
            `download="${filename}">⬇</a>`;
        }

        successCount++;
      } catch (err) {
        console.error("TTS 생성 실패:", err);
      }
    }

  scenarioBody?.addEventListener("click", async (e) => {
      const btn = e.target.closest(".scenario-dl-btn-eleven");
      if (!btn) return;

      e.preventDefault();
      const href = btn.dataset.href;
      const filename = btn.dataset.filename || "tts_output.wav";
      if (!href) return;

      const res = await fetch(href);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);

      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    });

    if (scenarioTtsStatus) {
      scenarioTtsStatus.textContent =
        `✅ TTS 처리 완료 (성공: ${successCount}개)` +
        (tableName ? "" : " / DDB 저장은 건너뜀");
    }
  });

      // ==================== Export TXT helpers ====================

    function collectScenarioRowsForExport() {
      if (!scenarioBody) return [];

      const allRows = Array.from(scenarioBody.querySelectorAll("tr"));

      // ✅ Only rows with checkbox checked
      const selectedRows = allRows.filter((row) => {
        const chk = row.querySelector(".scenario-row-check");
        return chk && chk.checked;
      });

      return selectedRows.map((row) => {
        const idInput   = row.querySelector(".scenario-input-id");
        const catInput  = row.querySelector(".scenario-input-category");
        const keyInput  = row.querySelector(".scenario-input-shortkey");
        const respInput = row.querySelector(".scenario-input-response");
        const urlCell   = row.querySelector(".scenario-url-cell, .scenario-cell-url");

        const scenarioId = (idInput?.value || "").trim();
        const category   = (catInput?.value || "").trim();
        const shortKey   = (keyInput?.value || "").trim();
        const response   = (respInput?.value || "").trim();

        let audioUrl = "";
        if (urlCell) {
          const link = urlCell.querySelector("a");
          if (link && link.href) {
            audioUrl = link.href;
          } else {
            audioUrl = (urlCell.textContent || "").trim();
          }
        }

        return {
          scenario_id:   scenarioId,
          category:      category,
          short_key:     shortKey,
          response_text: response,
          audio_url:     audioUrl,
        };
      });
    }

    async function exportScenarioRowsToTxt() {
      console.log("[export TXT] click handler started");

      const rows = collectScenarioRowsForExport();
      console.log("[export TXT] collected rows:", rows);

      if (!rows.length) {
        alert("내보낼 행이 없습니다.");
        return;
      }

      const tableName = (dynamoInput?.value || "").trim();
      console.log("[export TXT] tableName:", tableName);

      try {
        const resp = await fetch("/api/scenario/export-txt", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            table_name: tableName,
            rows: rows,
          }),
        });

        if (!resp.ok) {
          console.error("Export TXT failed:", resp.status, await resp.text());
          alert("TXT 내보내기 중 오류가 발생했습니다.");
          return;
        }

        const blob = await resp.blob();
        const url  = window.URL.createObjectURL(blob);

        const a = document.createElement("a");
        a.href = url;
        a.download = tableName
          ? tableName + "_scenario_rows.txt"
          : "scenario_rows.txt";

        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);

        console.log("[export TXT] download triggered");
      } catch (err) {
        console.error("Export TXT exception:", err);
        alert("TXT 내보내기 중 오류가 발생했습니다.");
      }
    }

    // Wire button
    if (scenarioExportTxtBtn) {
      console.log("[export TXT] wiring button");
      scenarioExportTxtBtn.addEventListener("click", (e) => {
        e.preventDefault();
        exportScenarioRowsToTxt().catch((err) => {
          console.error("[export TXT] unhandled error:", err);
        });
      });
    } else {
      console.warn("[export TXT] scenario-export-txt button not found");
    }

    // (Optional) expose to console if you want
    window.exportScenarioRowsToTxt = exportScenarioRowsToTxt;



});
</script>


  <div id="toast" style="
    visibility: hidden;
    min-width: 160px;
    background-color: #333;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 10px;
    position: fixed;
    z-index: 9999;
    left: 50%;
    bottom: 40px;
    transform: translateX(-50%);
    font-size: 0.9rem;
    opacity: 0;
    transition: opacity 0.4s ease, visibility 0.4s ease;
  ">
    Copied to clipboard!
  </div>

</body>
</html>
"""

# run:
if __name__ == "__main__":
    uvicorn.run(
        "__main__:app",      # module_name:app_variable
        host="0.0.0.0",
        port=5051,
        reload=True
    )