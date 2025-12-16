# phone_call/app.py
from fastapi import FastAPI, Form, HTTPException
import os
import time
import re
import boto3
import requests

app = FastAPI(title="Phone Call API")

# ====== ENV / CONFIG ======
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")

# Amazon Connect
CONNECT_INSTANCE_ID = os.getenv("CONNECT_INSTANCE_ID", "5b83741e-7823-4d70-952a-519d1ac05e63")
CONTACT_FLOW_ID = os.getenv("CONTACT_FLOW_ID", "b7c0c1be-a84a-489e-a428-e13505c85aca")
SOURCE_PHONE = os.getenv("SOURCE_PHONE", "+82269269343")  # your verified number

# DDB table to store per-phone greeting (like batch_intro_greeting.py)
DDB_TABLE = os.getenv("CALLER_TABLE", "PhoneIntro")

# your FishSpeech/TTS server running on the same NIPA box (we mounted it under /tts in the main app)
TTS_BASE = os.getenv("TTS_BASE", "http://localhost:8000/tts")
TTS_AUTH = os.getenv("TTS_AUTH", "Bearer YOUR_TOKEN")

TTS_BUCKET = os.getenv("TTS_BUCKET", "tts-bucket-250810")
KEY_PREFIX_GREETING = os.getenv("KEY_PREFIX_GREETING", "greetings/intro")


def _normalize_phone(num: str) -> str:
    """close to what you used earlier: 010-xxxx → +82xxxx"""
    s = re.sub(r"[^\d+]", "", num or "")
    if not s:
        return ""
    if s.startswith("+"):
        return s
    # assume Korea if starts with 0
    if s.startswith("0"):
        return "+82" + s[1:]
    return s


def _make_intro_text(display_name: str) -> str:
    n = (display_name or "").strip()
    if not n or n == "고객님":
        return "(friendly) 안녕하세요 고객님, 반갑습니다"
    n = re.sub(r"\s+", "", n)
    if not n.endswith(("고객님", "님")):
        n = f"{n}고객님"
    return f"(friendly) 안녕하세요 {n} 맞으시죠?"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/call_eleven")
def start_outbound_call(phone_number: str = Form(...)):
    """
    Start an outbound call via Amazon Connect.
    This is basically call_interrupt_input.py but exposed as HTTP.
    """
    dest = _normalize_phone(phone_number)
    if not dest:
        raise HTTPException(status_code=400, detail="Invalid phone number")

    connect = boto3.client("connect", region_name=AWS_REGION)
    resp = connect.start_outbound_voice_contact(
        DestinationPhoneNumber=dest,
        ContactFlowId=CONTACT_FLOW_ID,
        InstanceId=CONNECT_INSTANCE_ID,
        SourcePhoneNumber=SOURCE_PHONE,
        Attributes={
            "from": "web_api",
        },
    )
    return {"status": "ok", "contact_id": resp["ContactId"], "dest": dest}


@app.post("/call_gpt")
def start_outbound_call(phone_number: str = Form(...)):
    """
    Start an outbound call via Amazon Connect.
    This is basically call_interrupt_input.py but exposed as HTTP.
    """
    dest = _normalize_phone(phone_number)
    if not dest:
        raise HTTPException(status_code=400, detail="Invalid phone number")

    connect = boto3.client("connect", region_name=AWS_REGION)
    resp = connect.start_outbound_voice_contact(
        DestinationPhoneNumber=dest,
        ContactFlowId="47aa1c64-5a52-4593-84a6-43108efa9086",
        InstanceId=CONNECT_INSTANCE_ID,
        SourcePhoneNumber=SOURCE_PHONE,
        Attributes={
            "from": "web_api",
        },
    )
    return {"status": "ok", "contact_id": resp["ContactId"], "dest": dest}


@app.post("/generate-intro")
def generate_intro(
    phone_number: str = Form(...),
    display_name: str = Form(""),
):
    """
    Generate the intro TTS for a user and store the result in DynamoDB
    (single-row version of batch_intro_greeting.py).
    """
    phone = _normalize_phone(phone_number)
    if not phone:
        raise HTTPException(status_code=400, detail="Invalid phone number")

    text = _make_intro_text(display_name or "고객님")

    # 1) ask local TTS (which is actually mounted in the combined app)
    audio_url = ""
    audio_key = f"{KEY_PREFIX_GREETING}/{int(time.time())}.wav"
    try:
        r = requests.post(
            f"{TTS_BASE.rstrip('/')}/synthesize",
            json={
                "text": text,
                "key_prefix": "greetings/intro"  # 👈 force correct folder
            },
            headers={"Authorization": TTS_AUTH},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        audio_url = data.get("url") or data.get("s3_url") or ""
        audio_key = data.get("key", audio_key)
    except Exception:
        # if TTS is down, we still write DDB with a predictable key
        pass

    # 2) write to DynamoDB
    ddb = boto3.resource("dynamodb", region_name=AWS_REGION)
    table = ddb.Table(DDB_TABLE)
    table.update_item(
        Key={"phone_number": phone},
        UpdateExpression="SET display_name=:n, greeting_audio_s3=:g, updated_at=:t",
        ExpressionAttributeValues={
            ":n": display_name or "고객님",
            ":g": f"{TTS_BUCKET}/{audio_key}",
            ":t": int(time.time()),
        },
    )

    return {
        "status": "ok",
        "phone_number": phone,
        "display_name": display_name or "고객님",
        "audio_key": audio_key,
        "audio_url": audio_url,
    }
