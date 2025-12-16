import os
import json
import time
import urllib.request
from urllib.parse import urlparse, urlunparse

# ===== ENV =====
BRIDGE_BASE = os.getenv("BRIDGE_BASE", "https://honest-trivially-buffalo.ngrok-free.app")
AWS_REGION  = os.getenv("AWS_REGION", "ap-northeast-2")
TTS_BUCKET  = os.getenv("TTS_BUCKET", "tts-bucket-250810")

# Optional: fast fallback if bridge call fails
FALLBACK_INTRO = os.getenv(
    "FALLBACK_INTRO",
    f"https://{TTS_BUCKET}.s3.{AWS_REGION}.amazonaws.com/greetings/intro/fast_intro.wav"
)

# ===== Helpers =====
def _post_json(url: str, payload: dict, timeout: int = 20) -> dict:
    """POST JSON and return parsed response"""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)

def _normalize_s3_url(url: str) -> str:
    """
    Convert presigned S3 URL to public URL for Amazon Connect.
    - Strips query parameters (Signature, Expires, etc.)
    - Ensures correct region in hostname
    
    Example:
      IN:  https://bucket.s3.amazonaws.com/key?Signature=...&Expires=...
      OUT: https://bucket.s3.ap-northeast-2.amazonaws.com/key
    """
    if not url:
        return url
    
    p = urlparse(url)
    host = p.netloc
    
    # Fix region in hostname if needed
    if host.endswith("s3.amazonaws.com") and f".s3.{AWS_REGION}.amazonaws.com" not in host:
        bucket = host.split(".")[0]
        new_host = f"{bucket}.s3.{AWS_REGION}.amazonaws.com"
        p = p._replace(netloc=new_host)
    
    # Strip ALL query parameters (presigned URL parameters)
    # Amazon Connect needs clean public URLs without ?Signature=...&Expires=...
    p = p._replace(query="", fragment="")
    
    return urlunparse(p)

def _unwrap_params(event) -> dict:
    """
    Extract parameters from Connect event or direct invoke.
    """
    params = {}
    if isinstance(event, dict):
        if "Details" in event and "Parameters" in event["Details"]:
            params = dict(event["Details"]["Parameters"])
        else:
            params = dict(event)
    
    return {
        "text": params.get("text") or params.get("utterance") or "안녕하세요! 무엇을 도와드릴까요?",
        "mode": (params.get("mode") or params.get("engine") or "gpt_voice").strip(),
        "owner": params.get("owner") or "manager1",
        "voice": params.get("voice") or "alloy",  # GPT voice selection
        "use_cache": str(params.get("use_cache", "true")).lower() != "false",
        "voice_id": params.get("voice_id") or None,  # For ElevenLabs if needed
    }

# ===== Main Lambda Handler =====
def lambda_handler(event, context):
    """
    Lambda function for Amazon Connect GPT Voice integration.
    
    Modes:
      - gpt_voice (default): GPT Voice API (GPT brain + GPT voice)
      - gpt_voice_eleven: GPT Voice brain + ElevenLabs TTS
      - brain_gpt_eleven: GPT text API + ElevenLabs TTS (legacy)
    
    Returns Connect-compatible attributes with AudioS3Url0.
    """
    t0 = time.time()
    p = _unwrap_params(event)
    
    text = p["text"]    

    text = (p.get("text") or "").strip()

    if not text:
        # return a structured 'no-input' signal so the flow can reprompt
        return {
            "statusCode": 200,
            "body": json.dumps({
                "setAttributes": {
                    "ready": "false",
                    "need_reprompt": "true",
                    "reason": "empty_text"
                }
            })
        }
    
    mode = p["mode"].lower()
    owner = p["owner"]
    voice = p["voice"]
    use_cache = p["use_cache"]
    voice_id = p["voice_id"]
    
    # Map old mode names to new ones for backward compatibility
    mode_mapping = {
        "gpt_voice_instant_start": "gpt_voice",
        "gpt_voice_instant_fetch": "gpt_voice",
        "brain_gpt_voice": "gpt_voice",
        "voice": "gpt_voice",
    }
    mode = mode_mapping.get(mode, mode)
    
    print(f"[Lambda] Mode: {mode}, Text: {text[:50]}..., Owner: {owner}")
    
    # ===== GPT Voice (GPT brain + GPT voice) =====
    if mode in ("gpt_voice",):
        path = "/voice/brain/gpt-voice/start"
        payload = {
            "text": text,
            "voice": voice,
            "owner": owner
        }
        timeout = 15  # Allow up to 15 seconds for complete audio generation
    
    # ===== GPT Voice Brain + ElevenLabs TTS =====
    elif mode in ("gpt_voice_eleven", "eleven"):
        path = "/voice/brain/gpt-voice-eleven"
        payload = {
            "text": text,
            "user_text": text,
            "owner": owner,
            "use_cache": use_cache,
            "temperature": 0.6,  # Min temperature for GPT Voice
        }
        if voice_id:
            payload["voice_id"] = voice_id
        timeout = 30
    
    # ===== GPT Text + ElevenLabs (Legacy) =====
    else:
        path = "/voice/brain/gpt-eleven"
        payload = {
            "text": text,
            "owner": owner,
            "use_cache": use_cache
        }
        if voice_id:
            payload["voice_id"] = voice_id
        timeout = 30
    
    bridge_url = f"{BRIDGE_BASE}{path}"
    
    try:
        print(f"[Lambda] Calling: {bridge_url}")
        resp = _post_json(bridge_url, payload, timeout=timeout)
        print(f"[Lambda] Response: {json.dumps(resp)[:200]}...")
        
        # Extract audio URL from response
        audio_url = resp.get("audio_url")
        if not audio_url:
            # Fallback: try audio_urls list (legacy format)
            audio_urls = resp.get("audio_urls")
            if isinstance(audio_urls, list) and audio_urls:
                audio_url = audio_urls[0]
        
        if not audio_url:
            raise Exception("No audio_url in response")
        
        # Normalize the S3 URL
        audio_url = _normalize_s3_url(audio_url)
        
        # Build Connect attributes
        attrs = {
            "ready": "true",
            "AudioS3Url0": audio_url,
            "AudioS3UrlCount": "1",
            "HasMore": "false",
            "intent": "continue",
            "reply_text": resp.get("reply_text", text),
            "voice_id_used": resp.get("voice_id_used", ""),
            "status": resp.get("status", "complete"),
        }
        
        # Add timing information if available
        timings = resp.get("timings", {})
        if isinstance(timings, dict):
            for k in ("gpt_ms", "tts_ms", "upload_ms", "total_ms", "generation_ms", "s3_upload_ms"):
                if k in timings:
                    attrs[k] = str(timings[k])
        
        print(f"[Lambda] Success! Audio URL: {audio_url}")
        
    except Exception as e:
        print(f"[Lambda] ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Graceful fallback
        attrs = {
            "ready": "true",
            "AudioS3Url0": _normalize_s3_url(FALLBACK_INTRO),
            "AudioS3UrlCount": "1",
            "HasMore": "false",
            "intent": "continue",
            "reply_text": "죄송합니다. 시스템이 잠시 지연되고 있습니다.",
            "error": str(e)[:200],  # Truncate error to avoid attribute size limits
            "status": "error",
        }
    
    # Add Lambda execution time
    lambda_time = int((time.time() - t0) * 1000)
    attrs["lambda_total_ms"] = str(lambda_time)
    attrs["engine"] = "gpt-voice" if "gpt-voice" in path else "gpt-eleven"
    
    print(f"[Lambda] Completed in {lambda_time}ms")
    
    return {"setAttributes": attrs}


# ===== For Local Testing =====
if __name__ == "__main__":
    # Test different modes
    test_cases = [
        {
            "name": "GPT Voice (default)",
            "event": {
                "Details": {
                    "Parameters": {
                        "text": "안녕하세요",
                        "mode": "gpt_voice"
                    }
                }
            }
        },
        {
            "name": "GPT Voice + ElevenLabs",
            "event": {
                "Details": {
                    "Parameters": {
                        "text": "안녕하세요",
                        "mode": "gpt_voice_eleven"
                    }
                }
            }
        },
        {
            "name": "GPT Text + ElevenLabs (legacy)",
            "event": {
                "Details": {
                    "Parameters": {
                        "text": "안녕하세요",
                        "mode": "brain_gpt_eleven"
                    }
                }
            }
        }
    ]
    
    for test in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {test['name']}")
        print(f"{'='*60}")
        result = lambda_handler(test['event'], None)
        print(json.dumps(result, indent=2, ensure_ascii=False))