import os, json, boto3, re, hashlib
from botocore.config import Config

# ---------- Bedrock setup ----------
BEDROCK_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
brt = boto3.client(
    "bedrock-runtime",
    region_name=BEDROCK_REGION,
    config=Config(
        retries={"max_attempts": 3, "mode": "standard"},
        connect_timeout=1,
        read_timeout=2,
    ),
)

# S3 for cache warmup file (optional - can also bundle with Lambda)
s3 = boto3.client("s3")
CACHE_BUCKET = os.getenv("CACHE_BUCKET", None)
CACHE_KEY = os.getenv("CACHE_KEY", "cache_warmup.json")

SYSTEM_PROMPT = """Correct Korean ASR errors in car insurance promotional call responses.

CALL CONTEXT: Customer received call from 차집사 offering 7% commission for insurance signup (both direct/offline available, 95% conversion rate).

COMMON ASR ERRORS:
- "바빠요"(busy) ↔ "바꿔요"/"봐봐요" → prefer "바빠요" in rejection context
- "관심 없어요"(not interested) ↔ "관심 있어요"(interested)
- "괜찮아요" - check if acceptance or rejection from tone
- "통화 중" ↔ "통화 가능"

INTENTS: interested, busy, not_interested, transfer, unclear

Return JSON only:
{"intent": "<intent>", "normalized": "<corrected text>", "confidence": <0-1>}"""

# ---------- Global cache (persists across warm invocations) ----------
NORMALIZATION_CACHE = {}
CACHE_LOADED = False
USE_BEDROCK = os.getenv("USE_BEDROCK", "true").lower() == "true"
MAX_CACHE_SIZE = 10000

def get_cache_key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def normalize_text_fastpath(t: str):
    # ultra-fast regex router for the obvious cases
    if any(x in t for x in ("바빠", "바쁘")):   return {"intent":"busy","normalized":t,"confidence":0.95}
    if any(x in t for x in ("관심 없", "싫", "괜찮", "필요 없")): return {"intent":"not_interested","normalized":t,"confidence":0.8}
    if any(x in t for x in ("상담원", "사람", "연결", "직원")):   return {"intent":"transfer","normalized":t,"confidence":0.8}
    if any(x in t for x in ("바꿔", "변경", "교체")):            return {"intent":"interested","normalized":t,"confidence":0.8}
    return None  # unknown

def load_cache_warmup():
    """Load pre-computed normalizations on cold start"""
    global NORMALIZATION_CACHE, CACHE_LOADED
    if CACHE_LOADED:
        return
    try:
        if CACHE_BUCKET:
            print(f"Loading cache from s3://{CACHE_BUCKET}/{CACHE_KEY}")
            resp = s3.get_object(Bucket=CACHE_BUCKET, Key=CACHE_KEY)
            warmup_data = json.loads(resp["Body"].read().decode("utf-8"))
        else:
            warmup_data = {}
            cache_file = "/opt/cache_warmup.json"  # If using Lambda Layer
            if os.path.exists(cache_file):
                print(f"Loading cache from local file: {cache_file}")
                with open(cache_file, "r", encoding="utf-8") as f:
                    warmup_data = json.load(f)

        for text, result in warmup_data.items():
            NORMALIZATION_CACHE[get_cache_key(text)] = result

        CACHE_LOADED = True
        print(f"Cache loaded with {len(NORMALIZATION_CACHE)} entries")
    except Exception as e:
        print(f"Failed to load cache: {e}")
        CACHE_LOADED = True  # avoid retrying every invocation

def get_cached_normalization(text: str) -> dict | None:
    return NORMALIZATION_CACHE.get(get_cache_key(text))

def set_cached_normalization(text: str, result: dict):
    if len(NORMALIZATION_CACHE) >= MAX_CACHE_SIZE:
        # drop ~10% oldest keys (cheap LRU-ish)
        for k in list(NORMALIZATION_CACHE.keys())[: MAX_CACHE_SIZE // 10]:
            NORMALIZATION_CACHE.pop(k, None)
    NORMALIZATION_CACHE[get_cache_key(text)] = result

def parse_json_text(text: str) -> dict:
    # strip code fences if any and extract first JSON object
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    m = re.search(r"\{.*\}", cleaned, flags=re.S)
    if m:
        cleaned = m.group(0)
    try:
        return json.loads(cleaned)
    except Exception as e:
        print("Parse error:", e)
        return {"intent": "unclear", "normalized": text, "confidence": 0.0}

def bedrock_normalize(utterance: str) -> dict:
    cached = get_cached_normalization(utterance)
    if cached:
        print(f"✓ Cache HIT: {utterance[:30]}...")
        return cached

    print(f"✗ Cache MISS: {utterance[:30]}...")

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": [{"type": "text", "text": utterance}]}],
        "max_tokens": 80,
        "temperature": 0.0,
    }

    try:
        resp = brt.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(body).encode("utf-8"),
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(resp["body"].read())
        # Haiku commonly returns under outputText; also support content[] just in case
        text_out = payload.get("outputText", "").strip()
        if not text_out and "content" in payload:
            text_out = "".join(
                p.get("text", "") for p in payload["content"] if p.get("type") == "text"
            )
        result = parse_json_text(text_out)
        set_cached_normalization(utterance, result)
        return result
    except Exception as e:
        print("Bedrock error:", str(e))
        return {"intent": "unclear", "normalized": utterance, "confidence": 0.0}

# Load cache on cold start
load_cache_warmup()

def lambda_handler(event, context):
    session_state = event.get("sessionState", {}) or {}
    intent = session_state.get("intent", {"name": "Fallback", "state": "InProgress"})
    slots = intent.get("slots") or {}
    sess = session_state.get("sessionAttributes", {}) or {}

    first_text = event.get("inputTranscript") or ""
    invocation = event.get("invocationSource")
    THRESHOLD = float(os.getenv("INTENT_THRESHOLD", "0.6"))

    if invocation == "DialogCodeHook" and first_text:
        fast = normalize_text_fastpath(first_text)
        result = fast if (fast and not USE_BEDROCK) else bedrock_normalize(first_text) if USE_BEDROCK else (fast or {"intent":"unclear","normalized":first_text,"confidence":0.0})
        normalized = result.get("normalized") or first_text
        pred_intent = result.get("intent", "unclear")
        confidence = float(result.get("confidence", 0.0))

        # Optional: clarify for low confidence on actionable intents
        if pred_intent in ("busy", "not_interested", "interested", "transfer") and confidence < THRESHOLD:
            return {
                "sessionState": {
                    "sessionAttributes": sess,
                    "dialogAction": {"type": "ElicitIntent"},
                },
                "messages": [{
                    "contentType": "PlainText",
                    "content": "확실히 하려고요. 지금 바쁘신 건가요, 아니면 변경을 원하시는 건가요?"
                }]
            }

        # Stash debug/analytics safely
        sess["raw_utterance"] = first_text
        sess["normalized_utterance"] = normalized
        sess["pred_intent"] = pred_intent
        sess["pred_confidence"] = f"{confidence:.2f}"
        sess["cache_hit"] = str(get_cache_key(first_text) in NORMALIZATION_CACHE)
        sess["cache_size"] = str(len(NORMALIZATION_CACHE))

        # Optional: canonical action for downstream branching
        intent_to_action = {
            "busy": "user_busy",
            "not_interested": "reject",
            "interested": "proceed",
            "transfer": "handoff",
            "unclear": "clarify",
        }
        sess["action"] = intent_to_action.get(pred_intent, "clarify")

        # Update your slot
        slots["UserInput"] = {"value": {
            "originalValue": first_text,
            "interpretedValue": normalized
        }}

    return {
        "sessionState": {
            "sessionAttributes": sess,
            "dialogAction": {"type": "Delegate"},
            "intent": {"name": intent["name"], "state": "InProgress", "slots": slots},
        }
    }
