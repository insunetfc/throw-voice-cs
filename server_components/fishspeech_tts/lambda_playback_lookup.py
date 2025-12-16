
import os
import json
import hashlib
import boto3

DDB_TABLE = os.environ.get("RESPONSE_TABLE", "ResponseAudio")
REGION = os.environ.get("AWS_REGION", "ap-northeast-2")

ddb = boto3.resource("dynamodb", region_name=REGION)
table = ddb.Table(DDB_TABLE)

def h_short(text: str) -> str:
    return hashlib.blake2b(text.encode('utf-8'), digest_size=8).hexdigest()

def call_chatbot(text: str, intent_hint: str | None = None) -> str:
    """
    Replace this with your real (deterministic) chatbot call.
    For now, we assume your contact flow or upstream system already provides the chatbot reply in event['chatbot_reply'].
    """
    # If upstream gave it, just use it (lowest latency).
    provided = (text or "").strip()
    return provided

def handler(event, context):
    """
    Expected event:
    {
      "user_utterance": "string",       # raw user text
      "chatbot_reply": "string",        # optional; if provided, we skip calling the bot
      "intent_hint": "fee_question"     # optional
    }
    Returns:
    {
      "response_text": "...",           # the TTS text that the bot replied
      "response_hash": "d9f65a43dbf87876",
      "audio_s3_uri": "s3://bucket/ko-KR/d9f65a43dbf87876.wav",
      "found": true
    }
    """
    user_utterance = (event.get("user_utterance") or "").strip()
    bot_reply = (event.get("chatbot_reply") or "").strip()

    # If chatbot_reply not provided, call bot deterministically.
    if not bot_reply:
        bot_reply = call_chatbot(user_utterance, intent_hint=event.get("intent_hint"))

    rhash = h_short(bot_reply)
    res = table.get_item(Key={"response_hash": rhash})
    item = res.get("Item")

    result = {
        "response_text": bot_reply,
        "response_hash": rhash,
        "audio_s3_uri": item.get("audio_s3_uri") if item else "",
        "found": bool(item),
    }
    return {"statusCode": 200, "headers": {"Content-Type": "application/json"}, "body": json.dumps(result, ensure_ascii=False)}
