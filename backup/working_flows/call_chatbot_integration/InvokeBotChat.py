import json
import logging
import urllib.request
import re

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def clean_text(text):
    # Remove a leading period if it is not preceded by a complete sentence
    cleaned = re.sub(r'^\.\s*', '', text.strip())
    return cleaned

def lambda_handler(event, context):
    logger.info("===== Incoming event from Connect / Lex =====")
    logger.info(json.dumps(event, indent=2, ensure_ascii=False))

    # Try to get slot value from Lex
    user_input = (
        event.get("Details", {})
             .get("Parameters", {})
             .get("user_input", "")
             .lstrip(".")
             .strip()
    )
    # user_input = "광고내용 처음부터 알려줘"
    logger.info(f"[Captured user input] → '{user_input}'")
    
    # Cloud 9 chatbot API call
    url = "http://15.165.60.45:5000/chat"
    payload = json.dumps({
        "session_id": "38bb86ff-9a62-4e04-af1e-2dacdbda8f99",
        "question": user_input
    }).encode('utf-8')
    req = urllib.request.Request(
        url="http://15.165.60.45:5000/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode())
            answer = result.get("answer", "답변을 받을 수 없습니다.")
    except Exception as e:
        answer = f"오류가 발생했습니다: {str(e)}"
    
    # answer = str(answer) + str(user_input)
    # answer = clean_text(answer)
    logger.info(f"[Final answer for playback] → {answer}")

    return {
      "user_input": answer
    }
