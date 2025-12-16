import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

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

    logger.info(f"[Captured user input] → '{user_input}'")

    # Fallback message if input is empty
    if not user_input:
        response_message = "입력이 감지되지 않았습니다. 다시 말씀해 주세요."
    else:
        response_message = f"당신의 요청은 다음과 같습니다: {user_input}"

    logger.info(f"[Response for Connect] → '{response_message}'")

    return {
        "sessionAttributes": {},
        "actions": [
            {
                "type": "UpdateContactAttributes",
                "parameters": {
                    "attributes": {
                        "speech_result": response_message
                    }
                }
            }
        ]
    }
