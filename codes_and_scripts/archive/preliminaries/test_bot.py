import boto3
import json

# Replace these with your actual values
LEX_BOT_ID = "ZUZDJPUFTG"            # Your published bot
LEX_BOT_ALIAS_ID = "GSH2W8JQ0A"   # From create-bot-alias response
LEX_LOCALE_ID = "ko_KR"
LEX_REGION = "ap-northeast-1"

def lambda_handler(event, context):
    print("Received event:", json.dumps(event, ensure_ascii=False))

    # Simulate customer utterance text (in real flow, this would be passed by Connect or test payload)
    user_input = event.get("user_input", "예약하고 싶어요")
    session_id = event.get("session_id", "session-001")

    lex_client = boto3.client("lexv2-runtime", region_name=LEX_REGION)

    try:
        response = lex_client.recognize_text(
            botId=LEX_BOT_ID,
            botAliasId=LEX_BOT_ALIAS_ID,
            localeId=LEX_LOCALE_ID,
            sessionId=session_id,
            text=user_input
        )

        print("Lex response:", json.dumps(response, ensure_ascii=False))

        # Extract slot value
        slot_raw = response.get("sessionState", {}).get("intent", {}).get("slots", {}).get("UserInput")

        if slot_raw and slot_raw.get("value"):
            slot_value = slot_raw["value"].get("interpretedValue", "")
        else:
            slot_value = None


        return {
            "statusCode": 200,
            "body": json.dumps({
                "captured_user_input": slot_value,
                "lex_messages": response.get("messages", [])
            }, ensure_ascii=False)
        }

    except Exception as e:
        print("Error:", str(e))
        return {
            "statusCode": 500,
            "body": f"Failed to process: {str(e)}"
        }
