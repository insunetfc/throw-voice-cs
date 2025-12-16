import boto3

client = boto3.client("lexv2-models", region_name="ap-northeast-1")

response = client.update_slot(
    botId="ZUZDJPUFTG",
    botVersion="DRAFT",
    localeId="ko_KR",
    intentId="6ES2AIAS5J",
    slotId="TMGFQJ2O8Z",
    slotName="UserInput",
    slotTypeId="ND8KPJVHTZ",
    valueElicitationSetting={
        "slotConstraint": "Optional",
        "promptSpecification": {
            "messageGroups": [
                {
                    "message": {
                        "plainTextMessage": {
                            "value": "말씀해주세요."
                        }
                    }
                }
            ],
            "maxRetries": 0,
            "allowInterrupt": True
        }
    },
    slotCaptureSetting={
        "captureResponse": {
            "messageGroups": [
                {
                    "message": {
                        "plainTextMessage": {
                            "value": "알겠습니다. \"{UserInput}\" 이라고 하셨습니다."
                        }
                    }
                }
            ],
            "allowInterrupt": True
        }
    }
)

print("Slot updated successfully.")
