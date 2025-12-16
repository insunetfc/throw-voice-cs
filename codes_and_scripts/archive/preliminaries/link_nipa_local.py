import json
import os
import logging
import requests

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# http://109.61.127.25:8000/synthesize
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://al-consensus-habitat-bachelor.trycloudflare.com/synthesize")
S3_URL_ATTRIBUTE = "AudioS3Url"

def lambda_handler(event, context):
    logger.info(f"Incoming event: {json.dumps(event)}")

    text_input = (
        event.get("Details", {})
             .get("ContactData", {})
             .get("Attributes", {})
             .get("UserText", "Hello from Lambda")
    )

    try:
        response = requests.post(FASTAPI_URL, json={"text": text_input}, timeout=5)
        response.raise_for_status()
        data = response.json()
        s3_url = data.get("s3_url", "")
        return {"setAttributes": {S3_URL_ATTRIBUTE: s3_url}}

    except Exception as e:
        logger.error(f"Error calling FastAPI: {e}", exc_info=True)
        return {"setAttributes": {S3_URL_ATTRIBUTE: ""}}
