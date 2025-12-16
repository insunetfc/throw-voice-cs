import os, json, time, sys
import boto3
from botocore.exceptions import ClientError, BotoCoreError

# -------- config --------
REGION  = os.getenv("AWS_REGION", os.getenv("COMPANY_BUCKET_REGION", "ap-northeast-2"))
TABLE   = os.getenv("CACHE_TABLE", "ConnectPromptCache")
FN_NAME = os.getenv("ASYNC_FUNCTION_NAME", "InvokeBotLambda")

POLL_EVERY_SEC = int(os.getenv("POLL_EVERY_SEC", "2"))
TIMEOUT_SEC    = int(os.getenv("TIMEOUT_SEC", "180"))
CHAT_ON        = os.getenv("CHAT_ON", "false")

lambda_client = boto3.client("lambda", region_name=REGION)
dynamodb      = boto3.resource("dynamodb", region_name=REGION)
table         = dynamodb.Table(TABLE)

from decimal import Decimal

def de_decimalize(obj):
    """Recursively convert Decimal -> int/float for JSON serialization."""
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    if isinstance(obj, dict):
        return {k: de_decimalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [de_decimalize(x) for x in obj]
    return obj

def invoke_lambda(payload: dict) -> dict:
    resp = lambda_client.invoke(
        FunctionName=FN_NAME,
        InvocationType="RequestResponse",
        Payload=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )
    body = resp["Payload"].read()
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return json.loads(body)

def ddb_get(pk: str) -> dict | None:
    try:
        r = table.get_item(Key={"pk": pk}, ConsistentRead=True)
        return r.get("Item")
    except (ClientError, BotoCoreError) as e:
        print(f"[ddb] get_item error for pk={pk}: {e}")
        return None

def start_job(text: str) -> tuple[str, str, str]:
    event = {
        "action": "start",
        "Details": {
            "Parameters": {
                "user_input": text,
                "chat_on": CHAT_ON
            }
        }
    }
    print("==> START")
    res = invoke_lambda(event)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    pk    = res.get("pk")
    bucket= res.get("bucket") or (res.get("setAttributes") or {}).get("job_bucket")
    key   = res.get("key")    or (res.get("setAttributes") or {}).get("job_key")
    if not (pk and bucket and key):
        raise SystemExit("Start did not return pk/bucket/key; check Lambda logs.")
    return pk, bucket, key

def watch_until_ready(pk: str, bucket: str, key: str):
    print("\n==> WATCHING DDB (pk only; no Lambda check)")
    deadline = time.time() + TIMEOUT_SEC
    last_state = None
    while time.time() < deadline:
        item = ddb_get(pk)
        if not item:
            print("…no item yet, retrying")
            time.sleep(POLL_EVERY_SEC)
            continue

        raw_state = (item.get("state") or "pending").strip()
        state = raw_state.upper()
        if state != last_state:
            print(f"state={state}  prompt_arn={item.get('prompt_arn','')[:60]}  "
                  f"key={item.get('key', key)}")
            last_state = state

        if state in ("READY", "COMPLETED"):
            print("\n==> READY via DDB")
            print(json.dumps(de_decimalize({
                "pk": pk,
                "bucket": item.get("bucket", bucket),
                "key": item.get("key", key),
                "state": state,
                "prompt_arn": item.get("prompt_arn", ""),
                "final_text": item.get("final_text", "")[:200],
            }), ensure_ascii=False, indent=2))
            return

        time.sleep(POLL_EVERY_SEC)

    print("\n==> TIMEOUT (still not ready)")
    item = ddb_get(pk) or {}
    print(json.dumps(de_decimalize(item), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "아스파라거스를 좋아합니다"
    print(f"[cfg] region={REGION} table={TABLE} function={FN_NAME}")
    pk, bucket, key = start_job(text)
    # Immediately verify we can see 'pending' in DDB
    first = ddb_get(pk)
    print("\n==> FIRST READ (should be pending):")
    print(json.dumps(de_decimalize(first or {}), ensure_ascii=False, indent=2))
    # Then watch until 'ready'
    watch_until_ready(pk, bucket, key)
