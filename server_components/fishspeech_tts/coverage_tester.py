
#!/usr/bin/env python3
"""
Coverage tester that:
1) Reads response_templates_rich.csv
2) For each template, queries your deterministic chatbot with a probe utterance,
3) Confirms the chatbot returns EXACTLY the template's response (after strip),
4) Computes response_hash and fetches the audio_s3_uri from DynamoDB,
5) Writes a CSV report: coverage_report.csv

You need to implement CHATBOT_URL or `call_chatbot()` to hit your actual chatbot.
If you don't have an endpoint, you can stub it to return the `response` field from the CSV.
"""
import argparse
import csv
import hashlib
import os
import time
from typing import Dict, Any

import boto3
import requests
from botocore.exceptions import ClientError

DEFAULT_TABLE = "ResponseAudio"
DEFAULT_LOCALE = os.environ.get("RESPONSE_AUDIO_LOCALE", "ko-KR")

# --- Hash helpers (must match the seeding script) ---
def h_short(text: str) -> str:
    return hashlib.blake2b(text.encode('utf-8'), digest_size=8).hexdigest()

def call_chatbot(utterance: str, intent_hint: str = None) -> str:
    """
    Replace this stub with your *real* chatbot call.
    Example implementation for an HTTP endpoint expecting JSON:
        resp = requests.post(CHATBOT_URL, json={"text": utterance, "intent": intent_hint}, timeout=5)
        resp.raise_for_status()
        return resp.json()["response"]
    """
    CHATBOT_URL = os.environ.get("CHATBOT_URL")  # e.g. https://api.example.com/chat
    if CHATBOT_URL:
        resp = requests.post(CHATBOT_URL, json={"text": utterance, "intent": intent_hint}, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        # Adjust the extraction below if your payload differs
        return (data.get("response") or data.get("message") or "").strip()
    else:
        # Fallback stub: echo utterance (for local dry-runs without an endpoint)
        # Replace with deterministic lookup if you have a local function.
        return utterance.strip()

def fetch_audio_uri(ddb_table, response_text: str) -> Dict[str, Any]:
    resp_hash = h_short(response_text.strip())
    try:
        res = ddb_table.get_item(Key={"response_hash": resp_hash})
        item = res.get("Item")
        if not item:
            return {"found": False, "response_hash": resp_hash, "audio_s3_uri": ""}
        return {"found": True, "response_hash": resp_hash, "audio_s3_uri": item.get("audio_s3_uri", "")}
    except ClientError as e:
        raise

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/home/tiongsik/Python/outbound_calls/chatbot/data/response_templates_rich.csv", help="Path to response_templates_rich.csv")
    ap.add_argument("--table", default=DEFAULT_TABLE, help="DynamoDB table name")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "ap-northeast-2"))
    ap.add_argument("--probe_field", default="response",
                    help="CSV column to use as the probe utterance (default: 'response'). "
                         "If you have a separate 'probe' column of user utterances, set it here.")
    ap.add_argument("--out", default="coverage_report.csv", help="Output CSV path")
    args = ap.parse_args()

    ddb = boto3.resource("dynamodb", region_name=args.region)
    table = ddb.Table(args.table)

    # Read CSV
    rows = []
    with open(args.csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Test loop
    out_rows = []
    ok = 0
    for i, row in enumerate(rows, 1):
        intent = (row.get("intent") or "").strip()
        expected = (row.get("response") or "").strip()
        probe = (row.get(args.probe_field) or expected).strip()

        chatbot_reply = call_chatbot(probe, intent_hint=intent)
        matches = (chatbot_reply.strip() == expected)

        audio = fetch_audio_uri(table, expected)
        out_rows.append({
            "idx": i,
            "intent": intent,
            "context": (row.get("context") or "").strip(),
            "probe_used": probe,
            "expected_response": expected,
            "chatbot_reply": chatbot_reply,
            "exact_match": "YES" if matches else "NO",
            "response_hash": audio["response_hash"],
            "audio_found": "YES" if audio["found"] else "NO",
            "audio_s3_uri": audio["audio_s3_uri"],
        })
        if matches and audio["found"]:
            ok += 1

    # Write report
    fieldnames = ["idx", "intent", "context", "probe_used", "expected_response",
                  "chatbot_reply", "exact_match", "response_hash", "audio_found", "audio_s3_uri"]
    with open(args.out, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Coverage OK pairs (exact match + audio found): {ok}/{len(rows)}")
    print(f"Wrote report: {args.out}")

if __name__ == "__main__":
    main()
