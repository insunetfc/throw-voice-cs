
#!/usr/bin/env python3
"""
Robust DDB seeder (v2) for ResponseAudio:
- Accepts CSV with arbitrary column order and casing
- Handles UTF-8 BOM automatically
- Maps headers by normalized names: intent, response, context
- Verbose option to print each put
"""

import argparse
import csv
import hashlib
import os
import sys
import time

import boto3
from botocore.exceptions import ClientError

DEFAULT_TABLE = "ResponseAudio"
DEFAULT_BUCKET = os.environ.get("RESPONSE_AUDIO_BUCKET", "tts-bucket-250810")
DEFAULT_LOCALE = os.environ.get("RESPONSE_AUDIO_LOCALE", "ko-KR")
DEFAULT_PREFIX = os.environ.get("RESPONSE_AUDIO_PREFIX", "")  # optional subdir

def h_short(text: str) -> str:
    return hashlib.blake2b(text.encode('utf-8'), digest_size=8).hexdigest()

def h_sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def ensure_table(ddb, table_name: str):
    try:
        ddb.create_table(
            TableName=table_name,
            AttributeDefinitions=[{"AttributeName": "response_hash", "AttributeType": "S"}],
            KeySchema=[{"AttributeName": "response_hash", "KeyType": "HASH"}],
            BillingMode="PAY_PER_REQUEST",
            Tags=[{"Key": "app", "Value": "promo-calls"}],
        )
        print(f"Creating table {table_name} ...")
        waiter = ddb.meta.client.get_waiter("table_exists")
        waiter.wait(TableName=table_name)
        print("Table is active.")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            print(f"Table {table_name} already exists, continuing.")
        else:
            raise

def normalize_name(name: str) -> str:
    return (name or "").strip().lstrip("\ufeff").lower().replace(" ", "_")

def read_rows(csv_path: str):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            raw_headers = next(reader)
        except StopIteration:
            raise SystemExit("CSV is empty.")
        headers = [normalize_name(h) for h in raw_headers]
        # Build index map
        idx = {h: i for i, h in enumerate(headers)}
        # Accept synonyms
        def col(name, *alts):
            keys = [normalize_name(name)] + [normalize_name(a) for a in alts]
            for k in keys:
                if k in idx:
                    return idx[k]
            return None

        i_intent = col("intent")
        i_resp = col("response", "reply", "text", "tts_text")
        i_ctx = col("context", "category", "note")

        required_missing = [n for n,i in [("intent", i_intent), ("response", i_resp), ("context", i_ctx)] if i is None]
        if required_missing:
            raise SystemExit(f"CSV missing required columns (flex-match): {required_missing}. "
                             f"Got headers={headers}")

        for row in reader:
            # skip wholly empty lines
            if not any(cell.strip() for cell in row):
                continue
            # pad short rows
            if len(row) < len(headers):
                row = row + [""]*(len(headers)-len(row))
            yield {
                "intent": row[i_intent].strip(),
                "response": row[i_resp].strip(),
                "context": row[i_ctx].strip(),
            }

def seed_from_csv(table, rows, bucket: str, locale: str, prefix: str, verbose: bool = False):
    count = 0
    with table.batch_writer(overwrite_by_pkeys=["response_hash"]) as batch:
        for r in rows:
            response_text = r["response"]
            if not response_text:
                continue
            short = h_short(response_text)
            sha256 = h_sha256(response_text)
            s3_key = "/".join([p for p in [prefix.strip("/"), locale, f"{short}.wav"] if p])
            s3_uri = f"s3://{bucket}/{s3_key}"
            now = int(time.time())
            item = {
                "response_hash": short,
                "sha256_hash": sha256,
                "locale": locale,
                "intent": r["intent"],
                "context": r["context"],
                "response_text": response_text,
                "tts_text": response_text,
                "audio_s3_uri": s3_uri,
                "created_at": now,
                "updated_at": now,
            }
            batch.put_item(Item=item)
            count += 1
            if verbose:
                print(f"[PUT] {short} | intent={r['intent']} | s3={s3_uri}")
    print(f"Seeded {count} items into {table.table_name}.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/home/tiongsik/Python/outbound_calls/chatbot/data/response_templates_production_updated.csv")
    ap.add_argument("--table", default=DEFAULT_TABLE)
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--locale", default=DEFAULT_LOCALE)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "ap-northeast-2"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ddb = boto3.resource("dynamodb", region_name=args.region)
    ensure_table(ddb, args.table)
    table = ddb.Table(args.table)

    rows = list(read_rows(args.csv))
    if args.verbose:
        print(f"Rows parsed: {len(rows)}")
        # Peek first 3
        for peek in rows[:3]:
            print("[ROW]", peek)

    seed_from_csv(table, rows, args.bucket, args.locale, args.prefix, verbose=args.verbose)

if __name__ == "__main__":
    main()
