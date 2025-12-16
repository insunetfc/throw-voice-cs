#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill response-hash rows from existing utterance-hash rows in UtteranceCache.

- Source (scan): items where key_type is missing or 'utterance'
- Target (put):   item with utterance_hash = hash(normalized chatbot_response)
                  locale = same as source
                  key_type = 'response'
                  alias_of = source.utterance_hash
                  audio_s3_uri = source.audio_s3_uri
                  cached_text  = source.cached_text (what the audio says)
- Skips if response-hash row already exists (conditional put)
- Serial, resumable (checkpoint), with DRY_RUN

Env:
  AWS_REGION            (default: ap-northeast-2)
  UTT_CACHE_TABLE       (default: UtteranceCache)

Usage:
  python backfill_response_hash_rows.py --checkpoint .backfill.ckpt --dry-run
  python backfill_response_hash_rows.py --checkpoint .backfill.ckpt
"""

import os, re, sys, time, hashlib, unicodedata, argparse
from decimal import Decimal
from typing import Optional, Dict, Any

import boto3
import botocore

AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
UTT_CACHE_TABLE = os.getenv("UTT_CACHE_TABLE", "UtteranceCache")

session = boto3.session.Session(region_name=AWS_REGION)
ddb = session.client("dynamodb")

def normalize_utt(text: str) -> str:
    """Match your Lambda's normalize_utt: NFKC → strip → lower → collapse spaces → strip trailing .?!. """
    s = unicodedata.normalize("NFKC", text).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\.!\?]+$", "", s)
    return s

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def get_item(hash_hex: str, locale: str) -> Optional[Dict[str, Any]]:
    try:
        resp = ddb.get_item(
            TableName=UTT_CACHE_TABLE,
            Key={"utterance_hash": {"S": hash_hex}, "locale": {"S": locale}},
            ConsistentRead=False,
        )
        return resp.get("Item")
    except botocore.exceptions.BotoCoreError as e:
        print(f"[warn] ddb get error: {e}", file=sys.stderr)
        return None

def put_response_row(user_hash: str, locale: str, chatbot_response: str,
                     audio_s3_uri: str, created_at: int, dry_run: bool) -> str:
    resp_hash = sha256_hex(normalize_utt(chatbot_response))

    if resp_hash == user_hash:
        # Uncommon but possible; nothing to do.
        print(f"[skip] response_hash == user_hash ({resp_hash[:8]}), locale={locale}")
        return resp_hash

    item = {
        "utterance_hash": {"S": resp_hash},
        "locale": {"S": locale},
        "key_type": {"S": "response"},
        "audio_s3_uri": {"S": audio_s3_uri},
        "cached_text": {"S": chatbot_response},
        "chatbot_response": {"S": normalize_utt(chatbot_response)},
        "alias_of": {"S": user_hash},
        "approved_by": {"S": "backfill_response_hash_rows"},
        "created_at": {"N": str(created_at)},
        "status": {"S": "approved"},
        "updated_at": {"N": str(int(time.time()))},
        # intentionally NO original_utterance here
    }

    if dry_run:
        print(f"[dry-run] would put response row: hash={resp_hash[:8]} locale={locale}")
        return resp_hash

    try:
        ddb.put_item(
            TableName=UTT_CACHE_TABLE,
            Item=item,
            ConditionExpression="attribute_not_exists(utterance_hash) AND attribute_not_exists(locale)",
        )
        print(f"[ok] response row written: {resp_hash[:8]} locale={locale}")
    except botocore.exceptions.ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code == "ConditionalCheckFailedException":
            print(f"[exists] response row already present: {resp_hash[:8]} locale={locale}")
        else:
            raise
    return resp_hash

def scan_utterance_rows(limit: int = 1000, start_key=None):
    """Scan for sources: items with key_type missing OR key_type='utterance'."""
    filt = "attribute_not_exists(key_type) OR key_type = :utt"
    expr = {":utt": {"S": "utterance"}}
    kwargs = {
        "TableName": UTT_CACHE_TABLE,
        "FilterExpression": filt,
        "ExpressionAttributeValues": expr,
        "Limit": limit,
        "ProjectionExpression": "utterance_hash, locale, cached_text, audio_s3_uri, created_at",
    }
    if start_key:  # only pass if truthy
        kwargs["ExclusiveStartKey"] = start_key
    return ddb.scan(**kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=".backfill.ckpt", help="Path to checkpoint file (stores LastEvaluatedKey)")
    ap.add_argument("--batch", type=int, default=500, help="Scan page size (items per scan)")
    ap.add_argument("--max", type=int, default=0, help="Max total items to process (0 = all)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between puts (e.g., 0.01)")
    ap.add_argument("--dry-run", action="store_true", help="Do not write, just log actions")
    args = ap.parse_args()

    # load scan checkpoint (LastEvaluatedKey JSON) if present
    lek = None
    if os.path.exists(args.checkpoint):
        try:
            import json
            with open(args.checkpoint, "r", encoding="utf-8") as f:
                j = f.read().strip()
                if j:
                    lek = json.loads(j)
                    print(f"[info] resume from checkpoint (LEK present)")
        except Exception:
            pass

    processed = 0
    while True:
        try:
            resp = scan_utterance_rows(limit=args.batch, start_key=lek)
        except botocore.exceptions.ClientError as e:
            print(f"[error] scan failed: {e}")
            time.sleep(1.0)
            continue

        items = resp.get("Items", [])
        if not items:
            print("[info] scan: no items in this page")
        for it in items:
            try:
                user_hash = it["utterance_hash"]["S"]
                locale = it["locale"]["S"]
                chatbot_response = it.get("cached_text", {}).get("S") or ""
                audio_s3_uri = it.get("audio_s3_uri", {}).get("S") or ""
                created_at = int(it.get("created_at", {}).get("N", "0"))

                if not chatbot_response or not audio_s3_uri:
                    print(f"[skip] missing fields (cached_text/audio) for {user_hash[:8]} locale={locale}")
                    continue

                # Does response row already exist?
                resp_hash = sha256_hex(normalize_utt(chatbot_response))
                existing = get_item(resp_hash, locale)
                if existing:
                    print(f"[exists] response row already present: {resp_hash[:8]} locale={locale}")
                else:
                    put_response_row(user_hash, locale, chatbot_response, audio_s3_uri, created_at, args.dry_run)
                    if args.sleep > 0:
                        time.sleep(args.sleep)

                processed += 1
                if args.max and processed >= args.max:
                    print(f"[fin] reached max={args.max}")
                    return

            except botocore.exceptions.ClientError as e:
                code = e.response.get("Error", {}).get("Code")
                print(f"[warn] item error ({code}): {e}")
                time.sleep(0.2)
            except Exception as e:
                print(f"[warn] item error: {e}")
                time.sleep(0.2)

        # update LastEvaluatedKey checkpoint
        lek = resp.get("LastEvaluatedKey")
        import json
        with open(args.checkpoint, "w", encoding="utf-8") as f:
            f.write(json.dumps(lek or {}))
        if not lek:
            print("[fin] scan complete")
            break

if __name__ == "__main__":
    main()
