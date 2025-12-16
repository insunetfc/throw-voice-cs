#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Migrate S3 audio keys from a "bad" prefix (e.g., //ko-KR/) to a "good" prefix (e.g., ko-KR/),
then update DynamoDB UtteranceCache.audio_s3_uri to point to the new keys.

The script performs:
  1) S3 copy from s3://BUCKET/OLD_PREFIX -> s3://BUCKET/NEW_PREFIX (key-wise mapping)
  2) (optional) S3 delete of OLD_PREFIX after successful copy
  3) DynamoDB scan for items whose audio_s3_uri startswith s3://BUCKET/OLD_PREFIX and update to NEW_PREFIX

Keys & schema:
  - Table: UTT_CACHE_TABLE (default: UtteranceCache)
  - PK: (utterance_hash, locale)
  - Fields touched: audio_s3_uri, updated_at, (optional) prev_audio_s3_uri for rollback

Env (can be overridden by CLI flags):
  AWS_REGION        (default: ap-northeast-2)
  BUCKET            (required)
  OLD_PREFIX        (e.g., "//ko-KR/")
  NEW_PREFIX        (e.g., "ko-KR/")
  UTT_CACHE_TABLE   (default: UtteranceCache)

Usage examples:
  # Dry run (no writes), show plan
  python migrate_s3_prefix_and_update_ddb.py --bucket tts-bucket-250810 --old-prefix "//ko-KR/" --new-prefix "ko-KR/" --dry-run

  # Execute copy + DDB update (no delete of old keys)
  python migrate_s3_prefix_and_update_ddb.py --bucket tts-bucket-250810 --old-prefix "//ko-KR/" --new-prefix "ko-KR/"

  # After verifying, delete the old keys
  python migrate_s3_prefix_and_update_ddb.py --bucket tts-bucket-250810 --old-prefix "//ko-KR/" --new-prefix "ko-KR/" --delete-old
"""

import os
import re
import sys
import time
import argparse
from typing import Iterator, Tuple

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
UTT_CACHE_TABLE = os.getenv("UTT_CACHE_TABLE", "UtteranceCache")

def norm_prefix(p: str) -> str:
    """Normalize S3 prefix to not start with a slash; allow internal '//' by design."""
    if p.startswith("/"):
        p = p[1:]
    # Ensure trailing slash
    if p and not p.endswith("/"):
        p += "/"
    return p

def s3_iter_objects(s3, bucket: str, prefix: str) -> Iterator[dict]:
    """Yield S3 objects under the given prefix."""
    kwargs = {"Bucket": bucket, "Prefix": prefix}
    while True:
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []) or []:
            yield obj
        if resp.get("IsTruncated"):
            kwargs["ContinuationToken"] = resp.get("NextContinuationToken")
        else:
            break

def migrate_s3_objects(s3, bucket: str, old_prefix: str, new_prefix: str, dry_run: bool=False) -> Tuple[int, int]:
    """Copy each object from old_prefix to new_prefix. Returns (copied, skipped) counts.
       If destination exists, we overwrite by default (idempotent)."""
    copied = 0
    skipped = 0
    for obj in s3_iter_objects(s3, bucket, old_prefix):
        old_key = obj["Key"]
        if not old_key.startswith(old_prefix):
            continue
        suffix = old_key[len(old_prefix):]  # portion after old_prefix
        new_key = new_prefix + suffix
        if dry_run:
            print(f"[dry-run] copy s3://{bucket}/{old_key} -> s3://{bucket}/{new_key}")
            copied += 1
            continue
        try:
            s3.copy_object(
                Bucket=bucket,
                CopySource={"Bucket": bucket, "Key": old_key},
                Key=new_key,
                MetadataDirective="COPY",
                ContentType="audio/wav",  # keep it explicit
            )
            print(f"[ok] copied {old_key} -> {new_key}")
            copied += 1
        except ClientError as e:
            print(f"[ERR] copy failed for {old_key}: {e}")
            skipped += 1
    return copied, skipped

def delete_old_prefix(s3, bucket: str, old_prefix: str, dry_run: bool=False) -> int:
    """Delete all objects under old_prefix. Returns count deleted."""
    deleted = 0
    batch = []
    def flush_batch():
        nonlocal deleted, batch
        if not batch:
            return
        if dry_run:
            for it in batch:
                print(f"[dry-run] delete s3://{bucket}/{it['Key']}")
                deleted += 1
        else:
            try:
                s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": it["Key"]} for it in batch]})
                deleted += len(batch)
            except ClientError as e:
                print(f"[ERR] batch delete failed: {e}")
        batch = []

    for obj in s3_iter_objects(s3, bucket, old_prefix):
        batch.append({"Key": obj["Key"]})
        if len(batch) >= 1000:
            flush_batch()
    flush_batch()
    return deleted

def update_ddb_audio_paths(ddb, bucket: str, old_prefix: str, new_prefix: str, dry_run: bool=False, batch_size: int=200) -> Tuple[int, int]:
    """Scan UtteranceCache and update audio_s3_uri starting with old prefix to new prefix.
       Returns (updated, scanned) counts."""
    scanned = 0
    updated = 0
    lek = None
    old_uri_prefix = f"s3://{bucket}/{old_prefix}"
    new_uri_prefix = f"s3://{bucket}/{new_prefix}"
    proj = "utterance_hash, locale, audio_s3_uri"
    while True:
        scan_kwargs = {
            "TableName": UTT_CACHE_TABLE,
            "ProjectionExpression": proj,
            "Limit": batch_size,
        }
        if lek:
            scan_kwargs["ExclusiveStartKey"] = lek
        resp = ddb.scan(**scan_kwargs)
        items = resp.get("Items", [])
        for it in items:
            scanned += 1
            uh = it["utterance_hash"]["S"]
            loc = it["locale"]["S"]
            audio_uri = it.get("audio_s3_uri", {}).get("S", "")
            if not audio_uri or not audio_uri.startswith(old_uri_prefix):
                continue
            new_uri = new_uri_prefix + audio_uri[len(old_uri_prefix):]
            if dry_run:
                print(f"[dry-run] DDB update ({uh[:8]} {loc}): {audio_uri} -> {new_uri}")
                updated += 1
                continue
            try:
                ddb.update_item(
                    TableName=UTT_CACHE_TABLE,
                    Key={"utterance_hash": {"S": uh}, "locale": {"S": loc}},
                    UpdateExpression="SET audio_s3_uri = :u, updated_at = :t, prev_audio_s3_uri = :p",
                    ExpressionAttributeValues={
                        ":u": {"S": new_uri},
                        ":t": {"N": str(int(time.time()))},
                        ":p": {"S": audio_uri},
                    },
                )
                print(f"[ok] DDB updated ({uh[:8]} {loc}): {new_uri}")
                updated += 1
            except ClientError as e:
                print(f"[ERR] DDB update failed ({uh[:8]} {loc}): {e}")
        lek = resp.get("LastEvaluatedKey")
        if not lek:
            break
    return updated, scanned

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default=os.getenv("BUCKET"), required=False, help="S3 bucket name")
    ap.add_argument("--old-prefix", default=os.getenv("OLD_PREFIX"), required=False, help="Old/bad prefix (e.g., //ko-KR/)")
    ap.add_argument("--new-prefix", default=os.getenv("NEW_PREFIX"), required=False, help="New/good prefix (e.g., ko-KR/)")
    ap.add_argument("--delete-old", action="store_true", help="Delete old prefix after copy")
    ap.add_argument("--dry-run", action="store_true", help="Plan only; no writes")
    ap.add_argument("--batch", type=int, default=200, help="DDB scan page size")
    args = ap.parse_args()

    if not args.bucket:
        print("ERROR: --bucket (or BUCKET env) is required", file=sys.stderr)
        sys.exit(2)
    if not args.old_prefix or not args.new_prefix:
        print("ERROR: --old-prefix and --new-prefix are required (or set OLD_PREFIX/NEW_PREFIX env)", file=sys.stderr)
        sys.exit(2)

    bucket = args.bucket
    old_prefix = norm_prefix(args.old_prefix)
    new_prefix = norm_prefix(args.new_prefix)

    print(f"[info] bucket={bucket}")
    print(f"[info] old_prefix={old_prefix}")
    print(f"[info] new_prefix={new_prefix}")
    print(f"[info] table={UTT_CACHE_TABLE}")
    print(f"[info] dry_run={args.dry_run} delete_old={args.delete_old}")

    s3 = boto3.client("s3", region_name=AWS_REGION, config=BotoConfig(max_pool_connections=32))
    ddb = boto3.client("dynamodb", region_name=AWS_REGION)

    # 1) Copy S3 objects
    copied, skipped = migrate_s3_objects(s3, bucket, old_prefix, new_prefix, dry_run=args.dry_run)
    print(f"[sum] s3 copied={copied} (skipped/logged={skipped})")

    # 2) Update DDB audio_s3_uri
    updated, scanned = update_ddb_audio_paths(ddb, bucket, old_prefix, new_prefix, dry_run=args.dry_run, batch_size=args.batch)
    print(f"[sum] ddb scanned={scanned} updated={updated}")

    # 3) Optionally delete old keys
    if args.delete_old:
        deleted = delete_old_prefix(s3, bucket, old_prefix, dry_run=args.dry_run)
        print(f"[sum] s3 deleted={deleted}")

if __name__ == "__main__":
    main()
