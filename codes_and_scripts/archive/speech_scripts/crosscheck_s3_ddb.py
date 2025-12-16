'''
crosscheck_s3_ddb.py
- Verifies that every DynamoDB item (canonical ko-KR/<response_hash>.wav) actually exists in S3.
- Optionally lists/deletes legacy 'approved/responses/' objects (orphans).
- Dry-run by default. Nothing is deleted unless --delete-legacy is passed.

Examples:
  # Just verify DDB → S3 existence (ko-KR locale)
  python crosscheck_s3_ddb.py --table UtteranceCache --bucket tts-bucket-250810 --locale ko-KR

  # Also scan legacy prefix and write CSV reports
  python crosscheck_s3_ddb.py --table UtteranceCache --bucket tts-bucket-250810 --locale ko-KR \
      --scan-legacy --legacy-prefix approved/responses/ --write-csv report.csv

  # Delete legacy orphans after confirming report looks good
  python crosscheck_s3_ddb.py --table UtteranceCache --bucket tts-bucket-250810 --locale ko-KR \
      --scan-legacy --legacy-prefix approved/responses/ --delete-legacy

Notes:
- Requires AWS credentials with read on DDB + S3, and delete if you use --delete-legacy.
- Uses paginated scans; you can limit with --max-items.
'''

import argparse, csv, sys
from collections import defaultdict

import boto3
from boto3.dynamodb.conditions import Attr
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

def build_key(locale: str, response_hash: str, dest_prefix: str) -> str:
    dest_prefix = (dest_prefix or "").strip("/")
    if not dest_prefix:
        return f"{locale}/{response_hash}.wav"
    # If prefix already ends with locale, don't duplicate
    if dest_prefix.split("/")[-1] == locale:
        return f"{dest_prefix}/{response_hash}.wav"
    return f"{dest_prefix}/{locale}/{response_hash}.wav"

def ddb_iter_items(tbl, locale: str, max_items: int = 0):
    fe = Attr("locale").eq(locale) & Attr("response_hash").exists()
    proj = "utterance_hash, locale, response_hash, audio_s3_uri, updated_at"
    scanned = 0
    lek = None
    while True:
        scan_kwargs = {"FilterExpression": fe, "ProjectionExpression": proj}
        if lek:
            scan_kwargs["ExclusiveStartKey"] = lek
        page = tbl.scan(**scan_kwargs)
        items = page.get("Items", [])
        for it in items:
            yield it
            scanned += 1
            if max_items and scanned >= max_items:
                return
        lek = page.get("LastEvaluatedKey")
        if not lek:
            break

def list_s3_prefix(s3, bucket: str, prefix: str):
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for c in resp.get("Contents", []):
            yield c["Key"]
        token = resp.get("NextContinuationToken")
        if not token:
            break

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="UtteranceCache", help="DynamoDB table name (e.g., UtteranceCache)")
    ap.add_argument("--bucket", default="tts-bucket-250810", help="S3 bucket (e.g., tts-bucket-250810)")
    ap.add_argument("--locale", default="ko-KR", help="Locale to check (default: ko-KR)")
    ap.add_argument("--dest-prefix", default="ko-KR", help="Canonical prefix (default: ko-KR)")
    ap.add_argument("--region", default=None, help="AWS region (optional; uses default profile if omitted)")
    ap.add_argument("--scan-legacy", action="store_true", help="Also scan legacy prefix for orphans")
    ap.add_argument("--legacy-prefix", default="approved/responses/", help="Legacy prefix to scan for orphans")
    ap.add_argument("--delete-legacy", action="store_true", help="Delete legacy orphans (DANGEROUS). Dry-run otherwise.")
    ap.add_argument("--max-items", type=int, default=0, help="Limit number of DDB items to check (debug)")
    ap.add_argument("--write-csv", default="", help="Write a CSV report of results")
    args = ap.parse_args()

    cfg = BotoConfig(max_pool_connections=32)
    ddb = boto3.resource("dynamodb", region_name=args.region, config=cfg)
    s3 = boto3.client("s3", region_name=args.region, config=cfg)
    tbl = ddb.Table(args.table)

    # ----- Pass 1: DDB → S3 existence verification -----
    print(f"[ddb] scanning items for locale={args.locale} (max_items={args.max_items or 'all'})")
    total = ok = missing = 0
    rows = []

    for it in ddb_iter_items(tbl, args.locale, max_items=args.max_items):
        total += 1
        rh = it["response_hash"]
        expected_key = build_key(args.locale, rh, args.dest_prefix)

        # HEAD the expected key
        exists = False
        try:
            s3.head_object(Bucket=args.bucket, Key=expected_key)
            exists = True
        except ClientError as e:
            exists = False

        if exists:
            ok += 1
            rows.append(["OK", it["utterance_hash"], rh, expected_key, it.get("audio_s3_uri","")])
        else:
            missing += 1
            rows.append(["MISSING", it["utterance_hash"], rh, expected_key, it.get("audio_s3_uri","")])

    print(f"[ddb] checked={total} ok={ok} missing={missing}")

    # ----- Pass 2: Legacy orphan scan (optional) -----
    orphan_keys = []
    if args.scan_legacy:
        legacy_prefix = args.legacy_prefix.strip("/")
        print(f"[s3 ] scanning legacy prefix: s3://{args.bucket}/{legacy_prefix}/")
        count = 0
        for key in list_s3_prefix(s3, args.bucket, legacy_prefix + "/"):
            count += 1
            # Treat all under legacy as deletable since DDB no longer points there
            orphan_keys.append(key)
        print(f"[s3 ] legacy objects found: {count}")

        # Optional delete
        if args.delete_legacy and orphan_keys:
            print("[del] deleting legacy orphans... (this may take a while)")
            CHUNK = 1000
            for i in range(0, len(orphan_keys), CHUNK):
                to_del = [{"Key": k} for k in orphan_keys[i:i+CHUNK]]
                s3.delete_objects(Bucket=args.bucket, Delete={"Objects": to_del, "Quiet": True})
            print(f"[del] deleted: {len(orphan_keys)}")

    # ----- CSV report -----
    if args.write_csv:
        with open(args.write_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["result", "utterance_hash", "response_hash", "expected_canonical_key", "ddb_audio_s3_uri"])
            w.writerows(rows)
        print(f"[out] wrote report: {args.write_csv}")

    # Summary
    print("\\n[summary]")
    print(f"  DDB items checked:   {total}")
    print(f"  OK in S3:            {ok}")
    print(f"  Missing in S3:       {missing}")
    if args.scan_legacy:
        print(f"  Legacy objects seen: {len(orphan_keys)}")
        if args.delete_legacy:
            print(f"  Legacy deleted:      {len(orphan_keys)}")

if __name__ == '__main__':
    sys.exit(main())