#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DynamoDB duplicate cleaner (DRY-RUN by default), tolerant of missing key_type.

- Scans by locale, requires response_hash to exist
- Groups by response_hash
- Chooses a single winner per response_hash (latest updated_at; tie-break: prefer '/ko-KR/' in audio_s3_uri)
- Prints KEEP/DELETE plan (and optionally writes CSV)
- Only deletes if --apply is provided (otherwise dry run)
"""

import argparse
import csv
import sys
from collections import defaultdict

import boto3
from boto3.dynamodb.conditions import Attr

def scan_items(tbl, locale, require_key_type=None, projection=None):
    """
    Scan DDB for given locale; require response_hash to exist.
    Optionally require key_type == <value> if require_key_type is provided.
    """
    fe = Attr("locale").eq(locale) & Attr("response_hash").exists()
    if require_key_type:
        fe = fe & Attr("key_type").eq(require_key_type)

    scan_kwargs = {"FilterExpression": fe}
    if projection:
        scan_kwargs["ProjectionExpression"] = projection

    client = tbl.meta.client
    paginator = client.get_paginator("scan")
    for page in paginator.paginate(TableName=tbl.name, **scan_kwargs):
        for it in page.get("Items", []):
            yield it

def pick_winner(group):
    """Keep: max(updated_at), tie-break prefers '/ko-KR/' in audio_s3_uri."""
    def key_fn(it):
        updated = int(it.get("updated_at", 0) or 0)
        uri = (it.get("audio_s3_uri") or "")
        ko_bias = 1 if "/ko-KR/" in uri else 0
        return (updated, ko_bias)
    return max(group, key=key_fn)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="UtteranceCache")
    ap.add_argument("--region", default="ap-northeast-2")
    ap.add_argument("--locale", default="ko-KR")
    ap.add_argument("--require-key-type", default="", help="Optional: only include rows with this key_type")
    ap.add_argument("--write-plan", default="", help="Optional CSV path to write KEEP/DELETE plan")
    ap.add_argument("--apply", action="store_true", help="Apply deletes (default: dry-run)")
    args = ap.parse_args()

    ddb = boto3.resource("dynamodb", region_name=args.region)
    tbl = ddb.Table(args.table)

    projection = "utterance_hash, locale, response_hash, audio_s3_uri, updated_at"
    print(f"[scan] table={args.table} region={args.region} locale={args.locale}"
          f"{' key_type='+args.require_key_type if args.require_key_type else ''} (this may take a moment)")
    groups = defaultdict(list)
    cnt = 0
    for it in scan_items(tbl, args.locale,
                         require_key_type=(args.require_key_type or None),
                         projection=projection):
        groups[it["response_hash"]].append(it)
        cnt += 1
    print(f"[scan] collected {cnt} rows with response_hash in locale={args.locale}")

    # Build plan
    plan = []
    dupe_groups = 0
    delete_rows = 0

    for resp_hash, items in groups.items():
        if len(items) <= 1:
            continue
        dupe_groups += 1
        keep = pick_winner(items)
        keep_key = (keep["utterance_hash"], keep["locale"])
        print(f"\n[resp] {resp_hash}")
        print(f"  KEEP   uh={keep['utterance_hash']} loc={keep['locale']} "
              f"upd={keep.get('updated_at','')} uri={keep.get('audio_s3_uri','')}")
        plan.append(["KEEP", resp_hash, keep["utterance_hash"], keep["locale"],
                     keep.get("updated_at",""), keep.get("audio_s3_uri","")])

        for it in items:
            if (it["utterance_hash"], it["locale"]) == keep_key:
                continue
            delete_rows += 1
            print(f"  DELETE uh={it['utterance_hash']} loc={it['locale']} "
                  f"upd={it.get('updated_at','')} uri={it.get('audio_s3_uri','')}")
            plan.append(["DELETE", resp_hash, it["utterance_hash"], it["locale"],
                         it.get("updated_at",""), it.get("audio_s3_uri","")])

    print(f"\n[summary] duplicate response_hash groups: {dupe_groups}")
    print(f"[summary] rows to delete: {delete_rows}")
    print(f"[summary] mode: {'APPLY (will delete)' if args.apply else 'DRY-RUN (no changes)'}")

    # Optional: write plan
    if args.write_plan:
        with open(args.write_plan, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["action", "response_hash", "utterance_hash", "locale", "updated_at", "audio_s3_uri"])
            w.writerows(plan)
        print(f"[plan ] wrote: {args.write_plan}")

    # Apply deletes (only if requested)
    if args.apply:
        if delete_rows == 0:
            print("[apply] nothing to delete.")
            return
        confirm = input("Type 'delete' to proceed: ").strip().lower()
        if confirm != "delete":
            print("[apply] abort.")
            return
        with tbl.batch_writer() as batch:
            for action, resp_hash, uh, loc, upd, uri in plan:
                if action == "DELETE":
                    batch.delete_item(Key={"utterance_hash": uh, "locale": loc})
        print("[apply] delete requests submitted.")

if __name__ == "__main__":
    sys.exit(main())
