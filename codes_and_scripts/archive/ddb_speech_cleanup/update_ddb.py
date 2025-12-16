import os, re, unicodedata, hashlib, argparse, datetime, sys
import boto3
from botocore.exceptions import ClientError
from boto3.dynamodb.types import TypeSerializer

# ---------------- Normalizer (MUST match Lambda) ----------------
def normalize_for_key(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\.!\?]+$", "", s)  # strip trailing punctuation often unstable in ASR
    return s

def short_hash(text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return h[:16]

def hash_from_text(text: str) -> str:
    return short_hash(normalize_for_key(text))

def iso_now() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# ---------------- Helpers ----------------
def pick_response_text(item: dict) -> str | None:
    # Try common field names for the response text
    for key in ["ChatAnswer", "response_text", "chatbot_answer", "answer", "ResponseText"]:
        if key in item and isinstance(item[key], str) and item[key].strip():
            return item[key]
    # Some datasets may store under 'reply' or 'text'
    for key in ["reply", "text"]:
        if key in item and isinstance(item[key], str) and item[key].strip():
            return item[key]
    return None

def best_audio_fields(item: dict) -> tuple[str | None, str | None]:
    """
    Returns (response_audio_uri, audio_s3_uri) to set.
    Prefers response_audio_uri when present; mirrors to audio_s3_uri.
    """
    rau = item.get("response_audio_uri")
    a = item.get("audio_s3_uri")
    if rau:
        return rau, rau
    return None, a

# ---------------- Single-item migration (utterance) ----------------
def migrate_utterance_item(table, it: dict, region: str, dry_run: bool = False) -> tuple[str, dict]:
    """
    Moves utterance item to normalized key and normalizes original_utterance.
    Returns (action_string, normalized_item_dict)
    """
    orig = it.get("original_utterance")
    locale = it.get("locale")
    old_h = it.get("utterance_hash")
    if not orig or not locale or not old_h:
        return "skip_missing", it

    norm = normalize_for_key(orig)
    new_h = hash_from_text(orig)

    # A) Key unchanged -> just UPDATE normalized original_utterance (idempotent)
    if new_h == old_h:
        if norm != orig:
            if dry_run:
                print(f"[DRY] UPDATE ({old_h},{locale}) set original_utterance='{norm}'")
                new_item = dict(it)
                new_item["original_utterance"] = norm
                new_item["normalized_at"] = iso_now()
                new_item["legacy_hash"] = old_h
                return "updated_in_place", new_item
            table.update_item(
                Key={"utterance_hash": old_h, "locale": locale},
                UpdateExpression="SET original_utterance = :norm, normalized_at = :ts, legacy_hash = :lh",
                ExpressionAttributeValues={":norm": norm, ":ts": iso_now(), ":lh": old_h},
                ConditionExpression="attribute_exists(utterance_hash) AND attribute_exists(locale)"
            )
            new_item = dict(it)
            new_item["original_utterance"] = norm
            new_item["normalized_at"] = iso_now()
            new_item["legacy_hash"] = old_h
            return "updated_in_place", new_item
        else:
            return "skipped_same", it

    # B) Key changed
    # 1) Check if target exists
    tgt = table.get_item(Key={"utterance_hash": new_h, "locale": locale}, ConsistentRead=True).get("Item")

    if tgt:
        # Merge policy: prefer response_audio_uri; keep status; carry any missing fields
        expr_parts, vals = [], {}
        rau, a = best_audio_fields(it)
        if rau and rau != tgt.get("response_audio_uri"):
            expr_parts += ["response_audio_uri = :rau", "audio_s3_uri = :rau"]
            vals[":rau"] = rau
        elif a and not tgt.get("audio_s3_uri"):
            expr_parts.append("audio_s3_uri = :a")
            vals[":a"] = a

        if it.get("status") and it["status"] != tgt.get("status"):
            expr_parts.append("status = :st")
            vals[":st"] = it["status"]

        expr_parts.append("original_utterance = :norm")
        vals[":norm"] = norm
        expr_parts.append("normalized_at = :ts")
        vals[":ts"] = iso_now()
        expr_parts.append("legacy_hash = :lh")
        vals[":lh"] = old_h

        if dry_run:
            print(f"[DRY] MERGE target ({new_h},{locale}); DELETE old ({old_h},{locale})")
        else:
            if expr_parts:
                table.update_item(
                    Key={"utterance_hash": new_h, "locale": locale},
                    UpdateExpression="SET " + ", ".join(expr_parts),
                    ExpressionAttributeValues=vals,
                    ConditionExpression="attribute_exists(utterance_hash) AND attribute_exists(locale)"
                )
            table.delete_item(
                Key={"utterance_hash": old_h, "locale": locale},
                ConditionExpression="attribute_exists(utterance_hash) AND attribute_exists(locale)"
            )
        new_item = dict(it)
        new_item["utterance_hash"] = new_h
        new_item["original_utterance"] = norm
        new_item["normalized_at"] = iso_now()
        new_item["legacy_hash"] = old_h
        return "merged_into_existing_and_deleted_old", new_item
    else:
        # Target missing: create atomically, then delete old
        if dry_run:
            print(f"[DRY] PUT new ({new_h},{locale}); DELETE old ({old_h},{locale})")
            new_item = dict(it)
            new_item["utterance_hash"] = new_h
            new_item["original_utterance"] = norm
            new_item["normalized_at"] = iso_now()
            new_item["legacy_hash"] = old_h
            return "put_new_and_deleted_old", new_item

        client = boto3.client("dynamodb", region_name=region)
        ser = TypeSerializer()
        new_item = dict(it)
        new_item["utterance_hash"] = new_h
        new_item["original_utterance"] = norm
        new_item["normalized_at"] = iso_now()
        new_item["legacy_hash"] = old_h
        # Prefer response_audio_uri when present
        rau, a = best_audio_fields(new_item)
        if rau:
            new_item["audio_s3_uri"] = rau

        def av(d):
            return {k: ser.serialize(v) for (k, v) in d.items() if v is not None}

        try:
            client.transact_write_items(
                TransactItems=[
                    {
                        "Put": {
                            "TableName": table.name,
                            "Item": av(new_item),
                            "ConditionExpression": "attribute_not_exists(utterance_hash) AND attribute_not_exists(locale)"
                        }
                    },
                    {
                        "Delete": {
                            "TableName": table.name,
                            "Key": av({"utterance_hash": old_h, "locale": locale}),
                            "ConditionExpression": "attribute_exists(utterance_hash) AND attribute_exists(locale)"
                        }
                    }
                ]
            )
            return "put_new_and_deleted_old", new_item
        except ClientError as e:
            print(f"[ERR] transact {old_h}->{new_h}: {e}", file=sys.stderr)
            return "failed", it

# ---------------- Ensure response-hash row ----------------
def upsert_response_row(table, source_item: dict, dry_run: bool = False) -> str:
    locale = source_item.get("locale")
    if not locale:
        return "skip_missing_response_locale"

    resp_text = pick_response_text(source_item)
    if not resp_text:
        return "skip_missing_response_text"

    resp_norm = normalize_for_key(resp_text)
    resp_hash = short_hash(resp_norm)

    rau, a = best_audio_fields(source_item)

    # Build upsert via UpdateItem (creates if not exists)
    update_parts = [
        "response_text = :rt",
        "response_text_normalized = :rtn",
        "last_linked_from = :src",
        "updated_at = :ts"
    ]
    values = {
        ":rt": resp_text,
        ":rtn": resp_norm,
        ":src": source_item.get("utterance_hash"),
        ":ts": iso_now(),
    }

    # Prefer response_audio_uri, mirror into audio_s3_uri
    if rau:
        update_parts += ["response_audio_uri = :rau", "audio_s3_uri = :rau"]
        values[":rau"] = rau
    elif a:
        update_parts.append("audio_s3_uri = :a")
        values[":a"] = a

    # Carry status if available
    if source_item.get("status"):
        update_parts.append("status = :st")
        values[":st"] = source_item["status"]

    if dry_run:
        print(f"[DRY] UPSERT response-row ({resp_hash},{locale}) linked from {source_item.get('utterance_hash')}")
        return "response_upserted_dry"

    table.update_item(
        Key={"utterance_hash": resp_hash, "locale": locale},
        UpdateExpression="SET " + ", ".join(update_parts),
        ExpressionAttributeValues=values
        # No condition -> upsert
    )
    return "response_upserted"

# ---------------- Table scan driver ----------------
def main():
    ap = argparse.ArgumentParser(description="Normalize utterance rows and ensure response-hash reuse rows exist.")
    ap.add_argument("--table", default="UtteranceCache", help="DynamoDB table name (e.g., UtteranceCache)")
    ap.add_argument("--region", default=os.getenv("AWS_REGION", "ap-northeast-2"))
    ap.add_argument("--dry-run", action="store_true", help="Only print intended actions")
    ap.add_argument("--limit", type=int, default=None, help="Stop after N items for testing")
    ap.add_argument("--approved-only", action="store_true", help="Only process status=approved utterance rows")
    args = ap.parse_args()

    dynamodb = boto3.resource("dynamodb", region_name=args.region)
    table = dynamodb.Table(args.table)

    counters = {k: 0 for k in [
        "scanned","processed",
        "updated_in_place","merged_into_existing_and_deleted_old","put_new_and_deleted_old",
        "skipped_same","skip_missing","failed",
        "response_upserted","response_upserted_dry",
        "skip_missing_response_locale","skip_missing_response_text"
    ]}

    start_key = None
    while True:
        scan_kwargs = {"Limit": 500, "ConsistentRead": True}
        if start_key:
            scan_kwargs["ExclusiveStartKey"] = start_key
        resp = table.scan(**scan_kwargs)
        items = resp.get("Items", [])
        if not items:
            break

        for it in items:
            counters["scanned"] += 1
            if args.limit and counters["processed"] >= args.limit:
                break

            if args.approved_only and it.get("status") != "approved":
                counters["skipped_same"] += 1
                continue

            # Migrate utterance row
            action, normalized_item = migrate_utterance_item(table, it, region=args.region, dry_run=args.dry_run)
            counters["processed"] += 1
            counters[action] = counters.get(action, 0) + 1

            # Ensure response-hash row exists & linked
            resp_action = upsert_response_row(table, normalized_item, dry_run=args.dry_run)
            counters[resp_action] = counters.get(resp_action, 0) + 1

        if args.limit and counters["processed"] >= args.limit:
            break

        start_key = resp.get("LastEvaluatedKey")
        if not start_key:
            break

    print("\n=== Summary ===")
    for k in ["scanned","processed",
              "updated_in_place","merged_into_existing_and_deleted_old","put_new_and_deleted_old",
              "skipped_same","skip_missing","failed",
              "response_upserted","response_upserted_dry",
              "skip_missing_response_locale","skip_missing_response_text"]:
        print(f"{k:42s} {counters.get(k,0)}")

if __name__ == "__main__":
    main()