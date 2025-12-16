# Runtime: Python 3.11
# Permissions required on this Lambda role:
#   - connect:GetContactAttributes
#   - dynamodb:Query (and optionally GetItem/Scan if you use them)
#
# Env vars:
#   TRANSCRIPTS_TABLE = contactTranscriptSegments
#   SINGLE_TURN       = true | false   (default true -> Close after capture)
#   USE_CONNECT_ATTRS = true | false   (default true -> prefer Connect attributes)
#   MAX_TEXT_LEN      = int (default 2000)

import os
import json
import boto3
from typing import Tuple, Dict, Any
from boto3.dynamodb.conditions import Key
from datetime import datetime
import unicodedata, re

# --- AWS clients/resources ---
connect = boto3.client("connect")
dynamodb = boto3.resource("dynamodb")

# --- Config ---
TRANSCRIPTS_TABLE = os.environ.get("TRANSCRIPTS_TABLE", "contactTranscriptSegments")
SINGLE_TURN = os.environ.get("SINGLE_TURN", "true").lower() == "true"
USE_CONNECT_ATTRS = os.environ.get("USE_CONNECT_ATTRS", "true").lower() == "true"
MAX_TEXT_LEN = int(os.environ.get("MAX_TEXT_LEN", "2000"))
# Placeholders / transient messages you don't want to treat as valid text
PLACEHOLDERS = ["[Speech detected - processing...]"]

# ----------------- Normalization -----------------

def _canon(text: str) -> str:
    """Normalize Unicode (NFKC), collapse whitespace, and strip trailing full-stops/ellipses."""
    t = unicodedata.normalize("NFKC", str(text))
    t = re.sub(r"\s+", " ", t).strip()
    # Remove trailing full stops / ellipses in KR/EN punctuation
    t = re.sub(r"[\.。․·…‥]+$", "", t)
    return t

def normalize_text(t: str) -> str:
    """Apply canonical normalization and enforce MAX_TEXT_LEN."""
    return _canon(t)[:MAX_TEXT_LEN]

# ----------------- Entry -----------------

def lambda_handler(event, context):
    print("DialogCodeHook invoked")
    print(json.dumps(event, ensure_ascii=False))

    intent = (event.get("sessionState") or {}).get("intent") or {}
    slots = intent.get("slots") or {}
    sess_attrs = (event.get("sessionState") or {}).get("sessionAttributes") or {}

    # 1) Get best raw transcript (no normalization yet)
    raw_text, source = extract_best_transcript_raw(event)
    print(f"Using transcript from {source}: {raw_text!r}")

    intent_name = intent.get("name") or "CaptureUtterance"

    # Only override truly empty or weird system intents
    if intent_name not in ("CaptureUtterance", "NotInterestedIntent"):
        print(f"Intent '{intent_name}' not recognized → forcing to 'CaptureUtterance'")
        intent_name = "CaptureUtterance"

    # 2) Early silence guard with ONE-SHOT REPROMPT
    text = (raw_text or "").strip()
    if not text:
        # Get current reprompt count
        reprompt_count_str = sess_attrs.get("reprompt_count", "0")
        try:
            reprompt_count = int(reprompt_count_str) if reprompt_count_str else 0
        except (ValueError, TypeError):
            reprompt_count = 0

        print(f"⚠️ No speech detected. Current reprompt_count: {reprompt_count}")

        if reprompt_count == 0:
            # First silence -> increment counter and request reprompt
            sess_attrs["reprompt_count"] = "1"
            sess_attrs["reprompt_needed"] = "true"
            print("→ Setting reprompt_count=1, requesting reprompt")

            return {
                "sessionState": {
                    "dialogAction": {"type": "Close"},
                    "intent": {
                        "name": intent_name,
                        "state": "Failed"
                    },
                    "sessionAttributes": sess_attrs
                },
                "messages": []
            }
        elif reprompt_count == 1:
            # Second silence after reprompt -> mark no-input and end
            sess_attrs["error_no_input"] = "true"
            sess_attrs["reprompt_count"] = "2"
            print("→ Second silence after reprompt, setting count=2")

            return {
                "sessionState": {
                    "dialogAction": {"type": "Close"},
                    "intent": {
                        "name": intent_name,
                        "state": "Failed"
                    },
                    "sessionAttributes": sess_attrs
                },
                "messages": []
            }
        else:
            # reprompt_count >= 2 (guard)
            sess_attrs["error_no_input"] = "true"
            print(f"→ Unexpected reprompt_count={reprompt_count}, ending call")

            return {
                "sessionState": {
                    "dialogAction": {"type": "Close"},
                    "intent": {
                        "name": intent_name,
                        "state": "Failed"
                    },
                    "sessionAttributes": sess_attrs
                },
                "messages": []
            }

    # 3) Fill UserInput slot from transcript (force backfill even if Lex didn't capture it)
    if not get_slot_value(slots, "UserInput"):
        normalized = normalize_text(text)
        slots["UserInput"] = {
            "value": {
                "originalValue": raw_text,
                "interpretedValue": normalized
            }
        }
        print(f"✓ Filled UserInput slot (normalized): {normalized!r}")
    else:
        normalized = normalize_text(get_slot_value(slots, "UserInput"))

    # 4) Reset reprompt flags on successful speech
    for k in ("reprompt_count", "reprompt_needed", "error_no_input"):
        if k in sess_attrs:
            del sess_attrs[k]

    # 5) Close single-turn or continue dialog
    if SINGLE_TURN and get_slot_value(slots, "UserInput"):
        return close_intent(
            event,
            intent_name,   # normalized
            normalized,
            sess_attrs
        )
    else:
        return delegate_intent(
            event,
            intent_name,   # normalized
            slots,
            sess_attrs
        )

# ----------------- Helpers -----------------

def extract_best_transcript_raw(event: Dict[str, Any]) -> Tuple[str, str]:
    """
    Return (raw_text, source) with preference:
    1) Connect contact attributes (live_transcript)
    2) DynamoDB latest FINAL for this contact
    3) Lex inputTranscript
    """
    # 1) Connect attributes first (fast path for barge-in)
    if USE_CONNECT_ATTRS:
        cid, iid = extract_connect_ids(event)
        if cid and iid:
            t = get_connect_live_transcript(iid, cid)
            if t:
                return t, "connect"

    # 2) DynamoDB latest FINAL (contactTranscriptSegments)
    cid, _ = extract_connect_ids(event)
    if cid:
        t = get_latest_final_from_ddb(cid)
        if t:
            return t, "ddb"

    # 3) Fallback to Lex's own transcript
    t = (event.get("inputTranscript") or "").strip()
    if t:
        return t, "lex"

    return "", "none"

def extract_connect_ids(event: Dict[str, Any]):
    """Lex V2 → Connect requestAttributes provide these by default (if wired)."""
    req_attrs = event.get("requestAttributes") or {}
    return (
        req_attrs.get("x-amz-lex:connect-contact-id"),
        req_attrs.get("x-amz-lex:connect-instance-id"),
    )

def get_connect_live_transcript(instance_id: str, contact_id: str) -> str:
    """Read live transcript published by your transcriber via UpdateContactAttributes."""
    try:
        resp = connect.get_contact_attributes(
            InstanceId=instance_id,
            InitialContactId=contact_id
        )
        attrs = resp.get("Attributes") or {}
        t = (attrs.get("live_transcript") or "").strip()
        status = (attrs.get("transcript_status") or "").strip().upper()
        if not t or t in PLACEHOLDERS:
            return ""
        print(f"[Connect] status={status or 'N/A'}")
        return t
    except Exception as e:
        print(f"get_contact_attributes failed: {e}")
        return ""

def get_latest_final_from_ddb(contact_id: str) -> str:
    """
    Table: contactTranscriptSegments
    Expected attributes per item:
      - ContactId (PK)
      - IsPartial (bool)
      - EndTime (number) or LoggedOn (ISO timestamp)
      - Transcript (string)
    Strategy:
      - Query all items for the ContactId (paginated)
      - Filter IsPartial == False
      - Sort newest by EndTime (fallback: LoggedOn) and return its Transcript
    """
    try:
        table = dynamodb.Table(TRANSCRIPTS_TABLE)
        items = []
        kwargs = {
            "KeyConditionExpression": Key("ContactId").eq(contact_id),
            "ConsistentRead": True
        }
        while True:
            resp = table.query(**kwargs)
            items.extend(resp.get("Items") or [])
            lek = resp.get("LastEvaluatedKey")
            if not lek:
                break
            kwargs["ExclusiveStartKey"] = lek

        finals = [it for it in items if not it.get("IsPartial")]
        if not finals:
            return ""

        def sort_key(it):
            if "EndTime" in it:
                try:
                    return float(it["EndTime"])
                except Exception:
                    pass
            if "LoggedOn" in it:
                try:
                    # e.g. 2025-10-24T00:44:12.084447Z
                    return datetime.fromisoformat(it["LoggedOn"].replace("Z", "+00:00")).timestamp()
                except Exception:
                    pass
            return 0.0

        finals.sort(key=sort_key, reverse=True)
        t = (finals[0].get("Transcript") or "").strip()
        return t
    except Exception as e:
        print(f"DDB lookup failed: {e}")
        return ""

def get_slot_value(slots: Dict[str, Any], name: str) -> str:
    v = (slots or {}).get(name) or {}
    v = v.get("value") or {}
    return v.get("interpretedValue") or v.get("originalValue")

def delegate_intent(event: Dict[str, Any], intent_name: str, slots: Dict[str, Any], sess_attrs: Dict[str, Any]):
    return {
        "sessionState": {
            "dialogAction": {"type": "Delegate"},
            "intent": {"name": intent_name, "state": "InProgress", "slots": slots or {}},
            "sessionAttributes": sess_attrs
        }
    }

def close_intent(event, intent_name: str, message: str, sess_attrs: Dict[str, Any]):
    """Single-turn: immediately close with the captured text."""
    return {
        "sessionState": {
            "dialogAction": {"type": "Close"},
            "intent": {
                "name": intent_name,
                "state": "Fulfilled",
                "confirmationState": "None",
                "slots": {
                    "UserInput": {
                        "value": {
                            "originalValue": message,
                            "interpretedValue": message
                        }
                    }
                }
            },
            "sessionAttributes": sess_attrs
        }
    }
