#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch approval script for inbox items
Reads all unapproved items from DynamoDB and processes them through the approval system
Similar to generate_preset.py but for inbox approval workflow
"""

import os, time, argparse, json, hashlib, uuid
import boto3
from botocore.exceptions import ClientError
from typing import List, Dict, Any, Optional
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache
from decimal import Decimal
import boto3
from boto3.dynamodb.types import TypeDeserializer
from boto3.dynamodb.conditions import Attr

# ---------- Config ----------
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
INBOX_TABLE_NAME = os.getenv("INBOX_TABLE_NAME", "SttInboxNew")  # Replace with actual table name
TTS_BUCKET = os.getenv("TTS_BUCKET", "tts-bucket-250810")
TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://localhost:8000")
TTS_TOKEN = os.getenv("TTS_TOKEN", "")

# DynamoDB setup
ddb = boto3.resource("dynamodb", region_name=AWS_REGION)
inbox_table = ddb.Table(INBOX_TABLE_NAME)
utterance_cache_table = ddb.Table(os.getenv("UTTERANCE_CACHE_TABLE_NAME", "UtteranceCache"))

# S3 setup  
s3 = boto3.client("s3", region_name=AWS_REGION)

# Thread-safe counters
class Counter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()
    
    def increment(self):
        with self.lock:
            self.value += 1
            return self.value

# Global counters
processed_counter = Counter()
failed_counter = Counter()
success_counter = Counter()

def convert_dynamodb_item(item):
    """Convert DynamoDB item with Decimal types to regular Python types"""
    converted = {}
    for key, value in item.items():
        if isinstance(value, Decimal):
            # Convert Decimal to appropriate Python type
            converted[key] = float(value) if value % 1 != 0 else int(value)
        elif isinstance(value, dict):
            # Recursively convert nested dicts
            converted[key] = convert_dynamodb_item(value)
        elif isinstance(value, list):
            # Convert list items
            converted[key] = [convert_dynamodb_item(v) if isinstance(v, dict) else (float(v) if isinstance(v, Decimal) and v % 1 != 0 else int(v) if isinstance(v, Decimal) else v) for v in value]
        else:
            converted[key] = value
    return converted

def generate_hash(text: str) -> str:
    """Generate SHA256 hash for text"""
    normalized = normalize_for_key(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

def get_unapproved_inbox_items(limit: Optional[int] = None, 
                              contact_id: Optional[str] = None,
                              locale: Optional[str] = None,
                              inbox_ids: Optional[List[str]] = None,
                              date_after: Optional[str] = None,
                              date_before: Optional[str] = None,
                              text_contains: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Scan DynamoDB table for unapproved inbox items with flexible filtering
    """
    print("Scanning DynamoDB for unapproved inbox items...")
    
    # If specific inbox_ids provided, fetch those directly
    if inbox_ids:
        print(f"Fetching specific inbox_ids: {inbox_ids}")
        items = []
        for inbox_id in inbox_ids:
            try:
                response = inbox_table.get_item(Key={"inbox_id": inbox_id})
                if 'Item' in response:
                    item = convert_dynamodb_item(response['Item'])
                    # Check if it's actually unapproved
                    status = item.get('review_status')
                    if not status or status not in ['approved', 'approved_dual_hash_fixed', 'approved_both']:
                        items.append(item)
                        print(f"  Found unapproved: {inbox_id}")
                    else:
                        print(f"  Skipping already approved: {inbox_id} (status: {status})")
                else:
                    print(f"  Not found: {inbox_id}")
            except Exception as e:
                print(f"  Error fetching {inbox_id}: {e}")
        return items
    
    # Use a simpler scan approach without complex filter expressions to avoid Decimal issues
    try:
        print("Using simple scan to avoid Decimal comparison issues...")
        
        scan_params = {}
        if limit:
            scan_params["Limit"] = limit
            
        response = inbox_table.scan(**scan_params)
        all_items = []
        
        # Get all items first, then filter in Python
        while True:
            raw_items = response.get('Items', [])
            for raw_item in raw_items:
                converted_item = convert_dynamodb_item(raw_item)
                all_items.append(converted_item)
            
            if 'LastEvaluatedKey' not in response:
                break
                
            scan_params['ExclusiveStartKey'] = response['LastEvaluatedKey']
            response = inbox_table.scan(**scan_params)
        
        print(f"Scanned {len(all_items)} total items, now filtering...")
        
        # Filter in Python to avoid DynamoDB Decimal issues
        filtered_items = []
        for item in all_items:
            # Check if unapproved
            status = item.get('review_status')
            if status and status in ['approved', 'approved_dual_hash_fixed', 'approved_both']:
                continue  # Skip approved items
            
            # Apply optional filters
            if contact_id and item.get('contact_id') != contact_id:
                continue
                
            if locale and item.get('locale') != locale:
                continue
                
            # Date filters
            if date_after or date_before:
                turn_ts = item.get('turn_ts')
                if turn_ts:
                    try:
                        if date_after:
                            import datetime
                            if isinstance(date_after, str):
                                date_obj = datetime.datetime.fromisoformat(date_after.replace('Z', '+00:00'))
                                timestamp_after = int(date_obj.timestamp())
                            else:
                                timestamp_after = int(date_after)
                            if turn_ts < timestamp_after:
                                continue
                                
                        if date_before:
                            import datetime
                            if isinstance(date_before, str):
                                date_obj = datetime.datetime.fromisoformat(date_before.replace('Z', '+00:00'))
                                timestamp_before = int(date_obj.timestamp())
                            else:
                                timestamp_before = int(date_before)
                            if turn_ts > timestamp_before:
                                continue
                    except Exception as e:
                        print(f"Date filter error for item {item.get('inbox_id', 'unknown')}: {e}")
                        continue
            
            # Text content filter
            if text_contains:
                utterance = item.get('utterance_text', '') or ''
                response_text = item.get('proposed_response_text', '') or ''
                if text_contains.lower() not in utterance.lower() and text_contains.lower() not in response_text.lower():
                    continue
            
            filtered_items.append(item)
            
            # Apply limit after filtering
            if limit and len(filtered_items) >= limit:
                break
        
        print(f"Found {len(filtered_items)} unapproved items after filtering")
        return filtered_items
        
    except Exception as e:
        print(f"Error scanning DynamoDB: {e}")
        return []

def generate_tts_audio(text: str, locale: str = "ko-KR", voice_style: str = "agent") -> Optional[str]:
    """
    Generate TTS audio using your existing TTS server
    Returns S3 URI or None if failed
    """
    try:
        headers = {"Content-Type": "application/json"}
        if TTS_TOKEN:
            headers["Authorization"] = f"Bearer {TTS_TOKEN}"
            
        # Use your existing /synthesize endpoint
        payload = {
            "text": text,
            "sample_rate": 8000,
            "key_prefix": f"approved/responses/{generate_hash(text)}"
        }
        
        response = requests.post(
            f"{TTS_BASE_URL}/synthesize",
            json=payload,
            headers=headers,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            bucket = data.get("bucket")
            key = data.get("key")
            if bucket and key:
                return f"s3://{bucket}/{key}"
        else:
            print(f"TTS request failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"TTS generation error: {e}")
        return None

def cache_response_audio_under_both_hashes(user_input: str, chatbot_response: str,
                                         response_audio_uri: str, locale: str,
                                         contact_id: str, inbox_id: str) -> Dict[str, Any]:
    """
    Cache the response audio under both user input hash and response hash in UtteranceCache table
    Following the exact pattern from existing approved entries with approval_type='chatbot_response'
    """
    try:
        print(f"    Debug: user_input parameter = '{user_input}'")
        print(f"    Debug: chatbot_response parameter = '{chatbot_response[:50]}...'")
        
        user_hash = generate_hash(normalize_for_key(user_input))
        response_hash = generate_hash(normalize_for_key(chatbot_response))
        current_time = int(time.time())
        
        # Cache under user utterance hash (for matching user input)
        try:
            cache_item = {
                "utterance_hash": user_hash,
                "locale": locale,
                "approval_type": "chatbot_response",  # Match existing schema pattern
                "approved_by": "batch_approval_script",
                "audio_s3_uri": response_audio_uri,
                "cached_text": chatbot_response,  # The actual response text that was cached as audio
                "chatbot_response": normalize_for_key(chatbot_response),  # The chatbot's response text
                "contact_id": contact_id,
                "created_at": current_time,
                "inbox_id": inbox_id,
#                 "normalized_response": normalize_for_key(chatbot_response),
#                 "normalized_utterance": normalize_for_key(user_input),
                "notes": "batch approval - user hash cache",
                "num_hits": Decimal('0'),  # Use Decimal instead of int/float
                "original_utterance": user_input,  # This should be the original user utterance from SttInboxNew
                "response_hash": response_hash,
                "status": "approved",
                "updated_at": Decimal(str(current_time))  # Convert to Decimal for DynamoDB
            }
            
            print(f"    Debug: About to cache user entry with original_utterance = '{cache_item['original_utterance']}'")
            utterance_cache_table.put_item(Item=cache_item)
            print(f"    Cached under user hash: {user_hash}")
        except Exception as e:
            print(f"    ERROR: Failed to cache under user hash {user_hash}: {e}")
            return {"success": False, "error": f"User hash cache failed: {e}"}
        
#         # Cache under response hash (for reusing same response audio)
#         try:
#             utterance_cache_table.put_item(Item={
#                 "utterance_hash": response_hash,  # Using response_hash as the key
#                 "locale": locale,
#                 "approval_type": "chatbot_response",
#                 "approved_by": "batch_approval_script",
#                 "audio_s3_uri": response_audio_uri,
#                 "cached_text": chatbot_response,  # Same response text
#                 "chatbot_response": chatbot_response,
#                 "contact_id": contact_id,
#                 "created_at": current_time,
#                 "inbox_id": inbox_id,
#                 "normalized_response": normalize_for_key(chatbot_response),
#                 "normalized_utterance": normalize_for_key(chatbot_response),  # For response cache, treat response as the "utterance"
#                 "notes": "batch approval - response hash cache",
#                 "num_hits": Decimal('0'),  # Use Decimal
#                 "original_utterance": chatbot_response,  # For response hash cache, the "original utterance" is the response text
#                 "response_hash": response_hash,
#                 "status": "approved",
#                 "updated_at": Decimal(str(current_time))  # Use Decimal
#             })
#             print(f"    Cached under response hash: {response_hash}")
#         except Exception as e:
#             print(f"    ERROR: Failed to cache under response hash {response_hash}: {e}")
#             return {"success": False, "error": f"Response hash cache failed: {e}"}
        
        return {
            "success": True,
            "user_hash": user_hash,
            "response_hash": response_hash,
            "cached_uri": response_audio_uri,
            "cached_both": True
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def approve_single_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Approve a single inbox item - mirrors your Lambda approval logic
    """
    inbox_id = item.get("inbox_id")
    if not inbox_id:
        return {"success": False, "error": "Missing inbox_id", "inbox_id": "unknown"}
    
    try:
        # Extract required fields - match SttInboxNew table structure
        user_input = item.get("utterance_text", "")  # SttInboxNew uses utterance_text
        chatbot_response = item.get("proposed_response_text", "")  # SttInboxNew uses proposed_response_text
        locale = item.get("locale", "ko-KR")
        contact_id = item.get("contact_id", "unknown")
        
        if not user_input or not chatbot_response:
            return {
                "success": False, 
                "error": "Missing utterance_text or proposed_response_text",
                "inbox_id": inbox_id
            }
        
        print(f"Processing {inbox_id[:8]}... User: \"{user_input[:50]}\" Response: \"{chatbot_response[:50]}\"")
        
        # Generate TTS audio
        response_audio_uri = generate_tts_audio(
            text=chatbot_response,
            locale=locale,
            voice_style="agent"
        )
        
        if not response_audio_uri:
            return {
                "success": False,
                "error": "TTS generation failed", 
                "inbox_id": inbox_id
            }
        
        # Cache under both hashes
        cache_result = cache_response_audio_under_both_hashes(
            user_input=user_input,
            chatbot_response=chatbot_response,
            response_audio_uri=response_audio_uri,
            locale=locale,
            contact_id=contact_id,
            inbox_id=inbox_id
        )
        
        if not cache_result.get("success"):
            return {
                "success": False,
                "error": f"UtteranceCache failed: {cache_result.get('error')}",
                "inbox_id": inbox_id
            }
        
        # Update inbox item status
        try:
            inbox_table.update_item(
                Key={"inbox_id": inbox_id},
                UpdateExpression="SET review_status = :status, approved_at = :ts, response_audio_uri = :uri, dual_hash_cached = :dual",
                ExpressionAttributeValues={
                    ":status": "approved_dual_hash_batch",
                    ":ts": int(time.time()),
                    ":uri": response_audio_uri,
                    ":dual": True
                }
            )
        except Exception as update_error:
            print(f"Warning: Inbox update failed for {inbox_id}: {update_error}")
            # Don't fail the whole operation just because status update failed
        
        return {
            "success": True,
            "inbox_id": inbox_id,
            "response_audio_uri": response_audio_uri,
            "user_hash": cache_result.get("user_hash"),
            "response_hash": cache_result.get("response_hash"),
            "dual_hash_cached": cache_result.get("success", False)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "inbox_id": inbox_id
        }
    

def normalize_for_key(s: str) -> str:
    """Normalize text for consistent hashing"""
    import unicodedata, re
    s = unicodedata.normalize("NFKC", s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\.!\?]+$", "", s)  # strip trailing punctuation often unstable in ASR
    return s

def process_batch(items: List[Dict[str, Any]], max_workers: int = 5) -> List[Dict[str, Any]]:
    """
    Process multiple inbox items concurrently
    """
    results = []
    
    print(f"Processing {len(items)} items with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(approve_single_item, item): item 
            for item in items
        }
        
        # Process completed tasks
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result = future.result()
                results.append(result)
                
                # Update counters
                processed_counter.increment()
                if result.get("success"):
                    count = success_counter.increment()
                    print(f"✓ [{count}/{len(items)}] {result.get('inbox_id', 'unknown')[:8]} approved")
                else:
                    count = failed_counter.increment()
                    print(f"✗ [{processed_counter.value}/{len(items)}] {result.get('inbox_id', 'unknown')[:8]} failed: {result.get('error')}")
                    
            except Exception as e:
                failed_counter.increment()
                inbox_id = item.get("inbox_id", "unknown")
                print(f"✗ [{processed_counter.value + 1}/{len(items)}] {inbox_id[:8]} exception: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "inbox_id": inbox_id
                })
    
    return results

def bulk_update_review_status(inbox_ids: List[str], new_status: str = "approved") -> Dict[str, Any]:
    """
    Bulk update review_status for multiple inbox items
    
    Args:
        inbox_ids: List of inbox_id strings to update
        new_status: Status to set (default: "approved")
    
    Returns:
        Dictionary with success/failure counts and details
    """
    success_count = 0
    failed_count = 0
    failed_items = []
    
    print(f"Updating review_status to '{new_status}' for {len(inbox_ids)} items...")
    
    for inbox_id in inbox_ids:
        try:
            inbox_table.update_item(
                Key={"inbox_id": inbox_id},
                UpdateExpression="SET review_status = :status, updated_at = :ts",
                ExpressionAttributeValues={
                    ":status": new_status,
                    ":ts": Decimal(str(int(time.time())))
                }
            )
            success_count += 1
            print(f"  ✓ Updated {inbox_id[:8]}")
            
        except Exception as e:
            failed_count += 1
            failed_items.append({"inbox_id": inbox_id, "error": str(e)})
            print(f"  ✗ Failed to update {inbox_id[:8]}: {e}")
    
    result = {
        "success_count": success_count,
        "failed_count": failed_count, 
        "failed_items": failed_items,
        "total": len(inbox_ids)
    }
    
    print(f"\nBulk update complete: {success_count} successful, {failed_count} failed")
    return result

def auto_approve_processed_items(processed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Automatically approve all successfully processed items
    """
    successful_inbox_ids = [
        result.get("inbox_id") 
        for result in processed_results 
        if result.get("success") and result.get("inbox_id")
    ]
    
    if not successful_inbox_ids:
        return {"message": "No successful items to approve", "count": 0}
    
    print(f"\nAuto-approving {len(successful_inbox_ids)} successfully processed items...")
    return bulk_update_review_status(successful_inbox_ids, "approved")

def interactive_selection_mode(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Interactive mode to manually select which items to approve
    """
    if not items:
        return []
    
    print(f"\n=== Interactive Selection Mode ===")
    print(f"Found {len(items)} unapproved items")
    print("\nOptions:")
    print("  [a]ll - Select all items")
    print("  [n]one - Select none") 
    print("  [r]ange - Select by range (e.g., 1-10)")
    print("  [i]ds - Enter specific inbox_ids")
    print("  [m]anual - Review each item individually")
    print("  [f]ilter - Apply text filters")
    
    choice = input("\nHow would you like to select items? ").strip().lower()
    
    if choice in ['a', 'all']:
        return items
    elif choice in ['n', 'none']:
        return []
    elif choice in ['r', 'range']:
        return select_by_range(items)
    elif choice in ['i', 'ids']:
        return select_by_ids(items)
    elif choice in ['m', 'manual']:
        return manual_selection(items)
    elif choice in ['f', 'filter']:
        return filter_selection(items)
    else:
        print("Invalid choice, defaulting to none")
        return []

def select_by_range(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select items by numeric range"""
    print(f"\nItems 1-{len(items)}:")
    for i, item in enumerate(items[:20], 1):  # Show first 20
        inbox_id = item.get('inbox_id', 'unknown')[:8]
        utterance = item.get('utterance_text', '')[:60]
        print(f"  {i:2d}. {inbox_id} | \"{utterance}\"")
    if len(items) > 20:
        print(f"  ... and {len(items) - 20} more items")
    
    range_str = input("\nEnter range (e.g., 1-5, 3,7,9, or 10-): ").strip()
    try:
        selected_indices = parse_range(range_str, len(items))
        selected = [items[i-1] for i in selected_indices if 1 <= i <= len(items)]
        print(f"Selected {len(selected)} items")
        return selected
    except Exception as e:
        print(f"Invalid range: {e}")
        return []

def select_by_ids(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select by specific inbox_ids"""
    print("\nEnter inbox_ids (comma or space separated):")
    print("Example: cdbbca982487, a1b2c3d4e5f6, ...")
    
    ids_str = input("Inbox IDs: ").strip()
    if not ids_str:
        return []
    
    # Parse IDs (comma, space, or newline separated)
    import re
    target_ids = re.split(r'[,\s\n]+', ids_str)
    target_ids = [id.strip() for id in target_ids if id.strip()]
    
    selected = []
    for item in items:
        inbox_id = item.get('inbox_id', '')
        # Match full ID or partial ID (first 8+ chars)
        for target_id in target_ids:
            if inbox_id == target_id or inbox_id.startswith(target_id):
                selected.append(item)
                print(f"  Matched: {inbox_id}")
                break
    
    print(f"Selected {len(selected)} items")
    return selected

def manual_selection(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Review each item individually"""
    print(f"\nManual review mode - {len(items)} items")
    print("For each item, choose: [y]es, [n]o, [q]uit, [s]kip remaining")
    
    selected = []
    for i, item in enumerate(items, 1):
        inbox_id = item.get('inbox_id', 'unknown')
        utterance = item.get('utterance_text', '')
        response = item.get('proposed_response_text', '')
        contact_id = item.get('contact_id', 'unknown')
        
        print(f"\n--- Item {i}/{len(items)} ---")
        print(f"ID: {inbox_id}")
        print(f"Contact: {contact_id}")
        print(f"User: \"{utterance}\"")
        print(f"Response: \"{response}\"")
        
        while True:
            choice = input("Approve this item? [y/n/q/s]: ").strip().lower()
            if choice in ['y', 'yes']:
                selected.append(item)
                print("  → Added to approval list")
                break
            elif choice in ['n', 'no']:
                print("  → Skipped")
                break
            elif choice in ['q', 'quit']:
                print(f"Quit manual selection. Selected {len(selected)} items so far.")
                return selected
            elif choice in ['s', 'skip']:
                print(f"Skipping remaining items. Selected {len(selected)} items.")
                return selected
            else:
                print("Please enter y, n, q, or s")
    
    print(f"Manual selection complete. Selected {len(selected)} items.")
    return selected

def filter_selection(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter items by text content"""
    print(f"\nText filter mode")
    print("Filter by text content in user input or response")
    
    filter_text = input("Enter text to search for: ").strip().lower()
    if not filter_text:
        return items
    
    filtered = []
    for item in items:
        utterance = item.get('utterance_text', '').lower()
        response = item.get('proposed_response_text', '').lower()
        
        if filter_text in utterance or filter_text in response:
            filtered.append(item)
    
    print(f"Found {len(filtered)} items containing \"{filter_text}\"")
    
    # Show matches
    for i, item in enumerate(filtered[:10], 1):
        inbox_id = item.get('inbox_id', 'unknown')[:8]
        utterance = item.get('utterance_text', '')
        print(f"  {i}. {inbox_id} | \"{utterance[:60]}\"")
    if len(filtered) > 10:
        print(f"  ... and {len(filtered) - 10} more matches")
    
    return filtered

def parse_range(range_str: str, max_val: int) -> List[int]:
    """Parse range string like '1-5', '3,7,9', '10-' """
    indices = set()
    
    parts = range_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Range like "1-5" or "10-"
            if part.endswith('-'):
                # "10-" means from 10 to end
                start = int(part[:-1])
                indices.update(range(start, max_val + 1))
            elif part.startswith('-'):
                # "-5" means from 1 to 5
                end = int(part[1:])
                indices.update(range(1, end + 1))
            else:
                # "1-5" means from 1 to 5
                start, end = map(int, part.split('-'))
                indices.update(range(start, end + 1))
        else:
            # Single number
            indices.add(int(part))
    
    return sorted(list(indices))
    """
    Process multiple inbox items concurrently
    """
    results = []
    
    print(f"Processing {len(items)} items with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_item = {
            executor.submit(approve_single_item, item): item 
            for item in items
        }
        
        # Process completed tasks
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result = future.result()
                results.append(result)
                
                # Update counters
                processed_counter.increment()
                if result.get("success"):
                    count = success_counter.increment()
                    print(f"✓ [{count}/{len(items)}] {result.get('inbox_id', 'unknown')[:8]} approved")
                else:
                    count = failed_counter.increment()
                    print(f"✗ [{processed_counter.value}/{len(items)}] {result.get('inbox_id', 'unknown')[:8]} failed: {result.get('error')}")
                    
            except Exception as e:
                failed_counter.increment()
                inbox_id = item.get("inbox_id", "unknown")
                print(f"✗ [{processed_counter.value + 1}/{len(items)}] {inbox_id[:8]} exception: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "inbox_id": inbox_id
                })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Batch approve unapproved inbox items")
    parser.add_argument("--table-name", default="SttInboxNew", help="DynamoDB inbox table name")
    parser.add_argument("--cache-table-name", default="UtteranceCache", help="UtteranceCache table name")
    parser.add_argument("--limit", type=int, help="Max number of items to process")
    parser.add_argument("--contact-id", help="Filter by specific contact_id")
    parser.add_argument("--locale", help="Filter by specific locale (e.g., ko-KR)")
    parser.add_argument("--inbox-ids", help="Comma-separated list of specific inbox_ids to process")
    parser.add_argument("--date-after", help="Only process items after this date (ISO format: 2024-01-01)")
    parser.add_argument("--date-before", help="Only process items before this date (ISO format: 2024-12-31)")
    parser.add_argument("--text-contains", help="Only process items containing this text")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive selection mode")
    parser.add_argument("--max-workers", type=int, default=1, help="Number of concurrent workers")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without doing it")
    # parser.add_argument("--tts-url", default="http://localhost:8000", help="TTS server base URL")
    parser.add_argument("--tts-url", default="https://honest-trivially-buffalo.ngrok-free.app:8000", help="TTS server base URL")
    parser.add_argument("--tts-token", default="", help="TTS server auth token")
    parser.add_argument("--auto-approve", default=True, action="store_true", help="Automatically set review_status=approved after successful processing")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging to troubleshoot data mapping")
    parser.add_argument("--test-single", help="Test with a single inbox_id and show detailed field mapping")
    args = parser.parse_args()
    
    # Update globals with args
    global INBOX_TABLE_NAME, TTS_BASE_URL, TTS_TOKEN, utterance_cache_table
    INBOX_TABLE_NAME = args.table_name
    TTS_BASE_URL = args.tts_url.rstrip("/")
    TTS_TOKEN = args.tts_token
    
    # Update UtteranceCache table reference
    if args.cache_table_name != "UtteranceCache":
        utterance_cache_table = ddb.Table(args.cache_table_name)
    
    if args.test_single:
        print(f"=== Testing single inbox item: {args.test_single} ===")
        test_items = get_unapproved_inbox_items(inbox_ids=[args.test_single])
        if not test_items:
            print("Item not found or already approved")
            return
        
        item = test_items[0]
        print(f"Raw item data:")
        for key, value in item.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: '{value[:100]}...'")
            else:
                print(f"  {key}: {value}")
        
        print(f"\nField mapping for UtteranceCache:")
        utterance_text = item.get("utterance_text", "")
        proposed_response = item.get("proposed_response_text", "")
        print(f"  utterance_text -> original_utterance: '{utterance_text}'")
        print(f"  proposed_response_text -> cached_text: '{proposed_response[:100]}...'")
        
        if not utterance_text or not proposed_response:
            print(f"ERROR: Missing required fields!")
            print(f"  utterance_text present: {bool(utterance_text)}")
            print(f"  proposed_response_text present: {bool(proposed_response)}")
        else:
            print(f"Fields look good for processing.")
        return
    
    # Get unapproved items with enhanced filtering
    start_time = time.time()
    inbox_ids_list = None
    if args.inbox_ids:
        inbox_ids_list = [id.strip() for id in args.inbox_ids.split(',') if id.strip()]
    
    unapproved_items = get_unapproved_inbox_items(
        limit=args.limit,
        contact_id=args.contact_id,
        locale=args.locale,
        inbox_ids=inbox_ids_list,
        date_after=args.date_after,
        date_before=args.date_before,
        text_contains=args.text_contains
    )
    
    if not unapproved_items:
        print("No unapproved items found!")
        return
    
    print(f"Found {len(unapproved_items)} unapproved items")
    
    # Show sample data
    if unapproved_items:
        sample = unapproved_items[0]
        print(f"Sample item keys: {list(sample.keys())}")
        print(f"Sample inbox_id: {sample.get('inbox_id', 'N/A')}")
        print(f"Sample utterance: {sample.get('utterance_text', 'N/A')[:100]}")
        print(f"Sample response: {sample.get('proposed_response_text', 'N/A')[:100]}")
    
    if args.dry_run:
        print(f"\n=== DRY RUN ===")
        print(f"Would process {len(unapproved_items)} items")
        for i, item in enumerate(unapproved_items[:10], 1):
            inbox_id = item.get("inbox_id", "unknown")
            utterance = item.get("utterance_text", "")
            response = item.get("proposed_response_text", "")
            print(f"{i:2d}. {inbox_id[:8]} | U: \"{utterance[:40]}\" | R: \"{response[:40]}\"")
        if len(unapproved_items) > 10:
            print(f"... and {len(unapproved_items) - 10} more items")
        return
    
    # Interactive selection mode
    if args.interactive:
        unapproved_items = interactive_selection_mode(unapproved_items)
        if not unapproved_items:
            print("No items selected.")
            return
    
    # Final confirmation
    if not args.interactive:  # Skip if already confirmed in interactive mode
        confirmation = input(f"\nProcess {len(unapproved_items)} items? (y/N): ").strip().lower()
        if confirmation not in ['y', 'yes']:
            print("Cancelled.")
            return
    
    # Process items
    print(f"\nStarting batch approval...")
    results = process_batch(unapproved_items, max_workers=args.max_workers)
    
    # Auto-approve processed items if requested
    if args.auto_approve:
        auto_approve_result = auto_approve_processed_items(results)
        print(f"\nAuto-approval result: {auto_approve_result}")
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n=== SUMMARY ===")
    print(f"Total items: {len(unapproved_items)}")
    print(f"Processed: {processed_counter.value}")
    print(f"Successful: {success_counter.value}")
    print(f"Failed: {failed_counter.value}")
    print(f"Duration: {duration:.1f} seconds")
    print(f"Rate: {len(unapproved_items)/duration:.1f} items/sec")
    
    # Show failed items
    failed_results = [r for r in results if not r.get("success")]
    if failed_results:
        print(f"\nFailed items ({len(failed_results)}):")
        for result in failed_results[:10]:
            inbox_id = result.get("inbox_id", "unknown")
            error = result.get("error", "Unknown error")
            print(f"  {inbox_id[:8]}: {error}")
        if len(failed_results) > 10:
            print(f"  ... and {len(failed_results) - 10} more failures")
    
    print("\nBatch approval complete!")

if __name__ == "__main__":
    main()