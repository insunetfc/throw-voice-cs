#!/usr/bin/env python3
"""Quick diagnostic to list all KVS streams and their status"""

import boto3
from datetime import datetime, timezone

REGION = "ap-northeast-2"  # Adjust to your region

kvs = boto3.client("kinesisvideo", region_name=REGION)

print(f"Checking KVS streams in {REGION}...\n")

try:
    resp = kvs.list_streams(MaxResults=100)
    streams = resp.get("StreamInfoList", [])
    
    if not streams:
        print("❌ No streams found")
    else:
        print(f"✓ Found {len(streams)} streams:\n")
        
        now = datetime.now(timezone.utc)
        for s in streams:
            name = s.get("StreamName", "")
            status = s.get("Status", "")
            created = s.get("CreationTime")
            
            age = (now - created).total_seconds() if created else 0
            age_str = f"{age:.0f}s ago"
            
            is_connect = "connect-live" in name
            marker = "🔵" if is_connect else "⚪"
            
            print(f"{marker} {name}")
            print(f"   Status: {status}")
            print(f"   Created: {created} ({age_str})")
            
            if is_connect and "-contact-" in name:
                contact_id = name.split("-contact-")[-1]
                print(f"   Contact ID: {contact_id}")
            print()

except Exception as e:
    print(f"❌ Error: {e}")
    print("\nCheck:")
    print("1. AWS credentials are configured")
    print("2. IAM permissions include kinesisvideo:ListStreams")
    print("3. Region is correct")
