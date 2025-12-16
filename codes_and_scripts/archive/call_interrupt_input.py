import boto3
import time
from datetime import datetime
import hashlib
import os
import json

# Configs
REGION = "ap-northeast-2"
DESTINATION_PHONE = "+821043876322" #"+821072847990" #"+821029944042" #"+821043876322"
CONNECT_INSTANCE_ID = "5b83741e-7823-4d70-952a-519d1ac05e63"
SOURCE_PHONE = "+82269269343" #"+82269269342"
# CONTACT_FLOW_ID = "b5758a4d-edc4-4242-af82-1222f6f41608"
CONTACT_FLOW_ID = "b7c0c1be-a84a-489e-a428-e13505c85aca"
# CONTACT_FLOW_ID = "75e1c560-8b81-4ac8-af45-f07de76365b6"

CACHE_FILE = "processed_logs.json"
LOG_GROUP_NAME = "/aws/lambda/InvokeBotLambda"
LOG_DIR = "./logs"
POLL_INTERVAL = 5
TIMEOUT = 60

# Init AWS
connect = boto3.client("connect", region_name=REGION)
logs_client = boto3.client("logs", region_name=REGION)

# Ensure log directory exists
# os.makedirs(LOG_DIR, exist_ok=True)

# # Load cache
# if os.path.exists(CACHE_FILE):
#     with open(CACHE_FILE, "r") as f:
#         processed_logs = json.load(f)
# else:
#     processed_logs = {}

# # 1. Get current log streams before call
# before_streams = logs_client.describe_log_streams(
#     logGroupName=LOG_GROUP_NAME,
#     orderBy="LastEventTime",
#     descending=True,
#     limit=5
# )
# before_names = set(stream["logStreamName"] for stream in before_streams["logStreams"])

# 2. Start outbound call
response = connect.start_outbound_voice_contact(
    DestinationPhoneNumber=DESTINATION_PHONE,
    ContactFlowId=CONTACT_FLOW_ID,
    InstanceId=CONNECT_INSTANCE_ID,
    SourcePhoneNumber=SOURCE_PHONE,
    Attributes={"test_type": "voice"}
)
contact_id = response["ContactId"]
print(f"📞 Call started. Contact ID: {contact_id}")

# # 3. Poll for new log stream
# new_stream_name = None
# start_time = time.time()
# while time.time() - start_time < TIMEOUT:
#     time.sleep(POLL_INTERVAL)
#     stream_list = logs_client.describe_log_streams(
#         logGroupName=LOG_GROUP_NAME,
#         orderBy="LastEventTime",
#         descending=True,
#         limit=3
#     )
#     for stream in stream_list["logStreams"]:
#         name = stream["logStreamName"]
#         if name not in before_names:
#             new_stream_name = name
#             break
#     if new_stream_name:
#         print(f"📘 New log stream found: {new_stream_name}")
#         break
# else:
#     print("❌ Timeout: No new log stream found.")
#     exit()

# # 4. Deduplication via SHA256
# stream_hash = hashlib.sha256(new_stream_name.encode()).hexdigest()
# if stream_hash in processed_logs:
#     print("⚠️ This log stream has already been processed. Skipping.")
#     exit()

# # 5. Retrieve log events
# events = logs_client.get_log_events(
#     logGroupName=LOG_GROUP_NAME,
#     logStreamName=new_stream_name,
#     startFromHead=True
# )

# log_lines = []
# customer_input_line = ""
# reply_line = ""

# for event in events["events"]:
#     msg = event["message"]
#     log_lines.append(msg)
#     if "Customer Input" in msg:
#         customer_input_line = msg
#     if "reply" in msg.lower():
#         reply_line = msg

# # 6. Save logs
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# filename = os.path.join(LOG_DIR, f"call_log_{timestamp}.txt")

# with open(filename, "w") as f:
#     for line in log_lines:
#         f.write(line + "\n")

# print(f"\n✅ Logs saved to: {filename}")
# print("🧾 Extracted info:")
# print(customer_input_line)
# print(reply_line)

# # 7. Update cache
# processed_logs[stream_hash] = {
#     "logStreamName": new_stream_name,
#     "savedAs": filename
# }
# with open(CACHE_FILE, "w") as f:
#     json.dump(processed_logs, f, indent=2)
