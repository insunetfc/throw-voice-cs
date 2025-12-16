# file: smoke_wait_for_media.py
import os, time, io
import boto3, botocore.session, botocore.exceptions

REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
STREAM = os.environ["STREAM_NAME"]  # set to the newest 'connect-live-' stream name
MAX_WAIT_SEC = 120  # wait up to 2 minutes for first bytes

kvs = boto3.client("kinesisvideo", region_name=REGION)
ep = kvs.get_data_endpoint(StreamName=STREAM, APIName="GET_MEDIA")["DataEndpoint"]

media = botocore.session.get_session().create_client(
    "kinesis-video-media", region_name=REGION, endpoint_url=ep
)

print(f"Waiting for media on stream: {STREAM} in {REGION} ...")
t0 = time.time()
while True:
    try:
        res = media.get_media(StreamName=STREAM, StartSelector={"StartSelectorType": "NOW"})
        # Try to read a little; if no fragments yet, read() blocks briefly then returns b''/raises
        chunk = res["Payload"].read(512)
        if chunk:
            print(f"Got first {len(chunk)} bytes. Media is live.")
            break
    except botocore.exceptions.ClientError as e:
        # If stream isn't producing yet, retry
        pass
    if time.time() - t0 > MAX_WAIT_SEC:
        raise TimeoutError("No media arrived within MAX_WAIT_SEC. Start a call and try again.")
    time.sleep(1)
