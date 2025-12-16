#!/usr/bin/env python3
import os, sys, boto3, botocore.session, subprocess

AWS_REGION   = os.environ.get("AWS_REGION", "ap-northeast-2")
STREAM_NAME  = os.environ["STREAM_NAME"]
FIFO_PATH    = os.environ.get("WAV_FIFO", "/tmp/customer.wav")
OUT_RATE     = os.environ.get("SAMPLE_RATE", "8000")
OUT_CHANNELS = os.environ.get("OUT_CHANNELS", "1")

def main():
    kvs = boto3.client("kinesisvideo", region_name=AWS_REGION)
    ep  = kvs.get_data_endpoint(StreamName=STREAM_NAME, APIName="GET_MEDIA")["DataEndpoint"]
    media = botocore.session.get_session().create_client(
        "kinesis-video-media", region_name=AWS_REGION, endpoint_url=ep
    )
    res = media.get_media(StreamName=STREAM_NAME, StartSelector={"StartSelectorType":"NOW"})
    src = res["Payload"]  # streaming MKV (AAC inside)

    # Build a pipeline that reads MKV from stdin, decodes AAC, resamples to 8k mono S16,
    # and writes WAV to your FIFO.
    # Notes:
    # - matroskademux sometimes emits caps that need aacparse before avdec_aac
    # - queue elements make live streaming more stable
    pipeline = [
        "gst-launch-1.0", "-q",
        "fdsrc", "fd=0",
        "!", "matroskademux", "name=demux",
        "demux.audio_0", "!", "queue",
        "!", "decodebin",
        "!", "audioconvert",
        "!", "audioresample",
        "!", f"audio/x-raw,format=S16LE,channels={OUT_CHANNELS},rate={OUT_RATE}",
        "!", "filesink", f"location={FIFO_PATH}", "sync=true", "async=false"
    ]


    proc = subprocess.Popen(pipeline, stdin=subprocess.PIPE)

    try:
        while True:
            chunk = src.read(1024 * 64)
            if not chunk:
                break
            proc.stdin.write(chunk)
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()

if __name__ == "__main__":
    if not os.path.exists(FIFO_PATH):
        os.mkfifo(FIFO_PATH)
    main()
