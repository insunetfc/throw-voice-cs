#!/usr/bin/env python3
import os, sys, boto3, botocore.session, subprocess

AWS_REGION    = os.environ.get("AWS_REGION", "ap-northeast-2")
STREAM_NAME   = os.environ["STREAM_NAME"]             # pass via env
FIFO_PATH     = os.environ.get("WAV_FIFO", "/tmp/customer.wav")
OUT_RATE      = os.environ.get("SAMPLE_RATE", "8000") # Transcribe expects 8000
OUT_CHANNELS  = os.environ.get("OUT_CHANNELS", "1")

if not os.path.exists(FIFO_PATH):
    os.mkfifo(FIFO_PATH)


def main():
    kvs = boto3.client("kinesisvideo", region_name=AWS_REGION)
    ep  = kvs.get_data_endpoint(StreamName=STREAM_NAME, APIName="GET_MEDIA")["DataEndpoint"]
    media = botocore.session.get_session().create_client(
        "kinesis-video-media", region_name=AWS_REGION, endpoint_url=ep
    )
    res = media.get_media(StreamName=STREAM_NAME, StartSelector={"StartSelectorType":"NOW"})
    src = res["Payload"]  # streaming MKV (AAC inside in your case)

    # Decode MKV/AAC -> WAV PCM16 mono 8k using ffmpeg
    # We use pipe:0 as input (stdin) and write a WAV header + stream to FIFO.
    ff = subprocess.Popen(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-y",                         # <-- allow writing to existing FIFO
            "-analyzeduration", "10000000",
            "-probesize",      "10000000",
            "-fflags",         "+discardcorrupt",
            "-err_detect",     "ignore_err",
            # Let ffmpeg auto-detect MKV instead of forcing -f matroska
            "-i", "pipe:0",
            "-vn",
            "-ac", OUT_CHANNELS,          # mono
            "-ar", OUT_RATE,              # 8000 Hz
            "-sample_fmt", "s16",         # PCM16
            "-f", "wav",
            FIFO_PATH
        ],
        stdin=subprocess.PIPE
    )


    # Pump bytes from KVS -> ffmpeg stdin
    try:
        while True:
            chunk = src.read(1024 * 64)
            if not chunk:
                break
            ff.stdin.write(chunk)
    finally:
        try:
            ff.stdin.close()
        except Exception:
            pass
        ff.wait()

if __name__ == "__main__":
    if not os.path.exists(FIFO_PATH):
        os.mkfifo(FIFO_PATH)
    main()
