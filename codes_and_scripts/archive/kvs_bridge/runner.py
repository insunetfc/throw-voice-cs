#!/usr/bin/env python3
import os, time, subprocess, sys, signal, stat
import boto3, botocore.session, botocore.exceptions

# === Static configuration (edit these once) ===
AWS_REGION              = "ap-northeast-2"
KVS_PREFIX              = "connect-live-"
FIFO_PATH               = "/tmp/customer.wav"
APP_PATH                = "/home/work/VALL-E/asr_bridge/app.py"
SAMPLE_PATH             = "/home/work/VALL-E/asr_bridge/amazon-kinesis-video-streams-consumer-library-for-python/kvs_consumer_library_example.py"

# Lex / Connect / Transcribe parameters
CONNECT_INSTANCE_ID     = "5b83741e-7823-4d70-952a-519d1ac05e63"
LEX_BOT_ID              = "THVPHS55WP"
LEX_ALIAS_ID            = "GSH2W8JQ0A"
LEX_LOCALE              = "ko_KR"
TRANSCRIBE_LANGUAGE_CODE= "ko-KR"
SAMPLE_RATE             = "8000"
RMS_FLOOR               = "250.0"
SILENCE_MS              = "450"

# Orchestrator behaviour
POLL_SECONDS            = 2.0
LIVE_READ_BYTES         = 256

# === End static config ===

# --- helpers ---
def ensure_fifo(path):
    if os.path.exists(path):
        if not stat.S_ISFIFO(os.stat(path).st_mode):
            raise RuntimeError(f"{path} exists but is not a FIFO")
    else:
        os.mkfifo(path)

def list_streams(region, prefix):
    kvs = boto3.client("kinesisvideo", region_name=region)
    resp = kvs.list_streams(MaxResults=25)
    streams = resp.get("StreamInfoList", [])
    return [s for s in streams if s.get("StreamName", "").startswith(prefix)]

def latest_stream_name(region, prefix):
    streams = list_streams(region, prefix)
    if not streams:
        return None
    latest = sorted(streams, key=lambda s: s["CreationTime"], reverse=True)[0]
    return latest["StreamName"]

def stream_has_live_media(region, stream_name, test_bytes=LIVE_READ_BYTES):
    kvs = boto3.client("kinesisvideo", region_name=region)
    ep = kvs.get_data_endpoint(StreamName=stream_name, APIName="GET_MEDIA")["DataEndpoint"]
    media = botocore.session.get_session().create_client(
        "kinesis-video-media", region_name=region, endpoint_url=ep
    )
    try:
        res = media.get_media(StreamName=stream_name, StartSelector={"StartSelectorType": "NOW"})
        chunk = res["Payload"].read(test_bytes)
        return bool(chunk)
    except Exception:
        return False

def start_app_py():
    env = os.environ.copy()
    env.update({
        "AWS_REGION": AWS_REGION,
        "KVS_PREFIX": KVS_PREFIX,
        "WAV_FIFO": FIFO_PATH,
        "CONNECT_INSTANCE_ID": CONNECT_INSTANCE_ID,
        "LEX_BOT_ID": LEX_BOT_ID,
        "LEX_ALIAS_ID": LEX_ALIAS_ID,
        "LEX_LOCALE": LEX_LOCALE,
        "TRANSCRIBE_LANGUAGE_CODE": TRANSCRIBE_LANGUAGE_CODE,
        "SAMPLE_RATE": SAMPLE_RATE,
        "RMS_FLOOR": RMS_FLOOR,
        "SILENCE_MS": SILENCE_MS,
    })
    return subprocess.Popen([sys.executable, APP_PATH], env=env)

def start_sample_writer(stream_name):
    env = os.environ.copy()
    env.update({
        "AWS_REGION": AWS_REGION,
        "STREAM_NAME": stream_name,
        "WAV_FIFO": FIFO_PATH,
        "SAMPLE_RATE": SAMPLE_RATE,
    })
    return subprocess.Popen([sys.executable,
                             os.path.join(os.path.dirname(APP_PATH), "gst_bridge.py")],
                             env=env)


def terminate(p: subprocess.Popen):
    if p and p.poll() is None:
        try:
            p.terminate()
            p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            p.kill()

def main():
    print(f"[runner] region={AWS_REGION} prefix={KVS_PREFIX} fifo={FIFO_PATH}")
    ensure_fifo(FIFO_PATH)

    app_proc = start_app_py()
    print(f"[runner] started app.py pid={app_proc.pid}")

    active_stream = None
    writer_proc = None

    try:
        while True:
            if writer_proc and writer_proc.poll() is not None:
                print(f"[runner] writer exited rc={writer_proc.returncode}; waiting for next call")
                writer_proc = None
                active_stream = None

            latest = latest_stream_name(AWS_REGION, KVS_PREFIX)
            if latest and latest != active_stream:
                if stream_has_live_media(AWS_REGION, latest):
                    print(f"[runner] Detected LIVE stream {latest}")
                    writer_proc = start_sample_writer(latest)
                    active_stream = latest
                else:
                    pass
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("\n[runner] Ctrl-C, stopping...")
    finally:
        terminate(writer_proc)
        terminate(app_proc)

if __name__ == "__main__":
    main()
