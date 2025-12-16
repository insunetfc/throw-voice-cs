#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ov_cli.py — Tiny CLI to call your running OpenVoice (uvicorn) server and download the output file.

Usage:
  # Hi-Fi (24 kHz PCM)
  python ov_cli.py --text "안녕하세요, 테스트 음성입니다." --encode pcm --sr 24000 --out out_hifi.wav

  # Telephony (8 kHz μ-law)
  python ov_cli.py --text "안녕하세요, 테스트입니다." --encode mulaw --sr 8000 --out out_tel.wav

  # With a reference wav + emotion knobs
  python ov_cli.py --text "오늘 일정이 변경되었습니다." \
    --speaker-wav /abs/path/ref_kr_24k.wav \
    --encode pcm --sr 24000 \
    --emotion happy --intensity 0.5 --blend 0.5 --out out_happy.wav

Environment:
  export TTS_URL="http://127.0.0.1:8000"
  export TTS_TOKEN="your_token_here"
"""

import os
import sys
import json
import argparse
import pathlib
import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=os.environ.get("TTS_URL", "http://127.0.0.1:8000"), help="Base server URL")
    p.add_argument("--token", default=os.environ.get("TTS_TOKEN", ""), help="Bearer token")
    p.add_argument("--text", required=True, help="Synthesis text")
    p.add_argument("--speaker-wav", default=None, help="Optional reference speaker wav path (server-visible path)")
    p.add_argument("--encode", choices=["pcm", "mulaw"], default="pcm", help="pcm=24k PCM, mulaw=8k telephony")
    p.add_argument("--sr", type=int, default=24000, help="Sample rate")
    p.add_argument("--emotion", default=None, help="Optional: happy|sad|angry|neutral")
    p.add_argument("--intensity", type=float, default=None, help="Emotion intensity 0..1")
    p.add_argument("--blend", type=float, default=None, help="Emotion dry/wet 0..1")
    p.add_argument("--key-prefix", default="tts/tests", help="Server-side key prefix hint")
    p.add_argument("--extra", default=None, help="JSON string of extra fields")
    p.add_argument("--out", required=True, help="Where to save the resulting audio file")
    args = p.parse_args()

    synth_url = args.url.rstrip("/") + "/synthesize"
    headers = {"Content-Type": "application/json"}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"

    payload = {
        "text": args.text,
        "sample_rate": args.sr,
        "encode": args.encode,
        "key_prefix": args.key_prefix,
    }
    if args.speaker_wav:
        payload["speaker_wav"] = args.speaker_wav
    if args.emotion:
        payload["emotion"] = args.emotion
    if args.intensity is not None:
        payload["emotion_intensity"] = args.intensity
    if args.blend is not None:
        payload["emotion_blend"] = args.blend
    if args.extra:
        try:
            payload.update(json.loads(args.extra))
        except Exception as e:
            print("Failed parsing --extra JSON:", e, file=sys.stderr)

    try:
        resp = requests.post(synth_url, headers=headers, json=payload, timeout=120)
    except requests.exceptions.RequestException as e:
        print("Request failed:", e, file=sys.stderr)
        sys.exit(2)

    if resp.status_code >= 400:
        print("Server error:", resp.status_code, resp.text, file=sys.stderr)
        sys.exit(3)

    try:
        info = resp.json()
    except Exception:
        print("Non-JSON response:", resp.text[:2000], file=sys.stderr)
        sys.exit(4)

    url = info.get("url") or info.get("presigned_url") or info.get("file_url")
    if not url:
        local_path = info.get("file") or info.get("path") or info.get("key")
        if local_path and os.path.exists(local_path):
            data = open(local_path, "rb").read()
            pathlib.Path(args.out).write_bytes(data)
            print("Saved", args.out)
            return
        print("No downloadable URL in response:\n", json.dumps(info, ensure_ascii=False, indent=2))
        sys.exit(5)

    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("Download failed:", e, file=sys.stderr)
        print("Response JSON was:\n", json.dumps(info, ensure_ascii=False, indent=2))
        sys.exit(6)

    pathlib.Path(args.out).write_bytes(r.content)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
