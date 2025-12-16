#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fill missing audio_s3_uri for responses in response_bank_with_audio.csv

- Reuses TTS + S3 helpers from generate_preset.py
- For each row where audio_s3_uri is empty:
    * generate TTS via /synthesize
    * repair early-termination if needed
    * transcode to µ-law/8k/mono
    * upload to S3 with a stable key based on response_hash
    * fill audio_s3_uri (s3://...) and response_hash (if empty)
- Writes updated CSV as response_bank_with_audio_filled.csv
"""

import pandas as pd
import numpy as np
import time

from generate_preset import (  # type: ignore
    SAMPLE_RATE,
    AWS_REGION,
    TTS_BASE,
    TTS_TOKEN,
    s3,
    prepare_for_tts,
    synthesize_short,
    fetch_s3_bytes,
    repair_if_early_terminated,
    transcode_to_mulaw_8k_mono,
    to_regional_url,
    normalize_utt,
    utt_hash,
)

BANK_IN  = "/home/work/VALL-E/chatbot/data/response_bank_with_audio.csv"
BANK_OUT = "/home/work/VALL-E/chatbot/data/response_bank_with_audio_filled.csv"

S3_PREFIX = "chatbot-bank"  # change if you want another folder


def main():
    df = pd.read_csv(BANK_IN)

    # Make sure these columns exist
    if "audio_s3_uri" not in df.columns:
        df["audio_s3_uri"] = np.nan
    if "response_hash" not in df.columns:
        df["response_hash"] = np.nan
    if "locale" not in df.columns:
        df["locale"] = "ko-KR"

    print(f"[INFO] Loaded bank: {BANK_IN}, shape={df.shape}")

    changed = 0
    total_missing = df["audio_s3_uri"].isna() | (df["audio_s3_uri"].astype(str).str.strip() == "")

    for idx, row in df[total_missing].iterrows():
        text = str(row["response_text"]).strip()
        if not text:
            continue

        print(f"\n[GEN] idx={idx}, text={text[:40]}...")

        # 1) Ensure response_hash
        rh = row.get("response_hash")
        if isinstance(rh, str) and rh.strip():
            response_hash = rh.strip()
        else:
            # use same hashing normalization as your preset loader
            norm = normalize_utt(text)
            response_hash = utt_hash(norm)
            df.at[idx, "response_hash"] = response_hash

        # 2) Prepare text for TTS
        tts_text = prepare_for_tts(text)

        # 3) Short TTS via your existing endpoint
        info = synthesize_short(
            TTS_BASE,
            tts_text,
            key_prefix=f"{S3_PREFIX}/tmp",
            sr=SAMPLE_RATE,
            token=TTS_TOKEN,
        )

        bucket = info.get("bucket")
        src_key = info.get("key")
        if not bucket or not src_key:
            print(f"[ERROR] TTS failed for idx={idx}: {info}")
            continue

        # 4) Download raw WAV from the temp location
        raw = fetch_s3_bytes(bucket, src_key)

        # 5) Repair early termination if needed
        repaired_pcm = repair_if_early_terminated(
            raw,
            original_text=text,
            tts_base=TTS_BASE,
            sample_rate=SAMPLE_RATE,
            token=TTS_TOKEN,
            keep_original=False,
        )

        # 6) Transcode to µ-law/8k/mono
        final_raw = transcode_to_mulaw_8k_mono(repaired_pcm, sr_out=8000)

        # 7) Stable final key based on response_hash
        dst_key = f"{S3_PREFIX}/{response_hash}.wav"
        s3.put_object(
            Bucket=bucket,
            Key=dst_key,
            Body=final_raw,
            ContentType="audio/wav",
        )

        # 8) Delete temp key
        try:
            s3.delete_object(Bucket=bucket, Key=src_key)
        except Exception as e:
            print(f"[WARN] Could not delete {src_key}: {e}")

        # 9) Fill CSV fields
        s3_uri = f"s3://{bucket}/{dst_key}"
        http_url = to_regional_url(bucket, AWS_REGION, dst_key)

        df.at[idx, "audio_s3_uri"] = s3_uri
        # Optional: if you want an HTTP URL column too
        if "audio_http_url" not in df.columns:
            df["audio_http_url"] = np.nan
        df.at[idx, "audio_http_url"] = http_url

        changed += 1
        print(f"    -> {s3_uri}")

        # tiny sleep just to avoid hammering anything
        time.sleep(0.1)

    df.to_csv(BANK_OUT, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] Done. Updated {changed} rows, wrote: {BANK_OUT}")


if __name__ == "__main__":
    main()
