
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_preset_local.py
------------------------
Local-only OpenVoice generator that **bypasses S3** and saves WAVs to disk.
It reuses the same engine as your running app.py but does NOT call HTTP or upload.

Usage examples
--------------
# Basic (single utterance)
python generate_preset_local.py \
  --text "안녕하세요, 오늘은 테스트를 하고 있습니다." \
  --ref-wav ref_kr_24k.wav \
  --out-dir out_local

# With parameters
python generate_preset_local.py \
  --text "테스트 중입니다." --ref-wav ref_kr_24k.wav \
  --sr 24000 --temperature 0.8 --top-p 0.8 --rep 1.1 \
  --chunk-length 300 --gain-db 1.5 --pre-emphasis 0.85

# Small sweep over temperature/top-p
python generate_preset_local.py \
  --text "테스트 중입니다." --ref-wav ref_kr_24k.wav \
  --out-dir out_local_sweep \
  --sweep --temps 0.7 0.8 0.9 --tops 0.7 0.8 0.9 --reps 1.0 1.1

Notes
-----
- Requires: soundfile, numpy, torch
- Imports get_engine and _to_float_mono from your local app.py
- Writes linear PCM WAV at --sr (default 24000) to the chosen output folder
"""

import os
import sys
import uuid
import time
import argparse
from typing import List

import numpy as np
import torch
import soundfile as sf

# Reuse your engine and helper from app.py (must be importable from this script's cwd)
try:
    from app import get_engine, _to_float_mono
except Exception as e:
    print("ERROR: Could not import get_engine/_to_float_mono from app.py:", e, file=sys.stderr)
    sys.exit(1)


def pre_emphasis(x: np.ndarray, a: float) -> np.ndarray:
    """Simple high-boost to counter 'muffle'. a in [0.0, 0.97]."""
    if a <= 0.0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - a * x[:-1]
    # mild anti-clipping
    mx = float(np.max(np.abs(y))) or 1.0
    if mx > 1.0:
        y = y / mx
    return y.astype(np.float32)


def synth_once(text: str, ref_wav: str, sr: int,
               temperature: float, top_p: float, seed: int,
               repetition_penalty: float, max_new_tokens: int,
               chunk_length: int, use_memory_cache: bool,
               pre_emph: float, gain_db: float) -> np.ndarray:
    eng = get_engine()
    t0 = time.time()
    with torch.inference_mode():
        audio_f32, model_sr = eng.synthesize(
            text=text,
            speaker_wav=ref_wav,
            sr=sr,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            chunk_length=chunk_length,
            use_memory_cache=use_memory_cache,
        )
    wav, _ = _to_float_mono(audio_f32, model_sr, sr)

    if pre_emph and pre_emph > 0.0:
        wav = pre_emphasis(wav, pre_emph)

    if abs(gain_db) > 1e-6:
        wav = (10.0 ** (gain_db / 20.0)) * wav
        m = float(np.max(np.abs(wav))) or 1.0
        if m > 1.0:
            wav = wav / m

    dur = len(wav) / float(sr)
    print(f"[synth] len={len(wav)} sr={sr} dur={dur:.2f}s took={time.time()-t0:.2f}s")
    return wav


def save_wav(path: str, wav: np.ndarray, sr: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, wav, sr)
    print("  -> saved", path)


def main(argv: List[str] = None):
    p = argparse.ArgumentParser(description="Local-only OpenVoice generator (bypasses S3).")
    p.add_argument("--text", required=True, help="Synthesis text.")
    p.add_argument("--ref-wav", required=True, help="Speaker reference WAV (path).")
    p.add_argument("--out-dir", default="out_local", help="Output directory for WAV files.")
    p.add_argument("--sr", type=int, default=24000, help="Target sample rate for output WAV.")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rep", type=float, default=1.1, help="Repetition penalty.")
    p.add_argument("--max-new-tokens", type=int, default=0)
    p.add_argument("--chunk-length", type=int, default=300)
    p.add_argument("--no-mem-cache", action="store_true", help="Disable memory cache in engine.")
    p.add_argument("--pre-emphasis", type=float, default=0.0, dest="pre_emphasis",
                   help="0.0..0.97 high-boost; try 0.85 for clarity.")
    p.add_argument("--gain-db", type=float, default=0.0, help="+/- dB post gain.")
    # Sweep options
    p.add_argument("--sweep", action="store_true", help="Sweep temperatures/top-p/repetition grid.")
    p.add_argument("--temps", nargs="*", type=float, default=[0.7, 0.8, 0.9])
    p.add_argument("--tops", nargs="*", type=float, default=[0.7, 0.8, 0.9])
    p.add_argument("--reps", nargs="*", type=float, default=[1.0, 1.1])
    args = p.parse_args(argv)

    use_mem = not args.no_mem_cache

    if not os.path.isfile(args.ref_wav):
        print(f"ERROR: ref-wav not found: {args.ref_wav}", file=sys.stderr)
        sys.exit(2)

    if not args.sweep:
        wav = synth_once(
            text=args.text, ref_wav=args.ref_wav, sr=args.sr,
            temperature=args.temperature, top_p=args.top_p, seed=args.seed,
            repetition_penalty=args.rep, max_new_tokens=args.max_new_tokens,
            chunk_length=args.chunk_length, use_memory_cache=use_mem,
            pre_emph=args.pre_emphasis, gain_db=args.gain_db
        )
        basename = f"ov_{uuid.uuid4().hex}_sr{args.sr}_t{args.temperature}_p{args.top_p}_r{args.rep}.wav"
        save_wav(os.path.join(args.out_dir, basename), wav, args.sr)
        return 0

    # Sweep mode
    for t in args.temps:
        for p_ in args.tops:
            for r_ in args.reps:
                wav = synth_once(
                    text=args.text, ref_wav=args.ref_wav, sr=args.sr,
                    temperature=t, top_p=p_, seed=args.seed,
                    repetition_penalty=r_, max_new_tokens=args.max_new_tokens,
                    chunk_length=args.chunk_length, use_memory_cache=use_mem,
                    pre_emph=args.pre_emphasis, gain_db=args.gain_db
                )
                name = f"ov_sr{args.sr}_t{t}_p{p_}_r{r_}.wav".replace('.', '_')
                save_wav(os.path.join(args.out_dir, name), wav, args.sr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
