#!/usr/bin/env python3
import os
import sys
import subprocess
import numpy as np
import torch
import torchaudio
from pathlib import Path

# config (edit these)
ROOT = Path(".").resolve()
DAC_SCRIPT = ROOT / "fish_speech" / "models" / "dac" / "inference.py"
T2S_SCRIPT = ROOT / "fish_speech" / "models" / "text2semantic" / "inference.py"
DAC_CKPT = ROOT / "checkpoints" / "openaudio-s1-mini" / "codec.pth"
T2S_CKPT = ROOT / "checkpoints" / "openaudio-s1-mini"

def run_cmd(cmd):
    print(">>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)

def force_make_fake_npy_from_wav(wav_path: Path, out_npy: Path, target_sr=24000):
    """Fallback: if DAC script didn't produce fake.npy, make one ourselves."""
    print(f"[fallback] creating {out_npy} from {wav_path}")
    # load dac model
    from fish_speech.models.dac.inference import load_model
    model = load_model(str(DAC_CKPT))
    model.eval()

    wav, sr = torchaudio.load(str(wav_path))
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    with torch.no_grad():
        encoded = model.encode(wav)
    # adapt to return type
    if isinstance(encoded, dict) and "codes" in encoded:
        codes = encoded["codes"].cpu().numpy()
    else:
        codes = encoded.cpu().numpy()
    np.save(out_npy, codes)
    print(f"[fallback] saved {out_npy}")

def main():
    if len(sys.argv) < 3:
        print("usage: python run_fish_pipeline.py <ref_audio.wav> \"your text here\"")
        sys.exit(1)

    ref_audio = Path(sys.argv[1]).resolve()
    gen_text = sys.argv[2]
    prompt_text = "The text corresponding to reference audio"  # change to real transcript if you have it

    if not ref_audio.exists():
        raise FileNotFoundError(ref_audio)

    # STEP 1: run DAC inference on reference audio (this should create fake.wav, maybe fake.npy)
    run_cmd([
        sys.executable,
        str(DAC_SCRIPT),
        "-i",
        str(ref_audio),
        "--checkpoint-path",
        str(DAC_CKPT),
    ])

    fake_npy = Path("fake.npy")
    fake_wav = Path("fake.wav")

    # if the script didn't create fake.npy, make it ourselves
    if not fake_npy.exists():
        force_make_fake_npy_from_wav(ref_audio, fake_npy)

    # STEP 2: text2semantic -> will produce temp/codes_0.npy (in your case)
    print("[2] running text2semantic…")
    subprocess.run([
        sys.executable,
        str(T2S_SCRIPT),
        "--text", gen_text,
        "--prompt-text", prompt_text,
        "--prompt-tokens", str(fake_npy),
        "--checkpoint-path", str(T2S_CKPT),
        "--num-samples", "1",
    ], check=True)

    # STEP 2 output can be in "temp/codes_0.npy"
    codes_local = Path("codes_0.npy")
    codes_temp = Path("temp") / "codes_0.npy"
    if codes_temp.exists():
        codes_npy = codes_temp
    elif codes_local.exists():
        codes_npy = codes_local
    else:
        raise FileNotFoundError("Expected codes_0.npy (or temp/codes_0.npy) from text2semantic step but not found.")

    # STEP 3: decode back to audio with DAC
    print(f"[3] decoding {codes_npy} …")
    subprocess.run([
        sys.executable,
        str(DAC_SCRIPT),
        "-i",
        str(codes_npy),
        "--checkpoint-path",
        str(DAC_CKPT),
    ], check=True)

    print("\n✅ Done.")
    print(" - Reference used:", ref_audio)
    print(" - Prompt tokens :", fake_npy)
    print(" - Generated codes:", codes_npy)
    print(" - Listen to     : fake.wav")

if __name__ == "__main__":
    main()
