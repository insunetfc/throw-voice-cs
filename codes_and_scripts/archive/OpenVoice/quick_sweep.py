#!/usr/bin/env python3
import os
import torch
import soundfile as sf
from app import get_engine, _to_float_mono

eng = get_engine()

text = "안녕하세요, 오늘은 테스트를 하고 있습니다."
ref_wav = "ref_kr_24k.wav"

# param sweeps (adjustable)
temperatures = [0.7, 0.8, 0.9]
top_ps = [0.7, 0.8, 0.9]
reps = [1.0, 1.1]

os.makedirs("out_test", exist_ok=True)

for t in temperatures:
    for p in top_ps:
        for r in reps:
            with torch.inference_mode():
                audio_f32, model_sr = eng.synthesize(
                    text=text,
                    speaker_wav=ref_wav,
                    sr=24000,
                    temperature=t,
                    top_p=p,
                    seed=42,
                    repetition_penalty=r,
                    max_new_tokens=0,
                    chunk_length=300,
                    use_memory_cache=True,
                )
            float_mono, _ = _to_float_mono(audio_f32, model_sr, 24000)
            out = f"out_test/temp{t}_top{p}_rep{r}.wav"
            sf.write(out, float_mono, 24000)
            print("saved", out)
