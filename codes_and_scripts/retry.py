import wave

# reopen the existing file
with wave.open("gpt_voice.wav", "rb") as wf:
    audio = wf.readframes(wf.getnframes())

# write corrected header
with wave.open("gpt_voice_fixed.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)   # ✅ correct sample rate
    wf.writeframes(audio)

print("Saved gpt_voice_fixed.wav (24 kHz)")