# bridge_chat_to_tts.py
import os, time, uuid, json, requests
import soundfile as sf
from infer_tts import load_engine, _read_yaml, s3_upload_and_presign, _ensure_int16

CHAT_URL = "http://15.165.60.45:5000/chat"   # your endpoint

def tts_url_for(text: str, cfg_path: str = "./fishspeech_infer/config.yaml", outdir: str = "outputs") -> str:
    cfg = _read_yaml(cfg_path)
    engine = load_engine(cfg_path)
    audio_f32, sr = engine.synthesize(text=text)  # accepts temp/top_p if you want

    os.makedirs(outdir, exist_ok=True)
    fname = f"tts_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav"
    local_wav = os.path.join(outdir, fname)
    sf.write(local_wav, _ensure_int16(audio_f32), sr, subtype="PCM_16")

    bucket = cfg["s3"]["bucket"]
    key_prefix = cfg["s3"].get("key_prefix", "tts-out")
    expires = cfg.get("presign_expires", 3600)

    result = s3_upload_and_presign(local_wav, bucket=bucket, key_prefix=key_prefix, expires=expires)
    return result["url"]

def main():
    payload = {
        "session_id": "38bb86ff-9a62-4e04-af1e-2dacdbda8f99",
        "question": "안녕하세요"
    }
    r = requests.post(CHAT_URL, json=payload, timeout=20)
    r.raise_for_status()
    data = r.json()
    answer = data.get("answer") or data.get("response") or ""
    if not answer:
        raise RuntimeError(f"No answer in chat response: {data}")

    url = tts_url_for(answer)
    print(json.dumps({"answer": answer, "audio_url": url}, ensure_ascii=False))

if __name__ == "__main__":
    main()