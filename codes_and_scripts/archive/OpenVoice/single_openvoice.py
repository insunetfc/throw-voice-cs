# single_openvoice.py
import os, time, torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

# ---------- helpers ----------
def _sync(dev: str):
    # CUDA sync
    if isinstance(dev, str) and dev.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    # Apple MPS sync (safe guard)
    if dev == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()  # no local import; safe call
        except Exception:
            pass

def _ms_since(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0

# ---------- config ----------
ckpt_converter = 'checkpoints_v2/converter'
device = (
    "cuda:0" if torch.cuda.is_available()
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
          else "cpu")
)
output_dir = 'outputs_v2'
os.makedirs(output_dir, exist_ok=True)

text = "안녕하세요, 자동차 보험 비교 가입 도와드리는, 차집사 다이렉트, 차은하 팀장입니다. 잠시 통화 가능하실까요? 지금 이용하고 계신 업체 있으실 텐데요, 저희가 이번에, 보험사 연도 대상자 출신들로 팀을 재구성하면서, 수수료 7%의 조건으로 진행을 하고 있어서, 안내차 연락드렸습니다. 사고 건이 많거나 해서, 다이렉트 가입이 안 되시는 고객님들도, 오프라인으로 가입 가능하게 해드리고 있으며, 오프라인, 텔레마케팅, 비교사이트 가입 시 모두, 7% 수수료를 익일 오후에 바로 지급해드리고 있습니다. 수수료 조건도 좋고, 체결율도 95% 이상이라, 많은 분들이 함께하고 계신데요, 앞으로 딜러님, 사장님 담당은 제가 할 거라, 인사차 연락드렸습니다. 제 번호 저장해두셨다가, 견적 문의 있으실 때 연락주시면, 저희가 빠르게 진행 도와드리겠습니다. 명함, 문자로 남겨드릴게요. 감사합니다."
# text = "안녕하세요! 오늘은 날씨가 정말 좋네요."
speed = 1.4
reference_speaker = '/home/work/VALL-E/audio_samples/promotional_calls.wav'  # voice to clone

# ---------- load models (NOT TIMED) ----------
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# speaker embedding from reference (NOT TIMED)
target_se, _ = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)

# TTS model (NOT TIMED)
model = TTS(language='KR', device=device)  # use 'KR' if that works in your build; 'KO' on some versions
speaker_ids = model.hps.data.spk2id

# ---------- optional warmup (NOT COUNTED) ----------
try:
    _sync(device)
    any_spk = next(iter(speaker_ids.values()))
    with torch.inference_mode():
        model.tts_to_file("테스트.", any_spk, f'{output_dir}/_warmup.wav', speed=speed)
    _sync(device)
except Exception as e:
    print(f"[Warmup skipped] {e}")

# ---------- timing loop ----------
results = []
src_path = f'{output_dir}/tmp.wav'

for speaker_key, speaker_id in speaker_ids.items():
    skey = speaker_key.lower().replace('_', '-')
    source_se_path = f'checkpoints_v2/base_speakers/ses/{skey}.pth'
    if not os.path.exists(source_se_path):
        print(f"[Skip] source SE not found for {skey}: {source_se_path}")
        continue
    source_se = torch.load(source_se_path, map_location=device)

    # TTS timing
    _sync(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        model.tts_to_file(text, speaker_id, src_path, speed=speed)
    _sync(device)
    tts_ms = _ms_since(t0)

    # Tone color conversion timing
    save_path = f'{output_dir}/output_v2_{skey}.wav'
    encode_message = "@MyShell"
    _sync(device)
    t1 = time.perf_counter()
    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path,
        message=encode_message,
    )
    _sync(device)
    conv_ms = _ms_since(t1)

    total_ms = tts_ms + conv_ms
    results.append((skey, tts_ms, conv_ms, total_ms))
    print(f"[{skey}] TTS: {tts_ms:.1f} ms | Convert: {conv_ms:.1f} ms | Total: {total_ms:.1f} ms")

# ---------- summary ----------
if results:
    avg_tts  = sum(r[1] for r in results) / len(results)
    avg_conv = sum(r[2] for r in results) / len(results)
    avg_tot  = sum(r[3] for r in results) / len(results)
    print("\n=== Latency Summary (excluding model init/ckpt load) ===")
    print(f"Avg TTS:     {avg_tts:.1f} ms")
    print(f"Avg Convert: {avg_conv:.1f} ms")
    print(f"Avg Total:   {avg_tot:.1f} ms")
else:
    print("No speakers processed.")
