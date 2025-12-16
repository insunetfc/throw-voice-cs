
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid preset generator:
- Fillers (문자/확인/공감/시간벌기형/…): short TTS via /synthesize (single-shot)
- FAQ/chatbot answers: long TTS via /synthesize_stream_start + /synthesize_stream_batch, stitch parts, ensure WAV/8k/mono/µ-law

Requires: requests, boto3, python-dotenv, soundfile, ffmpeg in PATH (for safety transcode)
"""

import os, csv, time, argparse, subprocess, shutil, io, struct, hashlib, json
import requests
import boto3
from botocore.exceptions import ClientError
import soundfile as sf
import unicodedata

# ---------- Config ----------
DEFAULT_TTS_URL = os.getenv("TTS_URL", "https://cfd863800ab8.ngrok-free.app/synthesize").rstrip("/")
# Derive base for streaming endpoints if DEFAULT_TTS_URL already ends with /synthesize
TTS_BASE = DEFAULT_TTS_URL[:-len("/synthesize")] if DEFAULT_TTS_URL.endswith("/synthesize") else DEFAULT_TTS_URL
TTS_TOKEN       = os.getenv("TTS_TOKEN", "").strip()
SAMPLE_RATE     = int(os.getenv("FULL_SAMPLE_RATE", "8000"))
AWS_REGION      = os.getenv("AWS_REGION", "ap-northeast-2")
MAX_RETRIES     = 3
RETRY_SLEEP_SEC = 1.0
CHATBOT_URL     = os.getenv("CHATBOT_URL", "http://15.165.60.45:5000/chat")

FILLER_CATEGORIES = ("문자", "확인", "공감", "시간벌기형")  # short TTS path
MIN_TOKENS_BEFORE_EOS     = 40        # don't allow EOS until at least this many tokens
EOS_GRACE_MS              = 250       # after EOS is seen, keep generating ≥ this much audio
SILENCE_HANGOVER_MS       = 350       # require this much *continuous* silence before we stop
RMS_SILENCE_THRESH        = 0.010     # lower = less aggressive silence detection
TRIM_TAIL_DB              = 30        # if you trim silence anywhere, be gentle on the tail

# ---------- DynamoDB ----------
ddb = boto3.resource("dynamodb", region_name=AWS_REGION)
utt_tbl = ddb.Table("UtteranceCache")

# ---------- S3 ----------
s3 = boto3.client("s3", region_name=AWS_REGION)

# ---------- FAQ utterances only (answers come from chatbot) ----------
FAQS = [
    # 수수료 / 비용
    "수수료가 얼마인가요?",
    "왜 7%인가요?",
    "다른 비용은 없나요?",
    "실제로 제가 내야 하는 금액은 얼마예요?",
    # 가입 조건 / 자격
    "사고가 많은데 가입이 가능합니까?",
    "어떤 조건이 있어야 가입할 수 있나요?",
    "다이렉트로만 가능한가요, 오프라인도 되나요?",
    "제가 지금 보험이 있는데도 가입할 수 있나요?",
    # 지급 / 처리
    "수수료는 언제 지급되나요?",
    "처리 시간은 얼마나 걸리나요?",
    "바로 가입이 되나요?",
    "견적은 언제 받을 수 있나요?",
    # 신뢰도 / 성과
    "체결율이 어떻게 되나요?",
    "다른 사람들도 많이 이용하나요?",
    "이번에 새로 만든 팀은 어떤 팀인가요?",
    "믿을 만한가요?",
    # 담당자 / 연락처
    "앞으로 누가 담당하나요?",
    "담당자가 바뀌면 어떻게 되나요?",
    "연락처를 알려주세요.",
    "명함은 문자로 보내주실 수 있나요?",
    # 일반 응대
    "지금 바빠요.",
    "나중에 다시 전화 주세요.",
    "문자로 보내주세요.",
    "관심 없어요.",
    "이미 다른 데 가입했어요.",
    "그만 연락해 주세요."
]

# ---------- Filler categories (short TTS) ----------
CATEGORIES = {
    "문자": [
        "요청하신 내용을 문자로 바로 보내드리겠습니다.",
        "지금 바쁘시면, 핵심 내용만 문자로 안내드릴게요.",
        "필요하신 정보는 문자로 안내드리겠습니다."
    ],
    "test": ["네, 나중에 다시 연락드리겠습니다."],
    "확인": [
        "네, 고객님 맞으십니다.",
        "네, 말씀하신 내용 확인했습니다.",
        "네, 그렇게 진행하겠습니다.",
        "알겠습니다. 그대로 처리할게요.",
        "확인되었습니다, 잠시만 기다려 주세요.",
        "네, 지금 바로 도와드릴게요.",
        "네, 안내드린 대로 진행하겠습니다.",
        "네, 요청하신 부분 접수했습니다.",
        "네, 문제 없습니다.",
        "네, 이어서 진행하겠습니다.",
    ],
    "설명": [
        "안내드리자면, 이런 절차로 진행됩니다.",
        "간단히 말씀드리면요…",
        "먼저 한 가지 확인 후 설명드릴게요.",
        "순서대로 설명드리겠습니다.",
        "요약해서 말씀드리면요…",
        "자세히 안내드릴게요.",
        "참고로, 이 부분은 이렇게 이해하시면 됩니다.",
        "조금 더 구체적으로 설명드릴게요.",
        "핵심만 짚어서 말씀드리겠습니다.",
        "이어서 추가 설명 드릴게요.",
    ],
    "공감": [
        "아, 그러셨군요. 많이 불편하셨겠어요.",
        "네, 그 마음 이해합니다.",
        "말씀 주셔서 감사합니다.",
        "아, 그런 상황이면 답답하셨겠습니다.",
        "네, 그렇게 느끼실 수 있어요.",
        "공감합니다. 더 신경 쓰겠습니다.",
        "불편을 드려 죄송합니다.",
        "네, 충분히 이해했습니다.",
        "의견 주셔서 감사합니다. 반영해 보겠습니다.",
        "걱정되실 수 있겠습니다.",
    ],
    "시간벌기형": [
        "잠시만 기다려 주세요, 바로 확인하겠습니다.",
        "지금 조회 중입니다…",
        "곧 결과 안내드릴게요…",
        "확인까지 1~2초만 더 부탁드립니다…",
        "자료를 불러오는 중입니다…",
        "금방 연결해 드리겠습니다…",
        "처리 중입니다, 잠시만요…",
        "이어서 준비 중입니다…",
        "조금만 더 기다려 주세요…",
        "확인이 완료되는 대로 말씀드릴게요…",
    ],
#     "test": [
#         "안녕하세요~ 자동차 보험 비교 가입 도와드리는 차집사 다이렉트 차은하 팀장입니다. 잠시 통화 가능하실까요? 지금 이용하고 계신 업체 있으실텐데 저희가 이번에 보험사 연도대상자 출신들로 팀을 재구성 하면서 수수료 7%프로의 조건으로 진행을 하고 있어서 안내차 연락드렸습니다. 사고건이 많거나 해서 다이렉트 가입이 안되시는 고객님들도 OFF 라인으로 가입 가능하게 해드리고 OFF, TM, CM 가입시 모두 7%수수료를 익일오후에 바로 지급 해드리고 있습니다. 수수료 조건도 좋은데 체결율도 95% 이상이라 많은 분들이 함께 하고 계신데 앞으로 딜러님(사장님) 담당은 제가 할꺼라 인사차 연락드렸구요. 제 번호 저장해 두셨다가 견적문의 있으실때 연락주시면 저희가 빠르게 진행 도와드리겠습니다. 명함 문자로 남겨드릴게요~ 감사합니다.",
#     ],
    
}

# ---------- Neutral Messages for Promotional Calls ----------
NEUTRAL_MESSAGES = {
#     # General neutral responses that work for both positive/negative customer reactions
#     "general": [
#         "고객님의 말씀 잘 들었습니다. 검토 후 다시 연락드리겠습니다.",
#         "말씀해 주신 내용 확인했습니다. 적절한 안내를 위해 다시 연락드리겠습니다.",
#         "고객님께 맞는 상품 안내를 위해 검토 후 연락드리겠습니다.",
#         "더 정확한 정보 제공을 위해 확인 후 다시 연락드리겠습니다."
#     ],
    
#     # For busy customers
#     "busy_response": [
#         "바쁘신 중에 죄송합니다. 간단히 문자로 안내드리겠습니다.",
#         "시간 내어 주셔서 감사합니다. 정보를 문자로 보내드리겠습니다.",
#         "바쁘신 것 같으니 필요한 정보만 문자로 전달드리겠습니다."
#     ],
    
#     # For interested but cautious customers  
#     "consideration": [
#         "신중하게 검토하시는 것이 좋습니다. 자세한 자료를 문자로 보내드리겠습니다.",
#         "충분히 비교 검토하시길 바랍니다. 상세 정보를 문자로 안내드리겠습니다.",
#         "고민되시는 부분이 있으시군요. 명확한 정보를 문자로 제공해드리겠습니다."
#     ],
    
#     # For negative responses
#     "not_interested": [
#         "말씀 감사합니다. 혹시 관심 있으실 때를 위해 간단한 정보만 문자로 남겨드리겠습니다.",
#         "이해합니다. 참고용으로 기본 정보만 문자로 보내드리겠습니다.",
#         "알겠습니다. 나중에 필요하실 수도 있으니 연락처만 문자로 남겨드리겠습니다."
#     ]
    
    # "test": [
    #     "(happy) 의견 주셔서 감사합니다. 반영해 보겠습니다. 잠시만 기다려 주세요, 바로 확인하겠습니다.",
    #     "(friendly) 의견 주셔서 감사합니다. 반영해 보겠습니다. 잠시만 기다려 주세요, 바로 확인하겠습니다.",
    #     "(sad) 의견 주셔서 감사합니다. 반영해 보겠습니다. 잠시만 기다려 주세요, 바로 확인하겠습니다.",
    #     # "말씀해 주신 내용 확인했습니다. 적절한 안내를 위해 다시 연락드리겠습니다.",
    #     # "고객님께 맞는 상품 안내를 위해 검토 후 연락드리겠습니다.",
    #     # "더 정확한 정보 제공을 위해 확인 후 다시 연락드리겠습니다."
    # ],
    
    "promotional": [
        "(friendly) 네 저희는 다이렉트 자동차보험 비교 차집사 차은하 팀장입니다. 잠시 통화 가능하실까요? 지금 이용하고 계신 업체 있으실 텐데요, 저희가 이번에, 보험사 연도 대상자 출신들로 팀을 재구성하면서, 수수료 7%의 조건으로 진행을 하고 있어서, 안내차 연락드렸습니다. 사고 건이 많거나 해서, 다이렉트 가입이 안 되시는 고객님들도, 오프라인으로 가입 가능하게 해드리고 있으며, 오프라인, 텔레마케팅, 비교사이트 가입 시 모두, 7% 수수료를 익일 오후에 바로 지급해드리고 있습니다. 수수료 조건도 좋고, 체결율도 95% 이상이라, 많은 분들이 함께하고 계신데요, 앞으로 딜러님, 사장님 담당은 제가 할 거라, 인사차 연락드렸습니다. 제 번호 저장해두셨다가, 견적 문의 있으실 때 연락주시면, 저희가 빠르게 진행 도와드리겠습니다. 명함, 문자로 남겨드릴게요. 감사합니다.",
    ]
}

def generate_neutral_voices():
    """Generate all neutral voice variations"""
    neutral_rows = []
    
    for category, messages in NEUTRAL_MESSAGES.items():
        print(f"\n--- Generating neutral voices: {category} ---")
        
        for idx, text in enumerate(messages, start=1):
            print(f"Neutral {category} #{idx}: \"{text}\"")
            
            # Prepare text for TTS
            tts_text = prepare_for_tts(text)
            
            # Generate audio using short TTS
            info = synthesize_short(
                TTS_BASE, 
                tts_text, 
                key_prefix=f"neutral/{category}", 
                sr=SAMPLE_RATE, 
                token=TTS_TOKEN
            )
            
            bucket = info.get("bucket")
            src_key = info.get("key")
            
            if not bucket or not src_key:
                print(f"[ERROR] Failed to generate {category} #{idx}")
                continue
            
            # Define final key structure
            dst_key = f"neutral/{category}/{idx:02d}.wav"
            
            # Download and process audio
            raw = fetch_s3_bytes(bucket, src_key)
            
            # Apply repair if needed
            repaired_pcm = repair_if_early_terminated(
                raw,
                original_text=text,
                tts_base=TTS_BASE,
                sample_rate=SAMPLE_RATE,
                token=TTS_TOKEN,
                keep_original=False
            )
            
            # Ensure mu-law format for telephony
            final_raw = transcode_to_mulaw_8k_mono(repaired_pcm, sr_out=8000)
            
            # Upload to final location
            put_s3_bytes(bucket, dst_key, final_raw, content_type="audio/wav")
            
            # Clean up source file
            try:
                s3.delete_object(Bucket=bucket, Key=src_key)
            except Exception as e:
                print(f"[WARN] Could not delete {src_key}: {e}")
            
            final_url = to_regional_url(bucket, AWS_REGION, dst_key)
            print(f"    -> s3://{bucket}/{dst_key}")
            
            neutral_rows.append({
                "category": f"neutral_{category}",
                "index": idx,
                "text": text,
                "bucket": bucket,
                "final_key": dst_key,
                "final_url": final_url,
                "usage": "fallback_response"
            })
            
            # Add to UtteranceCache for potential reuse
            norm = normalize_utt(text)
            h = utt_hash(norm)
            utt_tbl.put_item(Item={
                "utterance_hash": h,
                "locale": "ko-KR",
                "normalized_utterance": norm,
                "audio_s3_uri": f"s3://{bucket}/{dst_key}",
                "status": "approved",
                "approved_by": "preset_loader",
                "created_at": int(time.time()),
                "num_hits": 0,
                "notes": f"neutral:{category} #{idx}"
            })
    
    return neutral_rows

def get_default_neutral_message():
    """Get the primary neutral message for cache misses"""
    # This should be the most versatile message that works for all scenarios
    return "고객님의 말씀 잘 들었습니다. 검토 후 다시 연락드리겠습니다."

def get_contextual_neutral_message(customer_response_type="general"):
    """
    Get appropriate neutral message based on customer response context
    customer_response_type: "general", "busy_response", "consideration", "not_interested"
    """
    if customer_response_type in NEUTRAL_MESSAGES:
        # Return first message from the category (you could randomize this)
        return NEUTRAL_MESSAGES[customer_response_type][0]
    return get_default_neutral_message()


# ---------- Helpers ----------
import io, re, numpy as np, soundfile as sf

def voiced_duration_sec(wav_bytes: bytes, sr_expected=8000, frame_ms=10, thr=0.02) -> float:
    """Approximate voiced duration using a simple RMS threshold."""
    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # (sr mismatch is rare here; good enough for detection)
    frame = max(1, int(sr * (frame_ms / 1000.0)))
    last_voiced = 0
    for i in range(0, len(audio) - frame, frame):
        rms = float(np.sqrt(np.mean(np.square(audio[i:i+frame])) + 1e-9))
        if rms > thr:
            last_voiced = i + frame
    return last_voiced / float(sr if sr else sr_expected)

_SENT_END_RE = re.compile(r"(?<=[\.!\?…])\s+")

def split_korean_sentences(text: str) -> list[str]:
    parts = _SENT_END_RE.split(text.strip())
    return [p for p in parts if p]

def _should_stop(now_tokens, saw_eos, ms_since_eos, ms_silence, have_audio_ms):
    # 1) Don't allow EOS too early
    if saw_eos and now_tokens < MIN_TOKENS_BEFORE_EOS:
        return False
    # 2) After EOS, insist on a grace window of extra audio
    if saw_eos and ms_since_eos < EOS_GRACE_MS:
        return False
    # 3) Only stop on silence if it's held long enough *and* we've generated enough speech
    if ms_silence >= SILENCE_HANGOVER_MS and have_audio_ms >= 1000:
        return True
    # 4) If EOS was seen long enough ago, it's okay to stop
    if saw_eos and ms_since_eos >= EOS_GRACE_MS:
        return True
    return False

def audio_rms(x):
    import numpy as np
    if x is None or len(x) == 0: return 0.0
    return float(np.sqrt((x.astype("float32") ** 2).mean() + 1e-9))

_ZWJ = "\u2060"  # WORD JOINER (prevents '겠' dropout)

def prepare_for_tts(text: str) -> str:
    """
    TTS-only preprocessing (do NOT use for hashing):
      - NFKC normalize & trim
      - Strip simple markdown markers
      - Make URLs speakable in Korean
      - Expand % -> '퍼센트'
      - Prevent '겠' skip with WORD JOINER
      - Gentle sentence-final tail
    """
    s = unicodedata.normalize("NFKC", text).strip()

    # 1) Strip simple markdown
    s = s.replace("**", "").replace("__", "").replace("*", "").replace("_", "").replace("`", "")

    # 2) URLs -> speakable
    def _speak_url(m):
        full = m.group(0)
        core = re.sub(r"^https?://", "", full, flags=re.I)
        core = re.sub(r"^www\.", "더블유 더블유 더블유 점 ", core, flags=re.I)
        core = core.replace(".com", " 닷컴")
        core = core.replace(".kr", " 점 케이알")
        core = core.replace(".", " 점 ")
        core = core.replace("/", " 슬래시 ")
        return core.strip()
    s = re.sub(r"https?://\S+|www\.\S+", _speak_url, s, flags=re.I)

    # 3) Safe glyph & numeric expansions
    s = s.replace("켇", "켜")                 # rare corruption guard
    s = re.sub(r"(\d+)\s*%", r"\1 퍼센트", s)  # 15% -> 15 퍼센트

    s = re.sub(
        r"[\U0001F600-\U0001F64F"
        r"\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF"
        r"\U0001F1E0-\U0001F1FF"
        r"\u2600-\u26FF\u2700-\u27BF]+",
        "",
        s
    )
    s = re.sub(r"[:;][\-\^]?[)D(]+", "", s)

    # 4) Collapse whitespace
    s = re.sub(r"\s+", " ", s)

    # 5) Prevent '겠' dropout before common endings (TTS-only)
    s = re.sub(r"겠(?=(다|습니다|어요|에요|지요|네요|군요|고요|죠))", "겠" + _ZWJ, s)

    # 6) Ensure a gentle sentence ending (so engines don't truncate)
    if not re.search(r"[.!?…~다요]$", s):
        s += "…"

    return s

def stitch_float_wavs(wavs: list[bytes]) -> bytes:
    """Concatenate several WAV byte blobs (PCM16/float) into one PCM16 WAV, with a tiny gap."""
    arrs = []
    sr_ref = None
    for wb in wavs:
        a, sr = sf.read(io.BytesIO(wb), dtype="float32", always_2d=False)
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr_ref is None:
            sr_ref = sr
        arrs.append(a)
        # 40 ms gap between sentences to avoid clicks / rushed joins
        gap = np.zeros(int((sr_ref or 8000) * 0.04), dtype="float32")
        arrs.append(gap)
    if arrs:
        arrs = arrs[:-1]  # drop last gap
    cat = np.concatenate(arrs) if arrs else np.zeros(0, dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, cat, sr_ref or 8000, subtype="PCM_16", format="WAV")
    return buf.getvalue()

def repair_if_early_terminated(raw_wav: bytes, original_text: str, tts_base: str, sample_rate: int, token: str, keep_original=False) -> bytes:
    """
    If speech ends too early (or likely risky ending like '겠습니다'), re-synthesize by sentence with short TTS and stitch.
    Returns repaired PCM WAV bytes (or the input if no repair needed).
    """
    v_sec = voiced_duration_sec(raw_wav, sr_expected=sample_rate)
    # total duration from header
    audio_len, sr = sf.read(io.BytesIO(raw_wav), dtype="float32", always_2d=False)
    total_sec = len(audio_len) / float(sr if sr else sample_rate)

    risky = ("겠습니다" in original_text)  # common KR polite tail that sometimes drops
    needs_repair = (v_sec < 0.7 * total_sec) or (risky and v_sec < total_sec - 0.3)
    if not needs_repair:
        return raw_wav  # looks fine

    print(f"[INFO] Early termination detected (voiced={v_sec:.2f}s/{total_sec:.2f}s). Repairing by sentences…")
    sents = split_korean_sentences(original_text)
    if not sents:
        sents = [original_text]

    parts = []
    for s in sents:
        tts_s = prepare_for_tts(s)
        info_seg = synthesize_short(tts_base, tts_s, key_prefix="repair", sr=sample_rate, token=token)
        b2, k2 = info_seg.get("bucket"), info_seg.get("key")
        if not b2 or not k2:
            continue
        seg_bytes = fetch_s3_bytes(b2, k2)
        parts.append(seg_bytes)
        if not keep_original:
            try:
                s3.delete_object(Bucket=b2, Key=k2)
            except Exception:
                pass
    if not parts:
        return raw_wav  # fallback: keep original

    return stitch_float_wavs(parts)


_ZWJ = "\u2060"  # WORD JOINER (prevents '겠' dropout)

def prepare_for_tts(text: str) -> str:
    """
    TTS-only preprocessing (do NOT use for hashing):
      - NFKC normalize & trim
      - Strip simple markdown markers
      - Make URLs speakable in Korean
      - Expand % -> '퍼센트'
      - Prevent '겠' skip with WORD JOINER
      - Gentle sentence-final tail
    """
    s = unicodedata.normalize("NFKC", text).strip()

    # 1) Strip simple markdown
    s = s.replace("**", "").replace("__", "").replace("*", "").replace("_", "").replace("`", "")

    # 2) URLs -> speakable
    def _speak_url(m):
        full = m.group(0)
        core = re.sub(r"^https?://", "", full, flags=re.I)
        core = re.sub(r"^www\.", "더블유 더블유 더블유 점 ", core, flags=re.I)
        core = core.replace(".com", " 닷컴")
        core = core.replace(".kr", " 점 케이알")
        core = core.replace(".", " 점 ")
        core = core.replace("/", " 슬래시 ")
        return core.strip()
    s = re.sub(r"https?://\S+|www\.\S+", _speak_url, s, flags=re.I)

    # 3) Safe glyph & numeric expansions
    s = s.replace("켇", "켜")                 # rare corruption guard
    s = re.sub(r"(\d+)\s*%", r"\1 퍼센트", s)  # 15% -> 15 퍼센트

    s = re.sub(
        r"[\U0001F600-\U0001F64F"
        r"\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF"
        r"\U0001F1E0-\U0001F1FF"
        r"\u2600-\u26FF\u2700-\u27BF]+",
        "",
        s
    )
    s = re.sub(r"[:;][\-\^]?[)D(]+", "", s)

    # 4) Collapse whitespace
    s = re.sub(r"\s+", " ", s)

    # 5) Prevent '겠' dropout before common endings (TTS-only)
    s = re.sub(r"겠(?=(다|습니다|어요|에요|지요|네요|군요|고요|죠))", "겠" + _ZWJ, s)

    # 6) Ensure a gentle sentence ending (so engines don't truncate)
    if not re.search(r"[.!?…~다요]$", s):
        s += "…"

    return s

def normalize_utt(text: str) -> str:
    return text.strip().lower()

def utt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def to_regional_url(bucket: str, region: str, key: str) -> str:
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"

def parse_wav_fmt_sr_ch(wav_bytes: bytes):
    bio = io.BytesIO(wav_bytes)
    if bio.read(4) != b'RIFF': return (None, None, None)
    _ = bio.read(4)
    if bio.read(4) != b'WAVE': return (None, None, None)
    while True:
        hdr = bio.read(8)
        if len(hdr) < 8: return (None, None, None)
        chunk_id, chunk_sz = hdr[:4], struct.unpack("<I", hdr[4:8])[0]
        if chunk_id == b'fmt ':
            fmt_data = bio.read(chunk_sz)
            if len(fmt_data) < 16: return (None, None, None)
            fmt_code   = struct.unpack("<H", fmt_data[0:2])[0]
            channels   = struct.unpack("<H", fmt_data[2:4])[0]
            sample_rate= struct.unpack("<I", fmt_data[4:8])[0]
            return (fmt_code, channels, sample_rate)
        else:
            bio.seek(chunk_sz + (chunk_sz % 2), io.SEEK_CUR)

def is_mulaw_8k_mono(wav_bytes: bytes) -> bool:
    fmt_code, ch, sr = parse_wav_fmt_sr_ch(wav_bytes)
    return (fmt_code == 7 and ch == 1 and sr == 8000)

def transcode_to_mulaw_8k_mono(in_wav_bytes: bytes, sr_out: int = 8000) -> bytes:
    if not has_ffmpeg():
        raise RuntimeError("ffmpeg not found in PATH; please install it.")
    # Add ~0.5s of silence tail at transcode time
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "wav", "-i", "pipe:0",
        "-ar", str(sr_out), "-ac", "1", "-c:a", "pcm_mulaw",
        "-af", "apad=pad_dur=0.5",          # ← tail padding at encode
        "-f", "wav", "pipe:1"
    ]
    proc = subprocess.run(cmd, input=in_wav_bytes, capture_output=True, check=True)
    return proc.stdout


def fetch_s3_bytes(bucket: str, key: str) -> bytes:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()

def put_s3_bytes(bucket: str, key: str, data: bytes, content_type: str = "audio/wav"):
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)

# ---------- TTS clients ----------
def _headers(token: str = ""):
    h = {"Content-Type": "application/json"}
    if token: h["Authorization"] = f"Bearer {token}"
    return h

def generate_neutral_by_temps(temps, base_category="test", message=None,
                              top_p=None, repetition_penalty=None,
                              max_new_tokens=None, chunk_length=None,
                              use_memory_cache=None):
    """
    For each temperature in `temps`, synthesize the neutral message once and store as:
    s3://<bucket>/neutral/<base_category>/T<temp>/<idx>.wav
    Returns a list of metadata rows, just like your other generators.
    """
    rows = []
    messages = [message] if message else NEUTRAL_MESSAGES.get(base_category, [])
    if not messages:
        print(f"[WARN] No messages found for category '{base_category}'")
        return rows

    for temp in temps:
        temp_tag = f"T{float(temp):.2f}".replace(".", "_")  # e.g., T0_80
        for idx, text in enumerate(messages, start=1):
            tts_text = prepare_for_tts(text)

            info = synthesize_short(
                TTS_BASE,
                tts_text,
                key_prefix=f"neutral/{base_category}/{temp_tag}",
                sr=SAMPLE_RATE,
                token=TTS_TOKEN,
                temperature=float(temp),
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                chunk_length=chunk_length,
                use_memory_cache=use_memory_cache,
            )

            bucket = info.get("bucket"); src_key = info.get("key")
            if not bucket or not src_key:
                print(f"[ERROR] Failed at temp={temp} #{idx}")
                continue

            # fetch, repair (if early stop), transcode to µ-law/8k/mono
            raw = fetch_s3_bytes(bucket, src_key)
            repaired_pcm = repair_if_early_terminated(
                raw, original_text=text, tts_base=TTS_BASE, sample_rate=SAMPLE_RATE,
                token=TTS_TOKEN, keep_original=False
            )
            final_raw = transcode_to_mulaw_8k_mono(repaired_pcm, sr_out=8000)

            dst_key = f"neutral/{base_category}/{temp_tag}/{idx:02d}.wav"
            put_s3_bytes(bucket, dst_key, final_raw, content_type="audio/wav")
            try:
                s3.delete_object(Bucket=bucket, Key=src_key)
            except Exception:
                pass

            final_url = to_regional_url(bucket, AWS_REGION, dst_key)
            print(f"[OK] temp={temp:.2f} → s3://{bucket}/{dst_key}")

            rows.append({
                "category": f"neutral_{base_category}_{temp_tag}",
                "index": idx,
                "text": text,
                "bucket": bucket,
                "final_key": dst_key,
                "final_url": final_url,
                "temperature": float(temp),
                "usage": "fallback_response"
            })
    return rows


def synthesize_short(tts_base: str, text: str, key_prefix: str, sr: int, token: str = "",
                     temperature: float | None = None, top_p: float | None = None,
                     repetition_penalty: float | None = None, max_new_tokens: int | None = None,
                     chunk_length: int | None = None, use_memory_cache: bool | None = None):
    url = f"{tts_base}/synthesize"
    payload = {
        "text": text,
        "sample_rate": sr,
        "key_prefix": key_prefix,
    }
    # only include provided knobs
    if temperature is not None:        payload["temperature"] = float(temperature)
    if top_p is not None:              payload["top_p"] = float(top_p)
    if repetition_penalty is not None: payload["repetition_penalty"] = float(repetition_penalty)
    if max_new_tokens is not None:     payload["max_new_tokens"] = int(max_new_tokens)
    if chunk_length is not None:       payload["chunk_length"] = int(chunk_length)
    if use_memory_cache is not None:   payload["use_memory_cache"] = bool(use_memory_cache)

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, headers=_headers(token), timeout=60)
            if resp.status_code == 200:
                return resp.json()
            last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(RETRY_SLEEP_SEC * attempt)
    raise RuntimeError(f"/synthesize failed: {last_err}")

def synthesize_long_stream(tts_base: str, text: str, sr: int, token: str = "") -> tuple[str, str]:
    """
    Start streaming job and return (job_id, bucket). You’ll poll parts and stitch.
    """
    url = f"{tts_base}/synthesize_stream_start"
    payload = {"text": text, "sample_rate": sr}
    resp = requests.post(url, json=payload, headers=_headers(token), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["job_id"], data["bucket"]

def poll_stream_batch(tts_base: str, job_id: str, start_idx: int = 0, limit: int = 8, token: str = "", expires: int = 600):
    url = f"{tts_base}/synthesize_stream_batch"
    params = {"job_id": job_id, "start_idx": start_idx, "limit": limit, "expires": expires}
    resp = requests.get(url, params=params, headers=_headers(token), timeout=30)
    resp.raise_for_status()
    return resp.json()

def download_wav(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

def stitch_wavs_to_bytes(urls: list[str]) -> bytes:
    """Download each WAV URL, read as float32 mono, concat, return WAV bytes (float PCM).
       We’ll µ-law-transcode later anyway."""
    floats = []
    sr_ref = None
    for u in urls:
        audio, sr = sf.read(io.BytesIO(download_wav(u)), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr_ref is None:
            sr_ref = sr
        elif sr != sr_ref:
            # resample lightly via soundfile+numpy if needed (rare for your server)
            import numpy as np
            t = np.linspace(0.0, 1.0, num=audio.size, endpoint=False, dtype=np.float32)
            # (simple resample omitted for brevity; your server outputs consistent SR)
        floats.append(audio)
    import numpy as np
    cat = np.concatenate(floats) if floats else np.zeros(0, dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, cat, sr_ref or 8000, subtype="PCM_16", format="WAV")
    return buf.getvalue()

# ---------- Chatbot ----------
def get_chatbot_answer(utt: str) -> str:
    payload = {"session_id": "preset-loader", "question": utt}
    r = requests.post(CHATBOT_URL, data=json.dumps(payload), headers={"Content-Type": "application/json"}, timeout=30)
    r.raise_for_status()
    try:
        data = r.json()
    except Exception:
        return r.text
    return data.get("answer") or data.get("response") or str(data)

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Hybrid synth presets (short fillers via /synthesize; FAQ via streaming).")
    parser.add_argument("--tts-url", default=DEFAULT_TTS_URL, help="app.py URL (e.g., http://host:8000/synthesize)")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE, help="target sample rate (default 8000)")
    parser.add_argument("--token", default=TTS_TOKEN, help="Bearer token if server requires auth")
    parser.add_argument("--region", default=AWS_REGION, help="AWS region for S3")
    parser.add_argument("--keep-original", action="store_true", help="Do not delete UUID source objects created server-side")
    parser.add_argument("--out-csv", default="filler_index.csv", help="Index CSV path")
    parser.add_argument("--force-transcode", action="store_true", help="Always transcode to µ-law even if already correct")
    parser.add_argument("--neutral-only", default=True, action="store_true", help="Generate only neutral voices")
    parser.add_argument("--sweep", action="store_true", help="run temperature sweep for test category")
    # change default temps to empty
    parser.add_argument("--temps", default="0.9", help="Comma-separated temperatures (used only with --sweep)")
    args = parser.parse_args()

    rows = []
    total = sum(len(v) for v in CATEGORIES.values())
    i = 0

    if not has_ffmpeg():
        print("[WARN] ffmpeg not found; non-µlaw inputs will be force-converted later.")
    
    # Generate neutral voices
    print("=== Generating Neutral Voices ===")
    neutral_rows = generate_neutral_voices()
    rows.extend(neutral_rows)

    # Also generate the neutral 'test' message across different temperatures for A/B comparison
    temps = [float(x) for x in (args.temps.split(",") if args.temps else []) if x.strip()]
    if args.sweep and temps:
        print(f"=== Generating Neutral 'test' across temperatures: {temps} ===")
        sweep_rows = generate_neutral_by_temps(
            temps,
            base_category="promotional",                                # <- use your real line’s bucket path
            message=NEUTRAL_MESSAGES["promotional"][0],                 # <- generate exactly THIS one
            top_p=0.8, repetition_penalty=1.25, max_new_tokens=128,
            chunk_length=128, use_memory_cache=False,
        )
        rows.extend(sweep_rows)
    
    if not args.neutral_only:

        # -------- Fillers (SHORT) --------
        print(f"Starting fillers → {TTS_BASE}/synthesize")
        for category, phrases in CATEGORIES.items():
            for idx, text in enumerate(phrases, start=1):
                i += 1
                print(f"[{i}/{total}] {category} #{idx}: \"{text}\" (short)")
                orig = text
                norm = normalize_utt(orig)         # key
                utt_h = utt_hash(norm)

                tts_text = prepare_for_tts(orig)
                info = synthesize_short(TTS_BASE, tts_text, key_prefix=category, sr=args.sample_rate, token=args.token)  # single-shot:contentReference[oaicite:3]{index=3}
                bucket = info.get("bucket"); src_key = info.get("key")
                if not bucket or not src_key:
                    raise RuntimeError(f"Server did not return bucket/key for {category} #{idx}: {info}")

                dst_key = f"{category}/{idx:02d}.wav"
                raw = fetch_s3_bytes(bucket, src_key)

                # ADD this block 👇
                repaired_pcm = repair_if_early_terminated(
                    raw,
                    original_text=orig,          # <- the original text (no tail mark)
                    tts_base=TTS_BASE,
                    sample_rate=args.sample_rate,
                    token=args.token,
                    keep_original=bool(args.keep_original),
                )

                must_transcode = args.force_transcode or (not is_mulaw_8k_mono(raw))
                if must_transcode:
                    raw = transcode_to_mulaw_8k_mono(raw, sr_out=8000)
                    print("    -> transcoded to µ-law/8k/mono")

                put_s3_bytes(bucket, dst_key, raw, content_type="audio/wav")

                if not args.keep_original:
                    try: s3.delete_object(Bucket=bucket, Key=src_key)
                    except ClientError as e: print(f"[WARN] Could not delete original {src_key}: {e}")

                final_url = to_regional_url(bucket, args.region, dst_key)
                print(f"    -> s3://{bucket}/{dst_key}")
                print(f"    -> URL: {final_url}")

                rows.append({
                    "category": category, "index": idx, "text": text,
                    "bucket": bucket, "final_key": dst_key, "final_url": final_url,
                    "src_key": src_key, "server_url": info.get("url") or info.get("s3_url"),
                    "sample_rate": info.get("sample_rate", args.sample_rate),
                    "transcoded": "yes" if must_transcode else "no",
                })

                # — DDB: utterance-only row (preset)
                norm = normalize_utt(text); h = utt_hash(norm)
                utt_tbl.put_item(Item={
                    "utterance_hash": h,
                    "locale": "ko-KR",
                    "normalized_utterance": norm,
                    "audio_s3_uri": f"s3://{bucket}/{dst_key}",
                    "status": "approved",
                    "approved_by": "preset_loader",
                    "created_at": int(time.time()),
                    "num_hits": 0,
                    "notes": f"preset:{category} #{idx}"
                })


        # -------- FAQ via Chatbot (LONG streaming path) --------
        print("\n--- Preloading FAQ answers via chatbot (streaming) ---")
        for utt in FAQS:
            print(f"FAQ: Q=\"{utt}\"  → asking chatbot…")
            answer_text = get_chatbot_answer(utt)  # your chatbot’s answer text
            answer_text = prepare_for_tts(answer_text)
            norm_utt  = normalize_utt(utt);   utt_h  = utt_hash(norm_utt)
            norm_resp = normalize_utt(answer_text); resp_h = utt_hash(norm_resp)

            # Start stream job
            job_id, bucket = synthesize_long_stream(TTS_BASE, answer_text, sr=args.sample_rate, token=args.token)  # streaming API:contentReference[oaicite:4]{index=4}

            # Poll parts until no more (simple bounded loop)
            part_idx = 0
            collected_urls = []
            max_empty_polls = 20
            empty_polls = 0
            while True:
                batch = poll_stream_batch(TTS_BASE, job_id, start_idx=part_idx, limit=6, token=args.token, expires=600)
                urls = batch.get("AudioS3Urls", []) or []
                if not urls:
                    empty_polls += 1
                    if empty_polls >= max_empty_polls:
                        break
                    time.sleep(0.3)
                    continue
                empty_polls = 0
                collected_urls.extend(urls)
                part_idx = int(batch.get("NextIndexOut", part_idx))
                has_more = str(batch.get("HasMore", "")).lower() == "true"
                if not has_more:
                    break

            # After polling collected_urls
            if not collected_urls:
                # 1) Try final.wav from the job path
                final_key_guess = f"{job_id}/final.wav"
                try:
                    s3.head_object(Bucket=bucket, Key=final_key_guess)
                    # If exists, download and use it
                    final_bytes = fetch_s3_bytes(bucket, final_key_guess)
                    final_raw = transcode_to_mulaw_8k_mono(final_bytes, sr_out=8000)
                    final_key = f"faq/{utt_h}.wav"
                    put_s3_bytes(bucket, final_key, final_raw, content_type="audio/wav")
                    print(f"    -> (fallback final.wav) s3://{bucket}/{final_key}")
                except Exception:
                    print(f"[WARN] No parts & no final.wav; falling back to /synthesize for: {utt}")
                    info2 = synthesize_short(TTS_BASE, answer_text, key_prefix="faq", sr=args.sample_rate, token=args.token)
                    bucket2 = info2.get("bucket")
                    src_key2 = info2.get("key")
                    if not src_key2 or not bucket2:
                        print(f"[ERROR] Short fallback failed for: {utt}")
                        continue

                    raw2 = fetch_s3_bytes(bucket2, src_key2)
                    repaired_pcm = repair_if_early_terminated(
                        raw2,
                        original_text=answer_text,
                        tts_base=TTS_BASE,
                        sample_rate=args.sample_rate,
                        token=args.token,
                        keep_original=bool(args.keep_original),
                    )

                    final_raw = transcode_to_mulaw_8k_mono(repaired_pcm, sr_out=8000)
                    final_key = f"faq/{utt_h}.wav"
                    put_s3_bytes(bucket2, final_key, final_raw, content_type="audio/wav")
                    if not args.keep_original:
                        try:
                            s3.delete_object(Bucket=bucket2, Key=src_key2)
                        except Exception as e:
                            print(f"[WARN] Could not delete original {src_key2}: {e}")


                # Write DDB row (utterance + response hashes)
                utt_tbl.put_item(Item={
                    "utterance_hash": utt_h,
                    "locale": "ko-KR",
                    "normalized_utterance": norm_utt,
                    "response_hash": resp_h,
                    "normalized_response": norm_resp,
                    "audio_s3_uri": f"s3://{bucket}/{final_key}",
                    "status": "approved",
                    "approved_by": "preset_loader",
                    "created_at": int(time.time()),
                    "num_hits": 0,
                    "notes": "faq bootstrap (fallback)"
                })
                continue  # go to next FAQ

            # Stitch parts → WAV bytes
            stitched_wav = stitch_wavs_to_bytes(collected_urls)
            # Transcode to μ-law/8k/mono (safety)
            repaired_pcm = repair_if_early_terminated(
                stitched_wav,
                original_text=answer_text,   # the chatbot answer BEFORE transcode
                tts_base=TTS_BASE,
                sample_rate=args.sample_rate,
                token=args.token,
                keep_original=bool(args.keep_original),
            )

            final_raw = transcode_to_mulaw_8k_mono(repaired_pcm, sr_out=8000)

            final_key = f"faq/{utt_h}.wav"
            put_s3_bytes(bucket, final_key, final_raw, content_type="audio/wav")
            print(f"    -> s3://{bucket}/{final_key}")

            # DDB row with both utterance + response hashes
            utt_tbl.put_item(Item={
                "utterance_hash": utt_h,
                "locale": "ko-KR",
                "normalized_utterance": norm_utt,
                "response_hash": resp_h,
                "normalized_response": norm_resp,
                "audio_s3_uri": f"s3://{bucket}/{final_key}",
                "status": "approved",
                "approved_by": "preset_loader",
                "created_at": int(time.time()),
                "num_hits": 0,
                "notes": "faq bootstrap"
            })

    # -------- Write index CSV --------
    if rows:
        with open("filler_index.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        print("\nDone. Wrote index CSV: filler_index.csv")
    else:
        print("No filler rows to write.")

if __name__ == "__main__":
    main()