import pandas as pd
import sys

def infer_intent(text: str) -> str:
    if not isinstance(text, str):
        return "other"
    t = text.strip()
    if any(k in t for k in ["안녕하세요", "여보세요", "네 말씀하세요"]):
        return "greeting"
    if any(k in t for k in ["정식 업체", "어디 회사", "차집사", "회사 맞아요", "홈페이지"]):
        return "about_company"
    if any(k in t for k in ["몇 %", "몇퍼", "몇 프로", "수수료", "7%", "7 %", "7 프로"]):
        return "fee_question"
    if any(k in t for k in ["관심 없", "패스할게요", "이번에는 패스", "전화 받기 싫", "나중에"]):
        return "rejection"
    if any(k in t for k in ["알겠습니다", "감사합니다", "보내드릴게요"]):
        return "positive"
    if any(k in t for k in ["어떻게 진행", "절차", "비교견적", "가입하", "가능한가요", "언제 받을 수"]):
        return "more_questions"
    if any(k in t for k in ["문자로", "카톡으로", "링크", "주소"]):
        return "fallback"
    return "other"

df = pd.read_csv("/home/tiongsik/Python/outbound_calls/chatbot/data/data/UtteranceCacheDataset.csv")
df["input_text"] = df["original_utterance"].fillna(df.get("normalized_utterance", ""))
df["response_text"] = df["chatbot_response"].fillna("")
df = df[(df["input_text"].str.strip() != "") & (df["response_text"].str.strip() != "")]

# infer intent same as before
df["intent"] = df["input_text"].apply(infer_intent)

# drop exact duplicate responses inside the same intent
df = df.sort_values("response_text")
df = df.drop_duplicates(subset=["intent", "response_text"])

df.to_csv("/home/tiongsik/Python/outbound_calls/chatbot/data/data/UtteranceCacheDataset_dedup.csv", index=False, encoding="utf-8")
