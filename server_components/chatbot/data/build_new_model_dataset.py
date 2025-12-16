import pandas as pd

# paths
INTENT_PATH = "/home/tiongsik/Python/outbound_calls/chatbot/data/data/intent_dataset.csv"  # user → intent (439 rows)
CACHE_PATH = "/home/tiongsik/Python/outbound_calls/chatbot/data/data/UtteranceCacheDataset.csv"  # user ↔ bot (1407 rows)
TEMPL_PATH = "/home/tiongsik/Python/outbound_calls/chatbot/data/backup/responses_csv/response_templates_production_updated.csv"  # intent → many responses
OUT_PATH = "/home/tiongsik/Python/outbound_calls/chatbot/data/response_intent_dataset.csv"

def guess_intent_from_user(text: str) -> str:
    if not text:
        return "other"
    t = text.strip()

    if any(k in t for k in ["안녕하세요", "여보세요", "네 말씀하세요"]):
        return "greeting"
    if any(k in t for k in ["정식 업체", "어디 회사", "차집사", "회사 맞아요", "홈페이지"]):
        return "about_company"
    if any(k in t for k in ["몇 %", "몇퍼", "몇 프로", "수수료", "7%", "7 %", "7 프로"]):
        return "fee_question"
    if any(k in t for k in ["관심 없", "패스할게요", "이번에는 패스", "전화 받기 싫", "바빠요", "나중에"]):
        return "rejection"
    if any(k in t for k in ["알겠습니다", "네 감사합니다", "보내드릴게요", "보내주세요"]):
        return "positive"
    if any(k in t for k in ["어떻게 진행", "비교견적", "가입하", "가능한가요", "언제 받을 수"]):
        return "more_questions"
    if any(k in t for k in ["문자로", "카톡으로", "링크", "주소"]):
        return "fallback"
    return "other"

def main():
    df_intent = pd.read_csv(INTENT_PATH)              # question, label
    df_cache = pd.read_csv(CACHE_PATH)                # original_utterance, chatbot_response, ...
    df_templates = pd.read_csv(TEMPL_PATH)            # intent, response, context, notes

    # which intents we actually have safe responses for
    safe_intents = set(df_templates["intent"].unique())

    rows = []

    # 1) from the hand-labeled intent dataset:
    for _, r in df_intent.iterrows():
        q = r["question"]
        intent = r["label"]
        if intent not in safe_intents:
            continue
        # pick ONE canonical response for this intent
        cand = df_templates[df_templates["intent"] == intent].sample(1).iloc[0]["response"]
        rows.append({
            "input_text": q,
            "target_text": cand,
            "intent": intent,
            "source": "intent_dataset",
        })

    # 2) from the cache (real user ↔ real bot)
    for _, r in df_cache.iterrows():
        user = r.get("original_utterance") or r.get("normalized_utterance") or ""
        bot  = r.get("chatbot_response") or ""
        user = str(user).strip()
        bot  = str(bot).strip()
        if not user or not bot:
            continue

        # infer intent based on user
        intent = guess_intent_from_user(user)
        if intent not in safe_intents:
            # still keep it but mark as other, OR skip
            # here we skip to keep training clean
            continue

        rows.append({
            "input_text": user,
            "target_text": bot,
            "intent": intent,
            "source": "utterance_cache",
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"wrote {len(df_out)} rows to {OUT_PATH}")

if __name__ == "__main__":
    main()
