import csv
import os

SRC_CSV = "/home/tiongsik/Python/outbound_calls/ddb/UtteranceCache_complete.csv"
DST_CSV = "/home/tiongsik/Python/outbound_calls/chatbot/data/UtteranceCacheDataset.csv"

# columns from your sample
# you can add/remove here depending on what you want to train on
KEEP_COLS = [
    "original_utterance",
    "normalized_utterance",
    "chatbot_response",
    "locale",
    "audio_s3_uri",
    "response_hash",
    "utterance_hash",
    "approval_type",
    "approved_by",
]

def is_too_similar(user_text: str, bot_text: str) -> bool:
    if not user_text or not bot_text:
        return False

    u = user_text.strip().lower()
    b = bot_text.strip().lower()

    # exact
    if u == b:
        return True

    # often your user asks "..." and bot replies same sentence w/o punctuation
    # e.g. "지금은 정식 업체 맞아요?" vs "지금은 정식 업체 맞아요"
    if u.rstrip("?.!").strip() == b.rstrip("?.!").strip():
        return True

    # if one is very short and contained in the other
    if (u in b or b in u) and min(len(u), len(b)) <= 12:
        return True

    # token overlap (loose)
    u_tokens = set(u.split())
    b_tokens = set(b.split())
    if u_tokens:
        overlap = len(u_tokens & b_tokens) / len(u_tokens)
        if overlap > 0.75:
            return True

    return False

def main():
    if not os.path.exists(SRC_CSV):
        raise FileNotFoundError(f"Source CSV not found: {SRC_CSV}")

    with open(SRC_CSV, "r", encoding="utf-8") as fsrc, \
         open(DST_CSV, "w", encoding="utf-8", newline="") as fdst:

        reader = csv.DictReader(fsrc)
        # intersect KEEP_COLS with actual CSV header, in case some columns don't exist
        header = reader.fieldnames or []
        out_cols = [c for c in KEEP_COLS if c in header]

        writer = csv.DictWriter(fdst, fieldnames=out_cols)
        writer.writeheader()

        total = 0
        written = 0
        dropped_similar = 0
        dropped_empty = 0

        for row in reader:
            total += 1

            # in your sample:
            # - user text lives in "original_utterance"      (e.g. "실례지만 추가 비용은 없는지 알려주세요")
            # - bot text lives in "chatbot_response"         (e.g. """추가 비용은 전혀 없습니다!...""")
            user_text = (
                row.get("original_utterance")
                or row.get("normalized_utterance")
                or ""
            )
            bot_text = (
                row.get("chatbot_response")
                or row.get("tts_text")
                or row.get("cached_text")
                or ""
            )

            # skip if either side is missing (can't train on it)
            if not user_text.strip() or not bot_text.strip():
                dropped_empty += 1
                continue

            # drop if too similar
            if is_too_similar(user_text, bot_text):
                dropped_similar += 1
                continue

            # write only selected columns
            out_row = {k: row.get(k, "") for k in out_cols}
            writer.writerow(out_row)
            written += 1

    print(f"total rows: {total}")
    print(f"written: {written}")
    print(f"dropped (empty): {dropped_empty}")
    print(f"dropped (similar): {dropped_similar}")
    print(f"output: {DST_CSV}")

if __name__ == "__main__":
    main()
