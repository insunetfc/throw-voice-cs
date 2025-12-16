import pandas as pd

# Input files
RESP_CSV = "/home/work/VALL-E/chatbot/data/response_intent_dataset.csv"   # input_text, target_text, intent, source
UTT_CSV  = "/home/work/VALL-E/chatbot/data/UtteranceCacheDataset.csv"     # original_utterance, chatbot_response, ...

# Output files
OUT_PAIRS = "/home/work/VALL-E/chatbot/data/multitask_pairs.csv"          # big training dataset
OUT_BANK  = "/home/work/VALL-E/chatbot/data/response_bank_large.csv"      # big response bank for retrieval

# ---------- 1) Load both sources ----------
df_resp = pd.read_csv(RESP_CSV)
df_utt  = pd.read_csv(UTT_CSV)

# ---------- 2) Normalize columns from response_intent_dataset ----------
# Rename target_text -> response_text
df_resp = df_resp.rename(columns={"target_text": "response_text"})

# Ensure 'source' exists
if "source" not in df_resp.columns:
    df_resp["source"] = "intent_dataset"

resp_cols = ["input_text", "response_text", "intent", "source"]
df_resp = df_resp[resp_cols]

# ---------- 3) Normalize columns from UtteranceCacheDataset ----------
# Here we do NOT have intent, so we leave it as None / NaN
df_utt_std = pd.DataFrame({
    "input_text": df_utt["original_utterance"].astype(str),
    "response_text": df_utt["chatbot_response"].astype(str),
    "intent": [None] * len(df_utt),
    "source": ["utterance_cache"] * len(df_utt),
})

# ---------- 4) Combine them ----------
df_all = pd.concat([df_resp, df_utt_std], ignore_index=True)

# Basic cleaning
for col in ["input_text", "response_text"]:
    df_all[col] = df_all[col].astype(str).str.strip()

df_all = df_all[(df_all["input_text"] != "") & (df_all["response_text"] != "")]

# Drop exact duplicates (optional but usually good)
df_all = df_all.drop_duplicates(subset=["input_text", "response_text", "intent", "source"])

# Save big training pairs
df_all.to_csv(OUT_PAIRS, index=False, encoding="utf-8-sig")
print(f"[+] Saved combined pairs to {OUT_PAIRS}, shape={df_all.shape}")

# ---------- 5) Build a BIG response bank ----------
# Each unique response_text, with usage count
bank = (
    df_all.groupby(["response_text"])
          .size()
          .reset_index(name="count")
          .sort_values("count", ascending=False)
)

bank.to_csv(OUT_BANK, index=False, encoding="utf-8-sig")
print(f"[+] Saved response bank to {OUT_BANK}, shape={bank.shape}")
