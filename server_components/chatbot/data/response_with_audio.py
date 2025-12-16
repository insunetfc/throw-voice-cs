import pandas as pd

BANK_IN   = "/home/work/VALL-E/chatbot/data/response_bank_large.csv"        # current big bank (response_text, count)
UTT_CSV   = "/home/work/VALL-E/chatbot/data/UtteranceCacheDataset.csv"      # original cache with audio
BANK_OUT  = "/home/work/VALL-E/chatbot/data/response_bank_with_audio.csv"   # new enriched bank

# Load both
bank = pd.read_csv(BANK_IN)
utt  = pd.read_csv(UTT_CSV)

# Keep only the audio-related columns from utterance cache
utt_small = (
    utt[["chatbot_response", "audio_s3_uri", "response_hash", "locale"]]
    .drop_duplicates(subset=["chatbot_response"])
)

# Join: response_text (bank) ↔ chatbot_response (utterance cache)
merged = bank.merge(
    utt_small,
    left_on="response_text",
    right_on="chatbot_response",
    how="left",
)

# We no longer need 'chatbot_response' column after merge
merged = merged.drop(columns=["chatbot_response"])

merged.to_csv(BANK_OUT, index=False, encoding="utf-8-sig")
print("Saved:", BANK_OUT, "shape:", merged.shape)
print(merged.head())
