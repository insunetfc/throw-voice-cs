"""
Retriever-based chatbot API with audio reuse.

- Dual-encoder model trained on multitask_pairs.csv
- Retrieves from LARGE response bank (response_bank_with_audio.csv)
- Reuses pre-generated TTS audio from UtteranceCacheDataset via:
    response_text -> audio_s3_uri, response_hash
- No rule-based intent → reply mapping; intent is auxiliary.
"""

import os
from typing import List, Dict, Any

import torch
import torch.nn as nn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertModel

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
RETRIEVER_CKPT_PATH = os.environ.get(
    "RETRIEVER_CKPT_PATH",
    "/home/work/VALL-E/chatbot/models/models_multitask/retriever_multitask_pairs.pt",
)

# IMPORTANT: this is the enriched bank with audio info
RESPONSE_BANK_CSV = os.environ.get(
    "RESPONSE_BANK_CSV",
    "/home/work/VALL-E/chatbot/data/response_bank_with_urls.csv",
)

SIM_THRESHOLD = float(os.environ.get("SIM_THRESHOLD", "0.45"))
TOP_K = int(os.environ.get("TOP_K", "5"))

FALLBACK_RESPONSE = os.environ.get(
    "FALLBACK_RESPONSE",
    "죄송합니다. 제가 정확히 이해하지 못했어요. 한 번만 더 말씀해 주시겠어요?",)

FALLBACK_RESPONSE_URL = os.environ.get(
    "FALLBACK_RESPONSE_URL",
    "https://tts-bucket-250810.s3.ap-northeast-2.amazonaws.com/connect/sessions/95b71cbf-8df6-452a-943c-e41b4a3c15fd/c2d462a9faa547b39b246c1adba4bd94/c46a9b9e16a44c8f8bb17957c8fe3c6b.wav",
)

DEFAULT_MAX_LEN = int(os.environ.get("MAX_LEN", "48"))


# -------------------------------------------------------------------
# Model definition (must match training)
# -------------------------------------------------------------------
class BertRetrieverMultiTask(nn.Module):
    def __init__(self, model_name: str, num_intents: int):
        super().__init__()
        self.bert_q = BertModel.from_pretrained(model_name)
        self.bert_r = BertModel.from_pretrained(model_name)
        hidden_size = self.bert_q.config.hidden_size
        self.intent_head = nn.Linear(hidden_size, num_intents)

    @staticmethod
    def mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode_query(self, ids, mask):
        out = self.bert_q(input_ids=ids, attention_mask=mask)
        return self.mean_pool(out.last_hidden_state, mask)

    def encode_response(self, ids, mask):
        out = self.bert_r(input_ids=ids, attention_mask=mask)
        return self.mean_pool(out.last_hidden_state, mask)


# -------------------------------------------------------------------
# Retrieval runtime
# -------------------------------------------------------------------
class RetrievalRuntime:
    def __init__(self, ckpt_path: str, bank_csv: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.intent2id: Dict[str, int] = ckpt["intent2id"]
        id2intent_raw = ckpt["id2intent"]
        if isinstance(next(iter(id2intent_raw.keys())), str):
            self.id2intent = {int(k): v for k, v in id2intent_raw.items()}
        else:
            self.id2intent = id2intent_raw

        self.model_name = ckpt.get("model_name", "klue/bert-base")
        self.max_len = ckpt.get("max_len", DEFAULT_MAX_LEN)

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertRetrieverMultiTask(self.model_name, num_intents=len(self.intent2id))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Load and embed response bank (with audio metadata)
        self.bank_df, self.responses, self.resp_embs = self._load_and_embed_bank(bank_csv)

        print(f"[init] Retriever loaded: {ckpt_path}")
        print(f"[init] Response bank size: {len(self.responses)}")

    def _encode_batch(self, texts: List[str]):
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    def _load_and_embed_bank(self, bank_csv: str):
        df_bank = pd.read_csv(bank_csv)

        if "response_text" not in df_bank.columns:
            raise ValueError("Response bank CSV must contain a 'response_text' column.")

        # Keep full DF for metadata (audio_s3_uri, response_hash, locale, count)
        bank_df = df_bank.copy()
        responses = bank_df["response_text"].astype(str).tolist()

        all_embs = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(responses), batch_size):
                batch = responses[i : i + batch_size]
                ids, mask = self._encode_batch(batch)
                emb = self.model.encode_response(ids, mask)
                all_embs.append(emb)

        resp_embs = torch.cat(all_embs, dim=0)
        resp_embs = nn.functional.normalize(resp_embs, dim=-1)

        return bank_df, responses, resp_embs

    @torch.inference_mode()
    def classify_intent(self, text: str) -> Dict[str, Any]:
        ids, mask = self._encode_batch([text])
        q_emb = self.model.encode_query(ids, mask)
        logits = self.model.intent_head(q_emb)

        probs = torch.softmax(logits, dim=-1)[0]
        top_idx = int(torch.argmax(probs).item())
        top_intent = self.id2intent[top_idx]
        top_prob = float(probs[top_idx].item())

        return {
            "predicted_intent": top_intent,
            "intent_prob": top_prob,
            "probs": {self.id2intent[i]: float(probs[i].item()) for i in range(len(self.id2intent))},
        }

    @torch.inference_mode()
    def respond(self, text: str, top_k: int | None = None) -> Dict[str, Any]:
        text_norm = (text or "").strip()
        if not text_norm:
            return {
                "text": text,
                "agent_response": FALLBACK_RESPONSE,
                "used_fallback": True,
                "reason": "empty_text",
            }

        # 1) encode query and predict intent
        ids, mask = self._encode_batch([text_norm])
        q_emb = self.model.encode_query(ids, mask)
        q_emb = nn.functional.normalize(q_emb, dim=-1)

        logits = self.model.intent_head(q_emb)
        probs = torch.softmax(logits, dim=-1)[0]
        top_intent_idx = int(torch.argmax(probs).item())
        top_intent = self.id2intent[top_intent_idx]
        top_intent_prob = float(probs[top_intent_idx].item())

        # 2) retrieval over bank
        sims = torch.matmul(q_emb, self.resp_embs.t()).squeeze(0)
        k_env = TOP_K
        if top_k is not None:
            try:
                k = int(top_k)
            except ValueError:
                k = k_env
        else:
            k = k_env
    
        k = max(1, min(k, len(self.responses)))
        scores, idxs = torch.topk(sims, k=k)

        scores_list = scores.tolist()
        idxs_list = idxs.tolist()

        top_score = scores_list[0]
        top_idx = idxs_list[0]
        top_resp = self.responses[top_idx]

        used_fallback = False
        chosen_text = top_resp
        chosen_audio_uri = None
        chosen_response_hash = None
        chosen_locale = None

        if top_score < SIM_THRESHOLD:
            used_fallback = True
            chosen_text = FALLBACK_RESPONSE
            chosen_audio_uri = FALLBACK_RESPONSE_URL
        else:
            row = self.bank_df.iloc[top_idx]
            if "audio_http_url" in row:
                chosen_audio_uri = row["audio_http_url"] if pd.notna(row["audio_http_url"]) else None
            if "response_hash" in row:
                chosen_response_hash = row["response_hash"] if pd.notna(row["response_hash"]) else None
            if "locale" in row:
                chosen_locale = row["locale"] if pd.notna(row["locale"]) else None

        top_candidates = []
        for rank, (s, j) in enumerate(zip(scores_list, idxs_list), start=1):
            row = self.bank_df.iloc[j]
            cand = {
                "rank": rank,
                "score": float(s),
                "response_text": self.responses[j],
            }
            if "audio_http_url" in row:
                cand["audio_http_url"] = row["audio_http_url"] if pd.notna(row["audio_http_url"]) else None
            if "response_hash" in row:
                cand["response_hash"] = row["response_hash"] if pd.notna(row["response_hash"]) else None
            if "locale" in row:
                cand["locale"] = row["locale"] if pd.notna(row["locale"]) else None
            top_candidates.append(cand)

        return {
            "text": text_norm,
            "predicted_intent": top_intent,
            "intent_prob": top_intent_prob,
            "top_k_candidates": top_candidates,
            "chosen_score": float(top_score),
            "used_fallback": used_fallback,
            # what your pipeline will actually speak:
            "agent_response": chosen_text,
            # audio metadata for reuse:
            "chosen_audio_s3_uri": chosen_audio_uri,
            "chosen_response_hash": chosen_response_hash,
            "chosen_locale": chosen_locale,
        }


# -------------------------------------------------------------------
# FastAPI
# -------------------------------------------------------------------
app = FastAPI(
    title="Retriever-based Chatbot with Audio",
    version="1.0.0",
)

class PredictIn(BaseModel):
    text: str
    top_k: int | None = None

class PredictBatchIn(BaseModel):
    texts: List[str]

RUNTIME: RetrievalRuntime = None


def _ensure_runtime():
    global RUNTIME
    if RUNTIME is None:
        RUNTIME = RetrievalRuntime(RETRIEVER_CKPT_PATH, RESPONSE_BANK_CSV)
        print("[chatbot] RUNTIME created")


@app.on_event("startup")
def _startup():
    _ensure_runtime()
    print("[startup] Retriever chatbot ready")


@app.get("/health")
def health():
    _ensure_runtime()
    return {
        "status": "ok",
        "bank_size": len(RUNTIME.responses),
        "has_audio_column": "audio_s3_uri" in RUNTIME.bank_df.columns,
    }


@app.post("/respond")
def respond(body: PredictIn):
    _ensure_runtime()
    return RUNTIME.respond(body.text, top_k=body.top_k)


@app.post("/predict")
def predict(body: PredictIn):
    _ensure_runtime()
    return RUNTIME.classify_intent(body.text)


@app.post("/batch")
def batch(body: PredictBatchIn):
    _ensure_runtime()
    return [RUNTIME.respond(t) for t in body.texts]


@app.get("/version")
def version():
    _ensure_runtime()
    return {
        "ckpt_path": RETRIEVER_CKPT_PATH,
        "response_bank_csv": RESPONSE_BANK_CSV,
        "num_responses": len(RUNTIME.responses),
        "sim_threshold": SIM_THRESHOLD,
        "top_k": TOP_K,
        "max_len": RUNTIME.max_len,
        "model_name": RUNTIME.model_name,
        "intents": list(RUNTIME.intent2id.keys()),
    }


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
