import os
import torch
import torch.nn as nn
import pandas as pd
from transformers import BertTokenizer, BertModel
from typing import Dict, List, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT_PATH = "/home/work/VALL-E/chatbot/models/models_multitask/retriever_multitask_pairs.pt"
BANK_CSV  = "/home/work/VALL-E/chatbot/data/response_bank_large.csv"
TOP_K = 5

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

    def forward(
        self,
        q_ids,
        q_mask,
        r_ids,
        r_mask,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_emb = self.encode_query(q_ids, q_mask)
        r_emb = self.encode_response(r_ids, r_mask)
        intent_logits = self.intent_head(q_emb)
        return q_emb, r_emb, intent_logits


def load_model_and_tokenizer():
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model_name = ckpt.get("model_name", "klue/bert-base")
    max_len = ckpt.get("max_len", 48)

    tokenizer = BertTokenizer.from_pretrained(model_name)

    intent2id = ckpt["intent2id"]
    id2intent = {int(k): v for k, v in ckpt["id2intent"].items()} if isinstance(next(iter(ckpt["id2intent"].keys())), str) else ckpt["id2intent"]

    model = BertRetrieverMultiTask(model_name, num_intents=len(intent2id))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    return model, tokenizer, intent2id, id2intent, max_len


def encode_texts(tokenizer, texts, max_len):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def prepare_response_bank(model, tokenizer, max_len):
    df_bank = pd.read_csv(BANK_CSV)
    if "response_text" not in df_bank.columns:
        raise ValueError("BANK_CSV must have 'response_text' column")

    responses = df_bank["response_text"].astype(str).tolist()
    print(f"Loaded {len(responses)} response candidates from bank")

    batch_size = 64
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(responses), batch_size):
            batch = responses[i : i + batch_size]
            ids, mask = encode_texts(tokenizer, batch, max_len)
            ids = ids.to(device)
            mask = mask.to(device)

            out = model.bert_r(input_ids=ids, attention_mask=mask)
            emb = mean_pool(out.last_hidden_state, mask)
            all_embs.append(emb)

    resp_embs = torch.cat(all_embs, dim=0)  # [N, D]
    resp_embs = nn.functional.normalize(resp_embs, dim=-1)

    return responses, resp_embs


def main():
    model, tokenizer, intent2id, id2intent, max_len = load_model_and_tokenizer()
    print("✅ Model & tokenizer loaded")

    responses, resp_embs = prepare_response_bank(model, tokenizer, max_len)
    print("✅ Response bank embeddings ready")

    while True:
        try:
            text = input("\nUser> ").strip()
        except EOFError:
            break

        if not text:
            continue
        if text.lower() in ["quit", "exit"]:
            break

        # encode query
        enc = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        q_ids = enc["input_ids"].to(device)
        q_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            q_emb, _, intent_logits = model(q_ids, q_mask, q_ids, q_mask)
            q_emb = nn.functional.normalize(q_emb, dim=-1)  # [1, D]

            # intent prediction
            intent_probs = torch.softmax(intent_logits, dim=-1).squeeze(0)
            top_intent_id = int(torch.argmax(intent_probs).item())
            top_intent = id2intent[top_intent_id]
            top_intent_prob = float(intent_probs[top_intent_id].item())

            # retrieval over response bank
            sims = torch.matmul(q_emb, resp_embs.t()).squeeze(0)  # [N]
            scores, idxs = torch.topk(sims, k=min(TOP_K, len(responses)))

        print(f"\nPredicted intent: {top_intent} ({top_intent_prob:.3f})")
        print("Top responses:")
        for rank, (score, idx) in enumerate(zip(scores.tolist(), idxs.tolist()), start=1):
            print(f"[{rank}] score={score:.3f}  {responses[idx]}")

        # final chosen response = rank 1
        print(f"\nChosen response:\n{responses[idxs[0]]}")


if __name__ == "__main__":
    main()
