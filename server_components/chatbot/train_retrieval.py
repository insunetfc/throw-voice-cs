import os
import math
import random
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup

# --------------- config ---------------
DATA_CSV = "/home/work/VALL-E/chatbot/data/multitask_pairs.csv"   # input_text, response_text, intent, source
MODEL_NAME = "klue/bert-base"

MAX_LEN = 48
BATCH_SIZE = 32
EPOCHS = 3
LR = 3e-5
WARMUP_RATIO = 0.1
SEED = 42

LAMBDA_RETR = 1.0
LAMBDA_INTENT = 0.5  # can tune later

SAVE_DIR = "/home/work/VALL-E/chatbot/models/models_multitask"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------- utils ---------------
def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer, intent2id: Dict[str, int]):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.intent2id = intent2id

    def __len__(self):
        return len(self.df)

    def encode_text(self, text: str):
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        q_text = str(row["input_text"])
        r_text = str(row["response_text"])
        intent = row["intent"]

        q_ids, q_mask = self.encode_text(q_text)
        r_ids, r_mask = self.encode_text(r_text)

        if pd.isna(intent):
            intent_id = -1  # mark as "no intent label"
        else:
            intent_id = self.intent2id[str(intent)]

        return {
            "q_ids": q_ids,
            "q_mask": q_mask,
            "r_ids": r_ids,
            "r_mask": r_mask,
            "intent_id": intent_id,
        }


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


def info_nce(q_emb, r_emb, temperature: float = 0.07):
    q_emb = nn.functional.normalize(q_emb, dim=-1)
    r_emb = nn.functional.normalize(r_emb, dim=-1)
    logits = torch.matmul(q_emb, r_emb.t()) / temperature
    labels = torch.arange(q_emb.size(0), device=q_emb.device)
    return nn.CrossEntropyLoss()(logits, labels)


def collate_fn(batch):
    q_ids = torch.stack([b["q_ids"] for b in batch])
    q_mask = torch.stack([b["q_mask"] for b in batch])
    r_ids = torch.stack([b["r_ids"] for b in batch])
    r_mask = torch.stack([b["r_mask"] for b in batch])
    intent_ids = torch.tensor([b["intent_id"] for b in batch], dtype=torch.long)
    return q_ids, q_mask, r_ids, r_mask, intent_ids


def main():
    set_seed(SEED)

    df = pd.read_csv(DATA_CSV)

    # basic cleaning
    df = df.dropna(subset=["input_text", "response_text"])
    df["input_text"] = df["input_text"].astype(str).str.strip()
    df["response_text"] = df["response_text"].astype(str).str.strip()
    df = df[(df["input_text"] != "") & (df["response_text"] != "")]

    # intents only from rows where intent is present
    intents = sorted(df["intent"].dropna().astype(str).unique().tolist())
    intent2id = {it: i for i, it in enumerate(intents)}
    id2intent = {i: it for it, i in intent2id.items()}
    print(f"Intents ({len(intents)}):", intents)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    dataset = PairDataset(df, tokenizer, intent2id)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = BertRetrieverMultiTask(MODEL_NAME, num_intents=len(intents)).to(device)

    total_steps = len(loader) * EPOCHS
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps,
    )

    ce_loss_fn = nn.CrossEntropyLoss()

    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for batch in loader:
            q_ids, q_mask, r_ids, r_mask, intent_ids = [
                x.to(device) for x in batch
            ]

            optimizer.zero_grad()

            q_emb, r_emb, intent_logits = model(q_ids, q_mask, r_ids, r_mask)

            # retrieval loss
            loss_retr = info_nce(q_emb, r_emb)

            # intent loss only where label exists (intent_id >= 0)
            mask = intent_ids >= 0
            if mask.any():
                loss_int = ce_loss_fn(intent_logits[mask], intent_ids[mask])
            else:
                loss_int = torch.tensor(0.0, device=device)

            loss = LAMBDA_RETR * loss_retr + LAMBDA_INTENT * loss_int
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                print(
                    f"step {global_step} | "
                    f"loss={loss.item():.4f} "
                    f"(retr={loss_retr.item():.4f}, intent={loss_int.item():.4f})"
                )

        avg_loss = running_loss / max(1, len(loader))
        print(f"Epoch {epoch+1}/{EPOCHS} - avg loss: {avg_loss:.4f}")

    ckpt = {
        "state_dict": model.state_dict(),
        "intent2id": intent2id,
        "id2intent": id2intent,
        "model_name": MODEL_NAME,
        "max_len": MAX_LEN,
    }
    save_path = os.path.join(SAVE_DIR, "retriever_multitask_pairs.pt")
    torch.save(ckpt, save_path)
    print("✅ saved checkpoint to", save_path)


if __name__ == "__main__":
    main()
