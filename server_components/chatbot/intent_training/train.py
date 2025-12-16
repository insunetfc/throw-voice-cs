import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd

# ============================================================
# 1. 환경 설정
# ============================================================
MODEL_NAME = "klue/bert-base"
SAVE_DIR = "model"
SAVE_PATH = os.path.join(SAVE_DIR, "intent_classification_model.pth")
NUM_LABELS = 8
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# 2. 데이터 로드
# ============================================================
# CSV 파일 예시
# columns: ['text', 'label']
# label 은 아래 8가지 중 하나여야 함
# 거절/보류, 정보요청, 신뢰확인, 절차문의, 서비스문의, 혜택문의, 이의제기, 긍정응답

df = pd.read_csv("./data/intent_dataset.csv")

label2id = {
    "거절/보류": 0,
    "정보요청": 1,
    "신뢰확인": 2,
    "절차문의": 3,
    "서비스문의": 4,
    "혜택문의": 5,
    "이의제기": 6,
    "긍정응답": 7
}
id2label = {v: k for k, v in label2id.items()}
df["label_id"] = df["label"].map(label2id)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["question"].tolist(), df["label_id"].tolist(), test_size=0.2, random_state=42
)

# ============================================================
# 3. Dataset 정의
# ============================================================
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

train_dataset = IntentDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = IntentDataset(val_texts, val_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 4. 모델 정의
# ============================================================
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
model = model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)
criterion = torch.nn.CrossEntropyLoss()

# ============================================================
# 5. 학습 루프
# ============================================================
def train_epoch(model, data_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def eval_epoch(model, data_loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    acc = correct / total
    return avg_loss, acc


# ============================================================
# 6. 학습 실행
# ============================================================
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    train_loss = train_epoch(model, train_loader, optimizer, scheduler)
    val_loss, val_acc = eval_epoch(model, val_loader)

    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# ============================================================
# 7. 모델 저장
# ============================================================
torch.save(model.state_dict(), SAVE_PATH.replace('.pth', '_trained.pth'))
print(f"\n✅ 모델 저장 완료: {SAVE_PATH}")

