"""
Improved app.py with context-aware response selection.

Key improvements:
1. Analyzes WHAT specifically the dealer is asking
2. Selects the most relevant response from templates
3. Still deterministic (same input = same output)
4. Better answers to specific questions
"""

import os, csv, hashlib
from typing import List, Dict, Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification

# ------------------------------
# Config
# ------------------------------
MODEL_PATH       = os.environ.get("MODEL_PATH", "/home/work/VALL-E/chatbot/model/final_model.pth")
TEMPLATE_CSV     = os.environ.get("TEMPLATE_CSV", "/home/work/VALL-E/chatbot/data/response_templates_rich.csv")
CONF_THRESHOLD   = float(os.environ.get("CONF_THRESHOLD", "0.70"))
MAX_LEN_FALLBACK = int(os.environ.get("MAX_LEN", "64"))

FACTS = {
    "fee_percent"  : os.environ.get("FEE_PERCENT", "7%"),
    "payout_timing": os.environ.get("PAYOUT_TIMING", "익일 오후"),
    "ars_number"   : os.environ.get("ARS_NUMBER", "1668-4007"),
}

# ------------------------------
# Context-Aware Response Selector
# ------------------------------
class ContextAnalyzer:
    """Analyzes what specifically the dealer is asking about"""
    
    def __init__(self):
        self.patterns = {
            # Fee sub-contexts
            'fee_amount': ['몇', '얼마', '프로', '퍼센트', '%', '수수료', '지급률'],
            'fee_timing': ['언제', '시점', '입금', '지급시', '익일', '바로', '즉시'],
            'fee_method': ['어떻게', '방법', '계좌', '이체', '지급방법'],
            'fee_tax': ['세금', '원천징수', '부가세', '공제', '3.3'],
            'fee_scope': ['다이렉트', '오프', '삼성', '포함', 'tm', 'cm'],
            
            # Company sub-contexts
            'company_identity': ['어디', '무슨회사', '뭐하는', '회사명', '업체명'],
            'company_legitimacy': ['정식', '등록', '인증', '신뢰', '진짜', '법인'],
            'company_relationship': ['제휴', '관계', '소속', '보험사', '직영'],
            
            # Service sub-contexts
            'service_process': ['절차', '과정', '순서', '어떻게', '방법'],
            'service_features': ['ars', '긴급출동', '연결', '서비스', '기능'],
            'service_coverage': ['범위', '대상', '포함', '가능', '지역'],
            
            # Positive sub-contexts
            'positive_interested': ['괜찮', '좋', '만족', '충분'],
            'positive_request': ['명함', '자료', '문자', '링크', '보내', '주세요'],
            'positive_commit': ['해볼게', '진행', '신청', '시작'],
            
            # Rejection sub-contexts
            'rejection_hard': ['안해', '필요없', '관심없', '그만'],
            'rejection_soft': ['바빠', '시간없', '나중에', '다음에'],
            'rejection_busy': ['고객응대', '통화중', '회의', '급해'],
            
            # Other sub-contexts
            'other_satisfied': ['만족', '괜찮', '좋아', '편해'],
            'other_committed': ['정해', '고정', '계속', '오래', '계약'],
            'other_hesitant': ['생각', '고민', '비교', '알아볼'],
        }
    
    def analyze(self, text: str, intent: str) -> str:
        """Find the most relevant sub-context"""
        t = text.replace(' ', '').lower()
        
        best_match = None
        max_matches = 0
        
        for context, keywords in self.patterns.items():
            # Only check contexts relevant to this intent
            if not context.startswith(intent.split('_')[0]):
                continue
            
            matches = sum(1 for kw in keywords if kw in t)
            if matches > max_matches:
                max_matches = matches
                best_match = context
        
        return best_match


class SmartResponseSelector:
    """Selects contextually appropriate responses"""
    
    def __init__(self, csv_path: str):
        self.analyzer = ContextAnalyzer()
        self.templates = self._load_templates(csv_path)
        
        # Map sub-contexts to template contexts (from your CSV)
        self.context_map = {
            'fee_amount': ['요약 응답', '사실 전달'],
            'fee_timing': ['사실 전달', '요약 응답'],
            'fee_method': ['요약 응답'],
            'fee_tax': ['요약 응답'],
            'fee_scope': ['확인 및 제안'],
            
            'company_identity': ['회사 소개', '서비스 소개'],
            'company_legitimacy': ['회사 소개'],
            'company_relationship': ['회사 소개', '차별점 강조'],
            
            'service_process': ['절차 안내', '프로세스 안내'],
            'service_features': ['기능 안내', '차별점 강조'],
            'service_coverage': ['절차 안내'],
            
            'positive_interested': ['감사+진행'],
            'positive_request': ['진행 확정'],
            'positive_commit': ['연결/진행'],
            
            'rejection_hard': ['완곡한 종료'],
            'rejection_soft': ['재접촉 제안'],
            'rejection_busy': ['재접촉 제안'],
            
            'other_satisfied': ['부담 완화'],
            'other_committed': ['부드러운 설득'],
            'other_hesitant': ['행동 유도'],
        }
    
    def _load_templates(self, csv_path: str) -> Dict:
        """Load templates organized by intent and context"""
        templates = {}
        
        if not os.path.isfile(csv_path):
            return templates
        
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                intent = row['intent'].strip()
                context = row['context'].strip()
                text = row['response'].strip()
                
                if intent not in templates:
                    templates[intent] = {}
                if context not in templates[intent]:
                    templates[intent][context] = []
                
                templates[intent][context].append(text)
        
        return templates
    
    def select(self, dealer_text: str, intent: str) -> Dict:
        """Select best response based on context"""
        
        # Analyze what specifically they're asking
        sub_context = self.analyzer.analyze(dealer_text, intent)
        
        # Get available templates for this intent
        intent_templates = self.templates.get(intent, {})
        
        if not intent_templates:
            return {
                'response': "죄송합니다. 다시 한번 말씀해 주시겠어요?",
                'context': 'fallback',
                'sub_context': sub_context
            }
        
        # Find matching template context
        selected_response = None
        selected_context = None
        
        if sub_context:
            # Try to find template matching the sub-context
            preferred_contexts = self.context_map.get(sub_context, [])
            
            for pref_ctx in preferred_contexts:
                if pref_ctx in intent_templates:
                    # Deterministic selection: always pick first
                    selected_response = intent_templates[pref_ctx][0]
                    selected_context = pref_ctx
                    break
        
        # Fallback: use first available template
        if not selected_response:
            first_context = list(intent_templates.keys())[0]
            selected_response = intent_templates[first_context][0]
            selected_context = first_context
        
        return {
            'response': selected_response,
            'context': selected_context,
            'sub_context': sub_context,
            'intent': intent
        }


# ------------------------------
# Enhanced fee_question handler
# ------------------------------
def build_fee_response(text: str, facts: Dict, base_response: str) -> str:
    """
    Enhance base response with specific details based on what was asked.
    This ensures fee questions always get factual answers.
    """
    t = text.replace(' ', '').lower()
    
    # Always ensure the percentage is mentioned
    if facts['fee_percent'] not in base_response:
        base_response = f"소개 수수료는 {facts['fee_percent']}입니다. " + base_response
    
    # Add timing if asked
    if any(k in t for k in ['언제', '시점', '입금', '익일']):
        if '익일' not in base_response and '오후' not in base_response:
            base_response += f" 지급 시점은 {facts['payout_timing']}입니다."
    
    # Add method if asked
    if any(k in t for k in ['어떻게', '방법', '계좌']):
        if '계좌' not in base_response:
            base_response += " 등록된 계좌로 이체해 드립니다."
    
    # Add tax info if asked
    if any(k in t for k in ['세금', '원천', '공제']):
        if '세금' not in base_response and '원천' not in base_response:
            base_response += " 세무 처리는 계약 형태에 따라 안내드립니다."
    
    return base_response


# ------------------------------
# Model Runtime
# ------------------------------
class IntentRuntime:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(model_path, map_location=self.device)

        self.label2id = ckpt["label2id"]
        self.id2label = ckpt["id2label"]
        self.tokenizer_name = ckpt.get("tokenizer_name", "klue/bert-base")
        self.max_len = ckpt.get("max_len", MAX_LEN_FALLBACK)

        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
        self.model = BertForSequenceClassification.from_pretrained(
            self.tokenizer_name, num_labels=len(self.label2id)
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Initialize smart response selector
        self.response_selector = SmartResponseSelector(TEMPLATE_CSV)
        
        print(f"[init] Model loaded: {model_path}")
        print(f"[init] Templates: {len(self.response_selector.templates)} intents")

    @torch.inference_mode()
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict intent with confidence scores"""
        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs_t = torch.softmax(logits, dim=1)[0]
        top_idx = int(torch.argmax(probs_t).item())
        probs = probs_t.tolist()
        
        return {
            "intent": self.id2label[top_idx],
            "confidence": float(probs[top_idx]),
            "probs": {self.id2label[i]: float(probs[i]) for i in range(len(probs))},
        }

    def policy_respond(self, text: str) -> Dict[str, Any]:
        """
        Main response generation with context awareness.
        
        Flow:
        1. Classify intent
        2. Override if needed (keyword-based)
        3. Analyze context (what specifically they're asking)
        4. Select best response template
        5. Enhance with facts if needed
        """
        # Step 1: Classify intent
        pred = self.predict(text)
        intent = pred["intent"]
        conf = pred["confidence"]
        
        # Step 2: Keyword-based overrides (your existing logic)
        t = text.replace(" ", "").lower()
        if any(k in t for k in ["%", "퍼센트", "수수료", "몇프로"]):
            intent = "fee_question"
        
        # Step 3: Handle low confidence
        if conf < CONF_THRESHOLD and intent != "fee_question":
            intent = "fallback"
        
        # Step 4: Select contextually appropriate response
        selection = self.response_selector.select(text, intent)
        reply = selection['response']
        
        # Step 5: Enhance fee responses with facts
        if intent == "fee_question":
            reply = build_fee_response(text, FACTS, reply)
        
        return {
            "text": text,
            "intent": pred["intent"],
            "confidence": conf,
            "final_intent": intent,
            "sub_context": selection.get('sub_context'),
            "template_context": selection.get('context'),
            "agent_response": reply,
            "probs": pred["probs"]
        }


# ------------------------------
# FastAPI
# ------------------------------
app = FastAPI(title="Intent Classifier with Context-Aware Responses", version="2.0.0")

class PredictIn(BaseModel):
    text: str

class PredictBatchIn(BaseModel):
    texts: List[str]

# NEW: eager init helper
def _ensure_runtime():
    """
    Make sure RUNTIME exists even when this app is *mounted*
    under another FastAPI app (mount doesn't run startup).
    """
    global RUNTIME
    if "RUNTIME" not in globals() or RUNTIME is None:
        RUNTIME = IntentRuntime(MODEL_PATH)
        print("[chatbot] RUNTIME created eagerly in import")

# keep the original startup for when you run this app alone
@app.on_event("startup")
def _startup():
    _ensure_runtime()
    print("[startup] Ready!")

@app.get("/health")
def health():
    _ensure_runtime()
    return {"status": "ok", "model": "loaded"}

@app.post("/predict")
def predict(body: PredictIn):
    _ensure_runtime()
    return RUNTIME.predict(body.text)

@app.post("/respond")
def respond(body: PredictIn):
    _ensure_runtime()
    return RUNTIME.policy_respond(body.text)

@app.post("/batch")
def batch(body: PredictBatchIn):
    _ensure_runtime()
    return [RUNTIME.policy_respond(t) for t in body.texts]

@app.get("/version")
def version():
    _ensure_runtime()
    return {
        "model_path": MODEL_PATH,
        "tokenizer": RUNTIME.tokenizer_name,
        "max_len": RUNTIME.max_len,
        "labels": list(RUNTIME.label2id.keys()),
        "facts": FACTS,
        "conf_threshold": CONF_THRESHOLD,
        "response_strategy": "context-aware"
    }

@app.get("/debug/contexts")
def debug_contexts():
    _ensure_runtime()
    return {
        "intents": list(RUNTIME.response_selector.templates.keys()),
        "context_patterns": list(RUNTIME.response_selector.analyzer.patterns.keys()),
        "context_mappings": RUNTIME.response_selector.context_map
    }

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)