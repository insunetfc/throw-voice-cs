import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

class IntentClassifier:
    """Intent classification inference with response mapping"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract metadata
        self.label2id = checkpoint['label2id']
        self.id2label = checkpoint['id2label']
        self.tokenizer_name = checkpoint['tokenizer_name']
        self.max_len = checkpoint['max_len']
        
        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
        self.model = BertForSequenceClassification.from_pretrained(
            self.tokenizer_name, 
            num_labels=len(self.label2id)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"Device: {self.device}")
        print(f"Classes: {list(self.id2label.values())}")
        
        # Response templates based on promotional scripts
        self.response_templates = self._load_response_templates()
    
    def _load_response_templates(self):
        """Load response templates for each intent"""
        return {
            "fee_question": [
                "보험료의 7%를 소개료로 익일 지급해드립니다.",
                "수수료는 7프로이며, 익일 오후에 바로 지급됩니다.",
                "OFF, TM, CM 가입 모두 7% 수수료를 익일에 바로 지급해드리고 있습니다."
            ],
            "about_company": [
                "저희는 40년 전통의 인슈넷 FC 자회사 차집사 다이렉트입니다.",
                "차집사는 인슈넷 FC 제휴사로 정식 등록된 보험 비교 서비스입니다.",
                "저희는 보험사 연도대상자 출신들로 구성된 전문 팀입니다."
            ],
            "more_questions": [
                "다이렉트 보험료를 10분 내 비교해드리고, AI ARS 1668-4007로 긴급출동 서비스도 제공합니다.",
                "가입 후에는 딜러님께 보험 관련 연락이 가지 않도록 AI 번호를 운영하고 있습니다.",
                "보험사 직접 연결 서비스와 사고 시 긴급출동 서비스를 제공하고 있습니다."
            ],
            "positive": [
                "감사합니다! 명함 문자로 남겨드릴게요. 앞으로 제가 담당이니 연락 주세요.",
                "감사합니다. 견적 문의 있으실 때 연락 주시면 빠르게 진행 도와드리겠습니다.",
                "좋습니다! CU 모바일 상품권도 드리니 첫 문의 부탁드립니다."
            ],
            "rejection": [
                "알겠습니다. 혹시 나중에 필요하시면 언제든 연락 주세요.",
                "네, 이해합니다. 명함만 남겨드릴게요. 나중에 필요하시면 편하게 연락 주세요.",
                "괜찮습니다. 좋은 하루 되세요. 감사합니다."
            ],
            "other": [
                "현재 거래처가 있으시군요. 저희는 수수료 7%와 체결율 95% 이상의 조건을 제공하고 있습니다.",
                "이해합니다. 혹시 비교해보시고 싶으시면 언제든 연락 주세요.",
                "네, 저희는 조건이 더 좋아 많은 분들이 함께 하고 계십니다. 한번 비교해보시는 것도 좋을 것 같습니다."
            ],
            "fallback": [
                "명함과 상세 자료를 문자로 보내드리겠습니다.",
                "카카오톡으로도 자료 전달 가능합니다. 상담원 연결도 도와드릴게요.",
                "네, 관련 자료 전부 보내드리고 담당자 직접 연결해드리겠습니다."
            ],
            "greeting": [
                "안녕하세요! 차집사 다이렉트입니다. 잠시 통화 가능하실까요?",
                "네, 안녕하세요. 자동차 보험 비교 가입 도와드리는 차집사입니다.",
                "안녕하세요~ 오늘 문자 보내드렸는데요, 잠깐 안내 말씀 드려도 될까요?"
            ]
        }
    
    def predict(self, text, return_probs=False):
        """Predict intent for a single text"""
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
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_id].item()
        
        intent = self.id2label[pred_id]
        
        if return_probs:
            all_probs = {self.id2label[i]: probs[0][i].item() 
                        for i in range(len(self.id2label))}
            return intent, confidence, all_probs
        
        return intent, confidence
    
    def predict_batch(self, texts):
        """Predict intents for multiple texts"""
        results = []
        for text in texts:
            intent, confidence = self.predict(text)
            results.append({
                'text': text,
                'intent': intent,
                'confidence': confidence
            })
        return pd.DataFrame(results)
    
    def get_response(self, text, intent=None):
        """Get appropriate response for dealer's text"""
        if intent is None:
            intent, confidence = self.predict(text)
        else:
            _, confidence = self.predict(text)
        
        # Get response template
        templates = self.response_templates.get(intent, ["죄송합니다. 다시 한번 말씀해 주시겠어요?"])
        response = templates[0]  # Can randomize or use logic to select
        
        return {
            'dealer_text': text,
            'detected_intent': intent,
            'confidence': confidence,
            'agent_response': response
        }
    
    def interactive_test(self):
        """Interactive testing mode"""
        print("\n" + "="*60)
        print("🤖 Interactive Intent Classification Test")
        print("Type 'quit' to exit")
        print("="*60 + "\n")
        
        while True:
            user_input = input("딜러 응답: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            intent, confidence, all_probs = self.predict(user_input, return_probs=True)
            response_data = self.get_response(user_input, intent)
            
            print(f"\n{'─'*60}")
            print(f"🎯 Intent: {intent} (Confidence: {confidence:.2%})")
            print(f"\n📊 All Probabilities:")
            for label, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 30)
                print(f"  {label:20s} {prob:6.2%} {bar}")
            print(f"\n💬 Agent Response:")
            print(f"  {response_data['agent_response']}")
            print(f"{'─'*60}\n")


def test_on_samples():
    """Test classifier on sample dealer responses"""
    classifier = IntentClassifier('./model/best_model.pth')
    
    test_samples = [
        "몇 퍼센트 주시는 거예요?",
        "차집사가 어디에요?",
        "고객이 거부하면 어떻게 되죠?",
        "오 괜찮은데요",
        "안 해요 지금은",
        "저 다른 데 하고 있어서요",
        "명함 좀 보내주세요",
        "안녕하세요",
    ]
    
    print("\n" + "="*60)
    print("📋 Testing on Sample Dealer Responses")
    print("="*60 + "\n")
    
    for sample in test_samples:
        result = classifier.get_response(sample)
        print(f"Dealer: {result['dealer_text']}")
        print(f"Intent: {result['detected_intent']} ({result['confidence']:.2%})")
        print(f"Agent: {result['agent_response']}")
        print(f"{'─'*60}\n")


if __name__ == "__main__":
    # Run tests
    test_on_samples()
    
    # Optional: Start interactive mode
    # classifier = IntentClassifier('./model/best_model.pth')
    # classifier.interactive_test()
