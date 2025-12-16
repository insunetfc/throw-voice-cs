import pandas as pd
import random

class SyntheticDataGenerator:
    """
    Generate synthetic training data based on promotional call scripts.
    Creates realistic dealer responses to different parts of the pitch.
    """
    
    def __init__(self):
        # Script key points that trigger specific dealer responses
        self.script_triggers = {
            'fee_mention': ['7%', '보험료의 7%', '수수료', '익일 지급'],
            'company_mention': ['차집사', '인슈넷', 'FC 제휴사', '다이렉트'],
            'service_mention': ['AI 1668-4007', '긴급출동', '보험사 직접 연결', '비교'],
            'benefit_mention': ['CU 5천원', 'CU편의점', '상품권', '체결율 95%']
        }
        
        # Synthetic response templates for each intent
        self.synthetic_templates = {
            'fee_question': [
                "몇 프로 준다고요?",
                "수수료가 정확히 얼마죠?",
                "{x}% 주는 거예요?",
                "언제 입금돼요?",
                "정산은 언제 되나요?",
                "다른 데보다 많이 주나요?",
                "조건이 어떻게 되죠?",
                "그럼 {x}%가 맞는 거죠?",
                "다이렉트도 똑같이 {x}%인가요?",
                "세금은 어떻게 되나요?",
                "지급 조건 좀 더 자세히 알려주세요",
                "익일이면 다음날이에요?",
                "계좌로 바로 들어오는 건가요?",
                "최소 금액 같은 거 있어요?",
                "월말 정산인가요?",
            ],
            'about_company': [
                "차집사가 어디예요?",
                "어느 회사죠?",
                "정식 업체 맞나요?",
                "보험사랑 무슨 관계죠?",
                "인슈넷이 뭐예요?",
                "제휴사라는 게 뭔가요?",
                "본사는 어디 있어요?",
                "등록된 회사예요?",
                "처음 듣는데요?",
                "어디 소속이세요?",
                "중개 업체인가요?",
                "보험사 직영이에요?",
                "법인 사업자예요?",
                "고객센터 번호 있어요?",
                "신뢰할 수 있는 곳인가요?",
            ],
            'more_questions': [
                "고객이 질문하면 어떻게 해요?",
                "보험 비교는 어떻게 하는 거예요?",
                "긴급출동이 뭐예요?",
                "AI 번호가 뭔가요?",
                "고객한테 뭐라고 설명해야 돼요?",
                "절차가 어떻게 되나요?",
                "상담 연결은 얼마나 걸려요?",
                "고객 정보는 뭐가 필요해요?",
                "견적은 몇 분 만에 나와요?",
                "모바일로도 되나요?",
                "계약서는 어떻게 받아요?",
                "사고 났을 때도 연락 가능해요?",
                "보장 내용 확인 가능한가요?",
                "여러 보험사 다 되나요?",
                "갱신 고객도 가능해요?",
            ],
            'positive': [
                "오 괜찮네요",
                "좋은 조건이네요",
                "한번 해볼게요",
                "명함 주세요",
                "문자로 보내주세요",
                "자료 좀 주실래요?",
                "그럼 진행해볼게요",
                "연락처 남겨주세요",
                "나중에 연락드릴게요",
                "생각보다 괜찮은데요",
                "고객한테 소개해볼게요",
                "링크 좀 주세요",
                "카톡으로 주세요",
                "일단 받아놓을게요",
                "저장해둘게요",
            ],
            'rejection': [
                "지금은 안 해요",
                "관심 없어요",
                "바빠서요",
                "나중에요",
                "됐어요",
                "안 할 거예요",
                "필요 없어요",
                "전화 끊을게요",
                "시간 없어요",
                "귀찮아요",
                "다음에 해요",
                "별로네요",
                "안 맞아요",
                "그만하세요",
                "고객들이 싫어해요",
            ],
            'other': [
                "저 다른 데 하고 있어요",
                "이미 거래처 있어요",
                "고정 업체 있어서요",
                "맡기는 곳이 있어요",
                "다른 데만 써요",
                "계속 하던 데가 있어서요",
                "파트너사 있어요",
                "제휴사 정해져 있어요",
                "기존 담당자 있어요",
                "믿는 곳이 있어서요",
                "바꿀 생각 없어요",
                "오래 거래하는 곳 있어요",
                "만족하는 업체 있어요",
                "한 군데만 써요",
                "정해진 루트가 있어요",
            ],
            'fallback': [
                "자료 좀 주세요",
                "어떻게 시작하나요?",
                "고객한테 뭐라고 해요?",
                "명함 있어요?",
                "링크 주세요",
                "문자로 보내주세요",
                "카톡 돼요?",
                "설명 자료 있어요?",
                "템플릿 같은 거요?",
                "상담원 번호 주세요",
                "담당자 연결해주세요",
                "이미지로 주실 수 있어요?",
                "PDF 있나요?",
                "공유해도 돼요?",
                "출력 가능해요?",
            ],
            'greeting': [
                "여보세요",
                "네 말씀하세요",
                "어디세요?",
                "누구시죠?",
                "안녕하세요",
                "네네",
                "예",
                "전화 받았어요",
                "말씀하세요",
                "어디 전화예요?",
            ]
        }
    
    def generate_variations(self, template, count=3):
        """Generate variations of a template"""
        variations = []
        
        if '{x}' in template:
            # Replace with common percentages
            percentages = ['5', '7', '10', '3']
            for pct in random.sample(percentages, min(count, len(percentages))):
                variations.append(template.replace('{x}', pct))
        else:
            # Just return the template multiple times with slight variations
            variations = [template] * min(count, 1)
        
        return variations
    
    def generate_synthetic_data(self, samples_per_class=30):
        """Generate synthetic training data"""
        synthetic_data = []
        
        for intent, templates in self.synthetic_templates.items():
            # Generate samples for this intent
            samples_generated = 0
            
            while samples_generated < samples_per_class:
                for template in templates:
                    if samples_generated >= samples_per_class:
                        break
                    
                    # Generate variations
                    variations = self.generate_variations(template, count=2)
                    
                    for var in variations:
                        if samples_generated >= samples_per_class:
                            break
                        
                        synthetic_data.append({
                            'question': var,
                            'label': intent
                        })
                        samples_generated += 1
        
        return pd.DataFrame(synthetic_data)
    
    def merge_with_existing(self, existing_csv, output_csv, samples_per_class=20):
        """Merge synthetic data with existing dataset"""
        
        # Load existing data
        existing_df = pd.read_csv(existing_csv)
        existing_df = existing_df.dropna(subset=['label'])
        
        print(f"📊 Existing dataset: {len(existing_df)} samples")
        print(existing_df['label'].value_counts())
        
        # Generate synthetic data
        print(f"\n🔄 Generating {samples_per_class} synthetic samples per class...")
        synthetic_df = self.generate_synthetic_data(samples_per_class=samples_per_class)
        
        print(f"\n📊 Synthetic dataset: {len(synthetic_df)} samples")
        print(synthetic_df['label'].value_counts())
        
        # Merge
        combined_df = pd.concat([existing_df, synthetic_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['question'])
        
        # Save
        combined_df.to_csv(output_csv, index=False)
        
        print(f"\n{'='*60}")
        print(f"✅ Combined dataset saved: {output_csv}")
        print(f"Total samples: {len(combined_df)}")
        print(f"Increase: +{len(combined_df) - len(existing_df)} samples")
        print(f"\n📊 Final distribution:")
        print(combined_df['label'].value_counts())
        print(f"{'='*60}")
        
        return combined_df


def create_conversation_flow_guide():
    """Create a guide for using the model in conversation flow"""
    
    flow_guide = """
    ╔════════════════════════════════════════════════════════════╗
    ║          차집사 프로모션 콜 - 대화 흐름 가이드                    ║
    ╚════════════════════════════════════════════════════════════╝
    
    1️⃣ GREETING (인사)
       딜러 응답: "여보세요", "안녕하세요"
       → 다음 액션: 스크립트 시작 (회사 소개 + 서비스 설명)
    
    2️⃣ ABOUT_COMPANY (회사 정보)
       딜러 응답: "차집사가 뭐예요?", "어디 회사예요?"
       → 다음 액션: 
          - 인슈넷 FC 제휴사 설명
          - 40년 전통, 보험사 연도대상자 출신 팀 강조
          - 정식 등록 업체임을 안내
    
    3️⃣ FEE_QUESTION (수수료 문의)
       딜러 응답: "몇 % 주는데요?", "수수료가 얼마죠?"
       → 다음 액션:
          - "7% 익일 지급" 명확히 전달
          - OFF/TM/CM 모두 동일 조건 강조
          - 체결율 95% 이상 언급
    
    4️⃣ MORE_QUESTIONS (서비스 문의)
       딜러 응답: "절차가 어떻게 되나요?", "고객이 질문하면?"
       → 다음 액션:
          - AI ARS 1668-4007 긴급출동 서비스 설명
          - 10분 내 비교 견적 제공
          - 보험사 직접 연결 서비스 설명
          - 딜러에게 보험 문의 안 감 강조
    
    5️⃣ POSITIVE (긍정 반응)
       딜러 응답: "괜찮네요", "명함 주세요", "해볼게요"
       → 다음 액션:
          - 명함/자료 문자 발송
          - CU 5천원권 혜택 재안내
          - 담당 상담원 연락처 공유
          - "첫 견적 문의 부탁드립니다" 마무리
    
    6️⃣ REJECTION (거절)
       딜러 응답: "안 해요", "관심 없어요", "바빠요"
       → 다음 액션:
          - 정중하게 수용
          - "명함만 남겨드릴게요" (부드럽게)
          - "나중에 필요하시면 연락주세요" 마무리
          - 더 이상 푸시하지 않기
    
    7️⃣ OTHER (기존 거래처)
       딜러 응답: "다른 데 하고 있어요", "거래처 있어요"
       → 다음 액션:
          - "비교만 해보세요" 제안
          - 7% + 체결율 95% 조건 차별화
          - "기존 거래처와 비교 후 결정하셔도 됩니다"
          - 명함 남기고 부드럽게 마무리
    
    8️⃣ FALLBACK (자료 요청)
       딜러 응답: "자료 주세요", "링크 주세요", "카톡으로?"
       → 다음 액션:
          - 즉시 명함/자료 문자 발송
          - 카카오톡 채널 안내 (있는 경우)
          - 상담원 직접 연결 제안
          - "필요하시면 바로 연락주세요" 안내
    
    ═══════════════════════════════════════════════════════════
    💡 Pro Tips:
    
    • 연속된 rejection → 정중하게 통화 종료
    • positive 후 fee_question → 적극 응대 (높은 관심도)
    • greeting → about_company 순서는 자연스러운 흐름
    • other 응답에 너무 푸시하지 말 것 (역효과)
    • fallback은 긍정 신호 - 적극 자료 제공
    
    ═══════════════════════════════════════════════════════════
    """
    
    print(flow_guide)
    
    # Save to file
    with open('./model/conversation_flow_guide.txt', 'w', encoding='utf-8') as f:
        f.write(flow_guide)
    
    print("✅ Conversation flow guide saved to: ./model/conversation_flow_guide.txt")


def main():
    """Main function to generate and merge synthetic data"""
    
    print("="*60)
    print("🤖 Synthetic Training Data Generator")
    print("="*60)
    
    generator = SyntheticDataGenerator()
    
    # Generate and merge with existing data
    combined_df = generator.merge_with_existing(
        existing_csv='./data/intent_dataset.csv',
        output_csv='./data/intent_dataset_enhanced.csv',
        samples_per_class=25  # Add 25 synthetic samples per class
    )
    
    # Create conversation flow guide
    print("\n" + "="*60)
    print("📖 Creating Conversation Flow Guide")
    print("="*60)
    create_conversation_flow_guide()
    
    print("\n" + "="*60)
    print("✅ All Done!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review: ./data/intent_dataset_enhanced.csv")
    print("2. Train: python train.py (use enhanced dataset)")
    print("3. Review: ./model/conversation_flow_guide.txt")
    print("="*60)


if __name__ == "__main__":
    main()