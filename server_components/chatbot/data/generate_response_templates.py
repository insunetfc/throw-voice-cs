import csv

# Define all response templates with metadata
TEMPLATES = [
    # ============================================================
    # GREETING - Opening statements
    # ============================================================
    {
        "intent": "greeting",
        "response": "안녕하세요, 자동차 보험 비교·가입 도와드리는 차집사 다이렉트입니다. 잠시 통화 가능하실까요?",
        "context": "formal_opening",
        "notes": "Standard formal opening"
    },
    {
        "intent": "greeting",
        "response": "안녕하세요, 차집사 다이렉트입니다. 보험료 비교 안내 간단히 드려도 될까요?",
        "context": "casual_opening",
        "notes": "Shorter, more casual"
    },
    {
        "intent": "greeting",
        "response": "네, 안녕하세요. 차집사 다이렉트 상담원입니다.",
        "context": "brief_opening",
        "notes": "Very brief response"
    },
    
    # ============================================================
    # ABOUT_COMPANY - Company information
    # ============================================================
    {
        "intent": "about_company",
        "response": "저희 차집사는 인슈넷 FC 제휴사로 여러 보험사와 연계해 최적의 자동차보험료를 찾아드립니다.",
        "context": "company_intro",
        "notes": "Basic company explanation"
    },
    {
        "intent": "about_company",
        "response": "차집사 다이렉트는 40년 전통의 인슈넷 FC 자회사로, 정식 등록된 보험 비교 서비스입니다.",
        "context": "legitimacy_emphasis",
        "notes": "Emphasize trust/legitimacy"
    },
    {
        "intent": "about_company",
        "response": "저희는 비교견적부터 가입까지 한 번에 도와드리며, 가입 후에는 AI ARS 1668-4007로 고객님을 보험사에 직접 연결해드립니다.",
        "context": "service_detail",
        "notes": "Service differentiation"
    },
    
    # ============================================================
    # FEE_QUESTION - Commission questions
    # ============================================================
    {
        "intent": "fee_question",
        "response": "소개 수수료는 7%이며 지급 시점은 익일 오후입니다. 원하시면 지금 비교 견적 접수해 드릴까요?",
        "context": "fee_summary",
        "notes": "Complete answer with CTA"
    },
    {
        "intent": "fee_question",
        "response": "네, 수수료는 건당 7%입니다. 정산은 보통 익일 오후에 입금됩니다.",
        "context": "fee_factual",
        "notes": "Direct factual answer"
    },
    {
        "intent": "fee_question",
        "response": "현재 조건은 7%이며, 지급은 익일 오후로 안내드립니다.",
        "context": "fee_brief",
        "notes": "Brief confirmation"
    },
    {
        "intent": "fee_question",
        "response": "수수료 7%, 익일 오후 지급입니다. 등록하신 계좌로 이체해 드립니다.",
        "context": "fee_with_method",
        "notes": "Include payment method"
    },
    
    # ============================================================
    # MORE_QUESTIONS - Service/process questions
    # ============================================================
    {
        "intent": "more_questions",
        "response": "AI ARS 1668-4007 번호로 가입 고객의 보험사에 직접 연결되며 긴급출동 안내도 가능합니다.",
        "context": "ars_feature",
        "notes": "ARS service explanation"
    },
    {
        "intent": "more_questions",
        "response": "견적은 보통 10분 내에 안내드리며 기존 보험과의 비교도 가능합니다.",
        "context": "process_timing",
        "notes": "Process and timeline"
    },
    {
        "intent": "more_questions",
        "response": "전담 상담원이 배정되어 청약까지 끝까지 도와드립니다. 필요한 서류는 문자로 안내드립니다.",
        "context": "support_detail",
        "notes": "Support process"
    },
    {
        "intent": "more_questions",
        "response": "고객님께 별도 연락이 가지 않도록 AI로 처리하며, 필요시 보험사와 직접 연결해드립니다.",
        "context": "customer_concern",
        "notes": "Address customer worry"
    },
    
    # ============================================================
    # POSITIVE - Interested/accepting responses
    # ============================================================
    # Sub-context: Initial interest
    {
        "intent": "positive",
        "response": "좋습니다! 비교 견적 바로 진행하고 결과를 문자로 보내드리겠습니다.",
        "context": "interest_action",
        "notes": "Move to action quickly"
    },
    {
        "intent": "positive",
        "response": "감사합니다. 담당자 연결해서 빠르게 안내드리겠습니다.",
        "context": "interest_transfer",
        "notes": "Transfer to specialist"
    },
    
    # Sub-context: Requesting materials
    {
        "intent": "positive",
        "response": "네, 명함과 자세한 자료를 지금 문자로 보내드리겠습니다. 검토 후 연락 주세요.",
        "context": "material_request",
        "notes": "Sending materials"
    },
    {
        "intent": "positive",
        "response": "알겠습니다. 자료와 담당자 연락처를 카톡으로 전달드리겠습니다.",
        "context": "kakao_request",
        "notes": "KakaoTalk delivery"
    },
    
    # Sub-context: Closing confirmation (THIS IS THE FIX!)
    {
        "intent": "positive",
        "response": "감사합니다! 좋은 하루 되세요.",
        "context": "closing_thanks",
        "notes": "PROPER CLOSING - use when dealer says thanks/goodbye"
    },
    {
        "intent": "positive",
        "response": "네, 감사합니다. 필요하시면 언제든 연락 주세요. 좋은 하루 보내세요!",
        "context": "closing_polite",
        "notes": "Polite closing with future opening"
    },
    
    # ============================================================
    # REJECTION - Not interested
    # ============================================================
    # Sub-context: Hard rejection
    {
        "intent": "rejection",
        "response": "알겠습니다. 명함만 문자로 남겨드릴게요. 필요하실 때 연락 주세요.",
        "context": "hard_rejection_soft",
        "notes": "Soft exit, leave door open"
    },
    {
        "intent": "rejection",
        "response": "네, 이해했습니다. 좋은 하루 되세요.",
        "context": "hard_rejection_close",
        "notes": "Clean close"
    },
    
    # Sub-context: Soft rejection (busy/later)
    {
        "intent": "rejection",
        "response": "알겠습니다. 편하신 시간에 다시 연락드려도 될까요?",
        "context": "soft_rejection_reschedule",
        "notes": "Attempt to reschedule"
    },
    {
        "intent": "rejection",
        "response": "네, 바쁘실 텐데 방해 드려 죄송합니다. 명함만 남겨둘게요.",
        "context": "soft_rejection_apologize",
        "notes": "Apologetic exit"
    },
    
    # Sub-context: Busy/timing issue
    {
        "intent": "rejection",
        "response": "네, 지금 바쁘시군요. 언제 다시 연락드리면 좋을까요?",
        "context": "busy_reschedule",
        "notes": "Acknowledge busy, reschedule"
    },
    
    # ============================================================
    # OTHER - Has existing provider
    # ============================================================
    {
        "intent": "other",
        "response": "이해합니다. 그래도 한 번 비교만 받아보시는 건 어떠세요? 수수료 7%와 높은 체결율이 강점입니다.",
        "context": "other_soft_pitch",
        "notes": "Gentle persuasion"
    },
    {
        "intent": "other",
        "response": "네, 기존 거래처 있으시군요. 혹시 비교견적만 받아보시고 결정하셔도 됩니다.",
        "context": "other_comparison",
        "notes": "Offer comparison"
    },
    {
        "intent": "other",
        "response": "알겠습니다. 혹시 조건 확인해보시고 싶으시면 언제든 연락 주세요.",
        "context": "other_soft_close",
        "notes": "Soft close, leave door open"
    },
    {
        "intent": "other",
        "response": "네, 이해합니다. 명함만 남겨드릴게요. 비교 필요하실 때 연락 주세요.",
        "context": "other_accept_close",
        "notes": "Accept their situation"
    },
    
    # ============================================================
    # FALLBACK - Unclear/confused
    # ============================================================
    {
        "intent": "fallback",
        "response": "죄송합니다. 제가 정확히 이해하지 못했어요. 한 번만 더 말씀해 주시겠어요?",
        "context": "clarification_request",
        "notes": "Polite clarification"
    },
    {
        "intent": "fallback",
        "response": "정확한 안내를 위해 궁금하신 점을 조금만 더 구체적으로 알려주실 수 있을까요?",
        "context": "specific_request",
        "notes": "Ask for specifics"
    },
    {
        "intent": "fallback",
        "response": "어떤 부분이 가장 궁금하신가요? 수수료, 절차, 서비스 중에서 알려드릴까요?",
        "context": "guided_choice",
        "notes": "Offer specific options"
    },
    {
        "intent": "fallback",
        "response": "혹시 담당자와 직접 통화하시겠어요? 더 자세히 안내드릴 수 있습니다.",
        "context": "escalate_human",
        "notes": "Escalate to human"
    },
]

def generate_csv(output_path: str = "response_templates_production.csv"):
    """Generate production CSV file"""
    
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['intent', 'response', 'context', 'notes'])
        writer.writeheader()
        
        for template in TEMPLATES:
            writer.writerow({
                'intent': template['intent'],
                'response': template['response'],
                'context': template['context'],
                'notes': template['notes']
            })
    
    print(f"✅ Generated: {output_path}")
    print(f"   Total templates: {len(TEMPLATES)}")
    
    # Print summary
    from collections import Counter
    intent_counts = Counter(t['intent'] for t in TEMPLATES)
    print(f"\n📊 Templates per intent:")
    for intent, count in intent_counts.items():
        print(f"   {intent:20s}: {count} templates")


if __name__ == "__main__":
    import sys
    
    # Allow optional output path as argument
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "response_templates_production.csv"
    
    generate_csv(csv_path)
