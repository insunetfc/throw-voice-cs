"""
Pre-warm normalization cache with common customer utterances
Run this during Lambda warm-up or as initialization
"""

import json

def extract_unique_base_utterances(augmented_file_path: str) -> list:
    """
    Extract unique base utterances (without prefixes like '음', '아', etc.)
    from augmented data
    """
    with open(augmented_file_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    
    # Common prefixes/fillers to remove
    prefixes = ['음', '아', '어', '잠깐만', '실례지만', '죄송하지만', '혹시', '지금은']
    
    unique_bases = set()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        
        # Try removing each prefix
        base = stripped
        for prefix in prefixes:
            if stripped.startswith(prefix + ' '):
                base = stripped[len(prefix)+1:]
                break
        
        unique_bases.add(base)
    
    return sorted(list(unique_bases))

def create_cache_warmup_data(augmented_file_path: str, output_file: str = 'cache_warmup.json'):
    """
    Create a JSON file with normalized mappings for cache pre-warming
    """
    unique_utterances = extract_unique_base_utterances(augmented_file_path)
    
    # Manual mappings based on your data patterns
    # These are the CANONICAL normalized forms
    canonical_mappings = {
        # Busy/Unavailable
        "회의중입니다": {"intent": "busy", "normalized": "회의중입니다", "confidence": 0.95},
        "지금 바빠서 나중에 부탁드립니다": {"intent": "busy", "normalized": "지금 바빠서 나중에 부탁드립니다", "confidence": 0.95},
        "지금은 통화가 어려워요": {"intent": "busy", "normalized": "지금은 통화가 어려워요", "confidence": 0.95},
        "회의 중이라 잠시 힘들어요": {"intent": "busy", "normalized": "회의 중이라 잠시 힘들어요", "confidence": 0.95},
        "상담중입니다": {"intent": "busy", "normalized": "상담중입니다", "confidence": 0.95},
        "제가 회의중이라서요": {"intent": "busy", "normalized": "제가 회의중이라서요", "confidence": 0.95},
        "회의 중이에요": {"intent": "busy", "normalized": "회의 중이에요", "confidence": 0.95},
        "운전 중이에요": {"intent": "busy", "normalized": "운전 중이에요", "confidence": 0.95},
        "지금 바빠요": {"intent": "busy", "normalized": "지금 바빠요", "confidence": 0.95},
        "지금 시간이 안 돼요": {"intent": "busy", "normalized": "지금 시간이 안 돼요", "confidence": 0.95},
        
        # Callback requests
        "나중에 전화드리겠습니다": {"intent": "busy", "normalized": "나중에 전화드리겠습니다", "confidence": 0.9},
        "나중에 다시 연락 드려도 될까요": {"intent": "busy", "normalized": "나중에 다시 연락 드려도 될까요", "confidence": 0.9},
        "나중에 다시 전화 주세요": {"intent": "busy", "normalized": "나중에 다시 전화 주세요", "confidence": 0.9},
        
        # Simple acknowledgments (ambiguous - could be acceptance or dismissal)
        "괜찮습니다": {"intent": "unclear", "normalized": "괜찮습니다", "confidence": 0.6},
        "네, 알겠습니다": {"intent": "unclear", "normalized": "네, 알겠습니다", "confidence": 0.6},
        "아닙니다": {"intent": "not_interested", "normalized": "아닙니다", "confidence": 0.8},
        
        # Questions about fees/commission
        "수수료가 얼마인가요?": {"intent": "interested", "normalized": "수수료가 얼마인가요?", "confidence": 0.9},
        "추가 비용은 없는지 알려주세요": {"intent": "interested", "normalized": "추가 비용은 없는지 알려주세요", "confidence": 0.9},
        "최종 부담액이 얼마인지요": {"intent": "interested", "normalized": "최종 부담액이 얼마인지요", "confidence": 0.9},
        "왜 7%인가요?": {"intent": "interested", "normalized": "왜 7%인가요?", "confidence": 0.85},
        "다른 비용은 없나요?": {"intent": "interested", "normalized": "다른 비용은 없나요?", "confidence": 0.9},
        
        # Eligibility questions
        "사고가 많은데 가입이 가능합니까?": {"intent": "interested", "normalized": "사고가 많은데 가입이 가능합니까?", "confidence": 0.9},
        "가입 가능 여부가 궁금합니다": {"intent": "interested", "normalized": "가입 가능 여부가 궁금합니다", "confidence": 0.9},
        "사고 이력이 있어도 괜찮나요": {"intent": "interested", "normalized": "사고 이력이 있어도 괜찮나요", "confidence": 0.9},
        
        # Process questions
        "다이렉트로만 가능한가요, 오프라인도 되나요?": {"intent": "interested", "normalized": "다이렉트로만 가능한가요, 오프라인도 되나요?", "confidence": 0.9},
        "처리 시간은 얼마나 걸리나요?": {"intent": "interested", "normalized": "처리 시간은 얼마나 걸리나요?", "confidence": 0.9},
        "어떻게 진행되나요?": {"intent": "interested", "normalized": "어떻게 진행되나요?", "confidence": 0.85},
        
        # Trust/verification
        "믿을 만한가요?": {"intent": "interested", "normalized": "믿을 만한가요?", "confidence": 0.8},
        "정식 등록업체 맞는지요": {"intent": "interested", "normalized": "정식 등록업체 맞는지요", "confidence": 0.85},
        "허가증이나 등록 정보 볼 수 있나요": {"intent": "interested", "normalized": "허가증이나 등록 정보 볼 수 있나요", "confidence": 0.85},
        
        # Already insured
        "지금 보험 있어요": {"intent": "not_interested", "normalized": "지금 보험 있어요", "confidence": 0.85},
        "이미 가입되어 있어요": {"intent": "not_interested", "normalized": "이미 가입되어 있어요", "confidence": 0.9},
        "제가 지금 보험이 있는데도 가입할 수 있나요?": {"intent": "interested", "normalized": "제가 지금 보험이 있는데도 가입할 수 있나요?", "confidence": 0.85},
        
        # Explicit rejection
        "관심 없어요": {"intent": "not_interested", "normalized": "관심 없어요", "confidence": 0.95},
        "이번에는 패스할게요": {"intent": "not_interested", "normalized": "이번에는 패스할게요", "confidence": 0.95},
        "연락 중단 요청드립니다": {"intent": "not_interested", "normalized": "연락 중단 요청드립니다", "confidence": 0.98},
        "이미 다른 데 가입했어요": {"intent": "not_interested", "normalized": "이미 다른 데 가입했어요", "confidence": 0.95},
        "그만 연락해 주세요": {"intent": "not_interested", "normalized": "그만 연락해 주세요", "confidence": 0.98},
        
        # Greetings
        "네, 안녕하세요": {"intent": "interested", "normalized": "네, 안녕하세요", "confidence": 0.8},
        "여보세요?": {"intent": "unclear", "normalized": "여보세요?", "confidence": 0.7},
        "네, 말씀하세요": {"intent": "interested", "normalized": "네, 말씀하세요", "confidence": 0.8},
        
        # Information requests
        "자세히 설명해 주세요": {"intent": "interested", "normalized": "자세히 설명해 주세요", "confidence": 0.9},
        "좀 더 알려주세요": {"intent": "interested", "normalized": "좀 더 알려주세요", "confidence": 0.9},
        "어떤 내용인가요?": {"intent": "interested", "normalized": "어떤 내용인가요?", "confidence": 0.85},
        
        # Deferred decision
        "생각해볼게요": {"intent": "not_interested", "normalized": "생각해볼게요", "confidence": 0.7},
        "나중에 연락드릴게요": {"intent": "not_interested", "normalized": "나중에 연락드릴게요", "confidence": 0.75},
        "가족하고 상의해야 해서요": {"intent": "busy", "normalized": "가족하고 상의해야 해서요", "confidence": 0.8},
    }
    
    # Add all variations with prefixes
    warmup_data = {}
    prefixes = ['', '음 ', '아 ', '어 ', '잠깐만 ', '실례지만 ', '죄송하지만 ', '혹시 ', '지금은 ']
    
    for base, mapping in canonical_mappings.items():
        for prefix in prefixes:
            variant = (prefix + base).strip()
            warmup_data[variant] = mapping.copy()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(warmup_data, f, ensure_ascii=False, indent=2)
    
    print(f"Created cache warmup file: {output_file}")
    print(f"Total entries: {len(warmup_data)}")
    return warmup_data

# Usage:
if __name__ == "__main__":
    warmup_data = create_cache_warmup_data('/home/tiongsik/Python/outbound_calls/codes_and_scripts/archive/prespeech_generation/augmented.txt')
    print(f"Sample entries: {list(warmup_data.items())[:5]}")