"""
Comprehensive test suite for intent classification and response generation.
Tests determinism, context-awareness, and response variety.
"""

import requests
import json
import time
from typing import Dict, List, Tuple
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================
URL = "https://honest-trivially-buffalo.ngrok-free.app/respond"
# URL = "https://f8d5daa0897b.ngrok-free.app/respond"

HEADERS = {
    "Content-Type": "application/json"
}

# ============================================================
# TEST CASES - Organized by Intent
# ============================================================
TEST_CASES = {
    "fee_question": {
        "amount": [
            "몇 프로에요?",
            "수수료 몇 %에요?",
            "얼마 주시나요?",
            "몇 퍼센트죠?",
        ],
        "timing": [
            "언제 입금돼요?",
            "지급 시점은 언제예요?",
            "익일이면 언제예요?",
            "바로 주는 거예요?",
        ],
        "method": [
            "어떻게 지급하나요?",
            "계좌로 들어오나요?",
            "지급 방법은요?",
        ],
        "tax": [
            "세금은 어떻게 되나요?",
            "원천징수 하나요?",
            "3.3% 떼나요?",
        ],
        "scope": [
            "다이렉트도 7%인가요?",
            "삼성도 포함돼요?",
            "오프라인도 되나요?",
        ]
    },
    "about_company": {
        "identity": [
            "차집사가 어디에요?",
            "어느 회사죠?",
            "회사 이름이 뭐예요?",
        ],
        "legitimacy": [
            "정식 업체예요?",
            "등록된 회사인가요?",
            "믿을 수 있나요?",
        ],
        "relationship": [
            "보험사랑 무슨 관계예요?",
            "제휴사예요?",
            "직영인가요?",
        ]
    },
    "more_questions": {
        "process": [
            "절차가 어떻게 되나요?",
            "어떻게 진행되나요?",
            "방법 좀 알려주세요",
        ],
        "features": [
            "ARS가 뭐예요?",
            "긴급출동도 되나요?",
            "어떤 서비스 있어요?",
        ]
    },
    "positive": {
        "interested": [
            "오 괜찮네요",
            "좋은데요",
            "괜찮은 것 같아요",
        ],
        "request": [
            "명함 주세요",
            "자료 좀 보내주세요",
            "문자로 주세요",
        ],
        "commit": [
            "그럼 해볼게요",
            "진행해주세요",
            "신청할게요",
        ]
    },
    "rejection": {
        "hard": [
            "안 해요",
            "필요 없어요",
            "관심 없어요",
        ],
        "soft": [
            "나중에 연락 주세요",
            "다음에 해요",
            "생각해볼게요",
        ],
        "busy": [
            "지금 바빠요",
            "고객 응대 중이에요",
            "시간 없어요",
        ]
    },
    "other": {
        "satisfied": [
            "다른 데 만족하고 있어요",
            "지금 하는 데 괜찮아요",
        ],
        "committed": [
            "이미 정해진 곳 있어요",
            "거래처 있어서요",
            "계약된 업체 있어요",
        ]
    },
    "greeting": {
        "basic": [
            "여보세요",
            "안녕하세요",
            "네 말씀하세요",
        ]
    }
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def make_request(text: str) -> Tuple[Dict, float]:
    """Make API request and return response + elapsed time"""
    payload = {"text": text}
    start_time = time.time()
    
    try:
        response = requests.post(URL, headers=HEADERS, data=json.dumps(payload), timeout=10)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            return response.json(), elapsed
        else:
            return {"error": f"HTTP {response.status_code}", "text": response.text}, elapsed
    
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        return {"error": "Request timeout"}, elapsed
    
    except Exception as e:
        elapsed = time.time() - start_time
        return {"error": str(e)}, elapsed


def print_response(text: str, result: Dict, elapsed: float, verbose: bool = False):
    """Pretty print a single response"""
    print(f"\n📝 Input: \"{text}\"")
    
    if "error" in result:
        print(f"   ❌ Error: {result['error']}")
        return
    
    intent = result.get('final_intent') or result.get('intent', 'unknown')
    confidence = result.get('confidence', 0)
    sub_context = result.get('sub_context', 'N/A')
    template_ctx = result.get('template_context', 'N/A')
    response = result.get('agent_response', 'N/A')
    
    print(f"   🎯 Intent: {intent} ({confidence:.2%})")
    print(f"   📊 Sub-context: {sub_context}")
    print(f"   📋 Template: {template_ctx}")
    print(f"   💬 Response: {response[:80]}{'...' if len(response) > 80 else ''}")
    print(f"   ⏱️  Time: {elapsed:.3f}s")
    
    if verbose:
        print(f"   🔍 Full response:")
        print(json.dumps(result, ensure_ascii=False, indent=4))


# ============================================================
# TEST 1: BASIC FUNCTIONALITY
# ============================================================

def test_basic_functionality():
    """Test basic API connectivity and response format"""
    print("\n" + "="*80)
    print("TEST 1: BASIC FUNCTIONALITY")
    print("="*80)
    
    test_text = "몇 프로에요?"
    print(f"\n🧪 Testing with: \"{test_text}\"")
    
    result, elapsed = make_request(test_text)
    
    if "error" in result:
        print(f"❌ FAILED: {result['error']}")
        return False
    
    print("✅ API is responding")
    
    # Check required fields
    required_fields = ['intent', 'confidence', 'agent_response']
    missing = [f for f in required_fields if f not in result]
    
    if missing:
        print(f"⚠️  Missing fields: {missing}")
    else:
        print("✅ All required fields present")
    
    print_response(test_text, result, elapsed, verbose=True)
    
    return True


# ============================================================
# TEST 2: DETERMINISM
# ============================================================

def test_determinism():
    """Test if same input gives same output"""
    print("\n" + "="*80)
    print("TEST 2: DETERMINISM (Same Input → Same Output)")
    print("="*80)
    
    test_cases = [
        "몇 프로에요?",
        "차집사가 어디에요?",
        "괜찮네요",
    ]
    
    results = {}
    
    for text in test_cases:
        print(f"\n🔁 Testing: \"{text}\" (3 attempts)")
        
        attempts = []
        for i in range(3):
            result, elapsed = make_request(text)
            if "error" not in result:
                attempts.append({
                    'intent': result.get('final_intent') or result.get('intent'),
                    'response': result.get('agent_response'),
                    'sub_context': result.get('sub_context'),
                })
            time.sleep(0.1)  # Small delay between requests
        
        # Check consistency
        if len(attempts) == 3:
            intents = [a['intent'] for a in attempts]
            responses = [a['response'] for a in attempts]
            
            intent_consistent = len(set(intents)) == 1
            response_consistent = len(set(responses)) == 1
            
            if intent_consistent and response_consistent:
                print(f"   ✅ DETERMINISTIC")
                print(f"      Intent: {intents[0]}")
                print(f"      Response: {responses[0][:60]}...")
            else:
                print(f"   ❌ NOT DETERMINISTIC")
                print(f"      Intents: {intents}")
                print(f"      Responses differ: {not response_consistent}")
            
            results[text] = intent_consistent and response_consistent
        else:
            print(f"   ❌ Failed to get 3 successful responses")
            results[text] = False
    
    passed = sum(results.values())
    total = len(results)
    print(f"\n📊 Result: {passed}/{total} passed")
    
    return passed == total


# ============================================================
# TEST 3: CONTEXT AWARENESS
# ============================================================

def test_context_awareness():
    """Test if different sub-contexts get different responses"""
    print("\n" + "="*80)
    print("TEST 3: CONTEXT AWARENESS (Different Questions → Different Answers)")
    print("="*80)
    
    # Test fee questions with different focuses
    print("\n📌 Testing fee_question with different contexts:")
    
    fee_tests = [
        ("몇 프로에요?", "Should focus on amount"),
        ("언제 입금돼요?", "Should focus on timing"),
        ("어떻게 지급하나요?", "Should focus on method"),
        ("세금은 어떻게 되나요?", "Should mention tax"),
    ]
    
    responses = []
    for text, expected in fee_tests:
        result, elapsed = make_request(text)
        if "error" not in result:
            response = result.get('agent_response', '')
            sub_ctx = result.get('sub_context', 'N/A')
            responses.append({
                'text': text,
                'expected': expected,
                'response': response,
                'sub_context': sub_ctx
            })
            print(f"\n   Input: \"{text}\"")
            print(f"   Expected: {expected}")
            print(f"   Sub-context: {sub_ctx}")
            print(f"   Response: {response[:70]}...")
        time.sleep(0.1)
    
    # Check if responses are different
    unique_responses = len(set(r['response'] for r in responses))
    unique_contexts = len(set(r['sub_context'] for r in responses if r['sub_context'] != 'N/A'))
    
    print(f"\n📊 Results:")
    print(f"   Unique responses: {unique_responses}/{len(responses)}")
    print(f"   Unique contexts detected: {unique_contexts}/{len(responses)}")
    
    if unique_responses >= 3:
        print(f"   ✅ CONTEXT-AWARE (Multiple different responses)")
    elif unique_responses == 1:
        print(f"   ❌ NOT CONTEXT-AWARE (All same response)")
    else:
        print(f"   ⚠️  PARTIALLY CONTEXT-AWARE ({unique_responses} different responses)")
    
    return unique_responses >= 3


# ============================================================
# TEST 4: INTENT CLASSIFICATION
# ============================================================

def test_intent_classification():
    """Test if different intents are correctly classified"""
    print("\n" + "="*80)
    print("TEST 4: INTENT CLASSIFICATION")
    print("="*80)
    
    test_samples = {
        "fee_question": ["몇 프로에요?", "수수료 얼마죠?"],
        "about_company": ["차집사가 어디에요?", "무슨 회사예요?"],
        "positive": ["괜찮네요", "명함 주세요"],
        "rejection": ["안 해요", "바빠요"],
        "other": ["다른 데 하고 있어요", "거래처 있어요"],
        "greeting": ["여보세요", "안녕하세요"],
    }
    
    results = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for expected_intent, texts in test_samples.items():
        print(f"\n📌 Testing {expected_intent}:")
        
        for text in texts:
            result, elapsed = make_request(text)
            
            if "error" not in result:
                detected = result.get('final_intent') or result.get('intent')
                confidence = result.get('confidence', 0)
                
                is_correct = detected == expected_intent
                results[expected_intent]["total"] += 1
                if is_correct:
                    results[expected_intent]["correct"] += 1
                
                status = "✅" if is_correct else "❌"
                print(f"   {status} \"{text}\" → {detected} ({confidence:.2%})")
            
            time.sleep(0.1)
    
    # Summary
    print(f"\n📊 Intent Classification Accuracy:")
    total_correct = 0
    total_samples = 0
    
    for intent, stats in results.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        total_correct += stats["correct"]
        total_samples += stats["total"]
        print(f"   {intent:20s}: {stats['correct']}/{stats['total']} ({accuracy:.1%})")
    
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f"\n   {'Overall':20s}: {total_correct}/{total_samples} ({overall_accuracy:.1%})")
    
    return overall_accuracy >= 0.8  # 80% threshold


# ============================================================
# TEST 5: RESPONSE VARIETY
# ============================================================

def test_response_variety():
    """Test if system has good variety in responses"""
    print("\n" + "="*80)
    print("TEST 5: RESPONSE VARIETY")
    print("="*80)
    
    all_responses = []
    all_hashes = set()
    
    for intent, contexts in TEST_CASES.items():
        for context, texts in contexts.items():
            # Test first text from each context
            if texts:
                text = texts[0]
                result, _ = make_request(text)
                
                if "error" not in result:
                    response = result.get('agent_response', '')
                    all_responses.append({
                        'intent': intent,
                        'context': context,
                        'text': text,
                        'response': response
                    })
                    # Create a simple hash of response
                    response_hash = hash(response)
                    all_hashes.add(response_hash)
                
                time.sleep(0.1)
    
    print(f"\n📊 Variety Analysis:")
    print(f"   Total test cases: {len(all_responses)}")
    print(f"   Unique responses: {len(all_hashes)}")
    print(f"   Variety ratio: {len(all_hashes)/len(all_responses):.1%}")
    
    # Group by intent to see variety within intents
    by_intent = defaultdict(list)
    for r in all_responses:
        by_intent[r['intent']].append(r['response'])
    
    print(f"\n📋 Variety by Intent:")
    for intent, responses in by_intent.items():
        unique = len(set(responses))
        total = len(responses)
        print(f"   {intent:20s}: {unique}/{total} unique responses")
    
    variety_ratio = len(all_hashes) / len(all_responses) if all_responses else 0
    return variety_ratio >= 0.7  # At least 70% unique responses


# ============================================================
# TEST 6: EDGE CASES
# ============================================================

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "="*80)
    print("TEST 6: EDGE CASES")
    print("="*80)
    
    edge_cases = [
        ("", "Empty string"),
        ("ㅋㅋㅋㅋ", "Only consonants"),
        ("아 뭐 어쩌라고", "Vague/unclear"),
        ("그게 그거고 저게 저거잖아요", "Nonsensical"),
        ("a b c d e f", "English letters"),
        ("123456", "Numbers only"),
    ]
    
    results = []
    
    for text, description in edge_cases:
        print(f"\n🧪 Testing: \"{text}\" ({description})")
        result, elapsed = make_request(text)
        
        if "error" in result:
            print(f"   ⚠️  Error: {result['error']}")
            results.append(False)
        else:
            intent = result.get('final_intent') or result.get('intent')
            confidence = result.get('confidence', 0)
            response = result.get('agent_response', '')
            
            # Should fallback to 'fallback' intent for unclear inputs
            is_handled = len(response) > 0
            
            print(f"   Intent: {intent} ({confidence:.2%})")
            print(f"   Response: {response[:60]}...")
            print(f"   {'✅ Handled' if is_handled else '❌ Not handled'}")
            
            results.append(is_handled)
        
        time.sleep(0.1)
    
    handled = sum(results)
    total = len(results)
    print(f"\n📊 Edge cases handled: {handled}/{total}")
    
    return handled >= total * 0.8  # 80% should be handled


# ============================================================
# MAIN TEST RUNNER
# ============================================================

def run_all_tests():
    """Run all tests and generate report"""
    print("\n" + "="*80)
    print("🧪 COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"Testing URL: {URL}")
    print("="*80)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Determinism", test_determinism),
        ("Context Awareness", test_context_awareness),
        ("Intent Classification", test_intent_classification),
        ("Response Variety", test_response_variety),
        ("Edge Cases", test_edge_cases),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results[test_name] = passed
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with exception: {e}")
            results[test_name] = False
        
        time.sleep(0.5)  # Delay between tests
    
    # Final Report
    print("\n" + "="*80)
    print("📊 FINAL REPORT")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status:12s} | {test_name}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print("="*80)
    print(f"Overall: {passed_count}/{total_count} tests passed ({passed_count/total_count*100:.1f}%)")
    print("="*80)
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Your system is working perfectly!")
    elif passed_count >= total_count * 0.8:
        print("\n👍 Most tests passed. Some minor issues to address.")
    else:
        print("\n⚠️  Several tests failed. Review the results above.")


# ============================================================
# QUICK TEST (FOR DEBUGGING)
# ============================================================

def quick_test(text: str):
    """Quick single test for debugging"""
    print(f"\n🔍 Quick Test: \"{text}\"")
    print("="*80)
    
    result, elapsed = make_request(text)
    print_response(text, result, elapsed, verbose=True)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Quick test mode with command line argument
        test_text = " ".join(sys.argv[1:])
        quick_test(test_text)
    else:
        # Run full test suite
        run_all_tests()