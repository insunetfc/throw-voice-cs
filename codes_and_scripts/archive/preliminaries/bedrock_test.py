import os, json, boto3, re

REGION = "ap-northeast-2"
MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

brt = boto3.client("bedrock-runtime", region_name=REGION)

SYSTEM_PROMPT = """
You are a speech transcription correction system for Korean promotional phone calls about car insurance.

CONTEXT:
The customer received this promotional call:
"안녕하세요~ 자동차 보험 비교 가입 도와드리는 차집사 다이렉트 차은하 팀장입니다. 잠시 통화 가능하실까요?"
(Offering car insurance comparison services with 7% commission, asking if they can talk)

TASK:
You receive short Korean customer utterances transcribed by ASR that may contain transcription errors. 
Correct the utterance based on the promotional call context and determine the customer's intent.

COMMON ASR CONFUSIONS IN PROMOTIONAL CALL CONTEXT:
- "바빠요" (I'm busy) ↔ "바꿔요" (transfer/change) ↔ "봐봐요" (look/see)
  → In promotional calls, "바빠요" (busy/rejection) is most likely
- "관심 없어요" (not interested) ↔ "관심 있어요" (interested)
- "괜찮아요" (I'm okay/no thanks) ↔ "괜찮아요" (sounds good)
- "통화 중이에요" (on another call) ↔ "통화 가능해요" (can talk)
- "안 돼요" (can't/no) ↔ "안녕하세요" (hello)

INTENT CATEGORIES:
- "interested": Customer wants to hear more, asks questions, or agrees to proceed
- "busy": Customer is currently busy but not explicitly rejecting
- "not_interested": Customer explicitly declines or shows no interest
- "transfer": Customer asks to transfer the call to someone else
- "unclear": Cannot determine with confidence

INSTRUCTIONS:
1. Consider the promotional call context (customer is being asked about car insurance)
2. Identify likely ASR errors based on common confusions
3. Choose the most contextually appropriate correction
4. Assign intent based on the corrected utterance
5. Rate confidence: 1.0 (certain), 0.7-0.9 (likely), 0.4-0.6 (unsure), <0.4 (very uncertain)

OUTPUT FORMAT (JSON only, no explanation):
{
  "intent": "interested|busy|not_interested|transfer|unclear",
  "normalized": "<corrected Korean text>",
  "confidence": <0.0 to 1.0>
}

EXAMPLES:
Input: "바꿔요"
Output: {"intent": "busy", "normalized": "바빠요", "confidence": 0.85}

Input: "관심 있어요"
Output: {"intent": "interested", "normalized": "관심 있어요", "confidence": 0.95}

Input: "봐봐요"
Output: {"intent": "busy", "normalized": "바빠요", "confidence": 0.75}

Input: "지금은 안 돼요"
Output: {"intent": "busy", "normalized": "지금은 안 돼요", "confidence": 0.9}

Now process the following customer utterance:
"""

def parse_json_text(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip("` \n")
    try:
        return json.loads(cleaned)
    except Exception as e:
        print("Parse error:", e, "on text:", cleaned)
        return {"intent": "other", "normalized": text, "confidence": 0.0}

def bedrock_call(utterance: str):
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": [{"type": "text", "text": utterance}]}],
        "max_tokens": 150
    }

    resp = brt.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(body).encode("utf-8"),
        contentType="application/json",
        accept="application/json",
    )
    payload = json.loads(resp["body"].read())
    print("RAW payload:", json.dumps(payload, indent=2, ensure_ascii=False))

    text_out = payload.get("outputText", "").strip()
    if not text_out and "content" in payload:
        text_out = "".join(p.get("text", "") for p in payload["content"] if p.get("type") == "text")

    parsed = parse_json_text(text_out)
    print("PARSED JSON:", parsed)
    return parsed

if __name__ == "__main__":
    for utt in ["지금 바빠요", "지금 바꿔요", "지금 봐봐요"]:
        print(f"\n--- Testing utterance: {utt} ---")
        bedrock_call(utt)
