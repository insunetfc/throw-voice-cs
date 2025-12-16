# Amazon Connect Korean Voice Bot Summary (as of 2025-08-06)

This document summarizes the completed setup of a Korean-language voice bot using Amazon Connect and Amazon Lex V2, with Lambda-based dynamic responses. The purpose is to preserve working knowledge for future reference.

---

## System Overview
- **Goal**: Build a minimal working prototype where a user can call in, speak a Korean phrase, and receive a spoken dynamic response based on that input.
- **Technology Stack**:
  - Amazon Connect (Tokyo region)
  - Amazon Lex V2 (Locale: ko_KR)
  - AWS Lambda (Python)
  - Text-to-Speech via Amazon Connect (Polly underneath)

---

## Components Used

### 1. **Amazon Lex V2**
- **Bot Name**: `FreeInputBotKorean`
- **Locale**: `ko_KR`
- **Intent**: `CaptureUtterance`
- **Slot**: `UserInput`
  - Type: `AMAZON.FreeFormInput`
  - Prompt: `한 말씀만 해주세요.`

### 2. **AWS Lambda**
- **Function Name**: `InvokeBotLambda`
- **Invocation Type**: Synchronous
- **Input**: Passed from Lex slot (`$.Lex.Slots.UserInput`)
- **Output Format**:
```python
return {
    "user_input": f"당신의 요청은 다음과 같습니다: {user_input}"
}
```

### 3. **Amazon Connect Flow**
- **Flow Name**: `SpeechInputFlow1`
- **Key Blocks**:
  1. Play Prompt (optional)
  2. Connect to Lex V2 Bot (`ko_KR` locale)
  3. Invoke Lambda with `user_input` slot
  4. MessageParticipant (TTS):
     - `Text`: `$.External.$.External.user_input`
     - `useDynamic: true`

---

## Flow Logic Summary
1. Customer speaks a Korean sentence (e.g., "시작해 줘")
2. Lex bot captures it as a `UserInput` slot
3. Connect invokes Lambda with the captured slot
4. Lambda returns a structured message
5. Connect uses a dynamic TTS block to play the returned message

> Note: The usage of `$.External.$.External.user_input` is intentional and works due to `useDynamic: true` being set — this double resolution allows the string returned by Lambda to be interpreted as a dynamic variable.

---

## Example Input/Output
| Spoken Phrase     | Captured Slot       | Lambda Response                                 | Playback                        |
|------------------|---------------------|--------------------------------------------------|----------------------------------|
| "시작해 줘"      | `시작해 줘`          | `당신의 요청은 다음과 같습니다: 시작해 줘`          | Spoken via TTS               |
| "안녕하세요"      | `안녕하세요`          | `당신의 요청은 다음과 같습니다: 안녕하세요`          | Spoken via TTS               |

---

## Special Behavior Notes
- Amazon Connect can access Lambda's return values via `$.External.*` without a `Set contact attributes` block **if** the Lambda returns a flat JSON object.
- Using `"Text": "$.External.$.External.user_input"` works due to double dynamic interpretation. This is a supported pattern when `useDynamic: true` is used.

---

## Required IAM Policies

### Amazon Connect
```json
AmazonConnectFullAccess
```
Optional fine-grained version:
```json
{
  "Effect": "Allow",
  "Action": [
    "connect:StartOutboundVoiceContact",
    "connect:DescribeContactFlow",
    "connect:UpdateContactAttributes",
    "connect:GetContactAttributes"
  ],
  "Resource": "*"
}
```

### Amazon Lex V2
```json
AmazonLexFullAccess
```
Optional fine-grained version:
```json
{
  "Effect": "Allow",
  "Action": [
    "lex:StartConversation",
    "lex:RecognizeText",
    "lex:RecognizeUtterance",
    "lex:DescribeBot",
    "lex:ListBots"
  ],
  "Resource": "*"
}
```

### AWS Lambda
```json
AWSLambdaBasicExecutionRole
AmazonConnectFullAccess
```
Optional fine-grained:
```json
{
  "Effect": "Allow",
  "Action": [
    "connect:UpdateContactAttributes"
  ],
  "Resource": "*"
}
```

### Amazon S3 (for prompt playback)
```json
AmazonS3FullAccess
```
Or restrict to a specific bucket:
```json
{
  "Effect": "Allow",
  "Action": [
    "s3:GetObject",
    "s3:ListBucket"
  ],
  "Resource": [
    "arn:aws:s3:::call-center-bucket-ifc",
    "arn:aws:s3:::call-center-bucket-ifc/*"
  ]
}
```

### (Optional) Amazon Polly (if pre-synthesizing)
```json
{
  "Effect": "Allow",
  "Action": [
    "polly:SynthesizeSpeech"
  ],
  "Resource": "*"
}
```

---

## Next Steps / Future Considerations
- Modularize Lambda to support S3 audio playback vs. TTS
- Test long TTS limits (~2,000 chars safe)
- Consider SSML or Polly caching
- Plan for transition away from Polly if required
- Extend documentation or automation if scale increases

---

## File Reference
- `SpeechInputFlow1.json` — Full exported contact flow
- `InvokeBotLambda` — Deployed Lambda handler
- `README.md` — Initial setup instructions (to be updated to reflect this summary)

---
