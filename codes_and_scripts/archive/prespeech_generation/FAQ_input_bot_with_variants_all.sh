#!/usr/bin/env bash
set -euo pipefail

# ------------ config ------------
FN_NAME="${1:-InvokeBotLambda}"
shift || true                 # remove FN_NAME; the rest go to generator
GEN_FLAGS=("$@")              # e.g., --punct-mode none --no-typos --no-dialects

AWS_REGION="${AWS_REGION:-ap-northeast-2}"
LIMIT="${LIMIT:-3}"
DRY_RUN="${DRY_RUN:-1}"
SLEEP_MS="${SLEEP_MS:-0}"

# prefer python3 if available
if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TS="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$SCRIPT_DIR/runs/$RUN_TS"
mkdir -p "$RUN_DIR"

GEN_V2="$SCRIPT_DIR/gen_korean_variants_v2.py"
GEN_V1="$SCRIPT_DIR/gen_korean_variants.py"

echo "[info] cwd: $PWD"
echo "[info] script dir: $SCRIPT_DIR"
echo "[info] run dir: $RUN_DIR"
echo "[info] FN_NAME=$FN_NAME  AWS_REGION=$AWS_REGION  LIMIT=$LIMIT  DRY_RUN=$DRY_RUN  SLEEP_MS=${SLEEP_MS}ms"
echo "[info] GEN_FLAGS: ${GEN_FLAGS[*]:-(none)}"

# ------------ inputs ------------
inputs=(
  "회의중입니다"
  "상담중입니다"
  "괜찮습니다"
  "아닙니다"
  "제가 회의중이라서요"
  "나중에 전화드리겠습니다"
  "수수료가 얼마인가요?"
  "왜 7%인가요?"
  "다른 비용은 없나요?"
  "실제로 제가 내야 하는 금액은 얼마예요?"
  "사고가 많은데 가입이 가능합니까?"
  "어떤 조건이 있어야 가입할 수 있나요?"
  "다이렉트로만 가능한가요, 오프라인도 되나요?"
  "제가 지금 보험이 있는데도 가입할 수 있나요?"
  "수수료는 언제 지급되나요?"
  "처리 시간은 얼마나 걸리나요?"
  "바로 가입이 되나요?"
  "견적은 언제 받을 수 있나요?"
  "체결율이 어떻게 되나요?"
  "다른 사람들도 많이 이용하나요?"
  "이번에 새로 만든 팀은 어떤 팀인가요?"
  "믿을 만한가요?"
  "앞으로 누가 담당하나요?"
  "담당자가 바뀌면 어떻게 되나요?"
  "연락처를 알려주세요."
  "명함은 문자로 보내주실 수 있나요?"
  "지금 바빠요."
  "나중에 다시 전화 주세요."
  "문자로 보내주세요."
  "관심 없어요."
  "이미 다른 데 가입했어요."
  "그만 연락해 주세요."
  "네, 안녕하세요."
  "여보세요?"
  "네, 말씀하세요."
  "누구세요?"
  "어디에서 전화하신 거예요?"
  "차집사요? 처음 들어보는데요."
  "차은하 팀장이세요? 안녕하세요."
  "지금 시간이 안 돼요."
  "회의 중이에요."
  "운전 중이에요."
  "언제 다시 전화 주시면 될까요?"
  "몇 분 정도 걸려요?"
  "간단하게 말씀해 주세요."
  "빨리 말씀해 주세요."
  "시간이 없어서요."
  "네, 들어보겠습니다."
  "어떤 내용인가요?"
  "자세히 설명해 주세요."
  "조건이 어떻게 되나요?"
  "좀 더 알려주세요."
  "궁금한데요."
  "괜찮은 것 같은데요."
  "혜택이 뭐가 있나요?"
  "다른 곳과 뭐가 다른가요?"
  "정말이에요?"
  "믿을 수 있나요?"
  "사기 아니에요?"
  "왜 이런 조건으로 하는 거예요?"
  "뭔가 수상한데요."
  "너무 좋은 조건 아닌가요?"
  "함정이 있는 거 아니에요?"
  "검증된 곳인가요?"
  "허가받은 업체예요?"
  "금융감독원 등록업체인가요?"
  "수수료가 정말 7%예요?"
  "숨겨진 비용은 없나요?"
  "다른 수수료는 안 받아요?"
  "왜 이렇게 높은 수수료를 주는 거예요?"
  "언제 받을 수 있어요?"
  "현금으로 주나요?"
  "계좌로 입금해 주나요?"
  "세금은 어떻게 되나요?"
  "수수료 지급 증명서 주나요?"
  "지금 보험 있어요."
  "이미 가입되어 있어요."
  "최근에 갱신했어요."
  "만족하고 있어서요."
  "바꿀 생각 없어요."
  "계약 기간이 남아있어요."
  "해지하면 손해 아닌가요?"
  "지금 것도 괜찮아요."
  "사고가 많은데 괜찮을까요?"
  "사고 이력이 있어도 되나요?"
  "보험료가 비쌀 텐데요?"
  "가입 거절당한 적 있어요."
  "다이렉트는 까다롭잖아요."
  "오프라인도 정말 되나요?"
  "어떻게 진행되나요?"
  "필요한 서류가 뭐예요?"
  "시간이 얼마나 걸려요?"
  "온라인으로 해야 하나요?"
  "직접 만나야 하나요?"
  "견적만 받아볼 수 있나요?"
  "비교해보고 싶어요."
  "계약서는 언제 받나요?"
  "필요 없어요."
  "안 할래요."
  "괜찮습니다."
  "생각해볼게요."
  "나중에 연락드릴게요."
  "다른 데 알아보고 있어서요."
  "가족하고 상의해야 해서요."
  "결정권자가 따로 있어요."
  "명함 주세요."
  "이메일 주소 있어요?"
  "카카오톡 되나요?"
  "언제 연락 주시면 돼요?"
  "제가 연락드릴게요."
  "번호 저장해둘게요."
  "회사 이름이 뭐라고요?"
  "정식 업체 맞아요?"
  "허가증 볼 수 있나요?"
  "홈페이지 있어요?"
  "다른 고객 후기 있나요?"
  "실제 지급 사례 있어요?"
  "보장받을 수 있나요?"
  "다시 한 번 말씀해 주세요."
  "잘 못 들었는데요."
  "천천히 말씀해 주세요."
  "무슨 말씀이세요?"
  "어떤 뜻이에요?"
  "구체적으로 어떻게 되는 거예요?"
  "알겠습니다."
  "네, 감사합니다."
  "생각해보겠습니다."
  "연락드리겠습니다."
  "나중에 통화해요."
  "수고하세요."
  "좋은 하루 되세요."
  "끊겠습니다."
  "전화 그만 끊을게요."
  "전화 받기 싫어요."
  "스팸 전화 그만해 주세요."
  "번호 어디서 구했어요?"
  "개인정보 어떻게 아신 거예요?"
  "귀찮아요."
  "화가 나네요."
  "신고할 거예요."
)

# ------------ pick generator ------------
GEN=""
if [[ -f "$GEN_V2" ]]; then
  GEN="$GEN_V2"
  echo "[info] using generator: $GEN_V2"
elif [[ -f "$GEN_V1" ]]; then
  GEN="$GEN_V1"
  echo "[info] using generator: $GEN_V1"
else
  echo "[error] No generator found next to this script."
  exit 1
fi

# ------------ seeds & expand (hardcoded under repo) ------------
SEEDS_FILE="$RUN_DIR/seeds.txt"
AUG_FILE="$RUN_DIR/augmented.txt"
RESP_DIR="$RUN_DIR/responses"
mkdir -p "$RESP_DIR"

printf "%s\n" "${inputs[@]}" > "$SEEDS_FILE"
echo "[info] seeds -> $SEEDS_FILE (count: ${#inputs[@]})"
echo "[debug] first 5 seeds:"
head -n 5 "$SEEDS_FILE" | nl

echo "[debug] running: $PY $(basename "$GEN") --seeds $(basename "$SEEDS_FILE") --limit-per-base $LIMIT ${GEN_FLAGS[*]:-}"
# run from SCRIPT_DIR so relative paths (if any) are consistent
(
  cd "$SCRIPT_DIR"
  set +e
  $PY "$GEN" --seeds "$SEEDS_FILE" --limit-per-base "$LIMIT" "${GEN_FLAGS[@]}" > "$AUG_FILE"
  GEN_RC=$?
  set -e
  if [[ $GEN_RC -ne 0 ]]; then
    echo "[error] generator exited with code $GEN_RC"
    exit $GEN_RC
  fi
)

VAR_CNT=$(wc -l < "$AUG_FILE" | tr -d ' ')
echo "[info] variants -> $AUG_FILE (lines: $VAR_CNT)"
echo "[debug] first 10 variants:"
head -n 10 "$AUG_FILE" | nl

if [[ "$VAR_CNT" -eq 0 ]]; then
  echo "[warn] generator produced 0 lines. Check GEN_FLAGS / Python."
  echo "       Example: --punct-mode none --no-typos --no-dialects"
  exit 0
fi

# ------------ invoke / dry-run ------------
i=0
while IFS= read -r utter; do
  ((i++))
  contact_id=$(printf "testv-%04d" "$i")
  payload=$(cat <<EOF
{
  "Details": {
    "Parameters": {},
    "ContactData": {
      "ContactId": "$contact_id",
      "CustomerInput": "$utter"
    }
  },
  "locale": "ko-KR"
}
EOF
)
  if [[ "$DRY_RUN" == "1" ]]; then
    printf "%s\n" "$payload" > "$RESP_DIR/payload-$i.json"
    echo "[$i/$VAR_CNT] DRY payload-$i.json  |  $utter"
  else
    aws lambda invoke \
      --region "$AWS_REGION" \
      --function-name "$FN_NAME" \
      --payload "$payload" \
      --cli-binary-format raw-in-base64-out \
      "$RESP_DIR/response-$i.json" >/dev/null
    echo "[$i/$VAR_CNT] response-$i.json  |  $utter"
  fi

  if [[ "$SLEEP_MS" -gt 0 ]]; then
    $PY - <<PY
import time
time.sleep($SLEEP_MS/1000.0)
PY
  fi
done < "$AUG_FILE"

echo "[done] total requests: $i  (LIMIT=$LIMIT)  outputs: $RESP_DIR"
