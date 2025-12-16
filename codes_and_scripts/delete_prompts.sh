# REQUIRED
INSTANCE_ID="eefed165-54dc-428e-a0f1-02c2ec35a22e"
REGION="ap-northeast-2"
MATCH="Dynamic TTS"      # substring to match in Description
DRY_RUN=0                # set to 0 to actually delete

# Get all prompt IDs (handles up to 1000; loop NextToken if you have more)
ALL_IDS=$(aws connect list-prompts \
  --instance-id "$INSTANCE_ID" --region "$REGION" --max-results 1000 \
  --query "PromptSummaryList[].Id" --output text)

for id in $ALL_IDS; do
  # Pull Name + Description for this prompt
  read -r NAME DESC <<EOF
$(aws connect describe-prompt \
  --instance-id "$INSTANCE_ID" --region "$REGION" --prompt-id "$id" \
  --query "[Prompt.Name, Prompt.Description]" --output text)
EOF

  if [[ "$DESC" == *"$MATCH"* ]]; then
    echo "[MATCH] $id  Name=\"$NAME\"  Desc=\"$DESC\""
    if [[ "$DRY_RUN" -eq 0 ]]; then
      aws connect delete-prompt --instance-id "$INSTANCE_ID" --region "$REGION" --prompt-id "$id"
      sleep 0.1
    fi
  fi
done
