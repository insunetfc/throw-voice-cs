BOT_ID="2M7RBQ3XRD"

# Get all versions of the bot
aws lexv2-models list-bot-versions \
  --bot-id $BOT_ID \
  --query "botVersionSummaries[*].botVersion" \
  --output text > bot-versions.txt

mkdir -p lex-backup

#!/usr/bin/env bash
set -euo pipefail

AWS_PROFILE="${AWS_PROFILE:-default}"
AWS_REGION="${AWS_REGION:-ap-northeast-2}"
OUTDIR="lex-backup"
mkdir -p "$OUTDIR"

echo "Listing bots..."
aws --profile "$AWS_PROFILE" --region "$AWS_REGION" lexv2-models list-bots \
  --query 'botSummaries[*].[botId,botName]' --output json > "$OUTDIR/bots.json"

jq -r '.[] | @tsv' "$OUTDIR/bots.json" | while IFS=$'\t' read -r BOT_ID BOT_NAME; do
  echo "==> Bot: $BOT_NAME ($BOT_ID)"

  # Optional: snapshot aliases for reference (not included in export zips)
  aws --profile "$AWS_PROFILE" --region "$AWS_REGION" lexv2-models list-bot-aliases \
    --bot-id "$BOT_ID" > "$OUTDIR/bot-${BOT_ID}-aliases.json" || true

  # Get all numbered versions as a JSON array, then print one per line
  aws --profile "$AWS_PROFILE" --region "$AWS_REGION" lexv2-models list-bot-versions \
    --bot-id "$BOT_ID" --max-results 1000 \
    --query 'botVersionSummaries[].botVersion' --output json \
    | jq -r '.[]' > "$OUTDIR/bot-${BOT_ID}-versions.txt" || true

  # If you also want to export DRAFT, uncomment this:
  # echo "DRAFT" >> "$OUTDIR/bot-${BOT_ID}-versions.txt"

  if ! [ -s "$OUTDIR/bot-${BOT_ID}-versions.txt" ]; then
    echo "   (No versions found)"
    continue
  fi

  # Iterate versions line-by-line (each line is exactly one version token)
  while IFS= read -r VER; do
    [ -z "$VER" ] && continue
    echo "   Exporting version: $VER"

    EXPORT_ID=$(
      aws --profile "$AWS_PROFILE" --region "$AWS_REGION" lexv2-models create-export \
        --file-format LexJson \
        --resource-specification "botExportSpecification={botId=${BOT_ID},botVersion=${VER}}" \
        --query 'exportId' --output text
    )

    # Wait for export to finish
    while :; do
      STATUS=$(aws --profile "$AWS_PROFILE" --region "$AWS_REGION" lexv2-models describe-export \
        --export-id "$EXPORT_ID" --query 'exportStatus' --output text || true)
      case "$STATUS" in
        Completed) break ;;
        Failed)
          echo "   !! Export failed for $BOT_ID v$VER"
          aws --profile "$AWS_PROFILE" --region "$AWS_REGION" lexv2-models describe-export \
            --export-id "$EXPORT_ID" > "$OUTDIR/bot-${BOT_ID}-v${VER}-export-failed.json" || true
          continue 2
          ;;
        *) sleep 4 ;;
      esac
    done

    URL=$(
      aws --profile "$AWS_PROFILE" --region "$AWS_REGION" lexv2-models describe-export \
        --export-id "$EXPORT_ID" --query 'downloadUrl' --output text
    )

    curl -fsSL "$URL" -o "$OUTDIR/bot-${BOT_ID}-v${VER}.zip"
    echo "   Saved: $OUTDIR/bot-${BOT_ID}-v${VER}.zip"
  done < "$OUTDIR/bot-${BOT_ID}-versions.txt"
done

echo "Done. Files are in $OUTDIR/"

