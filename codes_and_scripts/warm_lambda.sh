#!/bin/bash
REGION=ap-northeast-2
FUNC=InvokeBotLambda
ALIAS=production
PC=2

# Publish new version from updated $LATEST
VER=$(aws lambda publish-version \
  --region $REGION \
  --function-name $FUNC \
  --query Version --output text)

# Repoint alias to the new version
aws lambda update-alias \
  --region $REGION \
  --function-name $FUNC \
  --name $ALIAS \
  --function-version $VER

# Reapply provisioned concurrency
aws lambda put-provisioned-concurrency-config \
  --region $REGION \
  --function-name $FUNC \
  --qualifier $ALIAS \
  --provisioned-concurrent-executions $PC

echo "Updated alias $ALIAS to version $VER (with PC=$PC)"
