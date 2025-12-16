# Replace with your Connect instance ID (the UUID, not the ARN)
INSTANCE_ID="your-instance-id"

mkdir -p connect-backup

# List all contact flows in the instance
aws connect list-contact-flows \
  --instance-id $INSTANCE_ID \
  --query 'ContactFlowSummaryList[*].Id' \
  --output text > connect-backup/flow-ids.txt

# Loop through each flow ID and export definition
for flow_id in $(cat connect-backup/flow-ids.txt); do
  aws connect describe-contact-flow \
    --instance-id $INSTANCE_ID \
    --contact-flow-id $flow_id \
    > "connect-backup/contact-flow-${flow_id}.json"
done
