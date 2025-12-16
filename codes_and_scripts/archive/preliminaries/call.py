# local_trigger.py
import boto3

connect = boto3.client('connect', region_name='ap-northeast-2')

phone_numbers = ["+821043876322"]  # Replace with E.164 format numbers
contact_flow_id = "4542b882-ebda-48a5-883a-ad8d09f96c31"
instance_id = "eefed165-54dc-428e-a0f1-02c2ec35a22e"
source_number = "+13232056958"  # Your claimed outbound number

for number in phone_numbers:
    try:
        response = connect.start_outbound_voice_contact(
            DestinationPhoneNumber=number,
            ContactFlowId=contact_flow_id,
            InstanceId=instance_id,
            SourcePhoneNumber=source_number,
            Attributes={"Name": "LocalRun"}
        )
        print(f"✅ Call initiated to {number}. ContactId: {response['ContactId']}")

    except Exception as e:
        print(f"❌ Failed to call {number}: {str(e)}")
