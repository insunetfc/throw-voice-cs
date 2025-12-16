# Deployable in AWS Lambda function
import boto3

def lambda_handler(event, context):
    connect = boto3.client("connect", region_name="ap-northeast-2")
    
    response = connect.start_outbound_voice_contact(
        DestinationPhoneNumber='+821043876322',  # ← your phone number (test device)
        ContactFlowId='b5758a4d-edc4-4242-af82-1222f6f41608',  # ← your flow ID
        InstanceId='5b83741e-7823-4d70-952a-519d1ac05e63',     # ← your Connect instance ID
        SourcePhoneNumber='+442046234430',  # ← your claimed UK number
        Attributes={
            "test_type": "lex_korean"
        }
    )
    
    return {
        "status": "success",
        "contact_id": response["ContactId"]
    }