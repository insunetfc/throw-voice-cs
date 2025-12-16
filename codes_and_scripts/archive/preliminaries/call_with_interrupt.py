import boto3

connect = boto3.client('connect', region_name='ap-northeast-2')

response = connect.start_outbound_voice_contact(
    DestinationPhoneNumber='+821043876322',  # 👈 customer number
    ContactFlowId='b5758a4d-edc4-4242-af82-1222f6f41608',
    InstanceId='5b83741e-7823-4d70-952a-519d1ac05e63',
    SourcePhoneNumber='+442046234430',  # 👈 your verified Connect number
    Attributes={
        'context': 'promo',  # Optional context parameters
    }
)

print("Call started. Contact ID:", response['ContactId'])
