from lambda_functions.link_nipa_local import lambda_handler

event = {
    "Details": {
        "ContactData": {
            "Attributes": {
                "UserText": "Test offline mock"
            }
        }
    }
}

result = lambda_handler(event, None)
print(result)

# If you want to hear it:
import re, base64
match = re.match(r"data:audio/wav;base64,(.*)", result["setAttributes"]["AudioS3Url"])
if match:
    audio_data = base64.b64decode(match.group(1))
    with open("test.wav", "wb") as f:
        f.write(audio_data)
    print("Saved test.wav — play it to verify!")
