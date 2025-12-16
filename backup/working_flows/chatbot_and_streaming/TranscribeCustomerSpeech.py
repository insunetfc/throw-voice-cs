# to be used with bot alias
def lambda_handler(event, context):
    intent = event['sessionState']['intent']
    slots = intent.get('slots') or {}
    first_text = event.get('inputTranscript') or ""

    if event.get('invocationSource') == 'DialogCodeHook':
        if first_text and not (slots.get('UserInput') and slots['UserInput'].get('value')):
            slots['UserInput'] = {"value": {
                "originalValue": first_text,
                "interpretedValue": first_text
            }}
        return {
            "sessionState": {
                "dialogAction": {"type": "Delegate"},
                "intent": {"name": intent['name'], "state": "InProgress", "slots": slots}
            }
        }
