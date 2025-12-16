from flask import Flask, request, Response
import os

app = Flask(__name__)

CONNECT_DID = os.getenv("CONNECT_DID")  # optional

def twiml(xml_str: str) -> Response:
    return Response(xml_str, mimetype="text/xml")

@app.route("/voice_gather", methods=["POST", "GET"])
def voice_gather():
    # Short prompt; caller can speak over it. Twilio posts transcript to /handle_speech
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Gather input="speech" action="/handle_speech" method="POST" speechTimeout="auto">
    <Say language="ko-KR">안녕하세요. 말씀해 주세요.</Say>
  </Gather>
  <Say language="ko-KR">말씀이 들리지 않았습니다.</Say>
</Response>"""
    return twiml(xml)

@app.route("/handle_speech", methods=["POST"])
def handle_speech():
    speech = request.form.get("SpeechResult", "").strip()
    # Minimal: echo the result and (optionally) hand off to Connect
    if CONNECT_DID:
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say language="ko-KR">말씀하신 내용은 다음과 같습니다. {speech}</Say>
  <Dial>{CONNECT_DID}</Dial>
</Response>"""
    else:
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say language="ko-KR">말씀하신 내용은 다음과 같습니다. {speech}</Say>
</Response>"""
    return twiml(xml)

@app.route("/voice_stream", methods=["POST", "GET"])
def voice_stream():
    # Opens a bidirectional media stream to your WS server
    # Add a parameter "text" to seed TTS on the server if desired
    seed_text = request.args.get("text", "안녕하세요. 테스트 오디오를 재생합니다.")
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://YOUR_PUBLIC_WS_HOST/twilio">
      <Parameter name="text" value="{seed_text}"/>
    </Stream>
  </Connect>
</Response>"""
    return twiml(xml)

@app.route("/handoff_connect", methods=["POST"])
def handoff_connect():
    if not CONNECT_DID:
        return twiml("""<?xml version="1.0" encoding="UTF-8"?><Response><Say>CONNECT_DID not set.</Say></Response>""")
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Dial>{CONNECT_DID}</Dial>
</Response>"""
    return twiml(xml)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
