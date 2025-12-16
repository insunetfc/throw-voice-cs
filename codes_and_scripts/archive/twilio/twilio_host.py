# twilio_host.py
import json, base64, asyncio, websockets, audioop

STREAMS = {}          # streamSid -> websocket
CALL_TO_STREAM = {}   # callSid   -> streamSid

async def handle(ws, *args):
    sid = None
    call_sid = None
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            event = msg.get("event")
            if event == "start":
                sid = msg["start"]["streamSid"]
                call_sid = msg["start"].get("callSid")
                STREAMS[sid] = ws
                if call_sid:
                    CALL_TO_STREAM[call_sid] = sid
                print(f"[WS] start streamSid={sid} callSid={call_sid}")
            elif event == "stop":
                print(f"[WS] stop streamSid={sid} callSid={call_sid}")
                if sid:
                    STREAMS.pop(sid, None)
                if call_sid:
                    CALL_TO_STREAM.pop(call_sid, None)
    finally:
        if sid:
            STREAMS.pop(sid, None)
        if call_sid:
            CALL_TO_STREAM.pop(call_sid, None)

def ws_for(stream_sid):
    return STREAMS.get(stream_sid)

def ws_for_call(call_sid):
    return STREAMS.get(CALL_TO_STREAM.get(call_sid))

def stream_sid_for_call(call_sid):
    return CALL_TO_STREAM.get(call_sid)

def send_clear(ws, stream_sid):
    asyncio.create_task(ws.send(json.dumps({"event":"clear","streamSid":stream_sid})))

def send_mark(ws, stream_sid, name="chunk"):
    asyncio.create_task(ws.send(json.dumps({"event":"mark","streamSid":stream_sid,"mark":{"name":name}})))

def send_ulaw_media(ws, stream_sid, ulaw_bytes):
    payload = base64.b64encode(ulaw_bytes).decode()
    asyncio.create_task(ws.send(json.dumps({
        "event":"media","streamSid":stream_sid,"media":{"payload":payload}
    })))

async def main():
    async with websockets.serve(handle, "0.0.0.0", 8765, max_size=None):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
