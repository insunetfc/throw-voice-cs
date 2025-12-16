import os
import asyncio
import json
import websockets
from starlette.websockets import WebSocketDisconnect

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview"

async def gpt_ws_relay(client_ws):
    if not OPENAI_API_KEY:
        # if client is already gone, ignore
        try:
            await client_ws.send_text("GPT realtime not configured: missing OPENAI_API_KEY")
        except Exception:
            pass
        return

    async with websockets.connect(
        OPENAI_REALTIME_URL,
        extra_headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        ping_interval=20,
    ) as openai_ws:

        async def client_to_openai():
            try:
                async for msg in client_ws.iter_text():
                    # forward raw JSON from client to OpenAI
                    await openai_ws.send(msg)
            except WebSocketDisconnect:
                # client closed — stop reading
                pass
            except Exception as e:
                print(f"[GPT relay] client_to_openai error: {e}")
            finally:
                # tell OpenAI no more input
                try:
                    await openai_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                except Exception:
                    pass

        async def openai_to_client():
            try:
                async for raw in openai_ws:
                    # forward OpenAI events to the client
                    try:
                        await client_ws.send_text(raw)
                    except WebSocketDisconnect:
                        # client closed — stop sending
                        break
                    except RuntimeError:
                        # ASGI closed
                        break
            except Exception as e:
                print(f"[GPT relay] openai_to_client error: {e}")

        await asyncio.gather(client_to_openai(), openai_to_client(), return_exceptions=True)
