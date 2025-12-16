from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from rtc_backend.services.orchestrator_factory import build_orchestrator
from rtc_backend.state.session_store import session_store

router = APIRouter()

logger = logging.getLogger(__name__)

async def _iter_ws_audio(ws: WebSocket) -> AsyncIterator[bytes]:
    """
    Wrap WebSocket binary receives as an async iterator of audio frames.
    """
    try:
        while True:
            payload = await ws.receive_bytes()
            yield payload
    except WebSocketDisconnect:
        return


@router.websocket("/ws/fs-bridge/{call_id}")
async def fs_bridge_media(ws: WebSocket, call_id: str) -> None:
    """
    WebSocket endpoint for FreeSWITCH mod_audio_stream.

    FreeSWITCH dials:
      ws://<backend>:8000/ws/fs-bridge/{uuid}

    `call_id` here is the FreeSWITCH uuid (${uuid} in the dialplan).
    Any extra JSON metadata you pass from the dialplan is carried
    in the WebSocket stream (first text frame or similar), not as query params.
    """
    # If you want to reuse the orchestrator infra and treat call_id like contact_id:
    session = session_store.get(call_id)
    if not session:
        # For now just accept anyway; or you can enforce a 4404 like the Connect bridge
        logger.info("fs-bridge: no session found for call_id=%s, accepting anyway", call_id)

    await ws.accept()
    logger.info("fs-bridge connected call_id=%s", call_id)

    # Option A: reuse the orchestrator with call_id as key
    orchestrator = build_orchestrator(contact_id=call_id)

    try:
        async for out_chunk in orchestrator.handle_audio_stream(_iter_ws_audio(ws)):
            # For FS you might not need to send audio back, but you *can*
            # send_bytes here if you want to do TTS back over the same WS.
            await ws.send_bytes(out_chunk)
    except WebSocketDisconnect:
        logger.info("fs-bridge websocket disconnected call_id=%s", call_id)
    except Exception as exc:
        logger.exception("fs-bridge websocket error for %s: %s", call_id, exc)
        await ws.close(code=1011, reason="server_error")

async def _iter_ws_audio(ws: WebSocket) -> AsyncIterator[bytes]:
    """
    Wrap WebSocket binary receives as an async iterator of audio frames.
    """
    try:
        while True:
            payload = await ws.receive_bytes()
            yield payload
    except WebSocketDisconnect:
        return


@router.websocket("/ws/bridge")
async def bridge_media(ws: WebSocket) -> None:
    """
    Accepts PCM audio frames from Connect and streams GPT/STT/TTS output back.
    Requires `contactId` query parameter that matches `/voice/start`.
    """
    contact_id = ws.query_params.get("contactId")
    if not contact_id:
        await ws.close(code=4400, reason="contactId query parameter required")
        return

    session = session_store.get(contact_id)
    if not session:
        await ws.close(code=4404, reason="contact session not found")
        return

    await ws.accept()
    orchestrator = build_orchestrator(contact_id=contact_id)
    logger.info("bridge connected contactId=%s streamArn=%s", contact_id, session.stream_arn)
    try:
        async for out_chunk in orchestrator.handle_audio_stream(_iter_ws_audio(ws)):
            await ws.send_bytes(out_chunk)
    except WebSocketDisconnect:
        logger.info("bridge websocket disconnected contactId=%s", contact_id)
    except Exception as exc:  # pragma: no cover
        logger.exception("bridge websocket error for %s: %s", contact_id, exc)
        await ws.close(code=1011, reason="server_error")
