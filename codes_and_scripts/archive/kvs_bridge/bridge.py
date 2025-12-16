#!/usr/bin/env python3
"""
Optimized bridge for NIPA cloud containers.
Monitors KVS streams and writes transcripts to Connect attributes ASAP.

Deploy this on NIPA alongside your existing setup.
"""

import os, asyncio, logging, subprocess, signal, sys
from datetime import datetime, timezone
import boto3
from botocore.config import Config
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("bridge")

# Config
REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
INSTANCE_ALIAS = os.environ.get("CONNECT_INSTANCE_ALIAS", "insunetfc")
INSTANCE_ID = os.environ.get("CONNECT_INSTANCE_ID", "5b83741e-7823-4d70-952a-519d1ac05e63")
SCAN_INTERVAL = int(os.environ.get("SCAN_INTERVAL_SEC", "3"))
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "8000"))
LANG_CODE = os.environ.get("TRANSCRIBE_LANGUAGE_CODE", "ko-KR")

# Transcribe settings for end-of-speech detection
SILENCE_MS = int(os.environ.get("SILENCE_MS", "800"))  # 800ms silence = done speaking
RMS_FLOOR = float(os.environ.get("RMS_FLOOR", "200.0"))
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
SILENCE_FRAMES = SILENCE_MS // FRAME_MS

# AWS clients
boto_cfg = Config(
    retries={"max_attempts": 3, "mode": "standard"},
    connect_timeout=5,
    read_timeout=10
)
session = boto3.Session(region_name=REGION)
kvs_ctl = session.client("kinesisvideo", config=boto_cfg)
connect = session.client("connect", config=boto_cfg)

# State tracking
active_streams = {}  # {stream_name: task}
processed_contacts = set()  # Don't reprocess same contact

# Graceful shutdown
shutdown_event = asyncio.Event()

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    log.info(f"Received signal {signum}, shutting down...")
    shutdown_event.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def list_active_connect_streams():
    """Get all connect-live streams (any status for debugging)."""
    try:
        resp = kvs_ctl.list_streams(MaxResults=50)
        items = resp.get("StreamInfoList", []) or []
        
        # For debugging: return ALL connect streams, not just ACTIVE
        return [
            s for s in items
            if s.get("StreamName", "").startswith("connect-live--connect-")
        ]
    except Exception as e:
        log.error(f"List streams failed: {e}")
        return []


def extract_contact_id(stream_name: str) -> str:
    """Extract ContactId from stream name."""
    return stream_name.split("-contact-")[-1] if "-contact-" in stream_name else "unknown"


async def quick_transcribe(stream_name: str, contact_id: str):
    """
    Fast transcription pipeline:
    1. Read KVS stream
    2. Decode with ffmpeg
    3. Detect speech
    4. Send to Transcribe
    5. Write result to Connect immediately
    """
    log.info(f"[{contact_id}] Starting quick transcribe")
    
    try:
        # Get media endpoint
        ep = kvs_ctl.get_data_endpoint(
            StreamName=stream_name,
            APIName="GET_MEDIA"
        )["DataEndpoint"]
        
        media_client = session.client(
            "kinesis-video-media",
            endpoint_url=ep,
            config=boto_cfg
        )
        
        # Start GetMedia
        resp = media_client.get_media(
            StreamName=stream_name,
            StartSelector={"StartSelectorType": "NOW"}
        )
        payload = resp["Payload"]
        
        log.info(f"[{contact_id}] Connected to stream")
        
        # Launch ffmpeg to decode MKV → PCM
        ffmpeg_proc = subprocess.Popen(
            [
                "ffmpeg", "-loglevel", "error", "-hide_banner",
                "-f", "matroska", "-i", "pipe:0",
                "-vn", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(SAMPLE_RATE),
                "-f", "s16le", "pipe:1"
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Read KVS and write to ffmpeg in background
        async def pump_kvs():
            try:
                loop = asyncio.get_running_loop()
                while not shutdown_event.is_set():
                    chunk = await loop.run_in_executor(None, payload.read, 8192)
                    if not chunk:
                        await asyncio.sleep(0.01)
                        continue
                    ffmpeg_proc.stdin.write(chunk)
                    ffmpeg_proc.stdin.flush()
            except Exception as e:
                log.error(f"[{contact_id}] KVS pump error: {e}")
            finally:
                try:
                    ffmpeg_proc.stdin.close()
                except:
                    pass
        
        pump_task = asyncio.create_task(pump_kvs())
        
        # Read PCM from ffmpeg and transcribe
        transcript = await transcribe_pcm_stream(ffmpeg_proc.stdout, contact_id)
        
        # Cancel pump
        pump_task.cancel()
        try:
            ffmpeg_proc.kill()
        except:
            pass
        
        # Write result to Connect immediately
        if transcript:
            log.info(f"[{contact_id}] Transcript: {transcript}")
            
            try:
                connect.update_contact_attributes(
                    InstanceId=INSTANCE_ID,
                    InitialContactId=contact_id,
                    Attributes={
                        'live_transcript': transcript,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'source': 'bridge_transcribe'
                    }
                )
                log.info(f"[{contact_id}] ✓ Wrote to Connect attributes")
            except Exception as e:
                log.error(f"[{contact_id}] Failed to update Connect: {e}")
        else:
            log.warning(f"[{contact_id}] No transcript captured")
        
    except Exception as e:
        log.error(f"[{contact_id}] Transcription failed: {e}")
    finally:
        processed_contacts.add(contact_id)
        if stream_name in active_streams:
            del active_streams[stream_name]


async def transcribe_pcm_stream(pcm_stream, contact_id: str) -> str:
    """
    Read PCM from ffmpeg and send to Amazon Transcribe Streaming.
    Returns transcript when speech ends (silence detected).
    """
    try:
        from amazon_transcribe.client import TranscribeStreamingClient
        from amazon_transcribe.handlers import TranscriptResultStreamHandler
        
        client = TranscribeStreamingClient(region=REGION)
        stream = await client.start_stream_transcription(
            language_code=LANG_CODE,
            media_sample_rate_hz=SAMPLE_RATE,
            media_encoding="pcm",
            enable_partial_results_stabilization=True,
            partial_results_stability="medium"
        )
        
        final_transcript = ""
        partial_transcript = ""
        
        class TranscriptHandler(TranscriptResultStreamHandler):
            async def handle_transcript_event(self, transcript_event):
                nonlocal final_transcript, partial_transcript
                
                for result in transcript_event.transcript.results:
                    if result.is_partial:
                        for alt in result.alternatives:
                            partial_transcript = alt.transcript or ""
                            if partial_transcript:
                                log.debug(f"[{contact_id}] Partial: {partial_transcript}")
                    else:
                        for alt in result.alternatives:
                            final_transcript = alt.transcript or ""
                            if final_transcript:
                                log.info(f"[{contact_id}] Final: {final_transcript}")
        
        handler = TranscriptHandler(stream.output_stream)
        handler_task = asyncio.create_task(handler.handle_events())
        
        # Read PCM and send to Transcribe
        loop = asyncio.get_running_loop()
        frame_size = FRAME_SAMPLES * 2  # 20ms frames
        silent_count = 0
        speech_detected = False
        max_wait_frames = 1500  # 30 seconds max wait for speech (1500 * 20ms)
        frames_sent = 0
        
        async def send_audio():
            nonlocal silent_count, speech_detected, frames_sent
            
            while not shutdown_event.is_set():
                # Read 20ms frame
                frame_bytes = await loop.run_in_executor(None, pcm_stream.read, frame_size)
                
                if len(frame_bytes) < frame_size:
                    await asyncio.sleep(0.01)
                    continue
                
                # Send to Transcribe
                await stream.input_stream.send_audio_event(audio_chunk=frame_bytes)
                frames_sent += 1
                
                # Check for speech/silence
                pcm = np.frombuffer(frame_bytes, dtype=np.int16)
                rms = np.sqrt(np.mean(pcm.astype(np.float64) ** 2))
                
                if rms >= RMS_FLOOR:
                    # Speech detected!
                    if not speech_detected:
                        log.info(f"[{contact_id}] Speech detected at frame {frames_sent}")
                        speech_detected = True
                    silent_count = 0
                else:
                    # Silence
                    if speech_detected:
                        # Only count silence AFTER speech started
                        silent_count += 1
                        if silent_count >= SILENCE_FRAMES:
                            log.info(f"[{contact_id}] End of speech detected")
                            await stream.input_stream.end_stream()
                            break
                
                # Timeout if no speech after 30 seconds
                if not speech_detected and frames_sent >= max_wait_frames:
                    log.warning(f"[{contact_id}] No speech detected after 30s, stopping")
                    await stream.input_stream.end_stream()
                    break
            
            await stream.input_stream.end_stream()
        
        # Run both send and receive
        await asyncio.gather(send_audio(), handler_task)
        
        return final_transcript or partial_transcript
        
    except ImportError:
        log.error("amazon-transcribe-streaming not installed")
        log.error("Install with: pip install amazon-transcribe")
        return ""
    except Exception as e:
        log.error(f"[{contact_id}] Transcribe error: {e}")
        return ""


async def monitor_streams():
    """Main monitoring loop."""
    log.info("=" * 60)
    log.info("Bridge started on NIPA Cloud")
    log.info(f"Region: {REGION} | Instance: {INSTANCE_ALIAS}")
    log.info(f"Scan interval: {SCAN_INTERVAL}s")
    log.info("=" * 60)
    
    # Max age for streams (ignore streams older than this)
    MAX_STREAM_AGE_SECONDS = int(os.environ.get("MAX_STREAM_AGE_SEC", "300"))  # 5 minutes
    log.info(f"Max stream age: {MAX_STREAM_AGE_SECONDS}s")
    log.info("=" * 60)
    
    while not shutdown_event.is_set():
        try:
            streams = list_active_connect_streams()
            
            # Log what we found
            if streams:
                log.info(f"Scanning {len(streams)} active streams...")
            else:
                log.debug("No active streams found")
            
            for stream_info in streams:
                stream_name = stream_info.get("StreamName")
                contact_id = extract_contact_id(stream_name)
                
                # Check stream age
                age = (datetime.now(timezone.utc) - stream_info.get("CreationTime")).total_seconds()
                
                # Log every stream we see
                status = stream_info.get('Status')
                log.info(f"  Stream: {contact_id[:8]}... age: {age:.1f}s status: {status}")
                
                # Skip if not ACTIVE
                if status != "ACTIVE":
                    log.debug(f"  → Skipping (not ACTIVE)")
                    continue
                
                # Skip if already processing or processed
                if stream_name in active_streams or contact_id in processed_contacts:
                    log.debug(f"  → Skipping (already processing/processed)")
                    continue
                
                # Check stream age
                age = (datetime.now(timezone.utc) - stream_info.get("CreationTime")).total_seconds()
                
                # Skip old streams (likely finished calls)
                if age > MAX_STREAM_AGE_SECONDS:
                    log.debug(f"  → Skipping (age {age:.1f}s > {MAX_STREAM_AGE_SECONDS}s)")
                    processed_contacts.add(contact_id)  # Mark as processed to avoid checking again
                    continue
                
                log.info("=" * 60)
                log.info(f"NEW CALL: {contact_id} (age: {age:.1f}s)")
                log.info("=" * 60)
                
                # Start processing
                task = asyncio.create_task(quick_transcribe(stream_name, contact_id))
                active_streams[stream_name] = task
            
            # Status
            if active_streams:
                log.debug(f"Active: {len(active_streams)} | Processed: {len(processed_contacts)}")
            
        except Exception as e:
            log.error(f"Monitor error: {e}")
        
        await asyncio.sleep(SCAN_INTERVAL)


async def main():
    """Entry point."""
    try:
        await monitor_streams()
    except Exception as e:
        log.error(f"Fatal error: {e}")
    finally:
        log.info("Shutting down...")
        # Cancel active tasks
        for task in active_streams.values():
            task.cancel()
        await asyncio.gather(*active_streams.values(), return_exceptions=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Stopped by user")
    sys.exit(0)
