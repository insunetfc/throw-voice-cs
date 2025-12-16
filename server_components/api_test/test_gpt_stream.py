#!/usr/bin/env python3
"""
Test script for GPT Voice streaming with start/fetch pattern
This demonstrates the fix and shows sub-second latency.
"""

import requests
import base64
import io
import time
from pydub import AudioSegment
from pydub.playback import play

BASE_URL = "http://honest-trivially-buffalo.ngrok-free.app/voice"  # Change to your server URL

def test_gpt_voice_streaming():
    """
    Test the fixed GPT Voice streaming implementation.
    Expected behavior: First audio chunk in <1 second.
    """
    
    print("="*60)
    print("GPT Voice Streaming Test")
    print("="*60)
    
    # 1. Start the stream
    print("\n1. Starting stream...")
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{BASE_URL}/gpt/start",
            json={
                "text": "Say hello and introduce yourself briefly.",
                "voice": "alloy"  # verse, alloy, shimmer, echo
            },
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        session_id = data["session_id"]
        print(f"✅ Session started: {session_id}")
        print(f"   Status: {data['status']}")
        
    except Exception as e:
        print(f"❌ Failed to start stream: {e}")
        return
    
    # 2. Fetch audio deltas
    print("\n2. Fetching audio deltas...")
    next_index = 0
    audio_buffer = io.BytesIO()
    delta_count = 0
    first_delta_time = None
    
    while True:
        try:
            response = requests.get(
                f"{BASE_URL}/gpt/fetch/{session_id}",
                params={
                    "from_index": next_index,
                    "wait_ms": 100  # Wait up to 100ms for new data
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            # Process new deltas
            if data["deltas"]:
                if first_delta_time is None:
                    first_delta_time = time.time() - start_time
                    print(f"   ⚡ First delta received in {first_delta_time*1000:.0f}ms")
                
                for delta in data["deltas"]:
                    # Decode base64 PCM16 audio
                    pcm_bytes = base64.b64decode(delta)
                    audio_buffer.write(pcm_bytes)
                    delta_count += 1
                
                print(f"   📦 Received {len(data['deltas'])} deltas (total: {delta_count})")
            
            next_index = data["next_index"]
            status = data["status"]
            
            # Check completion
            if status == "completed":
                total_time = time.time() - start_time
                print(f"\n✅ Stream completed!")
                print(f"   Total time: {total_time:.2f}s")
                print(f"   Total deltas: {delta_count}")
                print(f"   Audio bytes: {audio_buffer.tell()}")
                if first_delta_time:
                    print(f"   First delta latency: {first_delta_time*1000:.0f}ms")
                break
            
            elif status == "error":
                error = data.get("error", "Unknown error")
                print(f"\n❌ Stream failed: {error}")
                return
            
            # If no deltas and still streaming, continue polling
            elif status in ["initializing", "streaming"]:
                continue
            
        except Exception as e:
            print(f"❌ Fetch error: {e}")
            return
    
    # 3. Play the audio
    if audio_buffer.tell() > 0:
        print("\n3. Playing audio...")
        audio_buffer.seek(0)
        
        try:
            # Convert PCM16 to AudioSegment
            audio = AudioSegment.from_raw(
                audio_buffer,
                sample_width=2,  # 16-bit = 2 bytes
                frame_rate=24000,  # GPT outputs 24kHz
                channels=1  # mono
            )
            
            print(f"   Duration: {len(audio)/1000:.2f}s")
            print(f"   Playing...")
            play(audio)
            print("   ✅ Playback complete")
            
        except Exception as e:
            print(f"   ⚠️  Playback error: {e}")
            print(f"   (Audio was received successfully, just couldn't play)")
    else:
        print("\n⚠️  No audio data received")
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)


def test_concurrent_streams():
    """Test multiple concurrent streams"""
    print("\n" + "="*60)
    print("Concurrent Streams Test")
    print("="*60)
    
    prompts = [
        "Count from 1 to 3",
        "Say the alphabet: A, B, C",
        "Name three colors"
    ]
    
    # Start all streams
    session_ids = []
    print("\nStarting 3 concurrent streams...")
    for i, prompt in enumerate(prompts):
        try:
            response = requests.post(
                f"{BASE_URL}/gpt/start",
                json={"text": prompt, "voice": "alloy"},
                timeout=5
            )
            session_id = response.json()["session_id"]
            session_ids.append(session_id)
            print(f"   {i+1}. Started: {session_id[:8]}... ({prompt})")
        except Exception as e:
            print(f"   {i+1}. ❌ Failed: {e}")
    
    # Check status of all streams
    print("\nChecking status after 2 seconds...")
    time.sleep(2)
    
    for i, session_id in enumerate(session_ids):
        try:
            response = requests.get(f"{BASE_URL}/gpt/status/{session_id}")
            data = response.json()
            print(f"   {i+1}. {data['status']} - {data['delta_count']} deltas")
        except Exception as e:
            print(f"   {i+1}. ❌ Error: {e}")


def test_latency_benchmark():
    """Benchmark latency across multiple requests"""
    print("\n" + "="*60)
    print("Latency Benchmark (10 requests)")
    print("="*60)
    
    latencies = []
    
    for i in range(10):
        print(f"\n{i+1}/10: ", end="", flush=True)
        start = time.time()
        
        try:
            # Start stream
            response = requests.post(
                f"{BASE_URL}/gpt/start",
                json={"text": "Say hello", "voice": "alloy"},
                timeout=5
            )
            session_id = response.json()["session_id"]
            
            # Wait for first delta
            next_index = 0
            first_delta_time = None
            
            while first_delta_time is None:
                response = requests.get(
                    f"{BASE_URL}/gpt/fetch/{session_id}",
                    params={"from_index": next_index, "wait_ms": 100},
                    timeout=10
                )
                data = response.json()
                
                if data["deltas"]:
                    first_delta_time = (time.time() - start) * 1000
                    latencies.append(first_delta_time)
                    print(f"{first_delta_time:.0f}ms ✅")
                    break
                
                if data["status"] in ["completed", "error"]:
                    print("❌ No deltas received")
                    break
                
                next_index = data["next_index"]
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Statistics
    if latencies:
        print("\n" + "-"*60)
        print("Results:")
        print(f"   Requests: {len(latencies)}")
        print(f"   Average: {sum(latencies)/len(latencies):.0f}ms")
        print(f"   Min: {min(latencies):.0f}ms")
        print(f"   Max: {max(latencies):.0f}ms")
        print(f"   Median: {sorted(latencies)[len(latencies)//2]:.0f}ms")
    
    print("="*60)


if __name__ == "__main__":
    import sys
    
    print("\n🔊 GPT Voice Streaming Test Suite\n")
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=2)
        print(f"✅ Server is running at {BASE_URL}\n")
    except:
        print(f"❌ Server not reachable at {BASE_URL}")
        print(f"   Please start your FastAPI server first.")
        sys.exit(1)
    
    # Run tests
    test_mode = sys.argv[1] if len(sys.argv) > 1 else "basic"
    
    if test_mode == "basic":
        test_gpt_voice_streaming()
    
    elif test_mode == "concurrent":
        test_concurrent_streams()
    
    elif test_mode == "benchmark":
        test_latency_benchmark()
    
    elif test_mode == "all":
        test_gpt_voice_streaming()
        test_concurrent_streams()
        test_latency_benchmark()
    
    else:
        print(f"Usage: {sys.argv[0]} [basic|concurrent|benchmark|all]")
        print(f"   basic: Single stream test (default)")
        print(f"   concurrent: Test multiple concurrent streams")
        print(f"   benchmark: Measure latency across 10 requests")
        print(f"   all: Run all tests")