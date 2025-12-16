#!/usr/bin/env python3
import argparse
import requests
import os

BRIDGE_BASE = "https://honest-trivially-buffalo.ngrok-free.app"

def register_voice(owner: str, ref_path: str, voice_name: str = None, set_active: bool = True) -> dict:
    """Register a new voice."""
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference file not found: {ref_path}")
    
    files = {"file": (os.path.basename(ref_path), open(ref_path, "rb"), "audio/wav")}
    data = {"owner": owner, "set_active": str(set_active).lower()}
    if voice_name:
        data["voice_name"] = voice_name
    
    url = f"{BRIDGE_BASE}/voice/eleven/register"
    r = requests.post(url, files=files, data=data, timeout=60)
    r.raise_for_status()
    return r.json()

def list_voices(owner: str) -> dict:
    """List all voices for an owner."""
    url = f"{BRIDGE_BASE}/voice/eleven/voices/{owner}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

def set_active_voice(owner: str, voice_id: str = None, voice_index: int = None) -> dict:
    """Set active voice for an owner."""
    url = f"{BRIDGE_BASE}/voice/eleven/voices/{owner}/set-active"
    params = {}
    if voice_id:
        params["voice_id"] = voice_id
    if voice_index is not None:
        params["voice_index"] = voice_index
    
    r = requests.post(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def delete_voice(owner: str, voice_id: str) -> dict:
    """Delete a voice."""
    url = f"{BRIDGE_BASE}/voice/eleven/voices/{owner}/{voice_id}"
    r = requests.delete(url, timeout=10)
    r.raise_for_status()
    return r.json()

def synthesize(owner: str, text: str, voice_id: str = None, use_cache: bool = True) -> dict:
    """Synthesize speech."""
    url = f"{BRIDGE_BASE}/voice/eleven"
    payload = {"text": text, "owner": owner, "use_cache": use_cache}
    if voice_id:
        payload["voice_id"] = voice_id
    
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def call_brain(user_text: str, owner: str, voice_id: str = None) -> dict:
    """Call GPT+ElevenLabs brain."""
    url = f"{BRIDGE_BASE}/voice/brain/gpt-eleven"
    payload = {"user_text": user_text, "owner": owner}
    if voice_id:
        payload["voice_id"] = voice_id
    
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def get_cache_stats() -> dict:
    """Get cache statistics."""
    url = f"{BRIDGE_BASE}/voice/eleven/cache/stats"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

def clear_cache(owner: str = None, voice_id: str = None) -> dict:
    """Clear cache."""
    url = f"{BRIDGE_BASE}/voice/eleven/cache/clear"
    params = {}
    if owner:
        params["owner"] = owner
    if voice_id:
        params["voice_id"] = voice_id
    
    r = requests.post(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def download(url: str, out_path: str):
    """Download audio file."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    print(f"✅ Saved to {out_path}")

def print_voices(data: dict):
    """Pretty print voice list."""
    print(f"\n{'='*70}")
    print(f"Voices for owner: {data['owner']} (Total: {data['count']})")
    print(f"{'='*70}")
    
    if not data['voices']:
        print("No voices registered yet.")
        return
    
    for i, voice in enumerate(data['voices']):
        status = "🟢 ACTIVE" if voice.get('active') else "⚪"
        print(f"\n[{i}] {status} {voice['name']}")
        print(f"    Voice ID: {voice['voice_id']}")
        print(f"    Registered: {voice.get('timestamp', 'unknown')}")
    
    print(f"\n{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(description="ElevenLabs Voice Management & Testing")
    
    # Common args with defaults
    parser.add_argument("--owner", default="manager1", help="Owner ID (default: manager1)")
    parser.add_argument("--base", help="Override bridge base URL")
    
    # Action selection - now with defaults!
    action = parser.add_mutually_exclusive_group(required=False)  # Changed to False
    action.add_argument("--register", nargs="?", const="default", metavar="AUDIO_FILE", 
                       help="Register new voice from audio file (default: use --ref path)")
    action.add_argument("--list", action="store_true", help="List all voices for owner")
    action.add_argument("--set-active", metavar="INDEX_OR_ID", help="Set active voice (index or voice_id)")
    action.add_argument("--delete", metavar="VOICE_ID", help="Delete a voice")
    action.add_argument("--synth", nargs="?", const="default", metavar="TEXT", 
                       help="Synthesize text (default: use --text)")
    action.add_argument("--brain", nargs="?", const="default", metavar="TEXT", 
                       help="Call GPT+Eleven brain (default: use --text)")
    action.add_argument("--cache-stats", action="store_true", help="Show cache statistics")
    action.add_argument("--clear-cache", action="store_true", help="Clear cache")
    
    # Additional options with defaults
    parser.add_argument("--ref", default="/home/tiongsik/Python/outbound_calls/files/audio/Promotional_Calls.wav",
                       help="Path to reference wav for cloning (default: Promotional_Calls.wav)")
    parser.add_argument("--text", default="복제된 음성으로 테스트합니다.", 
                       help="Text to synthesize (default: Korean test phrase)")
    parser.add_argument("--voice-name", help="Human-friendly name for new voice")
    parser.add_argument("--voice-id", help="Specific voice_id to use (overrides active)")
    parser.add_argument("--no-active", action="store_true", help="Don't set as active when registering")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache for this request")
    parser.add_argument("--out", default="./eleven_voice.wav", help="Output audio file path (default: ./eleven_voice.wav)")
    
    args = parser.parse_args()
    
    global BRIDGE_BASE
    if args.base:
        BRIDGE_BASE = args.base
    
    # If no action specified, default to --synth
    if not any([args.register, args.list, args.set_active, args.delete, 
                args.synth, args.brain, args.cache_stats, args.clear_cache]):
        args.synth = "default"
    
    try:
        # Register new voice
        if args.register is not None:
            # Use --ref path if --register has no argument or is "default"
            ref_path = args.ref if args.register == "default" else args.register
            print(f"📤 Registering voice from {ref_path}")
            result = register_voice(
                args.owner, 
                ref_path, 
                args.voice_name,
                set_active=not args.no_active
            )
            print(f"✅ Registered: {result['name']}")
            print(f"   Voice ID: {result['voice_id']}")
            print(f"   Active: {result['active']}")
            print(f"   Total voices: {result['total_voices']}")
        
        # List voices
        elif args.list:
            result = list_voices(args.owner)
            print_voices(result)
        
        # Set active voice
        elif args.set_active:
            # Try to parse as index first, otherwise treat as voice_id
            try:
                idx = int(args.set_active)
                result = set_active_voice(args.owner, voice_index=idx)
            except ValueError:
                result = set_active_voice(args.owner, voice_id=args.set_active)
            
            print(f"✅ Active voice set to: {result['active_voice']['name']}")
            print(f"   Voice ID: {result['active_voice']['voice_id']}")
            print(f"   Cache cleared: {result['cache_cleared']} items")
        
        # Delete voice
        elif args.delete:
            result = delete_voice(args.owner, args.delete)
            print(f"✅ Deleted voice: {result['deleted_voice']['name']}")
            print(f"   Remaining voices: {result['remaining_voices']}")
            print(f"   Cache cleared: {result['cache_cleared']} items")
        
        # Synthesize
        elif args.synth is not None:
            # Use --text if --synth has no argument or is "default"
            text = args.text if args.synth == "default" else args.synth
            print(f"🎤 Synthesizing: {text}")
            result = synthesize(
                args.owner, 
                text, 
                voice_id=args.voice_id,
                use_cache=not args.no_cache
            )
            
            cached = "⚡ CACHED" if result.get('cached') else "🔄 GENERATED"
            print(f"{cached}")
            print(f"Voice ID used: {result['voice_id_used']}")
            
            if result.get('audio_url'):
                download(result['audio_url'], args.out)
        
        # Brain (GPT + Eleven)
        elif args.brain is not None:
            # Use --text if --brain has no argument or is "default"
            text = args.text if args.brain == "default" else args.brain
            print(f"🧠 Calling brain with: {text}")
            result = call_brain(text, args.owner, voice_id=args.voice_id)
            
            print(f"\n{'='*70}")
            print(f"🤖 GPT Reply: {result['reply_text']}")
            print(f"{'='*70}")
            print(f"Voice ID: {result['voice_id_used']}")
            print(f"Cached: {result.get('cached', False)}")
            
            timings = result.get('timings', {})
            if timings:
                print(f"\n⏱️  Timings:")
                print(f"   GPT: {timings.get('gpt_ms')}ms")
                print(f"   TTS: {timings.get('tts_ms')}ms")
                print(f"   Total: {timings.get('total_ms')}ms")
            
            if result.get('audio_url'):
                download(result['audio_url'], args.out)
        
        # Cache stats
        elif args.cache_stats:
            result = get_cache_stats()
            print(f"\n{'='*70}")
            print(f"Cache Statistics")
            print(f"{'='*70}")
            print(f"Total cached items: {result['total_cached']}")
            
            if result.get('by_owner_and_voice'):
                print(f"\nBreakdown by owner and voice:")
                for owner, voices in result['by_owner_and_voice'].items():
                    print(f"\n  {owner}:")
                    for voice_name, count in voices.items():
                        print(f"    - {voice_name}: {count} items")
            print(f"\n{'='*70}\n")
        
        # Clear cache
        elif args.clear_cache:
            result = clear_cache(owner=args.owner if args.owner != "manager1" else None)
            print(f"✅ Cleared {result['cleared']} cached items")
    
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e.response.status_code}")
        print(f"   {e.response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()