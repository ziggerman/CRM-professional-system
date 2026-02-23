#!/usr/bin/env python3
"""
Local Voice Transcription Script
Uses faster-whisper for offline voice recognition (no API required)

Installation:
    pip install faster-whisper sounddevice numpy

Usage:
    python scripts/local_voice_transcribe.py <audio_file>
    python scripts/local_voice_transcribe.py --mic
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def transcribe_audio(file_path: str, model_size: str = "base") -> str:
    """
    Transcribe audio file using local Whisper model.
    
    Args:
        file_path: Path to audio file
        model_size: Model size (tiny, base, small, medium, large)
    
    Returns:
        Transcribed text
    """
    try:
        from faster_whisper import WhisperModel
        
        print(f"Loading Whisper model ({model_size})...")
        print("   (First run will download the model)")
        
        # Use CPU for now - for GPU use device="cuda"
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        
        print(f"Transcribing: {file_path}")
        
        segments, info = model.transcribe(
            file_path,
            beam_size=5,
            vad_filter=True,  # Voice activity detection
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        print(f"   Detected language: {info.language} ({info.language_probability:.2f})")
        
        full_text = ""
        for segment in segments:
            text = segment.text.strip()
            print(f"   [{segment.start:.1f}s - {segment.end:.1f}s] {text}")
            full_text += text + " "
        
        return full_text.strip()
        
    except ImportError:
        print("ERROR: faster-whisper not installed!")
        print("   Install with: pip install faster-whisper")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


async def transcribe_voice_async(file_path: str, model_size: str = "base") -> str:
    """Async wrapper for transcription."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, transcribe_audio, file_path, model_size)


def record_from_microphone(duration: int = 10, sample_rate: int = 16000) -> str:
    """
    Record audio from microphone and save to temp file.
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Audio sample rate
    
    Returns:
        Path to temporary audio file
    """
    try:
        import sounddevice as sd
        import numpy as np
        from datetime import datetime
        
        print(f"Recording for {duration} seconds...")
        
        # Record audio
        recording = sd.rec(
            frames=duration * sample_rate,
            samplerate=sample_rate,
            channels=1,
            dtype=np.int16
        )
        sd.wait()
        
        # Save to temp file
        temp_dir = Path(__file__).parent.parent / "uploads"
        temp_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_file = temp_dir / f"voice_{timestamp}.wav"
        
        # Save as WAV
        import wave
        with wave.open(str(temp_file), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(recording.tobytes())
        
        print(f"   Saved to: {temp_file}")
        return str(temp_file)
        
    except ImportError:
        print("ERROR: sounddevice not installed!")
        print("   Install with: pip install sounddevice numpy")
        return None
    except Exception as e:
        print(f"Recording error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Local Voice Transcription")
    parser.add_argument("file", nargs="?", help="Audio file to transcribe")
    parser.add_argument("--mic", action="store_true", help="Record from microphone")
    parser.add_argument("--duration", type=int, default=10, help="Recording duration in seconds")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    
    args = parser.parse_args()
    
    audio_file = None
    
    if args.mic:
        # Record from microphone
        audio_file = record_from_microphone(duration=args.duration)
        if not audio_file:
            print("Failed to record audio")
            return
    elif args.file:
        # Use provided file
        audio_file = args.file
        if not Path(audio_file).exists():
            print(f"File not found: {audio_file}")
            return
    else:
        print("Usage:")
        print("  python scripts/local_voice_transcribe.py <audio_file>")
        print("  python scripts/local_voice_transcribe.py --mic")
        return
    
    # Transcribe
    result = asyncio.run(transcribe_voice_async(audio_file, args.model))
    
    if result:
        print("\n=== Transcription result ===")
        print(result)
        print("=" * 40)
    else:
        print("Transcription failed")


if __name__ == "__main__":
    main()
