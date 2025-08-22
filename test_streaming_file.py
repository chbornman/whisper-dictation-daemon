#!/usr/bin/env python3
import sys
import subprocess
import tempfile
import numpy as np
import scipy.io.wavfile as wavfile
from faster_whisper import WhisperModel
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def convert_to_wav(mp3_file):
    """Convert MP3 to WAV using ffmpeg"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
        wav_path = tmp_wav.name
        
    cmd = [
        'ffmpeg', '-i', mp3_file,
        '-ar', '16000',  # 16kHz sample rate
        '-ac', '1',      # Mono
        '-f', 'wav',
        '-y',  # Overwrite output
        wav_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"ffmpeg error: {result.stderr}")
        raise Exception(f"Failed to convert MP3: {result.stderr}")
    
    return wav_path

def process_file_streaming(file_path):
    """Process audio file with streaming simulation"""
    
    # Convert if MP3
    if file_path.endswith('.mp3'):
        logger.info(f"Converting {file_path} to WAV...")
        wav_path = convert_to_wav(file_path)
    else:
        wav_path = file_path
    
    # Load audio
    sample_rate, audio = wavfile.read(wav_path)
    
    # Convert to float32
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    
    logger.info(f"Audio loaded: {len(audio)/sample_rate:.1f} seconds at {sample_rate}Hz")
    
    # Load model
    logger.info("Loading model...")
    model = WhisperModel("large-v3", device="cpu", compute_type="int8", num_workers=4)
    
    # Streaming parameters (matching our daemon)
    chunk_duration = 3.0
    stride_duration = 1.0
    chunk_samples = int(chunk_duration * sample_rate)
    stride_samples = int(stride_duration * sample_rate)
    
    # Process in streaming fashion
    position = 0
    all_text = []
    prev_text = ""
    
    while position + chunk_samples <= len(audio):
        # Get chunk
        chunk = audio[position:position + chunk_samples]
        
        # Transcribe
        logger.info(f"Processing chunk at {position/sample_rate:.1f}s...")
        segments, info = model.transcribe(
            chunk,
            language="en",
            beam_size=5,
            temperature=0.0,
            without_timestamps=False
        )
        
        # Extract text from middle window (0.5s to 1.5s)
        chunk_text = []
        for segment in segments:
            if segment.start >= 0.5 and segment.start < 1.5:
                text = segment.text.strip()
                if text:
                    chunk_text.append(text)
        
        if chunk_text:
            output = " ".join(chunk_text)
            all_text.append(output)
            print(f"[{position/sample_rate:.1f}s] {output}")
        
        # Move forward by stride
        position += stride_samples
    
    # Process remaining audio
    if position < len(audio):
        remaining = audio[position:]
        segments, _ = model.transcribe(remaining, language="en", beam_size=5)
        for segment in segments:
            text = segment.text.strip()
            if text:
                all_text.append(text)
                print(f"[final] {text}")
    
    print("\n=== Complete Transcription ===")
    print(" ".join(all_text))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: test_streaming_file.py <audio_file.mp3>")
        sys.exit(1)
    
    process_file_streaming(sys.argv[1])