#!/usr/bin/env python3
"""
Unified Whisper Streaming Daemon with multiple algorithms
"""
import argparse
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
import socket
import threading
import queue
from pathlib import Path
from collections import deque

import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd
from faster_whisper import WhisperModel

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

class LocalAgreementBuffer:
    """
    Implements the Local Agreement-n algorithm for streaming transcription.
    Confirms text when n consecutive chunks agree on a prefix.
    """
    def __init__(self, n=2):
        self.n = n  # Number of agreements needed
        self.history = deque(maxlen=n)
        self.confirmed_text = ""
        
    def update(self, new_text):
        """Update with new transcription and return confirmed text if any"""
        self.history.append(new_text)
        
        if len(self.history) < self.n:
            return None  # Not enough history yet
        
        # Find common prefix among all texts in history
        common_prefix = self.find_common_prefix(list(self.history))
        
        # Check if we have new confirmed text
        if common_prefix and len(common_prefix) > len(self.confirmed_text):
            # Find the new portion
            new_confirmed = common_prefix[len(self.confirmed_text):]
            self.confirmed_text = common_prefix
            return new_confirmed
        
        return None
    
    def find_common_prefix(self, texts):
        """Find the longest common prefix among all texts"""
        if not texts:
            return ""
        
        # Start with the first text
        prefix = texts[0]
        
        # Compare with each subsequent text
        for text in texts[1:]:
            # Find word-boundary aligned common prefix
            words1 = prefix.split()
            words2 = text.split()
            
            common_words = []
            for w1, w2 in zip(words1, words2):
                if w1 == w2:
                    common_words.append(w1)
                else:
                    break
            
            prefix = " ".join(common_words)
            
        return prefix

class WhisperStreamingDaemon:
    def __init__(self, algorithm="local-agreement", model_size="large-v3", no_streaming=False):
        self.sample_rate = 16000
        self.recording = False
        self.streaming = False
        self.interrupted = False
        self.model_loaded = False
        self.algorithm = algorithm
        self.model_size = model_size
        self.no_streaming = no_streaming  # If True, wait for complete recording
        
        # Algorithm-specific parameters
        if algorithm == "local-agreement":
            self.chunk_duration = 2.0  # Process 2-second chunks
            self.max_buffer_duration = 10.0  # Maximum buffer size
        elif algorithm == "sliding-window":
            self.chunk_duration = 3.0  # Total chunk size to process
            self.stride_duration = 1.0  # Move forward by 1 second each time
        else:  # simple
            self.chunk_duration = 5.0  # Simple 5-second chunks
        
        self.silence_threshold = 0.01
        
        # Queues for processing
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        
        # Buffer for continuous audio
        self.audio_buffer = []
        self.chunk_samples = int(self.chunk_duration * self.sample_rate)
        
        # Algorithm-specific state
        if algorithm == "local-agreement":
            self.local_agreement = LocalAgreementBuffer(n=2)
        elif algorithm == "sliding-window":
            self.stride_samples = int(self.stride_duration * self.sample_rate)
            self.last_chunk_text = ""
        
        # Socket for IPC
        if no_streaming:
            self.socket_path = f"/tmp/whisper_streaming_{algorithm}_nostream_daemon.sock"
        else:
            self.socket_path = f"/tmp/whisper_streaming_{algorithm}_daemon.sock"
        self.server_socket = None
        
        # Sound file paths
        self.start_sound = "/home/caleb/whisper-dictation-daemon/tr707-snare-drum-241412.mp3"
        self.stop_sound = "/home/caleb/whisper-dictation-daemon/echoed-hi-hats-89491.mp3"
        
        # Load model once at startup
        self.load_model()
        
        # Start processing threads
        self.start_processing_threads()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def load_model(self):
        """Load the Whisper model once at startup"""
        logger.info(f"Loading faster-whisper {self.model_size} model on CPU...")
        logger.info("CPU: 24-core Ryzen 9 9900X")
        
        try:
            self.model = WhisperModel(
                self.model_size, 
                device="cpu", 
                compute_type="int8",
                num_workers=4
            )
            self.model_loaded = True
            logger.info(f"Model loaded with {self.algorithm} algorithm!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            sys.exit(1)

    def start_processing_threads(self):
        """Start background threads for audio processing"""
        # Choose processor based on algorithm
        if self.algorithm == "local-agreement":
            processor = self.process_audio_local_agreement
        elif self.algorithm == "sliding-window":
            processor = self.process_audio_sliding_window
        else:
            processor = self.process_audio_simple
        
        self.audio_processor = threading.Thread(target=processor, daemon=True)
        self.audio_processor.start()
        
        # Thread for outputting transcriptions
        self.transcription_outputter = threading.Thread(target=self.output_transcriptions, daemon=True)
        self.transcription_outputter.start()

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        logger.info("Received shutdown signal")
        self.interrupted = True
        self.recording = False
        self.streaming = False
        if self.server_socket:
            self.server_socket.close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        sys.exit(0)

    def play_sound(self, sound_file):
        """Play a sound file using ffplay in background"""
        try:
            subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", sound_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            logger.error(f"Failed to play sound {sound_file}: {e}")

    def start_streaming_recording(self):
        """Start recording with streaming transcription"""
        # Play start sound
        self.play_sound(self.start_sound)
        
        if self.no_streaming:
            logger.info("Recording audio (non-streaming mode)... Press Ctrl to stop")
        else:
            logger.info(f"Starting streaming with {self.algorithm}... Press Ctrl to stop")
        
        self.recording = True
        self.streaming = not self.no_streaming  # Only stream if not in no_streaming mode
        self.audio_buffer = []
        
        # Reset algorithm-specific state
        if self.algorithm == "local-agreement":
            self.local_agreement = LocalAgreementBuffer(n=2)
        elif self.algorithm == "sliding-window":
            self.last_chunk_text = ""
        
        def audio_callback(indata, frames, time, status):
            if self.recording:
                # Add to buffer
                audio_chunk = indata.copy().flatten()
                self.audio_buffer.extend(audio_chunk)
                
                # If no_streaming mode, just collect audio, don't process yet
                if self.no_streaming:
                    return
                
                # Check if we have enough for processing
                if self.algorithm == "sliding-window":
                    if len(self.audio_buffer) >= self.chunk_samples:
                        # Extract full chunk
                        chunk = np.array(self.audio_buffer[:self.chunk_samples])
                        # Move forward by stride
                        self.audio_buffer = self.audio_buffer[self.stride_samples:]
                        self.audio_queue.put(chunk)
                else:
                    if len(self.audio_buffer) >= self.chunk_samples:
                        # Take the buffered audio
                        chunk_to_process = self.audio_buffer.copy()
                        
                        if self.algorithm == "local-agreement":
                            # Keep last 0.5 seconds for context
                            keep_samples = int(0.5 * self.sample_rate)
                            self.audio_buffer = self.audio_buffer[-keep_samples:]
                        else:  # simple
                            # Clear the buffer
                            self.audio_buffer = []
                        
                        self.audio_queue.put(np.array(chunk_to_process))
        
        with sd.InputStream(
            samplerate=self.sample_rate, 
            channels=1, 
            callback=audio_callback, 
            dtype=np.float32
        ):
            while self.recording and not self.interrupted:
                sd.sleep(100)
        
        # Handle end of recording differently for no_streaming mode
        if self.no_streaming:
            # Process the entire recording at once
            if self.audio_buffer:
                logger.info(f"Processing complete recording ({len(self.audio_buffer)/self.sample_rate:.1f}s)...")
                full_audio = np.array(self.audio_buffer)
                
                # Transcribe the entire audio
                try:
                    segments, info = self.model.transcribe(
                        full_audio,
                        language="en",
                        beam_size=5,
                        temperature=0.0,
                        vad_filter=True
                    )
                    
                    # Output all text at once
                    full_text = " ".join([segment.text.strip() for segment in segments])
                    if full_text:
                        self.transcription_queue.put(full_text)
                        logger.info("Transcription complete")
                        
                except Exception as e:
                    logger.error(f"Transcription error: {e}")
            
            self.transcription_queue.put(None)
        else:
            # Process any remaining audio for streaming mode
            if len(self.audio_buffer) > int(0.5 * self.sample_rate):
                final_chunk = np.array(self.audio_buffer)
                self.audio_queue.put(final_chunk)
            
            # Signal end of stream
            self.audio_queue.put(None)
        
        self.streaming = False
        
        # Play stop sound
        self.play_sound(self.stop_sound)
        
        logger.info("Recording stopped")

    def process_audio_local_agreement(self):
        """Process audio chunks with Local Agreement algorithm"""
        context_audio = []  # Keep growing context
        last_full_text = ""
        
        while True:
            try:
                chunk = self.audio_queue.get(timeout=1)
                
                if chunk is None:
                    # Output any remaining unconfirmed text at the end
                    if last_full_text and len(last_full_text) > len(self.local_agreement.confirmed_text):
                        remaining = last_full_text[len(self.local_agreement.confirmed_text):]
                        self.transcription_queue.put(remaining)
                    self.transcription_queue.put(None)
                    continue
                
                # Skip if too quiet
                if np.max(np.abs(chunk)) < self.silence_threshold:
                    continue
                
                # Add to context
                context_audio.extend(chunk)
                
                # Limit context size
                max_samples = int(self.max_buffer_duration * self.sample_rate)
                if len(context_audio) > max_samples:
                    context_audio = context_audio[-max_samples:]
                
                # Transcribe the full context
                logger.info(f"Processing {len(context_audio)/self.sample_rate:.1f}s of audio...")
                
                try:
                    segments, info = self.model.transcribe(
                        np.array(context_audio),
                        language="en",
                        beam_size=5,
                        temperature=0.0,
                        vad_filter=True,
                        without_timestamps=True
                    )
                    
                    # Collect full transcription
                    full_text = " ".join([segment.text.strip() for segment in segments])
                    
                    if full_text:
                        # Update local agreement buffer
                        confirmed_text = self.local_agreement.update(full_text)
                        
                        if confirmed_text:
                            self.transcription_queue.put(confirmed_text)
                            logger.info(f"Confirmed: {confirmed_text}")
                        
                        last_full_text = full_text
                        
                except Exception as e:
                    logger.error(f"Transcription error: {e}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio processing error: {e}")

    def process_audio_sliding_window(self):
        """Process audio with sliding window and timestamp-based deduplication"""
        while True:
            try:
                chunk = self.audio_queue.get(timeout=1)
                
                if chunk is None:
                    self.transcription_queue.put(None)
                    continue
                
                # Skip if too quiet
                if np.max(np.abs(chunk)) < self.silence_threshold:
                    continue
                
                logger.info(f"Processing {len(chunk)/self.sample_rate:.1f}s chunk...")
                
                try:
                    segments, info = self.model.transcribe(
                        chunk,
                        language="en",
                        beam_size=5,
                        temperature=0.0,
                        vad_filter=True,
                        without_timestamps=False  # Keep timestamps for alignment
                    )
                    
                    # Only output text from the middle portion
                    for segment in segments:
                        # Check if segment is in our target window (0.5s to 1.5s)
                        if segment.start >= 0.5 and segment.start < 1.5:
                            text = segment.text.strip()
                            if text:
                                self.transcription_queue.put(text + " ")
                                logger.info(f"Streamed: {text}")
                        
                except Exception as e:
                    logger.error(f"Transcription error: {e}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio processing error: {e}")

    def process_audio_simple(self):
        """Simple processing - transcribe chunks as they come"""
        while True:
            try:
                chunk = self.audio_queue.get(timeout=1)
                
                if chunk is None:
                    self.transcription_queue.put(None)
                    continue
                
                # Skip if too quiet
                if np.max(np.abs(chunk)) < self.silence_threshold:
                    continue
                
                logger.info(f"Processing {len(chunk)/self.sample_rate:.1f}s chunk...")
                
                try:
                    segments, info = self.model.transcribe(
                        chunk,
                        language="en",
                        beam_size=5,
                        temperature=0.0,
                        vad_filter=True,
                        without_timestamps=True
                    )
                    
                    # Collect text
                    text = " ".join([segment.text.strip() for segment in segments])
                    
                    if text:
                        self.transcription_queue.put(text + " ")
                        logger.info(f"Transcribed: {text}")
                        
                except Exception as e:
                    logger.error(f"Transcription error: {e}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio processing error: {e}")

    def output_transcriptions(self):
        """Output transcriptions as they become available"""
        while True:
            try:
                text = self.transcription_queue.get(timeout=1)
                
                if text is None:
                    logger.info("End of transcription stream")
                    continue
                
                # Type the text immediately
                self.type_text(text)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Transcription output error: {e}")

    def type_text(self, text):
        """Type text using wtype"""
        if text:
            try:
                subprocess.run(["wtype", text], check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to type text: {e}")

    def process_file(self, file_path):
        """Process an audio file for testing"""
        logger.info(f"Processing file: {file_path}")
        
        # Convert if MP3
        if file_path.endswith('.mp3'):
            logger.info("Converting MP3 to WAV...")
            wav_path = self.convert_to_wav(file_path)
        else:
            wav_path = file_path
        
        # Load audio
        sample_rate, audio = wavfile.read(wav_path)
        
        # Convert to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        
        logger.info(f"Audio loaded: {len(audio)/sample_rate:.1f} seconds")
        
        # Process based on algorithm
        if self.algorithm == "local-agreement":
            self.process_file_local_agreement(audio)
        elif self.algorithm == "sliding-window":
            self.process_file_sliding_window(audio)
        else:
            self.process_file_simple(audio)
    
    def convert_to_wav(self, mp3_file):
        """Convert MP3 to WAV using ffmpeg"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
            wav_path = tmp_wav.name
            
        cmd = [
            'ffmpeg', '-i', mp3_file,
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',      # Mono
            '-f', 'wav',
            '-y',  # Overwrite
            wav_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg error: {result.stderr}")
            raise Exception(f"Failed to convert MP3: {result.stderr}")
        
        return wav_path
    
    def process_file_local_agreement(self, audio):
        """Process file with local agreement"""
        local_agreement = LocalAgreementBuffer(n=2)
        position = 0
        chunk_samples = int(2.0 * self.sample_rate)
        context_audio = []
        last_full_text = ""
        
        while position < len(audio):
            # Get next chunk
            end = min(position + chunk_samples, len(audio))
            chunk = audio[position:end]
            context_audio.extend(chunk)
            
            # Limit context
            max_samples = int(10.0 * self.sample_rate)
            if len(context_audio) > max_samples:
                context_audio = context_audio[-max_samples:]
            
            # Transcribe
            segments, _ = self.model.transcribe(
                np.array(context_audio),
                language="en",
                beam_size=5,
                without_timestamps=True
            )
            
            full_text = " ".join([s.text.strip() for s in segments])
            
            if full_text:
                confirmed = local_agreement.update(full_text)
                if confirmed:
                    print(confirmed, end=" ", flush=True)
                last_full_text = full_text
            
            position = end
        
        # Output any remaining unconfirmed text at the end
        if last_full_text and len(last_full_text) > len(local_agreement.confirmed_text):
            remaining = last_full_text[len(local_agreement.confirmed_text):]
            print(remaining, end=" ", flush=True)
        
        print()  # Final newline
    
    def process_file_sliding_window(self, audio):
        """Process file with sliding window"""
        chunk_samples = int(3.0 * self.sample_rate)
        stride_samples = int(1.0 * self.sample_rate)
        position = 0
        
        while position + chunk_samples <= len(audio):
            chunk = audio[position:position + chunk_samples]
            
            segments, _ = self.model.transcribe(
                chunk,
                language="en",
                beam_size=5,
                without_timestamps=False
            )
            
            # Output middle portion
            for segment in segments:
                if segment.start >= 0.5 and segment.start < 1.5:
                    text = segment.text.strip()
                    if text:
                        print(text, end=" ", flush=True)
            
            position += stride_samples
        
        print()  # Final newline
    
    def process_file_simple(self, audio):
        """Simple file processing"""
        segments, _ = self.model.transcribe(
            audio,
            language="en",
            beam_size=5
        )
        
        for segment in segments:
            print(segment.text.strip(), end=" ", flush=True)
        print()

    def handle_client(self, client_socket):
        """Handle client requests"""
        try:
            data = client_socket.recv(1024).decode()
            if data == "STREAM_START":
                if not self.streaming and not self.recording:
                    threading.Thread(target=self.start_streaming_recording).start()
                    if self.no_streaming:
                        client_socket.send(b"RECORDING")
                    else:
                        client_socket.send(b"STREAMING")
                else:
                    if self.no_streaming:
                        client_socket.send(b"ALREADY_RECORDING")
                    else:
                        client_socket.send(b"ALREADY_STREAMING")
            elif data == "STREAM_STOP":
                self.recording = False
                client_socket.send(b"STOPPED")
            elif data == "STATUS":
                if self.recording:
                    status = "RECORDING" if self.no_streaming else "STREAMING"
                elif self.model_loaded:
                    status = "READY"
                else:
                    status = "LOADING"
                client_socket.send(status.encode())
        except Exception as e:
            logger.error(f"Client handling error: {e}")
        finally:
            client_socket.close()

    def start_daemon(self):
        """Start the daemon server"""
        logger.info(f"Starting Whisper daemon with {self.algorithm} algorithm...")
        
        # Remove existing socket
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        
        # Create Unix socket
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(5)
        
        logger.info(f"Daemon listening on {self.socket_path}")
        logger.info(f"Algorithm: {self.algorithm}, Model: {self.model_size}")
        
        while not self.interrupted:
            try:
                client_socket, _ = self.server_socket.accept()
                client_thread = threading.Thread(
                    target=self.handle_client, 
                    args=(client_socket,)
                )
                client_thread.daemon = True
                client_thread.start()
            except OSError:
                if not self.interrupted:
                    logger.error("Socket error")
                break

def main():
    parser = argparse.ArgumentParser(description='Whisper Streaming Daemon')
    parser.add_argument('--algorithm', '-a', 
                       choices=['local-agreement', 'sliding-window', 'simple'],
                       default='local-agreement',
                       help='Streaming algorithm to use (default: local-agreement)')
    parser.add_argument('--model', '-m',
                       default='large-v3',
                       help='Whisper model size (default: large-v3)')
    parser.add_argument('--no-streaming', '-n',
                       action='store_true',
                       help='Wait for complete recording before transcribing (non-streaming mode)')
    parser.add_argument('--file', '-f',
                       help='Process an audio file instead of starting daemon')
    parser.add_argument('--client', '-c',
                       choices=['start', 'stop', 'status', 'toggle'],
                       help='Send command to running daemon')
    
    args = parser.parse_args()
    
    # Client mode
    if args.client:
        if args.no_streaming:
            socket_path = f"/tmp/whisper_streaming_{args.algorithm}_nostream_daemon.sock"
        else:
            socket_path = f"/tmp/whisper_streaming_{args.algorithm}_daemon.sock"
        try:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(socket_path)
            
            if args.client == 'start':
                client.send(b"STREAM_START")
            elif args.client == 'stop':
                client.send(b"STREAM_STOP")
            elif args.client == 'status':
                client.send(b"STATUS")
            elif args.client == 'toggle':
                # First check status
                client.send(b"STATUS")
                status = client.recv(1024).decode()
                client.close()
                
                # Then toggle
                client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client.connect(socket_path)
                if status in ["STREAMING", "RECORDING"]:
                    client.send(b"STREAM_STOP")
                    print(f"Stopped {'recording' if status == 'RECORDING' else 'streaming'}")
                else:
                    client.send(b"STREAM_START")
                    print(f"Started {'recording' if args.no_streaming else 'streaming'}")
            
            response = client.recv(1024).decode()
            if args.client != 'toggle':
                print(f"Response: {response}")
            client.close()
            
        except FileNotFoundError:
            print(f"Daemon not running! Start with: python3 whisper_streaming.py -a {args.algorithm}")
        except Exception as e:
            print(f"Error: {e}")
        
        return
    
    # Create daemon
    daemon = WhisperStreamingDaemon(
        algorithm=args.algorithm, 
        model_size=args.model,
        no_streaming=args.no_streaming
    )
    
    # File processing mode
    if args.file:
        daemon.process_file(args.file)
    else:
        # Daemon mode
        daemon.start_daemon()

if __name__ == "__main__":
    main()