#!/usr/bin/env python3
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

import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd
from faster_whisper import WhisperModel

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

class WhisperStreamingDaemon:
    def __init__(self):
        self.sample_rate = 16000
        self.recording = False
        self.streaming = False
        self.interrupted = False
        self.model_loaded = False
        
        # Streaming parameters
        self.chunk_duration = 2.0  # Process 2-second chunks
        self.overlap_duration = 0.5  # 0.5 second overlap between chunks
        self.silence_threshold = 0.01  # Silence detection threshold
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        
        # Buffer for continuous audio
        self.audio_buffer = []
        self.chunk_samples = int(self.chunk_duration * self.sample_rate)
        self.overlap_samples = int(self.overlap_duration * self.sample_rate)
        
        # Socket for IPC
        self.socket_path = "/tmp/whisper_streaming_daemon.sock"
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
        logger.info("Loading faster-whisper large-v3 model on CPU...")
        logger.info("CPU: 24-core Ryzen 9 9900X")
        
        try:
            self.model = WhisperModel(
                "large-v3", 
                device="cpu", 
                compute_type="int8",
                num_workers=4
            )
            self.model_loaded = True
            logger.info("Model loaded and ready for streaming transcription!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            sys.exit(1)

    def start_processing_threads(self):
        """Start background threads for audio processing"""
        # Thread for processing audio chunks
        self.audio_processor = threading.Thread(target=self.process_audio_chunks, daemon=True)
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
        
        logger.info("Starting streaming transcription... Press Ctrl to stop")
        self.recording = True
        self.streaming = True
        self.audio_buffer = []
        
        def audio_callback(indata, frames, time, status):
            if self.recording:
                # Add to buffer for continuous recording
                audio_chunk = indata.copy().flatten()
                self.audio_buffer.extend(audio_chunk)
                
                # Check if we have enough for a chunk
                if len(self.audio_buffer) >= self.chunk_samples:
                    # Extract chunk with overlap from previous
                    chunk = np.array(self.audio_buffer[:self.chunk_samples])
                    
                    # Keep overlap for next chunk
                    self.audio_buffer = self.audio_buffer[self.chunk_samples - self.overlap_samples:]
                    
                    # Add to processing queue
                    self.audio_queue.put(chunk)
        
        with sd.InputStream(
            samplerate=self.sample_rate, 
            channels=1, 
            callback=audio_callback, 
            dtype=np.float32
        ):
            while self.recording and not self.interrupted:
                sd.sleep(100)
        
        # Process any remaining audio
        if len(self.audio_buffer) > 0:
            final_chunk = np.array(self.audio_buffer)
            self.audio_queue.put(final_chunk)
        
        # Signal end of stream
        self.audio_queue.put(None)
        self.streaming = False
        
        # Play stop sound
        self.play_sound(self.stop_sound)
        
        logger.info("Streaming stopped")

    def process_audio_chunks(self):
        """Process audio chunks in background thread"""
        while True:
            try:
                # Get audio chunk from queue
                chunk = self.audio_queue.get(timeout=1)
                
                if chunk is None:
                    # End of stream marker
                    self.transcription_queue.put(None)
                    continue
                
                # Skip if too quiet (silence)
                if np.max(np.abs(chunk)) < self.silence_threshold:
                    continue
                
                # Transcribe the chunk
                logger.info(f"Processing {len(chunk)/self.sample_rate:.1f}s chunk...")
                
                try:
                    segments, info = self.model.transcribe(
                        chunk,
                        language="en",
                        task="transcribe",
                        beam_size=5,
                        best_of=5,
                        temperature=0.0,
                        condition_on_previous_text=False,  # Don't condition on previous for real-time
                        vad_filter=True,  # Use VAD to filter out silence
                        vad_parameters=dict(
                            min_speech_duration_ms=100,
                            min_silence_duration_ms=100
                        )
                    )
                    
                    # Collect text from segments
                    text = " ".join([segment.text.strip() for segment in segments])
                    
                    if text:
                        self.transcription_queue.put(text)
                        
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
                    # End of stream
                    logger.info("End of transcription stream")
                    continue
                
                # Type the text immediately
                self.type_text(text + " ")
                logger.info(f"Streamed: {text}")
                
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

    def handle_client(self, client_socket):
        """Handle client requests"""
        try:
            data = client_socket.recv(1024).decode()
            if data == "STREAM_START":
                if not self.streaming:
                    threading.Thread(target=self.start_streaming_recording).start()
                    client_socket.send(b"STREAMING")
                else:
                    client_socket.send(b"ALREADY_STREAMING")
            elif data == "STREAM_STOP":
                self.recording = False
                client_socket.send(b"STOPPED")
            elif data == "STATUS":
                if self.streaming:
                    status = "STREAMING"
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
        logger.info("Starting Whisper streaming daemon...")
        
        # Remove existing socket
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        
        # Create Unix socket
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(5)
        
        logger.info(f"Streaming daemon listening on {self.socket_path}")
        logger.info("Model loaded - ready for real-time streaming transcription!")
        
        while not self.interrupted:
            try:
                client_socket, _ = self.server_socket.accept()
                # Handle each request in a separate thread
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
    daemon = WhisperStreamingDaemon()
    daemon.start_daemon()

if __name__ == "__main__":
    main()