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
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd
from faster_whisper import WhisperModel

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

class WhisperDaemon:
    def __init__(self):
        self.sample_rate = 16000
        self.recording = False
        self.interrupted = False
        self.model_loaded = False
        
        # Socket for IPC
        self.socket_path = "/tmp/whisper_daemon.sock"
        self.server_socket = None
        
        # Sound file paths
        self.start_sound = "/home/caleb/whisper-dictation-daemon/tr707-snare-drum-241412.mp3"
        self.stop_sound = "/home/caleb/whisper-dictation-daemon/echoed-hi-hats-89491.mp3"
        
        # Load model once at startup
        self.load_model()
        
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
            logger.info("Model loaded and ready for transcription!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            sys.exit(1)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        logger.info("Received shutdown signal")
        self.interrupted = True
        self.recording = False
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

    def record_audio(self):
        """Record audio from microphone"""
        # Play start sound
        self.play_sound(self.start_sound)

        logger.info("Recording... Press Ctrl to stop")
        self.recording = True
        audio_data = []

        def callback(indata, frames, time, status):
            if self.recording:
                audio_data.append(indata.copy())

        with sd.InputStream(
            samplerate=self.sample_rate, channels=1, callback=callback, dtype=np.float32
        ):
            while self.recording and not self.interrupted:
                sd.sleep(100)

        # Play stop sound
        self.play_sound(self.stop_sound)

        if audio_data:
            return np.concatenate(audio_data, axis=0)
        return None

    def transcribe_audio(self, audio_data):
        """Transcribe audio using the pre-loaded model"""
        logger.info("Transcribing with pre-loaded model...")
        
        # Flatten audio data and ensure correct format
        audio_data = audio_data.flatten().astype(np.float32)
        
        try:
            # Use pre-loaded model (instant!)
            segments, info = self.model.transcribe(
                audio_data,
                language="en",
                task="transcribe",
                beam_size=5,
                best_of=5,
                temperature=0.0
            )
            
            transcribed_text = " ".join([segment.text for segment in segments]).strip()
            logger.info(f"Transcription: {transcribed_text}")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""

    def type_text(self, text):
        """Type text using wtype"""
        if text:
            try:
                subprocess.run(["wtype", text], check=True)
                logger.info(f"Typed: {text}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to type text: {e}")

    def dictate(self):
        """Main dictation function"""
        audio_data = self.record_audio()
        if audio_data is not None and len(audio_data) > 0:
            text = self.transcribe_audio(audio_data)
            if text:
                self.type_text(text)
                return text
            else:
                logger.info("No speech detected")
        else:
            logger.info("No audio recorded")
        return None

    def handle_client(self, client_socket):
        """Handle client requests"""
        try:
            data = client_socket.recv(1024).decode()
            if data == "DICTATE":
                self.dictate()
                client_socket.send(b"OK")
            elif data == "STOP":
                self.recording = False
                client_socket.send(b"OK")
            elif data == "STATUS":
                status = "READY" if self.model_loaded else "LOADING"
                client_socket.send(status.encode())
        except Exception as e:
            logger.error(f"Client handling error: {e}")
        finally:
            client_socket.close()

    def start_daemon(self):
        """Start the daemon server"""
        logger.info("Starting Whisper daemon...")
        
        # Remove existing socket
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        
        # Create Unix socket
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(5)
        
        logger.info(f"Daemon listening on {self.socket_path}")
        logger.info("Model is loaded and ready for instant transcription!")
        
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
    daemon = WhisperDaemon()
    daemon.start_daemon()

if __name__ == "__main__":
    main()